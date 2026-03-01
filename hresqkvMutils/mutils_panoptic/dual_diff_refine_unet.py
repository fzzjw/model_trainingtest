# dual_diff_refine_unet.py
# ------------------------------------------------------------
# 用法：
#   1) 保持你原来的 Unet.py 不动
#   2) 本文件与 Unet.py 放在同一目录
#   3) from dual_diff_refine_unet import DualDiffRefineUNet
#
# 机制：
#   左右分支各取 tap 特征 -> (raw diff 或 QKV cross-attn 对齐后 diff)
#   -> 两个不同卷积头生成 cx_left/cx_right
#   -> 注入各自 decoder 的 inject_layer 位置（复用 Unet.py 的 external concat 思路）
# ------------------------------------------------------------

import torch
from torch import nn, Tensor
import torch.nn.functional as F

# 复用已有的 UNetConvBlock / UNetUpBlock（来自 Unet.py）
from MuTILs_Panoptic.mutils_panoptic.Unet import UNetConvBlock, UNetUpBlock


class UNetED(nn.Module):
    """
    UNet 的 Encoder-Decoder 拆分版本：
    - encode(): 返回 bottleneck、skip blocks、tap feature
    - decode(): 在 decoder 指定层注入 cx（external concat）
    """

    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        up_mode='upconv',
        external_concat_layer: int | None = None,   # decoder stage index: 0 最靠近 bottleneck
        external_concat_nc: int | None = None,      # cx 的通道数
    ):
        super().__init__()
        assert up_mode in ('upconv', 'upsample')
        self.depth = depth
        self.wf = wf
        self.padding = padding

        self.ecl = external_concat_layer
        self.ecnc = external_concat_nc
        self.econcat = self.ecl is not None
        if self.econcat and self.ecnc is None:
            raise ValueError("external_concat_nc must be set when external_concat_layer is not None")

        prev_channels = in_channels

        # encoder
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm))
            prev_channels = 2 ** (wf + i)

        # decoder
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            stage_index = depth - i - 2  # 0..depth-2，对应 up_path 的 index
            if self.econcat and (stage_index == self.ecl):
                in_size = prev_channels + self.ecnc
                stage2_in_size = prev_channels
            else:
                in_size = stage2_in_size = prev_channels

            self.up_path.append(
                UNetUpBlock(
                    in_size=in_size,
                    out_size=2 ** (wf + i),
                    up_mode=up_mode,
                    padding=padding,
                    batch_norm=batch_norm,
                    stage2_in_size=stage2_in_size,
                )
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def encode(
        self,
        x: Tensor,
        tap_depth: int | None = None,
        fetch_layers: list[int] | None = None,
        rd_start: int = 0,
    ):
        """
        tap_depth:
          - None 或 depth-1：取 bottleneck
          - 0..depth-2：取对应 encoder level 的 skip feature（pooling 前）
        """
        blocks = []
        tap = None
        feats: list[Tensor] = []
        rd = rd_start

        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)  # skip
                if tap_depth is not None and tap_depth == i:
                    tap = x
                x = F.avg_pool2d(x, 2)
            if fetch_layers is not None and rd in fetch_layers:
                feats.append(0. + x)
            rd += 1

        # bottleneck
        if tap_depth is None or tap_depth == len(self.down_path) - 1:
            tap = x

        return x, blocks, tap, feats, rd

    def decode(
        self,
        x: Tensor,
        blocks: list[Tensor],
        cx: Tensor | None = None,
        fetch_layers: list[int] | None = None,
        rd_start: int = 0,
    ):
        if self.econcat and cx is None:
            raise ValueError("Need cx when external concat is enabled")

        feats: list[Tensor] = []
        rd = rd_start
        for i, up in enumerate(self.up_path):
            if self.econcat and (i == self.ecl):
                # 保证拼接时空间尺寸一致
                if cx.shape[2:] != x.shape[2:]:
                    cx = F.interpolate(cx, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, cx], dim=1)

            x = up(x, blocks[-i - 1])
            if fetch_layers is not None and rd in fetch_layers:
                feats.append(0. + x)
            rd += 1

        return self.last(x), feats, rd


class CrossAttention2D(nn.Module):
    """
    QKV cross-attention：
    - Q 来自 q_map
    - K/V 来自 kv_map

    注意：建议在 bottleneck 等低分辨率特征上使用，避免 O((HW)^2) 计算开销过大。
    """

    def __init__(self, in_channels: int, d_model: int = 256, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.q_proj = nn.Conv2d(in_channels, d_model, 1, bias=False)
        self.k_proj = nn.Conv2d(in_channels, d_model, 1, bias=False)
        self.v_proj = nn.Conv2d(in_channels, d_model, 1, bias=False)

        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_k = nn.LayerNorm(d_model)
        self.ln_v = nn.LayerNorm(d_model)

        self.out_proj = nn.Conv2d(d_model, in_channels, 1, bias=False)

    def forward(self, q_map: Tensor, kv_map: Tensor) -> Tensor:
        b, c, h, w = q_map.shape
        q = self.q_proj(q_map).flatten(2).transpose(1, 2)   # B, HW, E
        k = self.k_proj(kv_map).flatten(2).transpose(1, 2)
        v = self.v_proj(kv_map).flatten(2).transpose(1, 2)

        q = self.ln_q(q)
        k = self.ln_k(k)
        v = self.ln_v(v)

        out, _ = self.attn(q, k, v, need_weights=False)     # B, HW, E
        out = out.transpose(1, 2).reshape(b, -1, h, w)      # B, E, H, W
        return self.out_proj(out)                            # B, C, H, W


class DiffRefineHead(nn.Module):
    """
    用 diff 特征生成 cx（供 decoder concat 注入）。

    提供 A/B 两个变体：
      - A: 3x3 + dilated 3x3 + 1x1
      - B: 5x5 + 3x3 + 1x1
    """

    def __init__(self, in_channels: int, out_channels: int, variant: str = "A", batch_norm: bool = False):
        super().__init__()

        def conv(in_ch, out_ch, k=3, d=1):
            pad = (k // 2) * d
            layers = [
                nn.Conv2d(in_ch, out_ch, k, padding=pad, dilation=d, bias=not batch_norm),
                nn.ReLU(inplace=True),
            ]
            if batch_norm:
                layers.insert(1, nn.BatchNorm2d(out_ch))
            return nn.Sequential(*layers)

        v = variant.upper()
        if v == "A":
            self.net = nn.Sequential(
                conv(in_channels, in_channels, k=3, d=1),
                conv(in_channels, in_channels, k=3, d=2),
                nn.Conv2d(in_channels, out_channels, 1),
            )
        elif v == "B":
            self.net = nn.Sequential(
                conv(in_channels, in_channels, k=5, d=1),
                conv(in_channels, in_channels, k=3, d=1),
                nn.Conv2d(in_channels, out_channels, 1),
            )
        else:
            raise ValueError("variant must be 'A' or 'B'")

    def forward(self, diff: Tensor) -> Tensor:
        return self.net(diff)


class DualDiffRefineUNet(nn.Module):
    """
    双分支 UNet + 差分细化注入。

    forward(x_left, x_right):
      1) 两侧 encoder 得到 tapL / tapR
      2) 计算差分（raw 或 qkv）
      3) 差分进入两套 refine head，生成 cxL / cxR
      4) 分别注入左右 decoder
      5) 融合 outL / outR（avg / left / right / concat_head）
    """

    def __init__(
        self,
        in_channels_left=1,
        in_channels_right=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        up_mode='upconv',
        tap_depth: int | None = None,     # 默认 bottleneck（depth-1）
        inject_layer: int = 0,            # decoder stage index，0 最靠近 bottleneck
        refine_channels: int | None = None,
        diff_mode: str = "raw",           # "raw" 或 "qkv"
        attn_d_model: int = 256,
        attn_heads: int = 8,
        attn_dropout: float = 0.0,
        qkv_alpha_init: float = -6.0,
        fuse: str = "avg",                # "avg" | "left" | "right" | "concat_head"
    ):
        super().__init__()

        self.depth = depth
        self.wf = wf
        self.tap_depth = (depth - 1) if tap_depth is None else tap_depth
        self.inject_layer = inject_layer

        # tap 特征通道：encoder 第 i 层输出通道 = 2**(wf+i)
        self.tap_channels = 2 ** (wf + self.tap_depth)
        self.refine_channels = self.tap_channels if refine_channels is None else refine_channels

        self.diff_mode = diff_mode.lower()
        self.fuse = fuse.lower()

        self.unet_left = UNetED(
            in_channels=in_channels_left,
            n_classes=n_classes,
            depth=depth,
            wf=wf,
            padding=padding,
            batch_norm=batch_norm,
            up_mode=up_mode,
            external_concat_layer=inject_layer,
            external_concat_nc=self.refine_channels,
        )
        self.unet_right = UNetED(
            in_channels=in_channels_right,
            n_classes=n_classes,
            depth=depth,
            wf=wf,
            padding=padding,
            batch_norm=batch_norm,
            up_mode=up_mode,
            external_concat_layer=inject_layer,
            external_concat_nc=self.refine_channels,
        )

        if self.diff_mode == "qkv":
            self.attn_l = CrossAttention2D(self.tap_channels, d_model=attn_d_model, num_heads=attn_heads, dropout=attn_dropout)
            self.attn_r = CrossAttention2D(self.tap_channels, d_model=attn_d_model, num_heads=attn_heads, dropout=attn_dropout)
            # 初期接近 raw-diff，随后逐步学习 qkv 差分
            self.qkv_alpha = nn.Parameter(torch.tensor(float(qkv_alpha_init)))
        elif self.diff_mode != "raw":
            raise ValueError("diff_mode must be 'raw' or 'qkv'")
        else:
            self.qkv_alpha = None

        # 两个“细化分支”
        self.refine_left = DiffRefineHead(self.tap_channels, self.refine_channels, variant="A", batch_norm=batch_norm)
        self.refine_right = DiffRefineHead(self.tap_channels, self.refine_channels, variant="B", batch_norm=batch_norm)

        if self.fuse == "concat_head":
            self.fuse_head = nn.Conv2d(n_classes * 2, n_classes, kernel_size=1)

    def _compute_diff(self, tapL: Tensor, tapR: Tensor):
        rawL = tapL - tapR
        rawR = tapR - tapL
        if self.diff_mode == "raw":
            return rawL, rawR

        r_to_l = self.attn_l(tapL, tapR)   # right -> left
        l_to_r = self.attn_r(tapR, tapL)   # left  -> right
        qkvL = tapL - r_to_l
        qkvR = tapR - l_to_r
        alpha = torch.sigmoid(self.qkv_alpha)
        return rawL + alpha * (qkvL - rawL), rawR + alpha * (qkvR - rawR)

    def forward(
        self,
        x_left: Tensor,
        x_right: Tensor,
        fetch_layers: list[int] | None = None,
        return_branches: bool = False,
        feature_branch: str = "left",
    ):
        bnL, skipsL, tapL, featsL, rdL = self.unet_left.encode(
            x_left, tap_depth=self.tap_depth, fetch_layers=fetch_layers, rd_start=0
        )
        bnR, skipsR, tapR, featsR, rdR = self.unet_right.encode(
            x_right, tap_depth=self.tap_depth, fetch_layers=fetch_layers, rd_start=0
        )

        # 保证左右 tap 的空间尺寸一致
        if tapL.shape[2:] != tapR.shape[2:]:
            tapR = F.interpolate(tapR, size=tapL.shape[2:], mode="bilinear", align_corners=False)

        diffL, diffR = self._compute_diff(tapL, tapR)

        cxL = self.refine_left(diffL)
        cxR = self.refine_right(diffR)

        outL, featsL_dec, _ = self.unet_left.decode(
            bnL, skipsL, cx=cxL, fetch_layers=fetch_layers, rd_start=rdL
        )
        outR, featsR_dec, _ = self.unet_right.decode(
            bnR, skipsR, cx=cxR, fetch_layers=fetch_layers, rd_start=rdR
        )

        if self.fuse == "left":
            out = outL
        elif self.fuse == "right":
            out = outR
        elif self.fuse == "avg":
            out = 0.5 * (outL + outR)
        elif self.fuse == "concat_head":
            out = self.fuse_head(torch.cat([outL, outR], dim=1))
        else:
            raise ValueError("fuse must be 'avg'|'left'|'right'|'concat_head'")

        feats = None
        if fetch_layers is not None:
            if feature_branch == "left":
                feats = featsL + featsL_dec
            elif feature_branch == "right":
                feats = featsR + featsR_dec
            else:
                raise ValueError("feature_branch must be 'left' or 'right'")

        if return_branches:
            if fetch_layers is not None:
                return out, outL, outR, feats
            return out, outL, outR
        if fetch_layers is not None:
            return out, feats
        return out


if __name__ == "__main__":
    # 简单 sanity check（建议在 GPU 上跑）
    model = DualDiffRefineUNet(
        in_channels_left=3,
        in_channels_right=3,
        n_classes=4,
        depth=5,
        wf=5,
        tap_depth=None,        # 默认 bottleneck
        inject_layer=0,        # 注入到最深 decoder stage
        diff_mode="qkv",       # "raw" 或 "qkv"
        fuse="concat_head",
    )
    x1 = torch.randn(2, 3, 256, 256)
    x2 = torch.randn(2, 3, 256, 256)
    y = model(x1, x2)
    print(y.shape)
