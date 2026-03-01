#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from MuTILs_Panoptic.configs.panoptic_model_configs import (
    RegionCellCombination,
    VisConfigs,
    collate_fn,
)
from MuTILs_Panoptic.mutils_panoptic.RegionDatasetLoaders import MuTILsDataset, get_cv_fold_slides
from MuTILs_Panoptic.utils.MiscRegionUtils import load_region_configs, load_trained_mutils_model

# Optional deps (recommended)
try:
    import rasterio.features
    from affine import Affine
    _HAS_RASTERIO = True
except Exception:
    _HAS_RASTERIO = False

try:
    from skimage.measure import find_contours
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


# -------------------------
# Utils
# -------------------------
def _move_data_to_device(batchdata, device):
    return [{k: v.to(device) for k, v in sample.items()} for sample in batchdata]


def _ensure_closed(ring: List[List[float]]) -> List[List[float]]:
    if not ring:
        return ring
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def _parse_offsets_from_roiname(roiname: str) -> Tuple[float, float]:
    """
    Parse patterns like:
      ... xmin104246_ymin48517 ...  (or XMIN / YMIN variants)
    """
    m_x = re.search(r"xmin(-?\d+)", roiname, flags=re.IGNORECASE)
    m_y = re.search(r"ymin(-?\d+)", roiname, flags=re.IGNORECASE)
    if not (m_x and m_y):
        return 0.0, 0.0
    return float(m_x.group(1)), float(m_y.group(1))


def _compute_scale(mtp, user_scale: Optional[float]) -> float:
    if user_scale is not None:
        return float(user_scale)
    # Prefer mpp ratio if available
    try:
        roi_mpp = mtp.dataset_params.get("roi_mpp")
        orig_mpp = mtp.dataset_params.get("original_mpp")
        if roi_mpp and orig_mpp:
            return float(roi_mpp) / float(orig_mpp)
    except Exception:
        pass
    # Fallback to side ratio if present
    try:
        roi_side = mtp.test_dataset_params.get("roi_side") or mtp.dataset_params.get("roi_side")
        orig_side = mtp.dataset_params.get("original_side")
        if roi_side and orig_side:
            return float(orig_side) / float(roi_side)
    except Exception:
        pass
    return 1.0


def _save_colored_pred_png(
    roi_pred: np.ndarray,
    class_id_to_name: Dict[int, str],
    out_png: Path,
) -> None:
    """
    Exactly the same idea as your reference code:
    make an RGB color mask using VisConfigs.REGION_COLORS and save as PNG.
    """
    color_mask = np.zeros((*roi_pred.shape, 3), dtype=np.uint8)
    for cls_id in np.unique(roi_pred):
        cls_name = class_id_to_name.get(int(cls_id), "EXCLUDE")
        color = VisConfigs.REGION_COLORS.get(cls_name, [128, 128, 128])
        color_mask[roi_pred == cls_id] = color
    Image.fromarray(color_mask).save(out_png)


# -------------------------
# Geometry canonicalization for dedup (no shapely required)
# -------------------------
def _canonical_ring(ring: List[List[float]], ndigits: int) -> Tuple[Tuple[float, float], ...]:
    """
    Normalize ring to be orientation/rotation-invariant (best-effort):
    - round coords
    - ensure closed
    - rotate to lexicographically smallest start
    - pick min(forward, reversed) representation
    """
    ring = _ensure_closed([[float(x), float(y)] for x, y in ring])
    if len(ring) < 4:
        return tuple((round(x, ndigits), round(y, ndigits)) for x, y in ring)

    pts = [(round(x, ndigits), round(y, ndigits)) for x, y in ring[:-1]]  # drop closing for rotation
    # rotate
    min_i = min(range(len(pts)), key=lambda i: pts[i])
    pts_fwd = pts[min_i:] + pts[:min_i]
    pts_rev = list(reversed(pts))
    min_j = min(range(len(pts_rev)), key=lambda i: pts_rev[i])
    pts_rev = pts_rev[min_j:] + pts_rev[:min_j]

    cand1 = tuple(pts_fwd + [pts_fwd[0]])
    cand2 = tuple(pts_rev + [pts_rev[0]])
    return min(cand1, cand2)


def _geom_signature(geom: Dict[str, Any], ndigits: int) -> Tuple:
    """
    Geometry-only signature (no class id) for dedup/conflict detection.
    Supports Polygon / MultiPolygon.
    """
    gtype = geom.get("type")
    coords = geom.get("coordinates", [])

    if gtype == "Polygon":
        rings = [_canonical_ring(r, ndigits) for r in coords]
        outer = rings[0] if rings else tuple()
        holes = tuple(sorted(rings[1:]))
        return ("Polygon", outer, holes)

    if gtype == "MultiPolygon":
        polys = []
        for poly in coords:
            rings = [_canonical_ring(r, ndigits) for r in poly]
            outer = rings[0] if rings else tuple()
            holes = tuple(sorted(rings[1:]))
            polys.append((outer, holes))
        polys = tuple(sorted(polys))
        return ("MultiPolygon", polys)

    # fallback
    return (gtype, str(coords)[:2000])


def _explode_to_polygons(geom: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a list of Polygon geometries (split MultiPolygon into many Polygons)."""
    gtype = geom.get("type")
    if gtype == "Polygon":
        return [geom]
    if gtype == "MultiPolygon":
        out = []
        for poly_coords in geom.get("coordinates", []):
            out.append({"type": "Polygon", "coordinates": poly_coords})
        return out
    return []


# -------------------------
# Polygonize backends
# -------------------------
def _mask_to_features_rasterio(
    mask: np.ndarray,
    class_map: Dict[int, str],
    *,
    min_area_px: float,
    min_points: int,
    x_offset: float,
    y_offset: float,
    scale: float,
    connectivity: int,
    dedup_policy: str,
    priority: List[str],
    dedup_rounding: int,
) -> List[Dict]:
    """
    Use rasterio.features.shapes to polygonize the integer label mask.
    It yields polygons bounding contiguous regions of the same value. 
    """
    if not _HAS_RASTERIO:
        raise RuntimeError("rasterio not available")

    # transform pixel (col,row) -> (x,y)
    transform = Affine(scale, 0.0, x_offset, 0.0, scale, y_offset)

    features: List[Dict] = []
    seen: Dict[Tuple, Dict] = {}  # signature -> kept feature (or merged)
    px_area_to_world = scale * scale if scale != 0 else 1.0

    # only polygonize non-zero
    shapes_iter = rasterio.features.shapes(
        mask.astype(np.int32),
        mask=(mask > 0),
        connectivity=int(connectivity),
        transform=transform,
    )

    for geom, value in shapes_iter:
        cls_id = int(value)
        if cls_id == 0:
            continue
        cls_name = class_map.get(cls_id, f"class_{cls_id}")

        for poly_geom in _explode_to_polygons(geom):
            # filter by points
            rings = poly_geom.get("coordinates", [])
            if not rings or len(rings[0]) < max(4, min_points):
                continue

            # filter by area in pixel units (approx: world_area / (scale^2))
            # (good enough for thresholding tiny specks)
            # Note: polygon area computation without shapely; do a simple shoelace on outer ring only.
            outer = rings[0]

            def _shoelace(r):
                a = 0.0
                for i in range(len(r) - 1):
                    x1, y1 = r[i]
                    x2, y2 = r[i + 1]
                    a += x1 * y2 - x2 * y1
                return abs(a) * 0.5

            world_area = _shoelace(outer)
            area_px = world_area / px_area_to_world
            if min_area_px > 0 and area_px < float(min_area_px):
                continue

            sig = _geom_signature(poly_geom, ndigits=dedup_rounding)

            if sig not in seen:
                feat = {
                    "type": "Feature",
                    "properties": {
                        "class_id": cls_id,
                        "class_name": cls_name,
                    },
                    "geometry": poly_geom,
                }
                seen[sig] = feat
                features.append(feat)
                continue

            # duplicated geometry encountered
            prev = seen[sig]
            prev_cls = prev["properties"].get("class_name", "UNKNOWN")
            if prev_cls == cls_name:
                prev["properties"]["dedup_count"] = int(prev["properties"].get("dedup_count", 1)) + 1
                continue

            # conflict: same geometry, different class
            prev["properties"]["class_conflict"] = True

            if dedup_policy == "merge":
                classes = set(prev["properties"].get("classes", [prev_cls]))
                classes.add(cls_name)
                prev["properties"]["classes"] = sorted(classes)
                continue

            # priority policy: keep class with higher priority (earlier in list)
            def _rank(name: str) -> int:
                try:
                    return priority.index(name)
                except ValueError:
                    return 10**9

            if _rank(cls_name) < _rank(prev_cls):
                # replace properties
                prev["properties"]["class_id"] = cls_id
                prev["properties"]["class_name"] = cls_name

    return features


def _mask_to_features_skimage(
    mask: np.ndarray,
    class_map: Dict[int, str],
    *,
    min_area_px: float,
    min_points: int,
    x_offset: float,
    y_offset: float,
    scale: float,
    pad: int = 1,
    drop_open: bool = True,
    dedup_rounding: int = 4,
) -> List[Dict]:
    """
    Fallback polygonization using find_contours.
    Important: contours intersecting the image edge are OPEN in skimage. 
    To avoid “突兀直线”，we drop open contours instead of force-closing them.
    """
    if not _HAS_SKIMAGE:
        raise RuntimeError("scikit-image not available")

    features: List[Dict] = []
    seen = set()

    for cls_id in np.unique(mask):
        cls_id = int(cls_id)
        if cls_id == 0:
            continue
        cls_name = class_map.get(cls_id, f"class_{cls_id}")
        binary = (mask == cls_id).astype(np.uint8)

        if pad > 0:
            binary = np.pad(binary, pad_width=pad, mode="constant", constant_values=0)

        contours = find_contours(binary, 0.5)

        for contour in contours:
            # contour is (row, col) -> (y, x)
            ring = []
            for r, c in contour:
                x = float(c - pad) * scale + x_offset
                y = float(r - pad) * scale + y_offset
                ring.append([x, y])

            if len(ring) < max(4, min_points):
                continue

            is_closed = (ring[0] == ring[-1])
            if drop_open and not is_closed:
                continue
            ring = _ensure_closed(ring)

            # area filter (approx in pixel units)
            def _shoelace(r):
                a = 0.0
                for i in range(len(r) - 1):
                    x1, y1 = r[i]
                    x2, y2 = r[i + 1]
                    a += x1 * y2 - x2 * y1
                return abs(a) * 0.5

            world_area = _shoelace(ring)
            area_px = world_area / (scale * scale if scale != 0 else 1.0)
            if min_area_px > 0 and area_px < float(min_area_px):
                continue

            geom = {"type": "Polygon", "coordinates": [ring]}
            sig = _geom_signature(geom, ndigits=dedup_rounding)
            if sig in seen:
                continue
            seen.add(sig)

            features.append(
                {
                    "type": "Feature",
                    "properties": {"class_id": cls_id, "class_name": cls_name},
                    "geometry": geom,
                }
            )

    return features


def mask_to_geojson_features(
    mask: np.ndarray,
    class_map: Dict[int, str],
    *,
    min_area: float,
    min_points: int,
    x_offset: float,
    y_offset: float,
    scale: float,
    use_rasterio: bool,
    connectivity: int,
    dedup_policy: str,
    priority: List[str],
    dedup_rounding: int,
) -> List[Dict]:
    if use_rasterio:
        return _mask_to_features_rasterio(
            mask,
            class_map,
            min_area_px=min_area,
            min_points=min_points,
            x_offset=x_offset,
            y_offset=y_offset,
            scale=scale,
            connectivity=connectivity,
            dedup_policy=dedup_policy,
            priority=priority,
            dedup_rounding=dedup_rounding,
        )
    return _mask_to_features_skimage(
        mask,
        class_map,
        min_area_px=min_area,
        min_points=min_points,
        x_offset=x_offset,
        y_offset=y_offset,
        scale=scale,
        pad=1,
        drop_open=True,
        dedup_rounding=dedup_rounding,
    )


# -------------------------
# Main export
# -------------------------
def export_roi_preds(args):
    cfg = load_region_configs(args.configs)
    mtp = cfg.MuTILsParams

    # slides to process
    train_slides, test_slides = get_cv_fold_slides(
        train_test_splits_path=Path(mtp.root, "train_test_splits"),
        fold=args.fold,
    )
    if args.split == "train":
        slides = train_slides
    elif args.split == "test":
        slides = test_slides
    else:
        slides = train_slides + test_slides

    # dataset / loader
    dataset = MuTILsDataset(root=mtp.root, slides=slides, **mtp.test_dataset_params)
    loader_kws = dict(collate_fn=collate_fn, shuffle=False, num_workers=0)
    if args.batch_size:
        loader_kws["batch_size"] = args.batch_size
    loader = DataLoader(dataset=dataset, **loader_kws)

    # model
    model = load_trained_mutils_model(args.ckpt, mtp)
    device = next(model.parameters()).device
    model.eval()

    class_map = RegionCellCombination.RREGION_CODES
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scale = _compute_scale(mtp, args.scale)

    # offsets
    base_xoff = float(args.x_offset)
    base_yoff = float(args.y_offset)

    # backend choice
    use_rasterio = (not args.no_rasterio) and _HAS_RASTERIO

    exported = 0
    with torch.no_grad():
        for batchdata, truth in loader:
            batchdata = _move_data_to_device(batchdata, device)
            preds = model(batchdata)
            roi_logits = preds["roi_region_logits"].detach().cpu().numpy()  # (B,C,H,W)

            for idx_in_batch in range(roi_logits.shape[0]):
                if args.limit and exported >= args.limit:
                    break

                roiname = str(truth[idx_in_batch]["roiname"])
                roi_pred = np.argmax(roi_logits[idx_in_batch], axis=0) + 1  # (H,W), 1..C

                if args.use_truth_exclude and "lowres_mask" in truth[idx_in_batch]:
                    truth_mask = truth[idx_in_batch]["lowres_mask"]
                    if torch.is_tensor(truth_mask):
                        truth_mask = truth_mask.cpu().numpy()
                    truth_mask = np.asarray(truth_mask)
                    if truth_mask.ndim == 3:
                        truth_mask = truth_mask[0]
                    roi_pred[truth_mask == 0] = 0

                # auto offsets from ROI name
                ax = ay = 0.0
                if args.auto_offset_from_roiname:
                    ax, ay = _parse_offsets_from_roiname(roiname)

                xoff = base_xoff + ax
                yoff = base_yoff + ay

                # ---- (ADDED) export colored PNG (same as your reference block) ----
                png_path = outdir / f"{Path(roiname).stem}_roi_pred.png"
                _save_colored_pred_png(roi_pred, class_map, png_path)

                # ---- export GeoJSON ----
                features = mask_to_geojson_features(
                    mask=roi_pred,
                    class_map=class_map,
                    min_points=args.min_points,
                    min_area=args.min_area,
                    x_offset=xoff,
                    y_offset=yoff,
                    scale=scale,
                    use_rasterio=use_rasterio,
                    connectivity=args.connectivity,
                    dedup_policy=args.dedup_policy,
                    priority=[s.strip() for s in args.priority.split(",") if s.strip()],
                    dedup_rounding=args.dedup_rounding,
                )

                fc = {"type": "FeatureCollection", "features": features}
                savename = outdir / f"{Path(roiname).stem}_roi_pred.geojson"
                with open(savename, "w", encoding="utf-8") as f:
                    json.dump(fc, f, ensure_ascii=False, indent=2)

                exported += 1

            if args.limit and exported >= args.limit:
                break

    print(f"Exported {exported} ROI predictions to {outdir}")
    if not use_rasterio:
        print(
            "NOTE: rasterio backend is OFF. If you still see '突兀直线', install rasterio/affine and rerun without --no-rasterio."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export ROI region predictions to GeoJSON polygons (scaled) + colored PNG mask."
    )

    # ---- keep your original arguments ----
    parser.add_argument(
        "--configs",
        type=str,
        default="configs/panoptic_model_configs.py",
        help="Path to panoptic_model_configs.py to load MuTILsParams.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to trained checkpoint (.pt).",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Fold id to pick train/test slides.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="all",
        help="Which split to export.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="export_geojson",
        help="Output directory for GeoJSON/PNG files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of ROIs to export (0 = no limit).",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.0,
        help="Discard polygons with area below this (in pixel units of the model mask).",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=3,
        help="Discard polygons with fewer vertices than this.",
    )
    parser.add_argument(
        "--x-offset",
        type=float,
        default=0.0,
        help="Optional X offset to add to all coordinates (after scaling).",
    )
    parser.add_argument(
        "--y-offset",
        type=float,
        default=0.0,
        help="Optional Y offset to add to all coordinates (after scaling).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Override coordinate scale factor; defaults to roi_mpp/original_mpp or original_side/roi_side.",
    )
    parser.add_argument(
        "--use-truth-exclude",
        action="store_true",
        help="If present, zero-out predictions where truth lowres_mask is 0 to avoid background bleed.",
    )

    # ---- additional args for robustness / dedup / conflicts ----
    parser.add_argument(
        "--auto-offset-from-roiname",
        action="store_true",
        help="Parse xmin/ymin from roiname (e.g. xmin104246_ymin48517) and add to x/y offset.",
    )
    parser.add_argument(
        "--no-rasterio",
        action="store_true",
        help="Disable rasterio polygonize and fallback to skimage.find_contours (not recommended).",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        default=8,
        choices=[4, 8],
        help="Connectivity for rasterio.features.shapes (4 or 8).",
    )
    parser.add_argument(
        "--dedup-policy",
        choices=["priority", "merge"],
        default="priority",
        help="If same geometry appears with different classes: keep by priority, or merge classes into properties.",
    )
    parser.add_argument(
        "--priority",
        type=str,
        default="Cancerous epithelium,Stroma,TILs,Normal epithelium,Blood,Junk/Debris,Other,Whitespace/Empty,Exclude",
        help="Comma-separated class_name priority list (earlier = higher priority). Used when --dedup-policy=priority.",
    )
    parser.add_argument(
        "--dedup-rounding",
        type=int,
        default=4,
        help="Rounding digits used when computing geometry signatures for dedup/conflict detection.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    export_roi_preds(parse_args())
