import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from skimage.measure import find_contours
from torch.utils.data import DataLoader

from MuTILs_Panoptic.configs.panoptic_model_configs import (
    RegionCellCombination,
    VisConfigs,
    collate_fn,
)
from MuTILs_Panoptic.mutils_panoptic.RegionDatasetLoaders import MuTILsDataset, get_cv_fold_slides
from MuTILs_Panoptic.utils.MiscRegionUtils import load_region_configs, load_trained_mutils_model


def _move_data_to_device(batchdata, device):
    """Move tensor fields to device."""
    return [{k: v.to(device) for k, v in sample.items()} for sample in batchdata]


def _ensure_closed(coords: List[List[float]]) -> List[List[float]]:
    if not coords:
        return coords
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def _signed_area(coords: Sequence[Sequence[float]]) -> float:
    area = 0.0
    n = len(coords)
    if n < 3:
        return 0.0
    for i in range(n - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def _point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    """Even-odd rule for point-in-polygon. Polygon expected closed."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n - 1):
        x1, y1 = polygon[i]
        x2, y2 = polygon[i + 1]
        if (y1 > y) != (y2 > y):
            x_at_y = x1 + (y - y1) * (x2 - x1) / ((y2 - y1) + 1e-12)
            if x < x_at_y:
                inside = not inside
    return inside


def mask_to_geojson_features(
    mask: np.ndarray,
    class_map: Dict[int, str],
    min_points: int = 3,
    min_area: float = 0.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    scale: float = 1.0,
) -> List[Dict]:
    """
    Convert an integer mask to GeoJSON features with hole handling and scaling.
    """
    features: List[Dict] = []
    seen = set()

    for cls_id in np.unique(mask):
        if cls_id == 0:
            continue  # skip exclude/background

        binary = (mask == cls_id).astype(np.uint8)
        contours = find_contours(binary, 0.5)

        outers: List[Dict] = []
        holes: List[Dict] = []

        for contour in contours:
            raw_ring = [[float(x), float(y)] for y, x in contour]
            raw_ring = _ensure_closed(raw_ring)
            if len(raw_ring) < min_points:
                continue

            area_raw = abs(_signed_area(raw_ring))
            if min_area > 0 and area_raw < min_area:
                continue

            scaled_ring = [
                [pt[0] * scale + x_offset, pt[1] * scale + y_offset]
                for pt in raw_ring
            ]
            scaled_ring = _ensure_closed(scaled_ring)

            signed = _signed_area(raw_ring)
            if signed >= 0:
                outers.append({"ring": scaled_ring, "holes": []})
            else:
                holes.append({"ring": scaled_ring})

        # assign holes to containing outers (if any)
        for hole in holes:
            hx = sum(p[0] for p in hole["ring"]) / len(hole["ring"])
            hy = sum(p[1] for p in hole["ring"]) / len(hole["ring"])
            placed = False
            for outer in outers:
                if _point_in_polygon((hx, hy), outer["ring"]):
                    outer["holes"].append(hole["ring"])
                    placed = True
                    break
            if not placed:
                # treat as its own polygon if no outer contains it
                outers.append({"ring": hole["ring"], "holes": []})

        cls_name = class_map.get(int(cls_id), f"class_{int(cls_id)}")
        for poly in outers:
                ring = poly["ring"]
                if len(ring) < min_points:
                    continue
                signature = _feature_signature(cls_id, [ring] + poly["holes"])
                if signature in seen:
                    continue
                seen.add(signature)
                features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "class_id": int(cls_id),
                        "class_name": cls_name,
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [ring] + poly["holes"],
                    },
                }
                )

    return features


def _feature_signature(class_id: int, rings: List[List[List[float]]], ndigits: int = 4):
    """Rounded coordinate signature to drop duplicated polygons."""
    rounded = []
    for ring in rings:
        rounded.append(tuple((round(pt[0], ndigits), round(pt[1], ndigits)) for pt in ring))
    return class_id, tuple(rounded)


def _compute_scale(mtp, user_scale: Optional[float]) -> float:
    if user_scale is not None:
        return float(user_scale)
    try:
        roi_mpp = mtp.dataset_params.get("roi_mpp")
        orig_mpp = mtp.dataset_params.get("original_mpp")
        if roi_mpp and orig_mpp:
            return float(roi_mpp) / float(orig_mpp)
    except Exception:
        pass
    try:
        roi_side = mtp.test_dataset_params.get("roi_side") or mtp.dataset_params.get("roi_side")
        orig_side = mtp.dataset_params.get("original_side")
        if roi_side and orig_side:
            return float(orig_side) / float(roi_side)
    except Exception:
        pass
    return 1.0


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

    exported = 0
    with torch.no_grad():
        for batchdata, truth in loader:
            batchdata = _move_data_to_device(batchdata, device)
            preds = model(batchdata)
            roi_logits = preds["roi_region_logits"].detach().cpu().numpy()
            # roi_logits shape: (B, C, H, W)
            for idx_in_batch in range(roi_logits.shape[0]):
                if args.limit and exported >= args.limit:
                    break

                roiname = truth[idx_in_batch]["roiname"]
                roi_pred = np.argmax(roi_logits[idx_in_batch], axis=0) + 1

                if args.use_truth_exclude and "lowres_mask" in truth[idx_in_batch]:
                    truth_mask = truth[idx_in_batch]["lowres_mask"]
                    if torch.is_tensor(truth_mask):
                        truth_mask = truth_mask.cpu().numpy()
                    truth_mask = np.asarray(truth_mask)
                    if truth_mask.ndim == 3:
                        truth_mask = truth_mask[0]
                    roi_pred[truth_mask == 0] = 0

                # save colored PNG using model's region colors
                color_mask = np.zeros((*roi_pred.shape, 3), dtype=np.uint8)
                for cls_id in np.unique(roi_pred):
                    cls_name = class_map.get(int(cls_id), "EXCLUDE")
                    color = VisConfigs.REGION_COLORS.get(cls_name, [128, 128, 128])
                    color_mask[roi_pred == cls_id] = color
                png_path = outdir / f"{Path(roiname).stem}_roi_pred.png"
                Image.fromarray(color_mask).save(png_path)

                features = mask_to_geojson_features(
                    mask=roi_pred,
                    class_map=class_map,
                    min_points=args.min_points,
                    min_area=args.min_area,
                    x_offset=args.x_offset,
                    y_offset=args.y_offset,
                    scale=scale,
                )
                fc = {"type": "FeatureCollection", "features": features}
                savename = outdir / f"{Path(roiname).stem}_roi_pred.geojson"
                with open(savename, "w", encoding="utf-8") as f:
                    json.dump(fc, f, ensure_ascii=False, indent=2)
                exported += 1
            if args.limit and exported >= args.limit:
                break

    print(f"Exported {exported} ROI predictions to {outdir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export ROI region predictions to GeoJSON polygons (scaled, with holes)."
    )
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
        help="Output directory for GeoJSON files.",
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
    return parser.parse_args()


if __name__ == "__main__":
    export_roi_preds(parse_args())
