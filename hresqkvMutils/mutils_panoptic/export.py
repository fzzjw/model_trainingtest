import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from skimage.measure import find_contours
from torch.utils.data import DataLoader

from MuTILs_Panoptic.configs.panoptic_model_configs import RegionCellCombination, collate_fn
from MuTILs_Panoptic.mutils_panoptic.RegionDatasetLoaders import (
    MuTILsDataset,
    get_cv_fold_slides,
)
from MuTILs_Panoptic.utils.MiscRegionUtils import load_region_configs, load_trained_mutils_model

# from configs.panoptic_model_configs import RegionCellCombination, collate_fn
# from mutils_panoptic.RegionDatasetLoaders import (
#     MuTILsDataset,
#     get_cv_fold_slides,
# )
# from utils.MiscRegionUtils import load_region_configs, load_trained_mutils_model


def _move_data_to_device(batchdata, device):
    """Move tensor fields to device."""
    return [{k: v.to(device) for k, v in sample.items()} for sample in batchdata]


def _polygon_area(coords: List[List[float]]) -> float:
    """Shoelace formula for simple polygons."""
    area = 0.0
    n = len(coords)
    if n < 3:
        return 0.0
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def mask_to_geojson_features(
    mask: np.ndarray,
    class_map: Dict[int, str],
    min_points: int = 3,
    min_area: float = 0.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> List[Dict]:
    """
    Convert an integer mask to GeoJSON features (one polygon per contour per class).
    """
    features: List[Dict] = []
    for cls_id in np.unique(mask):
        if cls_id == 0:
            continue  # skip exclude/background
        binary = (mask == cls_id).astype(np.uint8)
        contours = find_contours(binary, 0.5)
        for contour in contours:
            # contour is (y, x); convert to (x, y)
            coords = [[float(x + x_offset), float(y + y_offset)] for y, x in contour]
            if len(coords) < min_points:
                continue
            # close ring
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            if min_area > 0 and _polygon_area(coords) < min_area:
                continue
            cls_name = class_map.get(int(cls_id), f"class_{int(cls_id)}")
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "class_id": int(cls_id),
                        "class_name": cls_name,
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords],
                    },
                }
            )
    return features


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
    dataset = MuTILsDataset(
        root=mtp.root, slides=slides, **mtp.test_dataset_params
    )
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
                features = mask_to_geojson_features(
                    mask=roi_pred,
                    class_map=class_map,
                    min_points=args.min_points,
                    min_area=args.min_area,
                    x_offset=args.x_offset,
                    y_offset=args.y_offset,
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
        description="Export ROI region predictions to GeoJSON polygons."
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
        help="Discard polygons with area below this (in pixel units).",
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
        help="Optional X offset to add to all coordinates.",
    )
    parser.add_argument(
        "--y-offset",
        type=float,
        default=0.0,
        help="Optional Y offset to add to all coordinates.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    export_roi_preds(parse_args())
