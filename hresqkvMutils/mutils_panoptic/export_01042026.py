#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import sys
from contextlib import nullcontext
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

# Optional deps (nice terminal progress bar)
try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    _HAS_RICH = True
except Exception:
    _HAS_RICH = False


# -------------------------
# Utils
# -------------------------
def _move_data_to_device(batchdata, device):
    """Recursively move batch data to device."""
    if torch.is_tensor(batchdata):
        return batchdata.to(device)
    if isinstance(batchdata, (list, tuple)):
        return type(batchdata)(_move_data_to_device(x, device) for x in batchdata)
    if isinstance(batchdata, dict):
        return {k: _move_data_to_device(v, device) for k, v in batchdata.items()}
    return batchdata


def _get_class_name(class_map: Dict, class_id: int) -> str:
    """
    Helper to safely get class name whether class_map values are dicts or strings.
    """
    val = class_map.get(class_id)
    if val is None:
        return str(class_id)
    if isinstance(val, dict):
        return val.get("name", str(class_id))
    return str(val)


def _save_colored_pred_png(mask: np.ndarray, class_map: Dict[int, Any], outpath: Path):
    """
    Save a colored PNG visualization of the predicted region mask.
    Robust to class_map being int->str or int->dict.
    """
    # Create an RGB image buffer
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    unique_labels = np.unique(mask)
    for lbl in unique_labels:
        if lbl == 0:
            continue
        
        # Get class name safely
        c_name = _get_class_name(class_map, int(lbl))
        
        # Try to get color from VisConfigs using the class name
        # If not found, use a fallback grey or hash-based color
        color = VisConfigs.REGION_COLORS.get(c_name)
        if color is None:
            # Fallback: try to find by ID if VisConfigs uses IDs keys (rare but possible)
            color = VisConfigs.REGION_COLORS.get(int(lbl))
        
        if color is None:
            # Deterministic fallback color based on label ID
            color = [(int(lbl) * 50) % 255, (int(lbl) * 80) % 255, (int(lbl) * 110) % 255]
            
        color_mask[mask == lbl] = color

    im = Image.fromarray(color_mask, mode="RGB")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    im.save(str(outpath))


def _parse_offsets_from_roiname(roiname: str) -> Tuple[float, float]:
    """
    Parse xmin/ymin from roiname if present.
    Examples supported (case-insensitive):
      ... xmin12345_ymin67890 ...
      ... XMIN_12345__YMIN_67890 ...
    """
    s = roiname
    m1 = re.search(r"xmin[_=:\s\-]*([0-9]+)", s, flags=re.IGNORECASE)
    m2 = re.search(r"ymin[_=:\s\-]*([0-9]+)", s, flags=re.IGNORECASE)
    if m1 and m2:
        return float(m1.group(1)), float(m2.group(1))
    return 0.0, 0.0


def _compute_scale(mtp, explicit_scale: Optional[float]) -> float:
    """
    Compute scale factor for ROI mask coords -> desired coords.
    (已替换为 Code 2 的逻辑，支持字典读取)
    """
    if explicit_scale is not None:
        return float(explicit_scale)

    # 1. 优先尝试使用 mpp (microns per pixel) 计算比例
    try:
        roi_mpp = mtp.dataset_params.get("roi_mpp")
        orig_mpp = mtp.dataset_params.get("original_mpp")
        if roi_mpp and orig_mpp:
            return float(roi_mpp) / float(orig_mpp)
    except Exception:
        pass

    # 2. 其次尝试使用 side length (边长) 计算比例
    try:
        # test_dataset_params 有时会覆盖 dataset_params 的设置，优先查它
        roi_side = mtp.test_dataset_params.get("roi_side") or mtp.dataset_params.get("roi_side")
        orig_side = mtp.dataset_params.get("original_side")
        
        if roi_side and orig_side:
            return float(orig_side) / float(roi_side)
    except Exception:
        pass

    return 1.0


def _ring_signature(coords_xy: np.ndarray, rounding: int = 4) -> Tuple[Tuple[float, float], ...]:
    """
    Compute a canonical signature for a ring.
    """
    if coords_xy.ndim != 2 or coords_xy.shape[1] != 2:
        return tuple()

    pts = np.round(coords_xy.astype(np.float64), rounding)
    pts = [tuple(p) for p in pts.tolist()]
    if len(pts) == 0:
        return tuple()

    # Remove duplicate trailing point if closed
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]

    n = len(pts)
    if n == 0:
        return tuple()

    # All rotations
    def best_rotation(seq):
        best = None
        for i in range(n):
            rot = seq[i:] + seq[:i]
            if best is None or rot < best:
                best = rot
            if best is None: # safety
                best = rot
        return best

    fwd = best_rotation(pts)
    bwd = best_rotation(list(reversed(pts)))
    
    # Handle edge case where best_rotation returns None (empty list)
    if fwd is None: return tuple()
    if bwd is None: return tuple(fwd)
    
    return tuple(fwd) if fwd <= bwd else tuple(bwd)


def _feature_signature(feature: Dict[str, Any], rounding: int = 4) -> Tuple:
    """
    Signature for a Polygon/MultiPolygon feature based on rings.
    """
    geom = feature.get("geometry", {})
    gtype = geom.get("type", None)
    coords = geom.get("coordinates", None)

    if gtype == "Polygon":
        rings = coords or []
        ring_sigs = []
        for ring in rings:
            ring = np.asarray(ring)
            ring_sigs.append(_ring_signature(ring, rounding=rounding))
        return (gtype, tuple(ring_sigs))

    if gtype == "MultiPolygon":
        polys = coords or []
        poly_sigs = []
        for poly in polys:
            ring_sigs = []
            for ring in poly:
                ring = np.asarray(ring)
                ring_sigs.append(_ring_signature(ring, rounding=rounding))
            poly_sigs.append(tuple(ring_sigs))
        return (gtype, tuple(poly_sigs))

    return (gtype,)


def _mask_to_features_rasterio(
    mask: np.ndarray,
    class_map: Dict[int, Any],
    min_area_px: float,
    min_points: int,
    x_offset: float,
    y_offset: float,
    scale: float,
    connectivity: int,
    dedup_policy: str,
    priority: List[str],
    dedup_rounding: int,
) -> List[Dict[str, Any]]:
    """
    Polygonize mask using rasterio.features.shapes + affine transform.
    """
    if not _HAS_RASTERIO:
        raise RuntimeError("rasterio/affine not installed.")

    # rasterio expects a 2D array
    mask2d = np.asarray(mask).astype(np.int32)

    # Affine: x = xoff + scale * col; y = yoff + scale * row
    transform = Affine(scale, 0.0, x_offset, 0.0, scale, y_offset)

    features: List[Dict[str, Any]] = []
    sig2idx: Dict[Tuple, int] = {}

    # shapes yields (geom, value)
    for geom, val in rasterio.features.shapes(mask2d, connectivity=connectivity, transform=transform):
        v = int(val)
        if v <= 0:
            continue
        # area filter: rasterio polygon area will be in scaled units
        area_scaled = 0.0
        try:
            from shapely.geometry import shape as _shape
            shp = _shape(geom)
            area_scaled = float(shp.area)
        except Exception:
            area_scaled = None

        if area_scaled is not None:
            area_px_equiv = area_scaled / (scale * scale) if (scale != 0) else area_scaled
            if area_px_equiv < float(min_area_px):
                continue

        # --- FIX: Handle both dict and string in class_map safely ---
        c_name = _get_class_name(class_map, v)
        # ------------------------------------------------------------

        # Build feature
        props = {
            "class_id": v,
            "class_name": c_name,
        }
        feat = {"type": "Feature", "properties": props, "geometry": geom}

        # Basic point-count filter on outer ring
        coords = geom.get("coordinates", [])
        if geom.get("type") == "Polygon" and coords:
            outer = coords[0]
            if outer is not None and len(outer) < int(min_points):
                continue

        sig = _feature_signature(feat, rounding=dedup_rounding)
        if sig in sig2idx:
            # same geometry already seen
            old = features[sig2idx[sig]]
            old_props = old.get("properties", {})
            new_name = props.get("class_name")
            old_name = old_props.get("class_name")
            if new_name == old_name:
                # exact duplicate, skip
                continue

            # conflict: same geometry different class
            old_props["class_conflict"] = True
            if dedup_policy == "merge":
                # store all class names
                old_classes = old_props.get("classes", [])
                if not isinstance(old_classes, list):
                    old_classes = [old_classes]
                if old_name not in old_classes:
                    old_classes.append(old_name)
                if new_name not in old_classes:
                    old_classes.append(new_name)
                old_props["classes"] = old_classes
                old["properties"] = old_props
                continue

            # priority policy
            def prio(cname: str) -> int:
                try:
                    return priority.index(cname)
                except Exception:
                    return 10**9

            if prio(new_name) < prio(old_name):
                # replace winner
                old_props["class_id"] = v
                old_props["class_name"] = new_name
                old["properties"] = old_props
            continue

        sig2idx[sig] = len(features)
        features.append(feat)

    return features


def _mask_to_features_skimage(
    mask: np.ndarray,
    class_map: Dict[int, Any],
    min_area_px: float,
    min_points: int,
    x_offset: float,
    y_offset: float,
    scale: float,
) -> List[Dict[str, Any]]:
    """
    Polygonize using skimage.find_contours (fallback).
    """
    if not _HAS_SKIMAGE:
        raise RuntimeError("skimage not installed.")

    features: List[Dict[str, Any]] = []

    H, W = mask.shape
    for v in np.unique(mask):
        v = int(v)
        if v <= 0:
            continue
        binary = (mask == v).astype(np.uint8)

        # find_contours returns list of (N,2) arrays with coords (row, col)
        contours = find_contours(binary, 0.5)
        for cnt in contours:
            if cnt.shape[0] < int(min_points):
                continue

            # Drop open contours (touch boundary)
            rows = cnt[:, 0]
            cols = cnt[:, 1]
            if (rows.min() <= 0) or (rows.max() >= H - 1) or (cols.min() <= 0) or (cols.max() >= W - 1):
                continue

            # Convert to x,y with scaling and offsets
            xy = np.stack([cols, rows], axis=1)
            xy[:, 0] = x_offset + scale * xy[:, 0]
            xy[:, 1] = y_offset + scale * xy[:, 1]

            # Close ring
            ring = xy.tolist()
            if ring[0] != ring[-1]:
                ring.append(ring[0])

            # Area filter (shoelace) in scaled units -> convert to px-like
            area_scaled = 0.0
            try:
                x = np.asarray([p[0] for p in ring], dtype=np.float64)
                y = np.asarray([p[1] for p in ring], dtype=np.float64)
                area_scaled = 0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))
            except Exception:
                area_scaled = None

            if area_scaled is not None:
                area_px_equiv = area_scaled / (scale * scale) if (scale != 0) else area_scaled
                if area_px_equiv < float(min_area_px):
                    continue
            
            # --- FIX: Handle both dict and string in class_map safely ---
            c_name = _get_class_name(class_map, v)
            # ------------------------------------------------------------

            geom = {"type": "Polygon", "coordinates": [ring]}
            props = {
                "class_id": v,
                "class_name": c_name,
            }
            features.append({"type": "Feature", "properties": props, "geometry": geom})

    return features


def mask_to_geojson_features(
    mask: np.ndarray,
    class_map: Dict[int, Any],
    min_points: int,
    min_area: float,
    x_offset: float,
    y_offset: float,
    scale: float,
    use_rasterio: bool,
    connectivity: int,
    dedup_policy: str,
    priority: List[str],
    dedup_rounding: int,
) -> List[Dict[str, Any]]:
    """
    Convert mask into GeoJSON Features using chosen backend.
    """
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
    )


# -------------------------
# Main export
# -------------------------
def export_roi_preds(args):
    # load configs
    cfg = load_region_configs(args.configs)
    mtp = cfg.MuTILsParams

    # determine slides for split
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

    # optional PNG directory
    png_dir = None
    if args.save_png:
        png_dir = outdir / "png"
        png_dir.mkdir(parents=True, exist_ok=True)

    # progress bar (Rich)
    progress = None
    task_id = None
    total_rois = None
    try:
        total_rois = len(dataset)
    except Exception:
        total_rois = None
    if args.limit and args.limit > 0 and total_rois is not None:
        total_rois = min(total_rois, args.limit)

    enable_progress = _HAS_RICH and (not args.no_progress) and sys.stdout.isatty()
    if enable_progress:
        console = Console(force_interactive=True)
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        )
        progress.start()
        task_id = progress.add_task("Exporting ROIs", total=total_rois)

    exported = 0
    try:
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

                    # ---- optional colored PNG visualization ----
                    if args.save_png:
                        png_path = png_dir / f"{Path(roiname).stem}_roi_pred.png"
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

                    if progress is not None and task_id is not None:
                        progress.update(task_id, advance=1, description=f"Exporting {Path(roiname).stem}")

                if args.limit and exported >= args.limit:
                    break
    finally:
        if progress is not None:
            progress.stop()

    print(f"Exported {exported} ROI predictions to {outdir}")
    if not use_rasterio:
        print(
            "NOTE: rasterio backend is OFF. If you still see '突兀直线', install rasterio/affine and rerun without --no-rasterio."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export ROI region predictions to GeoJSON polygons (scaled) and (optionally) colored PNG masks."
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
        help="Path to trained checkpoint .pt file.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="CV fold index to use when reading train_test_splits.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "all"],
        default="train",
        help="Which split to export ROI predictions for.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for DataLoader inference.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="export_geojson",
        help="Output directory for GeoJSON and PNG.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of ROIs to export (0 = no limit).",
    )

    # ---- postprocess / exclude ----
    parser.add_argument(
        "--use-truth-exclude",
        action="store_true",
        help="If present, use truth lowres_mask == 0 to force roi_pred to 0 in those pixels.",
    )

    # ---- coord mapping ----
    parser.add_argument(
        "--x-offset",
        type=float,
        default=0.0,
        help="X offset applied to all exported polygon coordinates.",
    )
    parser.add_argument(
        "--y-offset",
        type=float,
        default=0.0,
        help="Y offset applied to all exported polygon coordinates.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Scale factor applied to mask pixel coordinates before adding offsets. If not set, compute from configs.",
    )
    parser.add_argument(
        "--auto-offset-from-roiname",
        action="store_true",
        help="If set, parse xmin/ymin from roiname and add to offsets.",
    )

    # ---- polygonization options ----
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.0,
        help="Minimum polygon area in approximate pixel units (filters small speckles).",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=3,
        help="Minimum number of points for a ring / contour.",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[4, 8],
        default=8,
        help="Pixel connectivity for rasterio polygonization.",
    )
    parser.add_argument(
        "--no-rasterio",
        action="store_true",
        help="Disable rasterio backend; use skimage.find_contours fallback.",
    )

    # ---- dedup / conflicts ----
    parser.add_argument(
        "--dedup-policy",
        type=str,
        choices=["priority", "merge"],
        default="priority",
        help="How to resolve same-geometry different-class conflicts: priority=keep highest priority class; merge=store all in properties.",
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

    # ---- NEW: optional png + rich progress ----
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Save colored PNG visualization (default: off).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Rich progress bar.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    export_roi_preds(parse_args())