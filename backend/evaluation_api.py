"""
evaluation_api.py
=================
FastAPI endpoints for evaluation. Import as a router in api.py:

    from evaluation_api import eval_router
    app.include_router(eval_router)

FIXES applied:
  - Uses actual boxes/scores from cache (requires api.py patch)
  - Uses actual origin/spacing/direction from cache
  - Proper error messages when cache is missing required fields
"""

from fastapi import APIRouter, HTTPException, Form, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Optional
from pathlib import Path
import csv
import json
import numpy as np
import torch
import math

def _sanitize_floats(obj):
    """
    Replace NaN / Inf with 0.0 so JSON serialization doesn't crash.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj

    elif isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]

    return obj

from evaluation import (
    load_luna16_annotations,
    evaluate_scan,
    evaluate_dataset,
    generate_gt_panoptic_masks,
    world_to_voxel,
    gt_nodule_to_voxel_box,
    volumetric_dice,
    volumetric_iou,
    #compute_panoptic_quality,
    GTNodule,
    ScanEvaluation,
)

eval_router = APIRouter(prefix="/eval", tags=["evaluation"])

# ──────────────────────────────────────────────
# Cache getter registration (used by api.py)
# ──────────────────────────────────────────────
_cache_getter = None

def register_cache_getter(getter_func):
    global _cache_getter
    _cache_getter = getter_func

def _get_cache(scan_id):
    if _cache_getter is None:
        raise RuntimeError(
            "Cache getter not registered. api.py must call register_cache_getter()."
        )
    return _cache_getter(scan_id)

# ──────────────────────────────────────────────
# Response models
# ──────────────────────────────────────────────

class SingleScanEvalResponse(BaseModel):
    seriesuid: str
    num_gt: int
    num_pred: int
    true_positives: int
    false_positives: int
    false_negatives: int
    sensitivity: float
    precision: float
    f1_score: float
    mean_dice: float
    mean_iou: float
    #panoptic_quality: float
    #segmentation_quality: float
    #recognition_quality: float
    matches: List[Dict]


class DatasetEvalResponse(BaseModel):
    num_scans: int
    total_gt_nodules: int
    total_predictions: int
    mean_sensitivity: float
    mean_precision: float
    mean_f1: float
    mean_dice: float
    mean_iou: float
   # mean_panoptic_quality: float
    CPM: float
    froc_sensitivities: Dict[str, float]


# ──────────────────────────────────────────────
# In-memory annotation store
# ──────────────────────────────────────────────
_annotations_store: Dict[str, List[GTNodule]] = {}


@eval_router.post("/load_annotations")
async def load_annotations(file: UploadFile = File(...)):
    """
    Upload LUNA16 annotations.csv to enable evaluation.
    """
    global _annotations_store
    try:
        content = (await file.read()).decode("utf-8")
        lines = content.strip().split("\n")
        reader = csv.DictReader(lines)

        _annotations_store = {}
        count = 0
        for row in reader:
            uid = row['seriesuid']
            nodule = GTNodule(
                seriesuid=uid,
                coord_x=float(row['coordX']),
                coord_y=float(row['coordY']),
                coord_z=float(row['coordZ']),
                diameter_mm=float(row['diameter_mm']),
            )
            _annotations_store.setdefault(uid, []).append(nodule)
            count += 1

        return {
            "status": "success",
            "num_series": len(_annotations_store),
            "num_nodules": count,
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to parse annotations: {e}")


@eval_router.post("/evaluate_scan/{scan_id}", response_model=SingleScanEvalResponse)
async def evaluate_single_scan(
    scan_id: str,
    seriesuid: str = Form(...),
    distance_thresh_mm: float = Form(15.0),
    iou_thresh_match: float = Form(0.1),
   # iou_thresh_pq: float = Form(0.5),
):
    """
    Evaluate a single scan that has already been inferred.
    Requires:
      - The scan must be in the segmentation cache (run /infer first)
      - Annotations must be loaded (run /eval/load_annotations first)
      - seriesuid must match a LUNA16 series

    FIX #3 & #4: Now reads actual boxes, scores, origin, spacing, direction
    from cache instead of using dummy values.
    """
    # Import from main api module
    #from api import get_from_cache

    if not _annotations_store:
        raise HTTPException(400, "No annotations loaded. POST to /eval/load_annotations first.")

    gt_nodules = _annotations_store.get(seriesuid, [])

    cache = _get_cache(scan_id)
    if cache is None:
        raise HTTPException(404, "Scan not in cache. Run /infer first.")

    # ── FIX #3: Validate required cache fields ──
    required_fields = ['semantic_mask', 'instance_mask', 'image_shape',
                       'voxel_spacing', 'origin', 'spacing_xyz', 'pred_boxes', 'pred_scores']
    missing = [f for f in required_fields if f not in cache]
    if missing:
        raise HTTPException(
            500,
            f"Cache missing required fields: {missing}. "
            f"Make sure api.py has been patched to store these during inference. "
            f"See api_patch_instructions.md for details."
        )

    semantic_mask = cache['semantic_mask']
    instance_mask = cache['instance_mask']
    image_shape = tuple(int(s) for s in cache['image_shape'])

    # ── FIX #3: Use actual origin/spacing/direction from cache ──
    origin = cache['origin']              # SimpleITK order (x,y,z)
    spacing_xyz = cache['spacing_xyz']    # SimpleITK order (x,y,z)
    direction = cache.get('direction', None)

    # ── FIX #4: Use actual boxes and scores from cache ──
    pred_boxes = cache['pred_boxes']     # tensor (N, 6)
    pred_scores = cache['pred_scores']   # tensor (N,)

    sr = evaluate_scan(
        pred_boxes=pred_boxes,
        pred_scores=pred_scores,
        pred_semantic_mask=semantic_mask,
        pred_instance_mask=instance_mask,
        gt_nodules=gt_nodules,
        image_shape=image_shape,
        origin=origin,
        spacing=spacing_xyz,
        direction=direction,
        distance_thresh_mm=distance_thresh_mm,
        iou_thresh_match=iou_thresh_match,
        iou_thresh_pq=iou_thresh_pq,
        seriesuid=seriesuid,
    )

    response = SingleScanEvalResponse(
    seriesuid=sr.seriesuid,
    num_gt=sr.num_gt,
    num_pred=sr.num_pred,
    true_positives=sr.true_positives,
    false_positives=sr.false_positives,
    false_negatives=sr.false_negatives,
    sensitivity=sr.sensitivity,
    precision=sr.precision,
    f1_score=sr.f1_score,
    mean_dice=sr.mean_dice,
    mean_iou=sr.mean_iou,
   # panoptic_quality=sr.panoptic_quality,
    #segmentation_quality=sr.segmentation_quality,
    #recognition_quality=sr.recognition_quality,
    matches=sr.matches,
)

    return _sanitize_floats(response.dict())


@eval_router.get("/annotations_status")
async def annotations_status():
    """Check if annotations are loaded and how many."""
    if not _annotations_store:
        return {"loaded": False, "num_series": 0, "num_nodules": 0}
    return {
        "loaded": True,
        "num_series": len(_annotations_store),
        "num_nodules": sum(len(v) for v in _annotations_store.values()),
    }


@eval_router.get("/metrics_info")
async def metrics_info():
    """
    Returns descriptions of all evaluation metrics used.
    Useful for the frontend tooltip/info display.
    """
    return {
        "metrics": [
            {
                "name": "FROC / CPM",
                "category": "Detection",
                "description": (
                    "Free-Response ROC. THE standard LUNA16 metric. "
                    "CPM = average sensitivity at 1/8, 1/4, 1/2, 1, 2, 4, 8 FPs per scan. "
                    "Range: 0-1, higher is better."
                ),
                "range": "0 to 1",
                "reference": "LUNA16 Challenge"
            },
            {
                "name": "Sensitivity (Recall)",
                "category": "Detection",
                "description": (
                    "Fraction of GT nodules that were detected. "
                    "TP / (TP + FN). High sensitivity = fewer missed nodules."
                ),
                "range": "0 to 1"
            },
            {
                "name": "Precision",
                "category": "Detection",
                "description": (
                    "Fraction of predictions that are true nodules. "
                    "TP / (TP + FP). High precision = fewer false alarms."
                ),
                "range": "0 to 1"
            },
            {
                "name": "F1 Score",
                "category": "Detection",
                "description": (
                    "Harmonic mean of sensitivity and precision. "
                    "Balances missed nodules vs false alarms."
                ),
                "range": "0 to 1"
            },
            {
                "name": "Dice Coefficient",
                "category": "Segmentation",
                "description": (
                    "Volumetric overlap between predicted pseudo-panoptic mask and "
                    "GT sphere mask. 2|A∩B| / (|A|+|B|). "
                    "Measures how well the box-based masks approximate nodule shape."
                ),
                "range": "0 to 1"
            },
            {
                "name": "IoU (Jaccard)",
                "category": "Segmentation",
                "description": (
                    "Intersection over Union between predicted and GT masks. "
                    "|A∩B| / |A∪B|. Stricter than Dice."
                ),
                "range": "0 to 1"
            },
            {
                "name": "3D Box IoU",
                "category": "Detection",
                "description": (
                    "3D Intersection over Union between predicted bounding boxes and "
                    "GT nodule bounding boxes (derived from sphere diameter)."
                ),
                "range": "0 to 1"
            },
        ]
    }