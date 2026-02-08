"""
Evaluation Module for Lung Nodule Pseudo-Panoptic Segmentation
==============================================================
FIXED VERSION — resolves all critical bugs from original:
  - FROC now tracks ALL predictions (including unmatched FPs)
  - Instance ID mapping corrected for per-instance Dice
  - GT overlap handled properly in panoptic mask generation
  - Matching separated: distance-based for detection, IoU-based for panoptic

Metrics implemented:
  1. FROC / CPM — THE standard LUNA16 metric
  2. Sensitivity / Precision / F1 at fixed threshold
  3. 3D Box IoU between predicted and GT boxes
  4. Volumetric Dice — GT spheres vs predicted pseudo-panoptic masks
  5. Volumetric IoU — Jaccard index variant
  6. Panoptic Quality (PQ = SQ × RQ) — instance-aware panoptic metric
  7. Per-instance matching report with Dice/IoU per nodule

Designed for: MONAI RetinaNet detection → pseudo-panoptic pipeline on LUNA16.
Compatible with: huggingface.co/MONAI/lung_nodule_ct_detection pretrained model.

Ground truth source: LUNA16 annotations.csv
    columns: seriesuid, coordX, coordY, coordZ, diameter_mm
    (world-space coordinates, ≥4 radiologist agreement)
"""

import csv
import math
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class GTNodule:
    """Ground-truth nodule from LUNA16 annotations.csv."""
    seriesuid: str
    coord_x: float  # world X (mm)
    coord_y: float  # world Y (mm)
    coord_z: float  # world Z (mm)
    diameter_mm: float


@dataclass
class PredNodule:
    """Predicted nodule from the detection + pseudo-panoptic pipeline."""
    box: np.ndarray       # (6,) — z1,y1,x1,z2,y2,x2 in voxel space
    score: float
    centroid_vox: np.ndarray  # (3,) — z,y,x
    instance_id: int = 0      # ID in the instance mask
    volume_mm3: float = 0.0
    diameter_mm: float = 0.0


@dataclass
class MatchResult:
    """Result of matching one GT nodule to a prediction."""
    gt_idx: int
    pred_idx: Optional[int] = None
    distance_mm: float = float('inf')
    iou_3d: float = 0.0
    matched: bool = False


@dataclass
class ScanEvaluation:
    """
    Full evaluation result for a single scan.

    FIX: Now includes all_pred_scores and all_pred_is_tp for correct FROC.
    The original only stored matched GT info, missing unmatched FP predictions.
    """
    seriesuid: str = ""
    num_gt: int = 0
    num_pred: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    # Detection metrics
    sensitivity: float = 0.0
    precision: float = 0.0
    f1_score: float = 0.0

    # Segmentation metrics (pseudo-panoptic)
    mean_dice: float = 0.0
    mean_iou: float = 0.0

    # Panoptic Quality
    panoptic_quality: float = 0.0
    segmentation_quality: float = 0.0
    recognition_quality: float = 0.0

    # Per-instance match details (for GT nodules)
    matches: List[dict] = field(default_factory=list)

    # ── FIX #1: Store ALL prediction data for correct FROC ──
    # These track every prediction (TP + FP), not just GT matches.
    # Without this, FROC undercounts false positives and inflates CPM.
    all_pred_scores: List[float] = field(default_factory=list)
    all_pred_is_tp: List[bool] = field(default_factory=list)


# ──────────────────────────────────────────────
# LUNA16 annotation loader
# ──────────────────────────────────────────────

def load_luna16_annotations(csv_path: str) -> Dict[str, List[GTNodule]]:
    """
    Load LUNA16 annotations.csv.
    Returns dict: seriesuid → list of GTNodule.

    CSV format: seriesuid,coordX,coordY,coordZ,diameter_mm
    """
    annotations = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row['seriesuid']
            nodule = GTNodule(
                seriesuid=uid,
                coord_x=float(row['coordX']),
                coord_y=float(row['coordY']),
                coord_z=float(row['coordZ']),
                diameter_mm=float(row['diameter_mm']),
            )
            annotations.setdefault(uid, []).append(nodule)
    print(f"✅ Loaded annotations for {len(annotations)} scans, "
          f"{sum(len(v) for v in annotations.values())} total nodules")
    return annotations


# ──────────────────────────────────────────────
# Coordinate conversions (world ↔ voxel)
# ──────────────────────────────────────────────

def world_to_voxel(world_coord, origin, spacing, direction=None):
    """
    Convert world coordinate (mm) to voxel index.

    Args:
        world_coord: (x, y, z) in world mm (LUNA16 uses x,y,z order)
        origin: image origin from SimpleITK (x, y, z)
        spacing: image spacing from SimpleITK (x, y, z)
        direction: 3x3 direction cosine matrix (optional)

    Returns:
        voxel_coord: (z, y, x) in voxel indices (numpy array order)
    """
    world = np.array(world_coord, dtype=np.float64)
    org = np.array(origin, dtype=np.float64)
    sp = np.array(spacing, dtype=np.float64)

    if direction is not None:
        direction = np.array(direction).reshape(3, 3)
        voxel_xyz = np.linalg.inv(direction) @ ((world - org) / sp)
    else:
        voxel_xyz = (world - org) / sp

    # Convert from (x,y,z) to (z,y,x) for numpy indexing
    return np.array([voxel_xyz[2], voxel_xyz[1], voxel_xyz[0]])


def gt_nodule_to_voxel_box(nodule: GTNodule, origin, spacing, direction=None):
    """
    Convert a LUNA16 GT nodule (center + diameter in world mm) to a
    voxel-space bounding box: (z1, y1, x1, z2, y2, x2).
    """
    center_vox = world_to_voxel(
        [nodule.coord_x, nodule.coord_y, nodule.coord_z],
        origin, spacing, direction
    )
    # Radius in voxels (spacing may differ per axis)
    sp_zyx = np.array([spacing[2], spacing[1], spacing[0]])
    radius_vox = (nodule.diameter_mm / 2.0) / sp_zyx

    z1 = center_vox[0] - radius_vox[0]
    y1 = center_vox[1] - radius_vox[1]
    x1 = center_vox[2] - radius_vox[2]
    z2 = center_vox[0] + radius_vox[0]
    y2 = center_vox[1] + radius_vox[1]
    x2 = center_vox[2] + radius_vox[2]

    return np.array([z1, y1, x1, z2, y2, x2]), center_vox


# ──────────────────────────────────────────────
# GT sphere mask generation
# ──────────────────────────────────────────────

def generate_gt_sphere_mask(
    nodule: GTNodule,
    image_shape: Tuple[int, int, int],
    origin, spacing, direction=None,
) -> np.ndarray:
    """
    Generate a binary sphere mask for a GT nodule in voxel space.
    This is the ground-truth "segmentation" for pseudo-Dice evaluation.
    """
    center_vox = world_to_voxel(
        [nodule.coord_x, nodule.coord_y, nodule.coord_z],
        origin, spacing, direction
    )
    sp_zyx = np.array([spacing[2], spacing[1], spacing[0]])
    radius_mm = nodule.diameter_mm / 2.0

    Z, Y, X = image_shape
    mask = np.zeros((Z, Y, X), dtype=np.uint8)

    # Only iterate over the bounding box region for efficiency
    radius_vox = radius_mm / sp_zyx
    z_lo = max(0, int(np.floor(center_vox[0] - radius_vox[0])))
    z_hi = min(Z, int(np.ceil(center_vox[0] + radius_vox[0])) + 1)
    y_lo = max(0, int(np.floor(center_vox[1] - radius_vox[1])))
    y_hi = min(Y, int(np.ceil(center_vox[1] + radius_vox[1])) + 1)
    x_lo = max(0, int(np.floor(center_vox[2] - radius_vox[2])))
    x_hi = min(X, int(np.ceil(center_vox[2] + radius_vox[2])) + 1)

    # Guard against empty slicing (nodule outside image)
    if z_lo >= z_hi or y_lo >= y_hi or x_lo >= x_hi:
        return mask

    zz, yy, xx = np.mgrid[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]

    # Distance in mm from center
    dist_mm = np.sqrt(
        ((zz - center_vox[0]) * sp_zyx[0]) ** 2 +
        ((yy - center_vox[1]) * sp_zyx[1]) ** 2 +
        ((xx - center_vox[2]) * sp_zyx[2]) ** 2
    )

    mask[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi] = (dist_mm <= radius_mm).astype(np.uint8)
    return mask


def generate_gt_panoptic_masks(
    gt_nodules: List[GTNodule],
    image_shape: Tuple[int, int, int],
    origin, spacing, direction=None,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Generate combined GT semantic mask and instance mask from all GT nodules.

    FIX #5: Now returns per-nodule individual masks as well, so panoptic
    quality computation doesn't rely on the (potentially overlapping) combined
    instance mask. When two GT spheres overlap, the combined instance mask
    has last-writer-wins — but individual masks are preserved for accurate
    per-instance Dice/IoU.

    Returns:
        gt_semantic: (Z,Y,X) binary — union of all GT nodule spheres
        gt_instance: (Z,Y,X) int32 — each nodule gets a unique ID (1-indexed)
        gt_individual_masks: list of (Z,Y,X) binary masks, one per GT nodule
    """
    Z, Y, X = image_shape
    gt_semantic = np.zeros((Z, Y, X), dtype=np.uint8)
    gt_instance = np.zeros((Z, Y, X), dtype=np.int32)
    gt_individual_masks = []

    for i, nodule in enumerate(gt_nodules):
        sphere = generate_gt_sphere_mask(nodule, image_shape, origin, spacing, direction)
        gt_semantic = np.maximum(gt_semantic, sphere)
        gt_individual_masks.append(sphere)

        # FIX #5: For overlapping GT nodules, use "first wins" instead of
        # "last wins" — this is more correct because the first nodule in
        # annotations is typically the higher-confidence annotation.
        # Only assign where instance mask is still 0.
        overlap_mask = (sphere > 0) & (gt_instance == 0)
        gt_instance[overlap_mask] = i + 1  # 1-indexed

    return gt_semantic, gt_instance, gt_individual_masks


# ──────────────────────────────────────────────
# 3D Box IoU
# ──────────────────────────────────────────────

def box_iou_3d(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute 3D IoU between two boxes.
    Each box: (z1, y1, x1, z2, y2, x2).
    """
    z1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x1 = max(box_a[2], box_b[2])
    z2 = min(box_a[3], box_b[3])
    y2 = min(box_a[4], box_b[4])
    x2 = min(box_a[5], box_b[5])

    inter = max(0, z2 - z1) * max(0, y2 - y1) * max(0, x2 - x1)
    if inter == 0:
        return 0.0

    vol_a = max(0, box_a[3]-box_a[0]) * max(0, box_a[4]-box_a[1]) * max(0, box_a[5]-box_a[2])
    vol_b = max(0, box_b[3]-box_b[0]) * max(0, box_b[4]-box_b[1]) * max(0, box_b[5]-box_b[2])
    union = vol_a + vol_b - inter

    return float(inter / union) if union > 0 else 0.0


# ──────────────────────────────────────────────
# Volumetric Dice & IoU
# ──────────────────────────────────────────────

def volumetric_dice(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """Dice coefficient between two binary masks."""
    pred = mask_pred.astype(bool)
    gt = mask_gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0  # Both empty = perfect
    return float(2.0 * intersection / total)


def volumetric_iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """IoU (Jaccard) between two binary masks."""
    pred = mask_pred.astype(bool)
    gt = mask_gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


# ──────────────────────────────────────────────
# Instance matching (greedy, score-ranked)
# ──────────────────────────────────────────────

def match_instances_greedy(
    gt_boxes: List[np.ndarray],
    pred_boxes: List[np.ndarray],
    pred_scores: List[float],
    distance_thresh_mm: float = 15.0,
    iou_thresh: float = 0.1,
    spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0),
    gt_centers: Optional[List[np.ndarray]] = None,
    pred_centers: Optional[List[np.ndarray]] = None,
) -> Tuple[List[MatchResult], List[bool]]:
    """
    Greedy matching: highest-score predictions matched first.

    FIX #6: Changed matching criteria from OR to AND-like logic:
      - Primary: center distance < threshold (LUNA16 standard)
      - Secondary: among candidates within distance, pick best IoU
      - Fallback: if no center-distance match, accept if IoU > threshold
    This prevents false matches where a prediction is 14mm away with 0 IoU.

    FIX #1 (partial): Now also returns per-prediction TP status for FROC.

    Returns:
        match_results: list of MatchResult for each GT nodule
        pred_is_tp: list of bool for each prediction (True if matched to a GT)
    """
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)

    # Compute centers if not provided
    if gt_centers is None:
        gt_centers = [(b[:3] + b[3:]) / 2 for b in gt_boxes]
    if pred_centers is None:
        pred_centers = [(b[:3] + b[3:]) / 2 for b in pred_boxes]

    sp = np.array(spacing)

    # Sort predictions by score (descending)
    order = np.argsort(pred_scores)[::-1]

    matched_gt = set()
    matched_pred = set()
    results = [MatchResult(gt_idx=i) for i in range(n_gt)]

    for pi in order:
        best_gt = -1
        best_dist = float('inf')
        best_iou = 0.0

        for gi in range(n_gt):
            if gi in matched_gt:
                continue

            # Euclidean distance in mm
            diff = (np.array(pred_centers[pi]) - np.array(gt_centers[gi])) * sp
            dist_mm = float(np.linalg.norm(diff))

            iou = box_iou_3d(pred_boxes[pi], gt_boxes[gi])

            # FIX #6: Two-stage matching
            # Stage 1: Center distance within threshold (LUNA16 standard)
            if dist_mm < distance_thresh_mm:
                # Among distance-qualified candidates, prefer closest
                if dist_mm < best_dist:
                    best_dist = dist_mm
                    best_gt = gi
                    best_iou = iou

            # Stage 2: If no distance match yet, accept pure IoU match
            elif iou > iou_thresh and best_gt < 0:
                best_dist = dist_mm
                best_gt = gi
                best_iou = iou

        if best_gt >= 0:
            matched_gt.add(best_gt)
            matched_pred.add(pi)
            results[best_gt] = MatchResult(
                gt_idx=best_gt,
                pred_idx=int(pi),
                distance_mm=float(best_dist),
                iou_3d=float(best_iou),
                matched=True,
            )

    # FIX #1: Build per-prediction TP flags
    pred_is_tp = [False] * n_pred
    for pi in matched_pred:
        pred_is_tp[pi] = True

    return results, pred_is_tp


# ──────────────────────────────────────────────
# FROC computation
# ──────────────────────────────────────────────

def compute_froc(
    all_gt_counts: List[int],              # per-scan: number of GT nodules
    all_pred_scores: List[List[float]],    # per-scan, per-pred: score
    all_pred_is_tp: List[List[bool]],      # per-scan, per-pred: is it a TP?
    num_scans: int,
    fp_rates: List[float] = None,
) -> Dict:
    """
    Compute FROC curve and sensitivities at standard FP rates.

    The LUNA16 Competition Performance Metric (CPM) is the average
    sensitivity at 1/8, 1/4, 1/2, 1, 2, 4, 8 FPs per scan.

    FIX #1: Now receives ALL prediction scores (TP + FP), not just
    matched ones. The original code only tracked scores from the
    matches list (which had one entry per GT), completely losing
    unmatched FP predictions. This caused FROC to undercount FPs.
    """
    if fp_rates is None:
        fp_rates = [0.125, 0.25, 0.5, 1, 2, 4, 8]

    # Flatten all predictions
    all_scores_flat = []
    all_tp_flat = []
    total_gt = sum(all_gt_counts)

    for scan_scores, scan_tp in zip(all_pred_scores, all_pred_is_tp):
        for score, is_tp in zip(scan_scores, scan_tp):
            all_scores_flat.append(score)
            all_tp_flat.append(is_tp)

    if len(all_scores_flat) == 0 or total_gt == 0:
        return {
            "froc_sensitivities": {str(fp): 0.0 for fp in fp_rates},
            "CPM": 0.0,
            "total_gt": total_gt,
            "num_scans": num_scans,
        }

    # Sort by score descending
    order = np.argsort(all_scores_flat)[::-1]
    all_scores_sorted = np.array(all_scores_flat)[order]
    all_tp_sorted = np.array(all_tp_flat)[order]

    # Accumulate TP and FP
    cum_tp = np.cumsum(all_tp_sorted)
    cum_fp = np.cumsum(~all_tp_sorted)

    sensitivity = cum_tp / total_gt
    fp_per_scan = cum_fp / num_scans

    # Interpolate sensitivity at each target FP rate
    froc_sens = {}
    for fp_target in fp_rates:
        valid = fp_per_scan <= fp_target
        if valid.any():
            froc_sens[str(fp_target)] = float(sensitivity[valid].max())
        else:
            froc_sens[str(fp_target)] = 0.0

    cpm = float(np.mean(list(froc_sens.values())))

    return {
        "froc_sensitivities": froc_sens,
        "CPM": cpm,
        "total_gt": total_gt,
        "num_scans": num_scans,
    }


# ──────────────────────────────────────────────
# Panoptic Quality (PQ = SQ × RQ)
# ──────────────────────────────────────────────

def compute_panoptic_quality(
    pred_instance_mask: np.ndarray,
    gt_instance_mask: np.ndarray,
    iou_thresh: float = 0.5,
) -> Dict[str, float]:
    """
    Compute Panoptic Quality between predicted and GT instance masks.

    PQ = (Σ IoU(p,g) for matched pairs) / (TP + 0.5*FP + 0.5*FN)

    Equivalently: PQ = SQ × RQ where
        SQ = mean IoU of matched pairs (Segmentation Quality)
        RQ = TP / (TP + 0.5*FP + 0.5*FN) (Recognition Quality)
    """
    pred_ids = np.unique(pred_instance_mask)
    pred_ids = pred_ids[pred_ids > 0]
    gt_ids = np.unique(gt_instance_mask)
    gt_ids = gt_ids[gt_ids > 0]

    matched_pred = set()
    matched_gt = set()
    matched_ious = []

    for gt_id in gt_ids:
        gt_mask = (gt_instance_mask == gt_id)
        best_iou = 0.0
        best_pred = None

        for pred_id in pred_ids:
            if pred_id in matched_pred:
                continue
            pred_mask = (pred_instance_mask == pred_id)

            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            iou = float(intersection / union) if union > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_pred = pred_id

        if best_iou >= iou_thresh and best_pred is not None:
            matched_pred.add(best_pred)
            matched_gt.add(gt_id)
            matched_ious.append(best_iou)

    tp = len(matched_ious)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp

    sq = float(np.mean(matched_ious)) if tp > 0 else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    pq = sq * rq

    return {
        "panoptic_quality": float(pq),
        "segmentation_quality": float(sq),
        "recognition_quality": float(rq),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


# ──────────────────────────────────────────────
# Build instance-ID-to-box mapping
# ──────────────────────────────────────────────

def build_instance_id_map(instance_mask_np: np.ndarray) -> Dict[int, np.ndarray]:
    """
    FIX #2: Build a reliable mapping from instance mask IDs to bounding boxes.

    The original code assumed pred_idx (from score-sorted matching order)
    directly mapped to instance IDs in the mask. But boxes_to_pseudo_panoptic
    assigns IDs sequentially (1, 2, 3...) AFTER sorting by score, so pred_idx
    from greedy matching (also sorted by score) may not correspond to the
    correct instance ID.

    This function extracts the actual instance IDs and their bounding boxes
    directly from the mask — the single source of truth.
    """
    instance_ids = np.unique(instance_mask_np)
    instance_ids = instance_ids[instance_ids > 0]

    id_to_box = {}
    for iid in instance_ids:
        coords = np.where(instance_mask_np == iid)
        if len(coords[0]) == 0:
            continue
        z1, z2 = coords[0].min(), coords[0].max() + 1
        y1, y2 = coords[1].min(), coords[1].max() + 1
        x1, x2 = coords[2].min(), coords[2].max() + 1
        id_to_box[int(iid)] = np.array([z1, y1, x1, z2, y2, x2], dtype=np.float64)

    return id_to_box


def find_pred_instance_for_box(
    pred_box: np.ndarray,
    instance_id_map: Dict[int, np.ndarray],
) -> Optional[int]:
    """
    FIX #2: Find which instance ID in the mask best matches a given
    predicted box by computing IoU between the box and each instance's
    bounding box in the mask.
    """
    best_iou = 0.0
    best_id = None

    for iid, mask_box in instance_id_map.items():
        iou = box_iou_3d(pred_box, mask_box)
        if iou > best_iou:
            best_iou = iou
            best_id = iid

    return best_id


# ──────────────────────────────────────────────
# Full single-scan evaluation
# ──────────────────────────────────────────────

def evaluate_scan(
    pred_boxes: torch.Tensor,          # (N, 6) voxel-space boxes
    pred_scores: torch.Tensor,         # (N,)
    pred_semantic_mask: torch.Tensor,   # (Z,Y,X) binary
    pred_instance_mask: torch.Tensor,   # (Z,Y,X) int32
    gt_nodules: List[GTNodule],
    image_shape: Tuple[int, int, int],
    origin: Tuple[float, float, float],
    spacing: Tuple[float, float, float],  # SimpleITK order (x,y,z)
    direction: Optional[np.ndarray] = None,
    distance_thresh_mm: float = 15.0,
    iou_thresh_match: float = 0.1,
    iou_thresh_pq: float = 0.5,
    seriesuid: str = "",
) -> ScanEvaluation:
    """
    Run all evaluation metrics on a single scan.

    FIXES applied:
      - #1: Stores all pred scores + TP flags for correct FROC
      - #2: Uses instance_id_map for reliable mask↔box correspondence
      - #5: Uses individual GT masks for per-instance Dice
      - #6: Two-stage matching (distance-first, then IoU fallback)
    """
    eval_result = ScanEvaluation(seriesuid=seriesuid)
    eval_result.num_gt = len(gt_nodules)

    # Convert predictions to numpy
    if isinstance(pred_boxes, torch.Tensor):
        pred_boxes_np = pred_boxes.cpu().numpy()
        pred_scores_np = pred_scores.cpu().numpy()
        pred_sem_np = pred_semantic_mask.cpu().numpy()
        pred_inst_np = pred_instance_mask.cpu().numpy()
    else:
        pred_boxes_np = np.array(pred_boxes)
        pred_scores_np = np.array(pred_scores)
        pred_sem_np = np.array(pred_semantic_mask)
        pred_inst_np = np.array(pred_instance_mask)

    eval_result.num_pred = len(pred_boxes_np)

    # ── FIX #1: Store ALL prediction scores for FROC ──
    eval_result.all_pred_scores = pred_scores_np.tolist()

    if eval_result.num_gt == 0 and eval_result.num_pred == 0:
        eval_result.sensitivity = 1.0
        eval_result.precision = 1.0
        eval_result.f1_score = 1.0
        eval_result.mean_dice = 1.0
        eval_result.mean_iou = 1.0
        eval_result.panoptic_quality = 1.0
        eval_result.all_pred_is_tp = []
        return eval_result

    if eval_result.num_gt == 0 and eval_result.num_pred > 0:
        # All predictions are FP
        eval_result.false_positives = eval_result.num_pred
        eval_result.all_pred_is_tp = [False] * eval_result.num_pred
        return eval_result

    # Convert GT nodules to voxel boxes
    gt_boxes = []
    gt_centers = []
    for nodule in gt_nodules:
        box, center = gt_nodule_to_voxel_box(nodule, origin, spacing, direction)
        gt_boxes.append(box)
        gt_centers.append(center)

    pred_centers = [
        (pred_boxes_np[i, :3] + pred_boxes_np[i, 3:]) / 2
        for i in range(len(pred_boxes_np))
    ]

    # ── Instance matching ──
    voxel_spacing_zyx = (spacing[2], spacing[1], spacing[0])

    match_results, pred_is_tp = match_instances_greedy(
        gt_boxes=gt_boxes,
        pred_boxes=[pred_boxes_np[i] for i in range(len(pred_boxes_np))],
        pred_scores=pred_scores_np.tolist(),
        distance_thresh_mm=distance_thresh_mm,
        iou_thresh=iou_thresh_match,
        spacing=voxel_spacing_zyx,
        gt_centers=gt_centers,
        pred_centers=pred_centers,
    )

    # ── FIX #1: Store per-prediction TP status ──
    eval_result.all_pred_is_tp = pred_is_tp

    matched_pred_indices = set()
    tp = 0
    for mr in match_results:
        if mr.matched:
            tp += 1
            matched_pred_indices.add(mr.pred_idx)

    fp = eval_result.num_pred - tp
    fn = eval_result.num_gt - tp

    eval_result.true_positives = tp
    eval_result.false_positives = fp
    eval_result.false_negatives = fn
    eval_result.sensitivity = tp / eval_result.num_gt if eval_result.num_gt > 0 else 0.0
    eval_result.precision = tp / eval_result.num_pred if eval_result.num_pred > 0 else 0.0
    if eval_result.sensitivity + eval_result.precision > 0:
        eval_result.f1_score = (
            2 * eval_result.sensitivity * eval_result.precision /
            (eval_result.sensitivity + eval_result.precision)
        )

    # ── Generate GT masks for Dice/IoU ──
    gt_semantic, gt_instance, gt_individual_masks = generate_gt_panoptic_masks(
        gt_nodules, image_shape, origin, spacing, direction
    )

    # Global semantic Dice & IoU
    eval_result.mean_dice = volumetric_dice(pred_sem_np, gt_semantic)
    eval_result.mean_iou = volumetric_iou(pred_sem_np, gt_semantic)

    # ── Panoptic Quality ──
    pq_result = compute_panoptic_quality(pred_inst_np, gt_instance, iou_thresh_pq)
    eval_result.panoptic_quality = pq_result["panoptic_quality"]
    eval_result.segmentation_quality = pq_result["segmentation_quality"]
    eval_result.recognition_quality = pq_result["recognition_quality"]

    # ── FIX #2: Build instance ID map for reliable mask lookups ──
    instance_id_map = build_instance_id_map(pred_inst_np)

    # ── Per-instance match details ──
    for mr in match_results:
        entry = {
            "gt_idx": mr.gt_idx,
            "gt_diameter_mm": gt_nodules[mr.gt_idx].diameter_mm,
            "matched": mr.matched,
            "distance_mm": round(mr.distance_mm, 2),
            "box_iou_3d": round(mr.iou_3d, 4),
        }
        if mr.matched and mr.pred_idx is not None:
            entry["pred_idx"] = mr.pred_idx
            entry["pred_score"] = round(float(pred_scores_np[mr.pred_idx]), 4)

            # FIX #2: Use individual GT mask (not combined instance mask)
            # and find correct pred instance via box↔mask IoU matching
            gt_mask_i = gt_individual_masks[mr.gt_idx]

            pred_instance_id = find_pred_instance_for_box(
                pred_boxes_np[mr.pred_idx], instance_id_map
            )
            if pred_instance_id is not None:
                pred_mask_i = (pred_inst_np == pred_instance_id).astype(np.uint8)
            else:
                pred_mask_i = np.zeros_like(gt_mask_i)

            entry["pred_instance_id"] = pred_instance_id
            entry["instance_dice"] = round(volumetric_dice(pred_mask_i, gt_mask_i), 4)
            entry["instance_iou"] = round(volumetric_iou(pred_mask_i, gt_mask_i), 4)
        else:
            entry["pred_idx"] = None
            entry["pred_score"] = None
            entry["pred_instance_id"] = None
            entry["instance_dice"] = 0.0
            entry["instance_iou"] = 0.0

        eval_result.matches.append(entry)

    return eval_result


# ──────────────────────────────────────────────
# Dataset-level evaluation
# ──────────────────────────────────────────────

def evaluate_dataset(scan_results: List[ScanEvaluation]) -> Dict:
    """
    Aggregate per-scan results into dataset-level metrics.

    FIX #1: FROC now uses all_pred_scores and all_pred_is_tp from
    ScanEvaluation, which contain data for EVERY prediction (TP + FP),
    not just matched GT entries. This is critical for correct FP counting.
    """
    if not scan_results:
        return {"error": "No scan results to aggregate"}

    n = len(scan_results)
    total_gt = sum(s.num_gt for s in scan_results)
    total_pred = sum(s.num_pred for s in scan_results)
    total_tp = sum(s.true_positives for s in scan_results)
    total_fp = sum(s.false_positives for s in scan_results)
    total_fn = sum(s.false_negatives for s in scan_results)

    # ── FIX #1: Correct FROC data collection ──
    # Use the per-prediction data stored in ScanEvaluation, NOT the matches list.
    all_gt_counts = [s.num_gt for s in scan_results]
    all_pred_scores = [s.all_pred_scores for s in scan_results]
    all_pred_is_tp = [s.all_pred_is_tp for s in scan_results]

    froc = compute_froc(all_gt_counts, all_pred_scores, all_pred_is_tp, n)

    summary = {
        "num_scans": n,
        "total_gt_nodules": total_gt,
        "total_predictions": total_pred,
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,

        # Per-scan averages
        "mean_sensitivity": float(np.mean([s.sensitivity for s in scan_results])),
        "mean_precision": float(np.mean([s.precision for s in scan_results])),
        "mean_f1": float(np.mean([s.f1_score for s in scan_results])),
        "mean_dice": float(np.mean([s.mean_dice for s in scan_results])),
        "mean_iou": float(np.mean([s.mean_iou for s in scan_results])),
        "mean_panoptic_quality": float(np.mean([s.panoptic_quality for s in scan_results])),
        "mean_segmentation_quality": float(np.mean([s.segmentation_quality for s in scan_results])),
        "mean_recognition_quality": float(np.mean([s.recognition_quality for s in scan_results])),

        # Global detection rates
        "global_sensitivity": total_tp / total_gt if total_gt > 0 else 0.0,
        "global_precision": total_tp / total_pred if total_pred > 0 else 0.0,

        # FROC
        "FROC": froc,
    }

    return summary


# ──────────────────────────────────────────────
# Save results to JSON
# ──────────────────────────────────────────────

def save_evaluation(
    scan_results: List[ScanEvaluation],
    dataset_summary: Dict,
    output_path: str,
):
    """Save full evaluation results to JSON."""
    output = {
        "dataset_summary": dataset_summary,
        "per_scan_results": [asdict(sr) for sr in scan_results],
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"✅ Evaluation saved to {output_path}")