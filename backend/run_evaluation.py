"""
run_evaluation.py
=================
End-to-end evaluation runner for the Lung Nodule Pseudo-Panoptic pipeline.
Runs inference on LUNA16 scans, compares against annotations.csv, and
produces a full evaluation report.

FIXES applied:
  - Passes actual scores and boxes (not reconstructed from mask)
  - Handles no-GT scans and no-detection scans correctly
  - Stores all prediction data for correct FROC

Usage:
    python run_evaluation.py \
        --data_dir /path/to/LUNA16/subset0 \
        --annotations /path/to/annotations.csv \
        --output_dir ./results \
        --score_threshold 0.3 \
        --max_scans 50

Directory structure expected:
    LUNA16/
    ├── annotations.csv
    ├── subset0/
    │   ├── 1.3.6.1.4...xyz.mhd
    │   └── 1.3.6.1.4...xyz.raw
    ├── subset1/
    ...

Compatible with: huggingface.co/MONAI/lung_nodule_ct_detection
"""

import argparse
import json
import csv
import time
import traceback
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from monai.apps.detection.networks.retinanet_network import (
    RetinaNet,
    resnet_fpn_feature_extractor,
)
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.networks.nets.resnet import resnet50
from monai.data import MetaTensor

from pseudo_panoptic import boxes_to_pseudo_panoptic
from evaluation import (
    load_luna16_annotations,
    evaluate_scan,
    evaluate_dataset,
    save_evaluation,
    ScanEvaluation,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pseudo-panoptic pipeline on LUNA16")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to LUNA16 subset directory (or parent with subset0..subset9)")
    parser.add_argument("--annotations", type=str, required=True,
                        help="Path to LUNA16 annotations.csv")
    parser.add_argument("--model_path", type=str, default="./models/model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Where to save evaluation results")
    parser.add_argument("--score_threshold", type=float, default=0.3,
                        help="Detection confidence threshold")
    parser.add_argument("--overlap_strategy", type=str, default="highest_score",
                        choices=["highest_score", "first_wins", "last_wins"])
    parser.add_argument("--distance_thresh_mm", type=float, default=15.0,
                        help="Max distance (mm) for GT<->pred matching")
    parser.add_argument("--iou_thresh_match", type=float, default=0.1,
                        help="Min 3D IoU for detection matching")
    parser.add_argument("--iou_thresh_pq", type=float, default=0.5,
                        help="IoU threshold for Panoptic Quality")
    parser.add_argument("--max_scans", type=int, default=None,
                        help="Limit number of scans to evaluate (for quick testing)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cpu', 'cuda', or 'auto'")
    parser.add_argument("--resampling_spacing", type=float, nargs=3, default=[2.0, 2.0, 2.0],
                        help="Resampling spacing (x,y,z) in mm")
    return parser.parse_args()


def resample_ct(img, out_spacing=(2.0, 2.0, 2.0)):
    """Resample CT to consistent spacing."""
    spacing = img.GetSpacing()
    size = img.GetSize()
    new_size = [int(size[i] * spacing[i] / out_spacing[i]) for i in range(3)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetDefaultPixelValue(-1024)
    return resampler.Execute(img)


def load_model(model_path: str, device: torch.device) -> RetinaNetDetector:
    """
    Load the MONAI RetinaNet detector.
    Configuration matches api.py and the pretrained HuggingFace model exactly:
      - ResNet50 backbone, 3D, FPN with layers [1,2]
      - 3 anchors: (6,8,4), (8,6,5), (10,10,6)
      - size_divisible=[16,16,8]
      - sliding window (96,96,96) with 0.25 overlap
    """
    print("🔧 Loading detection model...")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    checkpoint = checkpoint.get("model", checkpoint)

    backbone = resnet50(
        spatial_dims=3,
        n_input_channels=1,
        conv1_t_stride=[2, 2, 1],
        conv1_t_size=[7, 7, 7],
    )

    feature_extractor = resnet_fpn_feature_extractor(
        backbone=backbone,
        spatial_dims=3,
        pretrained_backbone=False,
        returned_layers=[1, 2],
    )

    network = RetinaNet(
        spatial_dims=3,
        num_classes=1,
        num_anchors=3,
        feature_extractor=feature_extractor,
        size_divisible=[16, 16, 8],
    )

    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=(1, 2, 4),
        base_anchor_shapes=(
            (6, 8, 4),
            (8, 6, 5),
            (10, 10, 6),
        ),
    )

    detector = RetinaNetDetector(
        network=network,
        anchor_generator=anchor_generator,
        spatial_dims=3,
        num_classes=1,
        size_divisible=[16, 16, 8],
    )

    detector.set_box_selector_parameters(
        score_thresh=0.02,
        nms_thresh=0.22,
        detections_per_img=300,
    )

    detector.set_sliding_window_inferer(
        roi_size=(96, 96, 96),
        sw_batch_size=1,
        overlap=0.25,
        mode="gaussian",
    )

    missing, unexpected = detector.network.load_state_dict(checkpoint, strict=False)
    if missing:
        print(f"⚠️  Missing keys: {len(missing)}")
    if unexpected:
        print(f"⚠️  Unexpected keys: {len(unexpected)}")

    detector.network.to(device)
    detector.eval()
    print(f"✅ Model loaded on {device}")
    return detector


def find_mhd_files(data_dir: str) -> list:
    """Find all .mhd files in data_dir (supports subset directories)."""
    data_path = Path(data_dir)
    mhd_files = sorted(data_path.glob("**/*.mhd"))
    print(f"📂 Found {len(mhd_files)} .mhd files in {data_dir}")
    return mhd_files


def run_inference_single(
    mhd_path: Path,
    detector: RetinaNetDetector,
    device: torch.device,
    resampling_spacing: tuple,
    score_threshold: float,
    overlap_strategy: str,
):
    """
    Run inference on a single scan.
    Returns dict with all data needed for evaluation, or None if no detections.
    """
    img = sitk.ReadImage(str(mhd_path))
    img = sitk.DICOMOrient(img, "RAS")

    img = resample_ct(img, resampling_spacing)

    # Store metadata AFTER resampling (this is the coordinate space
    # the model operates in and GT must be converted to)
    resampled_origin = img.GetOrigin()       # (x, y, z)
    resampled_spacing = img.GetSpacing()     # (x, y, z)
    resampled_direction = np.array(img.GetDirection()).reshape(3, 3)

    vol_np = sitk.GetArrayFromImage(img).astype(np.float32)
    vol_np = np.clip(vol_np, -1024, 300)
    vol_np = (vol_np + 1024) / (300 + 1024)
    vol_np = vol_np[np.newaxis, ...]  # (1, D, H, W)

    voxel_spacing = tuple(resampled_spacing)[::-1]  # (z,y,x)

    vol = MetaTensor(
        torch.from_numpy(vol_np),
        meta={
            "spacing": voxel_spacing,
            "pixdim": (1.0, *voxel_spacing),
        },
    ).unsqueeze(0).to(device)

    image_shape = vol.shape[-3:]

    with torch.no_grad():
        outputs = detector(vol)
        result = outputs[0]

        scores = None
        for key in ("label_scores", "labels_scores", "scores"):
            if key in result:
                scores = result[key]
                break
        if scores is None:
            raise RuntimeError(f"No score key found: {result.keys()}")

    boxes = result.get("boxes")

    if boxes is None or len(boxes) == 0:
        return None  # No detections

    keep = scores >= score_threshold
    boxes = boxes[keep]
    scores = scores[keep]

    if len(boxes) == 0:
        return None

    semantic_mask, instance_mask, instance_scores = boxes_to_pseudo_panoptic(
        boxes=boxes,
        scores=scores,
        image_shape=image_shape,
        overlap_strategy=overlap_strategy,
        score_thresh=score_threshold,
        mask_shape="refined",
        volume=vol[0, 0],
    )

    return {
        "boxes": boxes,          # actual detection boxes (N, 6) tensor
        "scores": scores,        # actual detection scores (N,) tensor
        "semantic_mask": semantic_mask,
        "instance_mask": instance_mask,
        "instance_scores": instance_scores,
        "image_shape": tuple(int(s) for s in image_shape),
        "origin": resampled_origin,       # (x,y,z) SimpleITK order
        "spacing": resampled_spacing,     # (x,y,z) SimpleITK order
        "direction": resampled_direction, # 3x3 matrix
    }


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load annotations
    annotations = load_luna16_annotations(args.annotations)

    # Load model
    detector = load_model(args.model_path, device)

    # Find scans
    mhd_files = find_mhd_files(args.data_dir)
    if args.max_scans:
        mhd_files = mhd_files[:args.max_scans]

    resampling_spacing = tuple(args.resampling_spacing)

    # Evaluate each scan
    scan_results = []
    errors = 0

    print(f"\n{'='*60}")
    print(f"  Evaluating {len(mhd_files)} scans")
    print(f"  Score threshold: {args.score_threshold}")
    print(f"  Distance threshold: {args.distance_thresh_mm} mm")
    print(f"  Resampling: {resampling_spacing}")
    print(f"{'='*60}\n")

    for i, mhd_path in enumerate(mhd_files):
        seriesuid = mhd_path.stem
        gt_nodules = annotations.get(seriesuid, [])

        print(f"[{i+1}/{len(mhd_files)}] {seriesuid[:40]}... "
              f"(GT: {len(gt_nodules)} nodules)", end=" ")

        t0 = time.time()

        try:
            inf_result = run_inference_single(
                mhd_path, detector, device,
                resampling_spacing, args.score_threshold, args.overlap_strategy,
            )

            if inf_result is None:
                # No detections — create result with 0 predictions
                sr = ScanEvaluation(
                    seriesuid=seriesuid,
                    num_gt=len(gt_nodules),
                    num_pred=0,
                    false_negatives=len(gt_nodules),
                    all_pred_scores=[],   # No predictions
                    all_pred_is_tp=[],    # No predictions
                )
                if len(gt_nodules) == 0:
                    # No GT and no predictions = perfect
                    sr.sensitivity = 1.0
                    sr.precision = 1.0
                    sr.f1_score = 1.0
                    sr.mean_dice = 1.0
                    sr.mean_iou = 1.0
                scan_results.append(sr)
                elapsed = time.time() - t0
                print(f"→ 0 preds ({elapsed:.1f}s)")
                continue

            sr = evaluate_scan(
                pred_boxes=inf_result["boxes"],
                pred_scores=inf_result["scores"],
                pred_semantic_mask=inf_result["semantic_mask"],
                pred_instance_mask=inf_result["instance_mask"],
                gt_nodules=gt_nodules,
                image_shape=inf_result["image_shape"],
                origin=inf_result["origin"],
                spacing=inf_result["spacing"],
                direction=inf_result["direction"],
                distance_thresh_mm=args.distance_thresh_mm,
                iou_thresh_match=args.iou_thresh_match,
                iou_thresh_pq=args.iou_thresh_pq,
                seriesuid=seriesuid,
            )
            scan_results.append(sr)

            elapsed = time.time() - t0
            print(f"→ {sr.num_pred} preds, TP={sr.true_positives}, "
                  f"Dice={sr.mean_dice:.3f}, PQ={sr.panoptic_quality:.3f} ({elapsed:.1f}s)")

        except Exception as e:
            errors += 1
            print(f"→ ERROR: {e}")
            traceback.print_exc()
            continue
        finally:
            # Free GPU memory between scans to prevent OOM on large datasets
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not scan_results:
        print("\n❌ No scans were successfully evaluated.")
        return

    # Dataset summary
    print(f"\n{'='*60}")
    print(f"  Computing dataset-level metrics...")
    print(f"{'='*60}\n")

    dataset_summary = evaluate_dataset(scan_results)

    # Print summary
    print("📊 EVALUATION SUMMARY")
    print(f"   Scans evaluated: {dataset_summary['num_scans']}")
    print(f"   Scans with errors: {errors}")
    print(f"   Total GT nodules: {dataset_summary['total_gt_nodules']}")
    print(f"   Total predictions: {dataset_summary['total_predictions']}")
    print()
    print(f"   DETECTION:")
    print(f"     Mean Sensitivity:  {dataset_summary['mean_sensitivity']:.4f}")
    print(f"     Mean Precision:    {dataset_summary['mean_precision']:.4f}")
    print(f"     Mean F1:           {dataset_summary['mean_f1']:.4f}")
    print(f"     Global Sensitivity:{dataset_summary['global_sensitivity']:.4f}")
    print(f"     Global Precision:  {dataset_summary['global_precision']:.4f}")
    print()
    print(f"   SEGMENTATION (Pseudo-Panoptic):")
    print(f"     Mean Dice:         {dataset_summary['mean_dice']:.4f}")
    print(f"     Mean IoU:          {dataset_summary['mean_iou']:.4f}")
    print()
    print(f"   PANOPTIC QUALITY:")
    print(f"     Mean PQ:           {dataset_summary['mean_panoptic_quality']:.4f}")
    print(f"     Mean SQ:           {dataset_summary['mean_segmentation_quality']:.4f}")
    print(f"     Mean RQ:           {dataset_summary['mean_recognition_quality']:.4f}")
    print()
    print(f"   FROC (LUNA16 Competition Metric):")
    froc = dataset_summary.get("FROC", {})
    print(f"     CPM Score:         {froc.get('CPM', 0):.4f}")
    for fp_rate, sens in froc.get("froc_sensitivities", {}).items():
        print(f"     Sens @ {fp_rate} FP/scan: {sens:.4f}")

    # Save full results
    output_path = output_dir / "evaluation_results.json"
    save_evaluation(scan_results, dataset_summary, str(output_path))

    # Save per-scan CSV for quick inspection
    csv_path = output_dir / "per_scan_metrics.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "seriesuid", "num_gt", "num_pred", "TP", "FP", "FN",
            "sensitivity", "precision", "f1", "dice", "iou",
            "PQ", "SQ", "RQ"
        ])
        for sr in scan_results:
            writer.writerow([
                sr.seriesuid, sr.num_gt, sr.num_pred,
                sr.true_positives, sr.false_positives, sr.false_negatives,
                f"{sr.sensitivity:.4f}", f"{sr.precision:.4f}", f"{sr.f1_score:.4f}",
                f"{sr.mean_dice:.4f}", f"{sr.mean_iou:.4f}",
                f"{sr.panoptic_quality:.4f}", f"{sr.segmentation_quality:.4f}",
                f"{sr.recognition_quality:.4f}",
            ])
    print(f"✅ Per-scan CSV saved to {csv_path}")

    # Also save a concise summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(dataset_summary, f, indent=2, default=str)
    print(f"\n✅ Summary saved to {summary_path}")


if __name__ == "__main__":
    main()