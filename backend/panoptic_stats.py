# panoptic_stats.py
import torch
import math


def compute_panoptic_stats(
    semantic_mask,        # (Z,Y,X)
    instance_mask,        # (Z,Y,X)
    instance_scores=None, # Tensor[K]
    voxel_spacing=None,   # (z,y,x) in mm
):
    """
    Compute global statistics (JSON-serializable).
    """

    stats = {}

    # ---------------- INSTANCE STATS ----------------
    instance_ids = torch.unique(instance_mask)
    instance_ids = instance_ids[instance_ids > 0]

    stats["num_instances"] = int(len(instance_ids))

    if len(instance_ids) > 0:
        volumes_vox = [
            int((instance_mask == iid).sum().item())
            for iid in instance_ids
        ]

        stats["avg_instance_volume_voxels"] = float(sum(volumes_vox) / len(volumes_vox))
        stats["max_instance_volume_voxels"] = int(max(volumes_vox))
        stats["min_instance_volume_voxels"] = int(min(volumes_vox))

        if voxel_spacing is not None:
            voxel_vol = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
            volumes_mm3 = [v * voxel_vol for v in volumes_vox]

            stats["avg_instance_volume_mm3"] = float(sum(volumes_mm3) / len(volumes_mm3))
            stats["max_instance_volume_mm3"] = float(max(volumes_mm3))
            stats["min_instance_volume_mm3"] = float(min(volumes_mm3))

            stats["instance_diameters_mm"] = [
                float(2 * ((3 * v / (4 * math.pi)) ** (1 / 3)))
                for v in volumes_mm3
            ]
    else:
        stats.update({
            "avg_instance_volume_voxels": 0.0,
            "max_instance_volume_voxels": 0,
            "min_instance_volume_voxels": 0,
        })

    # ---------------- SEMANTIC STATS ----------------
    total_voxels = semantic_mask.numel()
    nodule_voxels = int(semantic_mask.sum().item())

    stats["nodule_voxel_count"] = nodule_voxels
    stats["semantic_coverage_percent"] = float(
        100.0 * nodule_voxels / total_voxels if total_voxels > 0 else 0.0
    )

    # ---------------- CONFIDENCE STATS ----------------
    if instance_scores is not None and len(instance_scores) > 0:
        stats["mean_confidence"] = float(instance_scores.mean().item())
        stats["max_confidence"] = float(instance_scores.max().item())
        stats["min_confidence"] = float(instance_scores.min().item())
        stats["high_conf_instances"] = int((instance_scores > 0.5).sum().item())
    else:
        stats["mean_confidence"] = 0.0
        stats["high_conf_instances"] = 0

    return stats


def get_instance_details(
    instance_mask,
    instance_scores=None,
    voxel_spacing=None,
):
    """
    Per-instance clinical-style details.
    """

    details = []

    instance_ids = torch.unique(instance_mask)
    instance_ids = instance_ids[instance_ids > 0]

    for idx, iid in enumerate(instance_ids):
        mask = instance_mask == iid
        coords = torch.where(mask)

        volume_vox = int(mask.sum().item())

        zmin, zmax = coords[0].min().item(), coords[0].max().item()
        ymin, ymax = coords[1].min().item(), coords[1].max().item()
        xmin, xmax = coords[2].min().item(), coords[2].max().item()

        entry = {
            "instance_id": int(iid),
            "volume_voxels": volume_vox,
            "bbox": [zmin, ymin, xmin, zmax, ymax, xmax],
            "centroid": [
                float(coords[0].float().mean().item()),
                float(coords[1].float().mean().item()),
                float(coords[2].float().mean().item()),
            ],
        }

        if instance_scores is not None and idx < len(instance_scores):
            entry["confidence"] = float(instance_scores[idx].item())

        if voxel_spacing is not None:
            voxel_vol = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
            volume_mm3 = volume_vox * voxel_vol

            entry["volume_mm3"] = float(volume_mm3)
            entry["diameter_mm"] = float(
                2 * ((3 * volume_mm3 / (4 * math.pi)) ** (1 / 3))
            )

        details.append(entry)

    return details
