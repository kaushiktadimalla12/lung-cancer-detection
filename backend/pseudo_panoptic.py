# pseudo_panoptic.py — Improved with ellipsoidal masks + CT intensity refinement
import torch
import numpy as np
from scipy import ndimage


def _generate_ellipsoid_mask(shape_zyx, margin=0.0):
    """
    Generate a binary ellipsoid mask inscribed within a 3D box.
    
    Args:
        shape_zyx: (dz, dy, dx) shape of the bounding box region
        margin: shrink factor (0.0 = inscribed ellipsoid touching box walls,
                0.1 = 10% smaller than inscribed)
    
    Returns:
        np.ndarray bool of shape shape_zyx
    """
    dz, dy, dx = shape_zyx
    if dz <= 0 or dy <= 0 or dx <= 0:
        return np.zeros(shape_zyx, dtype=bool)

    # Center of the box
    cz, cy, cx = (dz - 1) / 2.0, (dy - 1) / 2.0, (dx - 1) / 2.0

    # Semi-axes (half the box dimensions, minus margin)
    rz = max(cz * (1.0 - margin), 0.5)
    ry = max(cy * (1.0 - margin), 0.5)
    rx = max(cx * (1.0 - margin), 0.5)

    # Create coordinate grids
    zz, yy, xx = np.ogrid[0:dz, 0:dy, 0:dx]

    # Ellipsoid equation: (z-cz)^2/rz^2 + (y-cy)^2/ry^2 + (x-cx)^2/rx^2 <= 1
    ellipsoid = (
        ((zz - cz) / rz) ** 2 +
        ((yy - cy) / ry) ** 2 +
        ((xx - cx) / rx) ** 2
    ) <= 1.0

    return ellipsoid


def _intensity_refine(volume_region, ellipsoid_mask, method="otsu"):
    """
    Refine mask within the ellipsoid using CT intensity values.
    Nodule tissue is typically denser (higher HU) than surrounding lung.
    
    Args:
        volume_region: 3D numpy array of CT values within the box
        ellipsoid_mask: boolean mask of the ellipsoid
        method: "otsu" for Otsu thresholding, "adaptive" for percentile-based
    
    Returns:
        refined boolean mask (same shape)
    """
    if not ellipsoid_mask.any():
        return ellipsoid_mask

    # Get intensity values inside the ellipsoid
    values = volume_region[ellipsoid_mask]

    if len(values) < 10:
        return ellipsoid_mask  # Too few voxels, keep ellipsoid as-is

    if method == "otsu":
        # Simple Otsu: split intensities into 2 classes
        # Nodule tissue should be the higher-intensity class
        sorted_vals = np.sort(values)
        n = len(sorted_vals)

        best_thresh = sorted_vals[n // 2]
        best_variance = 0.0

        # Test thresholds at percentile steps for efficiency
        for pct in range(10, 90, 5):
            idx = int(n * pct / 100)
            thresh = sorted_vals[idx]

            w0 = idx / n
            w1 = 1.0 - w0

            if w0 == 0 or w1 == 0:
                continue

            m0 = sorted_vals[:idx].mean()
            m1 = sorted_vals[idx:].mean()

            between_var = w0 * w1 * (m0 - m1) ** 2
            if between_var > best_variance:
                best_variance = between_var
                best_thresh = thresh

        # Keep the denser (higher intensity) class within the ellipsoid
        intensity_mask = volume_region >= best_thresh
        refined = ellipsoid_mask & intensity_mask

        # If refinement removed too much (< 20% of ellipsoid), 
        # the threshold was probably wrong — fall back to ellipsoid
        if refined.sum() < 0.20 * ellipsoid_mask.sum():
            return ellipsoid_mask

        return refined

    elif method == "adaptive":
        # Keep voxels above the 30th percentile within the ellipsoid
        p30 = np.percentile(values, 30)
        intensity_mask = volume_region >= p30
        refined = ellipsoid_mask & intensity_mask

        if refined.sum() < 0.25 * ellipsoid_mask.sum():
            return ellipsoid_mask
        return refined

    return ellipsoid_mask


def _morphological_cleanup(mask_3d, closing_radius=1, opening_radius=1):
    """
    Apply morphological closing then opening to smooth mask boundaries.
    """
    if not mask_3d.any():
        return mask_3d

    struct = ndimage.generate_binary_structure(3, 1)

    if closing_radius > 0:
        mask_3d = ndimage.binary_closing(mask_3d, structure=struct, iterations=closing_radius)
    if opening_radius > 0:
        mask_3d = ndimage.binary_opening(mask_3d, structure=struct, iterations=opening_radius)

    return mask_3d.astype(bool)


def boxes_to_pseudo_panoptic(
    boxes,                          # Tensor [N, 6] (z1,y1,x1,z2,y2,x2)
    image_shape,                    # (Z, Y, X)
    scores=None,                    # Tensor [N]
    score_thresh=0.3,
    overlap_strategy="highest_score",   # highest_score | first_wins | last_wins
    mask_shape="ellipsoid",             # "box" (original) | "ellipsoid" | "refined"
    volume=None,                        # Tensor/ndarray (Z,Y,X) CT values — needed for "refined"
    ellipsoid_margin=0.05,              # shrink ellipsoid 5% from box walls
    morphological_cleanup=True,         # smooth edges
):
    """
    Convert detection bounding boxes to pseudo-panoptic segmentation masks.
    
    Improvements over original box-filling:
        - "ellipsoid": inscribes an ellipsoid within each box (better Dice vs GT spheres)
        - "refined": ellipsoid + CT intensity thresholding (best quality, needs volume)
        - morphological cleanup smooths jagged boundaries
    
    Args:
        boxes:              (N, 6) tensor of bounding boxes in voxel coords
        image_shape:        (Z, Y, X) volume dimensions
        scores:             (N,) confidence scores per detection
        score_thresh:       minimum confidence to include
        overlap_strategy:   how to handle overlapping detections
        mask_shape:         "box" (legacy), "ellipsoid", or "refined" (best)
        volume:             3D CT volume for intensity-guided refinement (optional)
        ellipsoid_margin:   how much to shrink ellipsoid from box edges (0.0–0.3)
        morphological_cleanup: apply morphological smoothing
    
    Returns:
        semantic_mask:      (Z,Y,X) uint8 tensor — 1 where any nodule, 0 elsewhere
        instance_mask:      (Z,Y,X) int32 tensor — instance ID per voxel
        instance_scores:    tensor of scores per instance (or None)
    """
    device = boxes.device
    Z, Y, X = image_shape

    semantic_mask = torch.zeros((Z, Y, X), dtype=torch.uint8, device=device)
    instance_mask = torch.zeros((Z, Y, X), dtype=torch.int32, device=device)
    instance_scores = []

    # Sort boxes if needed
    if overlap_strategy == "highest_score" and scores is not None:
        order = torch.argsort(scores, descending=True)
        boxes = boxes[order]
        scores = scores[order]

    # Get numpy volume if doing refined masks
    vol_np = None
    if mask_shape == "refined" and volume is not None:
        if isinstance(volume, torch.Tensor):
            vol_np = volume.detach().cpu().numpy()
        else:
            vol_np = np.asarray(volume)
    elif mask_shape == "refined" and volume is None:
        # Fall back to ellipsoid if no volume provided
        mask_shape = "ellipsoid"

    instance_id = 1

    for i, box in enumerate(boxes):
        if scores is not None and scores[i] < score_thresh:
            continue

        z1, y1, x1, z2, y2, x2 = box.int()

        # Clamp to image bounds
        z1, y1, x1 = max(0, z1.item()), max(0, y1.item()), max(0, x1.item())
        z2, y2, x2 = min(Z, z2.item()), min(Y, y2.item()), min(X, x2.item())

        # Skip invalid boxes
        if z2 <= z1 or y2 <= y1 or x2 <= x1:
            continue

        dz, dy, dx = z2 - z1, y2 - y1, x2 - x1

        # ── Generate mask for this detection ──
        if mask_shape == "box":
            # Original behavior: fill entire box
            local_mask = np.ones((dz, dy, dx), dtype=bool)

        elif mask_shape in ("ellipsoid", "refined"):
            # Generate ellipsoid inscribed in the box
            local_mask = _generate_ellipsoid_mask((dz, dy, dx), margin=ellipsoid_margin)

            # Fallback: if ellipsoid is empty (very tiny box), fill the whole box
            if not local_mask.any():
                local_mask = np.ones((dz, dy, dx), dtype=bool)

            # Optionally refine with CT intensity
            if mask_shape == "refined" and vol_np is not None:
                vol_region = vol_np[z1:z2, y1:y2, x1:x2]
                local_mask = _intensity_refine(vol_region, local_mask, method="otsu")

            # Morphological cleanup
            if morphological_cleanup and min(dz, dy, dx) >= 5:
                local_mask = _morphological_cleanup(local_mask, closing_radius=1, opening_radius=1)

        else:
            local_mask = np.ones((dz, dy, dx), dtype=bool)

        # Convert to torch tensor on same device
        local_tensor = torch.from_numpy(local_mask.astype(np.uint8)).to(device)

        # ── Apply to global masks ──
        # Semantic mask: OR operation (any nodule voxel = 1)
        semantic_mask[z1:z2, y1:y2, x1:x2] = torch.max(
            semantic_mask[z1:z2, y1:y2, x1:x2],
            local_tensor
        )

        # Instance mask: handle overlap
        if overlap_strategy in ("highest_score", "first_wins"):
            region = instance_mask[z1:z2, y1:y2, x1:x2]
            # Only write where mask is active AND no prior instance
            write_mask = (local_tensor > 0) & (region == 0)
            instance_mask[z1:z2, y1:y2, x1:x2] = torch.where(
                write_mask,
                torch.tensor(instance_id, device=device, dtype=torch.int32),
                region,
            )
        else:  # last_wins
            write_mask = local_tensor > 0
            region = instance_mask[z1:z2, y1:y2, x1:x2]
            instance_mask[z1:z2, y1:y2, x1:x2] = torch.where(
                write_mask,
                torch.tensor(instance_id, device=device, dtype=torch.int32),
                region,
            )

        if scores is not None:
            instance_scores.append(scores[i].item())

        instance_id += 1

    if len(instance_scores) > 0:
        instance_scores = torch.tensor(instance_scores, device=device)
    else:
        instance_scores = None

    return semantic_mask, instance_mask, instance_scores