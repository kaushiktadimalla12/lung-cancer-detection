# pseudo_panoptic.py
import torch


def boxes_to_pseudo_panoptic(
    boxes,                 # Tensor [N, 6] (z1,y1,x1,z2,y2,x2)
    image_shape,           # (Z, Y, X)
    scores=None,           # Tensor [N]
    score_thresh=0.3,
    overlap_strategy="highest_score",  # highest_score | first_wins | last_wins
):
    """
    Convert detection boxes into pseudo-panoptic masks.

    Returns:
        semantic_mask  : uint8  (Z,Y,X)  {0,1}
        instance_mask  : int32  (Z,Y,X)  {0,1..K}
        instance_scores: Tensor[K] aligned with instance IDs
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

    instance_id = 1

    for i, box in enumerate(boxes):
        if scores is not None and scores[i] < score_thresh:
            continue

        z1, y1, x1, z2, y2, x2 = box.int()

        # Clamp to image bounds
        z1, y1, x1 = max(0, z1), max(0, y1), max(0, x1)
        z2, y2, x2 = min(Z, z2), min(Y, y2), min(X, x2)

        # Skip invalid boxes
        if z2 <= z1 or y2 <= y1 or x2 <= x1:
            continue

        # Semantic mask
        semantic_mask[z1:z2, y1:y2, x1:x2] = 1

        # Instance mask handling
        if overlap_strategy in ("highest_score", "first_wins"):
            region = instance_mask[z1:z2, y1:y2, x1:x2]
            instance_mask[z1:z2, y1:y2, x1:x2] = torch.where(
                region == 0,
                torch.tensor(instance_id, device=device, dtype=torch.int32),
                region,
            )
        else:  # last_wins
            instance_mask[z1:z2, y1:y2, x1:x2] = instance_id

        # Track score for this instance
        if scores is not None:
            instance_scores.append(scores[i].item())

        instance_id += 1

    if len(instance_scores) > 0:
        instance_scores = torch.tensor(instance_scores, device=device)
    else:
        instance_scores = None

    return semantic_mask, instance_mask, instance_scores
