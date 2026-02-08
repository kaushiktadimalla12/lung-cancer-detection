import torch
from monai.apps.detection.networks.retinanet_network import RetinaNet, resnet_fpn_feature_extractor
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.networks.nets.resnet import resnet50
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "model.pt"

# Load checkpoint to inspect
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
if isinstance(checkpoint, dict):
    checkpoint = checkpoint.get("model", checkpoint)

print("Checkpoint keys (first 20):")
for i, key in enumerate(list(checkpoint.keys())[:20]):
    print(f"  {key}: {checkpoint[key].shape if hasattr(checkpoint[key], 'shape') else 'N/A'}")

# Try to infer config from checkpoint
cls_weight = checkpoint.get('classification_head.cls_logits.weight')
bbox_weight = checkpoint.get('regression_head.bbox_reg.weight')

if cls_weight is not None:
    print(f"\nClassification head output channels: {cls_weight.shape[0]}")
    print(f"This suggests num_anchors = {cls_weight.shape[0]}")

if bbox_weight is not None:
    print(f"BBox regression head output channels: {bbox_weight.shape[0]}")
    coords_per_box = 6  # 3D bounding box
    print(f"This suggests num_anchors = {bbox_weight.shape[0] // coords_per_box}")

# Count FPN levels
fpn_keys = [k for k in checkpoint.keys() if 'fpn' in k and 'inner_blocks' in k]
num_fpn_levels = len(set([k.split('.')[2] for k in fpn_keys]))
print(f"\nNumber of FPN levels: {num_fpn_levels}")