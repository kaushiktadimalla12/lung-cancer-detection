"""
FastAPI Backend for Lung Nodule Pseudo-Panoptic Segmentation
Production-ready version with evaluation integration
"""

# =========================
# STANDARD IMPORTS
# =========================
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from pathlib import Path
import uuid
import traceback
import time
from collections import OrderedDict

import torch
import numpy as np
import SimpleITK as sitk

from fastapi.responses import Response
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from scipy import ndimage

# =========================
# LOCAL MODULES
# =========================
from pseudo_panoptic import boxes_to_pseudo_panoptic
from panoptic_stats import (
    compute_panoptic_stats,
    get_instance_details,
)

# =========================
# EVALUATION ROUTER (PATCH)
# =========================
from evaluation_api import eval_router, register_cache_getter

# =========================
# MONAI IMPORTS
# =========================
from monai.apps.detection.networks.retinanet_network import (
    RetinaNet,
    resnet_fpn_feature_extractor,
)
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.networks.nets.resnet import resnet50
from monai.data import MetaTensor

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "model.pt"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Cache configuration
MAX_CACHE_ITEMS = 10
MAX_CACHE_AGE_SECONDS = 3600

print(f"📂 Project root: {PROJECT_ROOT}")
print(f"📂 Model path: {MODEL_PATH}")
print(f"📂 Upload dir: {UPLOAD_DIR}")

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Lung Nodule Pseudo-Panoptic Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount evaluation router
app.include_router(eval_router)

# =========================
# RESPONSE SCHEMA
# =========================
class InferenceResponse(BaseModel):
    scan_id: str
    status: str
    num_instances: int
    image_shape: List[int]
    stats: Dict
    details: List[Dict]

# =========================
# IMPROVED CACHE WITH TTL
# =========================
class CacheEntry:
    def __init__(self, data):
        self.data = data
        self.timestamp = time.time()

segmentation_cache = OrderedDict()

def add_to_cache(scan_id: str, data: dict):
    """Add item to cache with LRU eviction."""
    if len(segmentation_cache) >= MAX_CACHE_ITEMS:
        oldest_key = next(iter(segmentation_cache))
        removed = segmentation_cache.pop(oldest_key)
        print(f"🗑️ Evicted scan {oldest_key} from cache (LRU)")
    
    segmentation_cache[scan_id] = CacheEntry(data)
    print(f"💾 Cached scan {scan_id} ({len(segmentation_cache)}/{MAX_CACHE_ITEMS})")

def get_from_cache(scan_id: str) -> Optional[dict]:
    """Get item from cache, checking TTL."""
    if scan_id not in segmentation_cache:
        return None
    
    entry = segmentation_cache[scan_id]
    age = time.time() - entry.timestamp
    
    if age > MAX_CACHE_AGE_SECONDS:
        segmentation_cache.pop(scan_id)
        print(f"🗑️ Evicted scan {scan_id} from cache (expired)")
        return None
    
    return entry.data

# Register cache accessor with evaluation router (avoids cross-import issues)
register_cache_getter(get_from_cache)

# =========================
# CT RESAMPLING
# =========================
def resample_ct(img, out_spacing=(2.0, 2.0, 2.0)):
    """Resample CT to consistent spacing."""
    spacing = img.GetSpacing()
    size = img.GetSize()

    new_size = [
        int(size[i] * spacing[i] / out_spacing[i])
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetDefaultPixelValue(-1024)

    resampled = resampler.Execute(img)
    
    actual_spacing = resampled.GetSpacing()
    print(f"✅ Resampled spacing: {actual_spacing}")
    
    return resampled

# =========================
# MODEL SINGLETON
# =========================
class ModelSingleton:
    _instance = None

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load_model()
        return cls._instance

    def load_model(self):
        print("🔧 Loading detection model...")
        
        checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
        checkpoint = checkpoint.get("model", checkpoint)
        
        print("🔍 Classification head shape:")
        cls_key = 'classification_head.cls_logits.weight'
        if cls_key in checkpoint:
            print(f"   {cls_key}: {checkpoint[cls_key].shape}")
            num_anchors = checkpoint[cls_key].shape[0]
            print(f"   → num_anchors = {num_anchors}")

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

        self.detector = RetinaNetDetector(
            network=network,
            anchor_generator=anchor_generator,
            spatial_dims=3,
            num_classes=1,
            size_divisible=[16, 16, 8],
        )

        self.detector.set_box_selector_parameters(
            score_thresh=0.02,
            nms_thresh=0.3,
            detections_per_img=300,
        )

        self.detector.set_sliding_window_inferer(
            roi_size=(96, 96, 96),
            sw_batch_size=1,
            overlap=0.5,
            mode="gaussian",
        )

        print("📥 Loading weights...")
        missing, unexpected = self.detector.network.load_state_dict(checkpoint, strict=False)
        if missing:
            print(f"⚠️ Missing keys: {len(missing)}")
        if unexpected:
            print(f"⚠️ Unexpected keys: {len(unexpected)}")

        self.detector.network.to(self.device)
        self.detector.eval()

        print(f"✅ Model loaded on {self.device}")

# =========================
# STARTUP
# =========================
@app.on_event("startup")
async def startup_event():
    ModelSingleton.get_instance()

# =========================
# UPLOAD ENDPOINT
# =========================
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    scan_id: Optional[str] = Form(None)
):
    ext = Path(file.filename).suffix.lower()

    if ext == ".mhd":
        scan_id = str(uuid.uuid4())
        mhd_path = UPLOAD_DIR / f"{scan_id}.mhd"

        try:
            content = (await file.read()).decode("ascii", errors="replace")
        except UnicodeDecodeError:
            content = (await file.read()).decode("utf-8", errors="replace")
        
        lines = [
            f"ElementDataFile = {scan_id}.raw" if l.startswith("ElementDataFile") else l
            for l in content.splitlines()
        ]
        mhd_path.write_text("\n".join(lines), encoding="ascii")
        
        # Return the original filename so frontend can extract seriesuid
        original_name = Path(file.filename).stem
        return {
            "status": "success",
            "scan_id": scan_id,
            "file_type": "mhd",
            "original_filename": original_name,
        }

    elif ext == ".raw":
        if not scan_id:
            raise HTTPException(400, "scan_id required for .raw upload")
        raw_path = UPLOAD_DIR / f"{scan_id}.raw"
        raw_path.write_bytes(await file.read())
        return {"status": "success", "scan_id": scan_id, "file_type": "raw"}

    raise HTTPException(400, "Unsupported file type")

# =========================
# INFERENCE ENDPOINT (PATCHED)
# =========================
@app.post("/infer", response_model=InferenceResponse)
async def infer(
    scan_id: str = Form(...),
    score_threshold: float = Form(0.1),
    overlap_strategy: str = Form("highest_score"),
):
    model = ModelSingleton.get_instance()

    mhd = UPLOAD_DIR / f"{scan_id}.mhd"
    raw = UPLOAD_DIR / f"{scan_id}.raw"
    if not mhd.exists() or not raw.exists():
        raise HTTPException(404, "Scan files missing")

    try:
        img = sitk.ReadImage(str(mhd))
        img = sitk.DICOMOrient(img, "RAS")
        img = resample_ct(img, (2.0, 2.0, 2.0))
        
        # Store actual spacing for stats
        actual_spacing = img.GetSpacing()
        voxel_spacing = tuple(actual_spacing)[::-1]  # Convert to (z,y,x)
        print(f"📏 Voxel spacing for stats: {voxel_spacing}")

        # ── EVAL PATCH: Store metadata for evaluation ──
        resampled_origin = img.GetOrigin()         # (x,y,z)
        resampled_spacing = img.GetSpacing()       # (x,y,z)
        resampled_direction = np.array(img.GetDirection()).reshape(3, 3)

        vol_np = sitk.GetArrayFromImage(img).astype(np.float32)
        vol_np = np.clip(vol_np, -1024, 300)
        vol_np = (vol_np + 1024) / (300 + 1024)
        vol_np = vol_np[np.newaxis, ...]  # (1, D, H, W)

        vol = MetaTensor(
            torch.from_numpy(vol_np),
            meta={
                "spacing": voxel_spacing,
                "pixdim": (1.0, *voxel_spacing),
            },
        ).unsqueeze(0).to(model.device)

        print(f"✅ Input shape: {vol.shape}")

        with torch.no_grad():
            outputs = model.detector(vol)
            result = outputs[0]
            
            scores = None
            if "label_scores" in result:
                scores = result["label_scores"]
                print("✅ Using 'label_scores' key")
            elif "labels_scores" in result:
                scores = result["labels_scores"]
                print("✅ Using 'labels_scores' key")
            elif "scores" in result:
                scores = result["scores"]
                print("✅ Using 'scores' key")
            else:
                raise RuntimeError(f"No score key found. Available keys: {result.keys()}")

        boxes = result.get("boxes")

        if boxes is None or len(boxes) == 0:
            return InferenceResponse(
                scan_id=scan_id,
                status="no_detections",
                num_instances=0,
                image_shape=list(vol.shape[-3:]),
                stats={},
                details=[],
            )

        print(f"✅ Found {len(boxes)} raw detections")
        
        keep = scores >= score_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        
        print(f"✅ After threshold filtering: {len(boxes)} detections")

        if len(boxes) == 0:
            return InferenceResponse(
                scan_id=scan_id,
                status="no_detections",
                num_instances=0,
                image_shape=list(vol.shape[-3:]),
                stats={},
                details=[],
            )

        semantic_mask, instance_mask, instance_scores = boxes_to_pseudo_panoptic(
            boxes=boxes,
            scores=scores,
            image_shape=vol.shape[-3:],
            overlap_strategy=overlap_strategy,
            score_thresh=score_threshold,
            mask_shape="refined",       # ellipsoid + CT intensity refinement
            volume=vol[0, 0],           # raw CT volume for intensity-guided segmentation
        )

        stats = compute_panoptic_stats(
            semantic_mask, 
            instance_mask,
            instance_scores=instance_scores,
            voxel_spacing=voxel_spacing
        )
        
        details = get_instance_details(
            instance_mask,
            instance_scores=instance_scores,
            voxel_spacing=voxel_spacing
        )

        original_volume = vol[0, 0].cpu().numpy()
        
        # ── EVAL PATCH: Store all data needed for evaluation ──
        # CRITICAL: Convert everything to numpy/native Python types.
        # Storing torch tensors in cache is fragile — downstream code
        # (evaluation, visualization) uses numpy operations like np.unique().
        cache_data = {
            'semantic_mask': semantic_mask.cpu().numpy(),    # (D,H,W) uint8
            'instance_mask': instance_mask.cpu().numpy(),    # (D,H,W) int32
            'original_volume': original_volume,              # (D,H,W) float32
            'image_shape': list(int(s) for s in vol.shape[-3:]),  # [D,H,W] plain ints
            'voxel_spacing': list(voxel_spacing),            # [sz,sy,sx] floats
            # Evaluation metadata
            'origin': tuple(float(v) for v in resampled_origin),          # (x,y,z) mm
            'spacing_xyz': tuple(float(v) for v in resampled_spacing),    # (x,y,z) mm
            'direction': resampled_direction.tolist(),                     # 3x3 as nested list
            'pred_boxes': boxes.cpu().numpy(),               # (N,6) float32 numpy
            'pred_scores': scores.cpu().numpy(),             # (N,) float32 numpy
        }
        
        add_to_cache(scan_id, cache_data)

        return InferenceResponse(
            scan_id=scan_id,
            status="success",
            num_instances=len(details),
            image_shape=list(vol.shape[-3:]),
            stats=stats,
            details=details,
        )

    except Exception as e:
        print(f"❌ ERROR in inference:")
        traceback.print_exc()
        raise HTTPException(500, str(e))

# =========================
# CLEANUP
# =========================
@app.delete("/cleanup/{scan_id}")
async def cleanup_scan(scan_id: str):
    mhd = UPLOAD_DIR / f"{scan_id}.mhd"
    raw = UPLOAD_DIR / f"{scan_id}.raw"
    
    deleted = []
    if mhd.exists():
        mhd.unlink()
        deleted.append("mhd")
    if raw.exists():
        raw.unlink()
        deleted.append("raw")
    
    if scan_id in segmentation_cache:
        segmentation_cache.pop(scan_id)
        deleted.append("cache")
    
    return {"status": "success", "deleted": deleted}

# Helper: safely convert cache values to numpy (handles both torch tensors and numpy arrays)
def _ensure_numpy(val):
    """Convert torch tensor or numpy array to numpy. No-op if already numpy."""
    if hasattr(val, 'cpu'):       # torch tensor
        return val.cpu().numpy()
    if hasattr(val, 'numpy'):     # torch tensor (another path)
        return val.numpy()
    return np.asarray(val)        # already numpy or list


# =========================
# SLICE VISUALIZATION
# =========================
@app.get("/visualize/{scan_id}/slice/{slice_idx}")
async def visualize_slice(
    scan_id: str,
    slice_idx: int,
    view: str = "axial",
    highlight_instance: Optional[int] = None
):
    cache = get_from_cache(scan_id)
    if cache is None:
        raise HTTPException(404, "Segmentation not found or expired")

    volume = _ensure_numpy(cache['original_volume'])
    semantic = _ensure_numpy(cache['semantic_mask'])
    instance = _ensure_numpy(cache['instance_mask'])

    if view == "axial":
        max_idx = volume.shape[0] - 1
        if slice_idx < 0 or slice_idx > max_idx:
            raise HTTPException(400, f"Slice {slice_idx} out of range [0, {max_idx}]")
        img = volume[slice_idx, :, :].T
        sem = semantic[slice_idx, :, :].T
        inst = instance[slice_idx, :, :].T
    elif view == "sagittal":
        max_idx = volume.shape[2] - 1
        if slice_idx < 0 or slice_idx > max_idx:
            raise HTTPException(400, f"Slice {slice_idx} out of range [0, {max_idx}]")
        img = volume[:, :, slice_idx].T
        sem = semantic[:, :, slice_idx].T
        inst = instance[:, :, slice_idx].T
    elif view == "coronal":
        max_idx = volume.shape[1] - 1
        if slice_idx < 0 or slice_idx > max_idx:
            raise HTTPException(400, f"Slice {slice_idx} out of range [0, {max_idx}]")
        img = volume[:, slice_idx, :].T
        sem = semantic[:, slice_idx, :].T
        inst = instance[:, slice_idx, :].T
    else:
        raise HTTPException(400, "Invalid view. Use 'axial', 'sagittal', or 'coronal'")
    
    fig = plt.figure(figsize=(18, 6), facecolor='#0f172a')
    gs = fig.add_gridspec(1, 3, hspace=0.05, wspace=0.05)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img, cmap='gray', origin='lower')
    ax1.set_title('Original CT', color='white', fontsize=14)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img, cmap='gray', origin='lower', alpha=0.85)
    
    if sem.max() > 0:
        sem_smooth = ndimage.gaussian_filter(sem.astype(float), sigma=0.5)
        sem_mask = np.ma.masked_where(sem_smooth < 0.1, sem_smooth)
        ax2.imshow(sem_mask, cmap='hot', alpha=0.6, origin='lower')
    
    ax2.set_title('Semantic Segmentation', color='white', fontsize=14)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img, cmap='gray', origin='lower', alpha=0.85)
    
    if inst.max() > 0:
        unique_instances = np.unique(inst)
        unique_instances = unique_instances[unique_instances > 0]
        
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
        for idx, inst_id in enumerate(unique_instances):
            inst_mask_binary = (inst == inst_id)
            inst_smooth = ndimage.gaussian_filter(inst_mask_binary.astype(float), sigma=0.5)
            inst_mask = np.ma.masked_where(inst_smooth < 0.1, inst_smooth)
            
            if highlight_instance == inst_id:
                ax3.imshow(inst_mask, cmap='spring', alpha=0.8, origin='lower')
            else:
                inst_colored = np.zeros((*inst_mask.shape, 4))
                color = colors[idx % 20]
                inst_colored[~inst_mask.mask] = [*color[:3], 0.5]
                ax3.imshow(inst_colored, origin='lower')
            
            y_coords, x_coords = np.where(inst_mask_binary)
            if len(y_coords) > 0:
                cy, cx = int(np.mean(y_coords)), int(np.mean(x_coords))
                bbox = dict(boxstyle='round,pad=0.5', 
                           facecolor='yellow' if highlight_instance == inst_id else 'cyan',
                           alpha=0.8, edgecolor='white', linewidth=2)
                ax3.text(cx, cy, f'#{inst_id}', color='black', fontsize=10,
                        fontweight='bold', ha='center', va='center', bbox=bbox)
    
    ax3.set_title('Instance Segmentation', color='white', fontsize=14)
    ax3.axis('off')
    
    fig.suptitle(f'{view.capitalize()} Slice {slice_idx}', color='white', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)
    buf.seek(0)
    
    return Response(content=buf.getvalue(), media_type="image/png")

# =========================
# MONTAGE VIEW
# =========================
@app.get("/visualize/{scan_id}/montage")
async def visualize_montage(
    scan_id: str,
    view: str = "axial",
    num_slices: int = 12
):
    cache = get_from_cache(scan_id)
    if cache is None:
        raise HTTPException(404, "Segmentation not found or expired")
    
    volume = _ensure_numpy(cache['original_volume'])
    instance = _ensure_numpy(cache['instance_mask'])
    
    if view == "axial":
        total_slices = volume.shape[0]
    elif view == "sagittal":
        total_slices = volume.shape[2]
    else:
        total_slices = volume.shape[1]
    
    slice_indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)
    
    rows, cols = 3, 4
    fig = plt.figure(figsize=(4*cols, 4*rows), facecolor='#0f172a')
    gs = fig.add_gridspec(rows, cols, hspace=0.15, wspace=0.05)
    
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    for idx, slice_idx in enumerate(slice_indices):
        row, col = idx // cols, idx % cols
        ax = fig.add_subplot(gs[row, col])
        
        if view == "axial":
            img = volume[slice_idx, :, :].T
            inst = instance[slice_idx, :, :].T
        elif view == "sagittal":
            img = volume[:, :, slice_idx].T
            inst = instance[:, :, slice_idx].T
        else:
            img = volume[:, slice_idx, :].T
            inst = instance[:, slice_idx, :].T
        
        ax.imshow(img, cmap='gray', origin='lower', alpha=0.9)
        
        unique_instances = np.unique(inst)
        unique_instances = unique_instances[unique_instances > 0]
        
        for inst_idx, inst_id in enumerate(unique_instances):
            mask = (inst == inst_id)
            if not mask.any():
                continue
            
            colored = np.zeros((*mask.shape, 4))
            color = colors[inst_idx % 20]
            colored[mask] = [*color[:3], 0.5]
            
            ax.imshow(colored, origin='lower')
        
        ax.set_title(f'Slice {slice_idx}', color='white', fontsize=11, fontweight='bold')
        ax.axis('off')
    
    fig.suptitle(f'{view.capitalize()} View - Pseudo-Panoptic Montage', 
                 color='white', fontsize=18, fontweight='bold')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)
    buf.seek(0)
    
    return Response(content=buf.getvalue(), media_type="image/png")

# =========================
# 3D VISUALIZATION
# =========================
@app.get("/visualize/{scan_id}/3d")
async def visualize_3d(
    scan_id: str,
    instance_id: Optional[int] = None
):
    cache = get_from_cache(scan_id)
    if cache is None:
        raise HTTPException(404, "Segmentation not found or expired")
    
    volume = _ensure_numpy(cache['original_volume'])
    instance = _ensure_numpy(cache['instance_mask'])
    
    fig = plt.figure(figsize=(20, 7), facecolor='#0f172a')
    gs = fig.add_gridspec(1, 3, hspace=0.05, wspace=0.1)
    
    if instance_id is not None:
        mask = (instance == instance_id)
        if not mask.any():
            raise HTTPException(404, f"Instance {instance_id} not found")
        title = f'3D MIP - Instance #{instance_id}'
        
        ax1 = fig.add_subplot(gs[0, 0])
        bg = volume.max(axis=0)
        ax1.imshow(bg, cmap='gray', origin='lower', alpha=0.7)
        overlay = mask.astype(float).max(axis=0)
        overlay_masked = np.ma.masked_where(overlay < 0.1, overlay)
        im1 = ax1.imshow(overlay_masked, cmap='spring', alpha=0.8, origin='lower', vmin=0, vmax=1)
        ax1.set_title('Axial MIP', color='white', fontsize=14, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        ax2 = fig.add_subplot(gs[0, 1])
        bg = volume.max(axis=1)
        ax2.imshow(bg, cmap='gray', origin='lower', alpha=0.7)
        overlay = mask.astype(float).max(axis=1)
        overlay_masked = np.ma.masked_where(overlay < 0.1, overlay)
        im2 = ax2.imshow(overlay_masked, cmap='spring', alpha=0.8, origin='lower', vmin=0, vmax=1)
        ax2.set_title('Coronal MIP', color='white', fontsize=14, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        ax3 = fig.add_subplot(gs[0, 2])
        bg = volume.max(axis=2)
        ax3.imshow(bg, cmap='gray', origin='lower', alpha=0.7)
        overlay = mask.astype(float).max(axis=2)
        overlay_masked = np.ma.masked_where(overlay < 0.1, overlay)
        im3 = ax3.imshow(overlay_masked, cmap='spring', alpha=0.8, origin='lower', vmin=0, vmax=1)
        ax3.set_title('Sagittal MIP', color='white', fontsize=14, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
    else:
        from matplotlib.colors import LinearSegmentedColormap
        unique_instances = np.unique(instance)
        unique_instances = unique_instances[unique_instances > 0]
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
        title = f'3D MIP - All Instances (Pseudo-Panoptic)'
        
        for ax_idx, (axis, ax_title) in enumerate([(0, 'Axial'), (1, 'Coronal'), (2, 'Sagittal')]):
            ax = fig.add_subplot(gs[0, ax_idx])
            bg = volume.max(axis=axis)
            ax.imshow(bg, cmap='gray', origin='lower', alpha=0.7)
            
            for idx, inst_id in enumerate(unique_instances):
                mask = (instance == inst_id)
                overlay = mask.astype(float).max(axis=axis)
                overlay_masked = np.ma.masked_where(overlay < 0.1, overlay)
                color = colors[idx % 20]
                cmap = LinearSegmentedColormap.from_list('instance', [(0,0,0,0), color])
                ax.imshow(overlay_masked, cmap=cmap, alpha=0.7, origin='lower', vmin=0, vmax=1)
            
            ax.set_title(f'{ax_title} MIP', color='white', fontsize=14, fontweight='bold')
            ax.axis('off')
    
    fig.suptitle(title, color='white', fontsize=18, fontweight='bold')
    
    if instance_id:
        fig.text(0.5, 0.02, f'Maximum Intensity Projection | Instance #{instance_id}', 
                ha='center', color='#94a3b8', fontsize=12)
    else:
        num_inst = len(unique_instances) if 'unique_instances' in dir() else 0
        fig.text(0.5, 0.02, 
                f'Pseudo-Panoptic Visualization | {num_inst} instances with unique colors', 
                ha='center', color='#94a3b8', fontsize=12)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)
    buf.seek(0)
    
    return Response(content=buf.getvalue(), media_type="image/png")

# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
async def health():
    model = ModelSingleton.get_instance()
    return {
        "status": "ok",
        "device": str(model.device),
        "cache_size": len(segmentation_cache),
        "max_cache": MAX_CACHE_ITEMS
    }