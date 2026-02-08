"""
FastAPI Backend for Lung Nodule Pseudo-Panoptic Segmentation
Production-ready version with all critical fixes
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
from backend.pseudo_panoptic import boxes_to_pseudo_panoptic
from backend.panoptic_stats import (
    compute_panoptic_stats,
    get_instance_details,
)

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
MAX_CACHE_ITEMS = 10  # Maximum number of scans to keep in memory
MAX_CACHE_AGE_SECONDS = 3600  # 1 hour

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

# OrderedDict to maintain insertion order for LRU eviction
segmentation_cache = OrderedDict()

def add_to_cache(scan_id: str, data: dict):
    """Add item to cache with LRU eviction."""
    # Remove oldest if cache is full
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
    
    # Verify spacing after resampling
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
            nms_thresh=0.22,
            detections_per_img=300,
        )

        self.detector.set_sliding_window_inferer(
            roi_size=(96, 96, 96),
            sw_batch_size=1,
            overlap=0.25,
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
# UPLOAD ENDPOINT (FIXED)
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

        # FIX #1: Proper .mhd handling
        try:
            # .mhd files are ASCII text with specific format
            content = (await file.read()).decode("ascii", errors="replace")
        except UnicodeDecodeError:
            # Fallback to UTF-8 with replacement
            content = (await file.read()).decode("utf-8", errors="replace")
        
        # Update ElementDataFile reference
        lines = [
            f"ElementDataFile = {scan_id}.raw" if l.startswith("ElementDataFile") else l
            for l in content.splitlines()
        ]
        mhd_path.write_text("\n".join(lines), encoding="ascii")
        
        return {"status": "success", "scan_id": scan_id, "file_type": "mhd"}

    elif ext == ".raw":
        if not scan_id:
            raise HTTPException(400, "scan_id required for .raw upload")
        raw_path = UPLOAD_DIR / f"{scan_id}.raw"
        raw_path.write_bytes(await file.read())
        return {"status": "success", "scan_id": scan_id, "file_type": "raw"}

    raise HTTPException(400, "Unsupported file type")

# =========================
# INFERENCE ENDPOINT (FIXED)
# =========================
@app.post("/infer", response_model=InferenceResponse)
async def infer(
    scan_id: str = Form(...),
    score_threshold: float = Form(0.3),
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
        
        # Store actual spacing for stats (should be 2.0, 2.0, 2.0 after resampling)
        actual_spacing = img.GetSpacing()
        voxel_spacing = tuple(actual_spacing)[::-1]  # Convert to (z,y,x)
        print(f"📏 Voxel spacing for stats: {voxel_spacing}")

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
        ).unsqueeze(0).to(model.device)  # (1, 1, D, H, W)

        print(f"✅ Input shape: {vol.shape}")

        with torch.no_grad():
            outputs = model.detector(vol)
            result = outputs[0]
            
            # FIX #3: Robust score key handling
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

        # FIX #2: Proper volume extraction (remove both batch and channel dims)
        # vol shape is (1, 1, D, H, W), we want (D, H, W)
        original_volume = vol[0, 0].cpu()  # Correct way to get (D, H, W)
        
        cache_data = {
            'semantic_mask': semantic_mask.cpu(),
            'instance_mask': instance_mask.cpu(),
            'original_volume': original_volume,
            'image_shape': vol.shape[-3:],
            'voxel_spacing': voxel_spacing,
        }
        
        # FIX #4: Use cache with LRU eviction
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

# =========================
# SLICE VISUALIZATION (FIXED)
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

    volume = cache['original_volume'].numpy()
    semantic = cache['semantic_mask'].numpy()
    instance = cache['instance_mask'].numpy()

    
    # FIX #6: Proper slice validation
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
    
    # Create figure
    fig = plt.figure(figsize=(18, 6), facecolor='#0f172a')
    gs = fig.add_gridspec(1, 3, hspace=0.05, wspace=0.05)
    
    # 1. Original CT
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img, cmap='gray', origin='lower')
    ax1.set_title('Original CT', color='white', fontsize=14)
    ax1.axis('off')
    
    # 2. Semantic (all nodules)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img, cmap='gray', origin='lower', alpha=0.85)
    
    if sem.max() > 0:
        sem_smooth = ndimage.gaussian_filter(sem.astype(float), sigma=0.5)
        sem_mask = np.ma.masked_where(sem_smooth < 0.1, sem_smooth)
        ax2.imshow(sem_mask, cmap='hot', alpha=0.6, origin='lower')
    
    ax2.set_title('Semantic Segmentation', color='white', fontsize=14)
    ax2.axis('off')
    
    # 3. Instance (individual nodules with actual masks)
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
# MONTAGE VIEW (FIXED - TRUE PSEUDO-PANOPTIC)
# =========================
@app.get("/visualize/{scan_id}/montage")
async def visualize_montage(
    scan_id: str,
    view: str = "axial",
    num_slices: int = 12
):
    """
    True pseudo-panoptic montage: each instance gets unique color.
    This is NOT semantic segmentation - instances are preserved.
    """
    cache = get_from_cache(scan_id)
    if cache is None:
        raise HTTPException(404, "Segmentation not found or expired")
    
    volume = cache['original_volume'].numpy()
    instance = cache['instance_mask'].numpy()
    
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
    
    # Generate consistent colors for all instances
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
        
        # TRUE PSEUDO-PANOPTIC: Each instance gets unique color
        unique_instances = np.unique(inst)
        unique_instances = unique_instances[unique_instances > 0]
        
        for inst_idx, inst_id in enumerate(unique_instances):
            mask = (inst == inst_id)
            if not mask.any():
                continue
            
            # Create colored overlay for this instance
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
# 3D VISUALIZATION (FIXED - TRUE PSEUDO-PANOPTIC)
# =========================
@app.get("/visualize/{scan_id}/3d")
async def visualize_3d(
    scan_id: str,
    instance_id: Optional[int] = None
):
    """
    True pseudo-panoptic 3D MIP: preserves instance identities.
    When viewing all instances, each gets a unique color.
    """
    cache = get_from_cache(scan_id)
    if cache is None:
        raise HTTPException(404, "Segmentation not found or expired")
    
    volume = cache['original_volume'].numpy()
    instance = cache['instance_mask'].numpy()
    
    fig = plt.figure(figsize=(20, 7), facecolor='#0f172a')
    gs = fig.add_gridspec(1, 3, hspace=0.05, wspace=0.1)
    
    if instance_id is not None:
        # Single instance view
        mask = (instance == instance_id)
        if not mask.any():
            raise HTTPException(404, f"Instance {instance_id} not found")
        title = f'3D MIP - Instance #{instance_id}'
        
        # Axial MIP
        ax1 = fig.add_subplot(gs[0, 0])
        bg = volume.max(axis=0)
        ax1.imshow(bg, cmap='gray', origin='lower', alpha=0.7)
        overlay = mask.astype(float).max(axis=0)
        overlay_masked = np.ma.masked_where(overlay < 0.1, overlay)
        im1 = ax1.imshow(overlay_masked, cmap='spring', alpha=0.8, origin='lower', vmin=0, vmax=1)
        ax1.set_title('Axial MIP', color='white', fontsize=14, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Coronal MIP
        ax2 = fig.add_subplot(gs[0, 1])
        bg = volume.max(axis=1)
        ax2.imshow(bg, cmap='gray', origin='lower', alpha=0.7)
        overlay = mask.astype(float).max(axis=1)
        overlay_masked = np.ma.masked_where(overlay < 0.1, overlay)
        im2 = ax2.imshow(overlay_masked, cmap='spring', alpha=0.8, origin='lower', vmin=0, vmax=1)
        ax2.set_title('Coronal MIP', color='white', fontsize=14, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Sagittal MIP
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
        # TRUE PSEUDO-PANOPTIC: All instances with unique colors
        unique_instances = np.unique(instance)
        unique_instances = unique_instances[unique_instances > 0]
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
        title = f'3D MIP - All Instances (Pseudo-Panoptic)'
        
        # Axial MIP
        ax1 = fig.add_subplot(gs[0, 0])
        bg = volume.max(axis=0)
        ax1.imshow(bg, cmap='gray', origin='lower', alpha=0.7)
        
        for idx, inst_id in enumerate(unique_instances):
            mask = (instance == inst_id)
            overlay = mask.astype(float).max(axis=0)
            overlay_masked = np.ma.masked_where(overlay < 0.1, overlay)
            
            # Create single-color colormap for this instance
            color = colors[idx % 20]
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list('instance', [(0,0,0,0), color])
            
            ax1.imshow(overlay_masked, cmap=cmap, alpha=0.7, origin='lower', vmin=0, vmax=1)
        
        ax1.set_title('Axial MIP', color='white', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Coronal MIP
        ax2 = fig.add_subplot(gs[0, 1])
        bg = volume.max(axis=1)
        ax2.imshow(bg, cmap='gray', origin='lower', alpha=0.7)
        
        for idx, inst_id in enumerate(unique_instances):
            mask = (instance == inst_id)
            overlay = mask.astype(float).max(axis=1)
            overlay_masked = np.ma.masked_where(overlay < 0.1, overlay)
            
            color = colors[idx % 20]
            cmap = LinearSegmentedColormap.from_list('instance', [(0,0,0,0), color])
            
            ax2.imshow(overlay_masked, cmap=cmap, alpha=0.7, origin='lower', vmin=0, vmax=1)
        
        ax2.set_title('Coronal MIP', color='white', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Sagittal MIP
        ax3 = fig.add_subplot(gs[0, 2])
        bg = volume.max(axis=2)
        ax3.imshow(bg, cmap='gray', origin='lower', alpha=0.7)
        
        for idx, inst_id in enumerate(unique_instances):
            mask = (instance == inst_id)
            overlay = mask.astype(float).max(axis=2)
            overlay_masked = np.ma.masked_where(overlay < 0.1, overlay)
            
            color = colors[idx % 20]
            cmap = LinearSegmentedColormap.from_list('instance', [(0,0,0,0), color])
            
            ax3.imshow(overlay_masked, cmap=cmap, alpha=0.7, origin='lower', vmin=0, vmax=1)
        
        ax3.set_title('Sagittal MIP', color='white', fontsize=14, fontweight='bold')
        ax3.axis('off')
    
    fig.suptitle(title, color='white', fontsize=18, fontweight='bold')
    
    # Add info text
    if instance_id:
        fig.text(0.5, 0.02, f'Maximum Intensity Projection | Instance #{instance_id}', 
                ha='center', color='#94a3b8', fontsize=12)
    else:
        num_instances = len(unique_instances)
        fig.text(0.5, 0.02, 
                f'Pseudo-Panoptic Visualization | {num_instances} instances with unique colors', 
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