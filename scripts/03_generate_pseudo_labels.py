import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.cuda.amp import autocast

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.default_config import (
    SPLITS_FILE, SPARSE_CACHE_DIR, PSEUDO_CACHE_DIR, CHECKPOINTS_DIR,
    PATCH_SIZE, NUM_CLASSES
)
from src.models.resnet_unet3d import ResEncoderUNet3D
from src.dataset.geodesic_tracker import reconstruct_geodesic_dense_mask
from src.postprocessing.anatomical_filter import filter_bilateral_ian

@torch.no_grad()
def sliding_window_predict(model, volume_np, device="cuda", patch_size=(64, 128, 128), overlap=0.5):
    model.eval()
    D, H, W = volume_np.shape
    pD, pH, pW = patch_size
    
    output_probs = np.zeros((NUM_CLASSES, D, H, W), dtype=np.float32)
    count_map = np.zeros((1, D, H, W), dtype=np.float32)
    
    step_d = max(1, int(pD * (1.0 - overlap)))
    step_h = max(1, int(pH * (1.0 - overlap)))
    step_w = max(1, int(pW * (1.0 - overlap)))
    
    d_starts = list(range(0, max(1, D - pD + 1), step_d))
    if d_starts[-1] != max(0, D - pD): d_starts.append(max(0, D - pD))
    h_starts = list(range(0, max(1, H - pH + 1), step_h))
    if h_starts[-1] != max(0, H - pH): h_starts.append(max(0, H - pH))
    w_starts = list(range(0, max(1, W - pW + 1), step_w))
    if w_starts[-1] != max(0, W - pW): w_starts.append(max(0, W - pW))
    
    for z in d_starts:
        for y in h_starts:
            for x in w_starts:
                patch_np = volume_np[z:z+pD, y:y+pH, x:x+pW]
                patch_tensor = torch.from_numpy(patch_np).unsqueeze(0).unsqueeze(0).to(device)
                with torch.amp.autocast('cuda'):
                    patch_out = model(patch_tensor)
                    if isinstance(patch_out, list):
                        patch_out = patch_out[0]
                    probs = torch.softmax(patch_out, dim=1)[0].cpu().numpy()
                output_probs[:, z:z+pD, y:y+pH, x:x+pW] += probs
                count_map[:, z:z+pD, y:y+pH, x:x+pW] += 1.0
                
    final_probs = output_probs / np.maximum(count_map, 1e-5)
    torch.cuda.empty_cache()
    return final_probs

def generate_pseudo_labels(teacher_checkpoint_path: str | Path = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("🔍 PHASE 3: GEODESIC PSEUDO-LABEL GENERATION ON 290 SPARSE SCANS")
    print("=" * 70)
    
    # Load model
    model = ResEncoderUNet3D(in_channels=1, num_classes=NUM_CLASSES, feature_dims=[24, 48, 96, 192, 256], deep_supervision=False)
    if teacher_checkpoint_path is None:
        teacher_checkpoint_path = CHECKPOINTS_DIR / "teacher_resenc_fold0" / "best_model.pth"
        
    print(f"Loading Teacher Checkpoint: {teacher_checkpoint_path}")
    checkpoint = torch.load(teacher_checkpoint_path, map_location=device)
    # Filter state dict if trained with deep supervision
    state_dict = {k: v for k, v in checkpoint["model_state_dict"].items() if not k.startswith("ds_head")}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    sparse_pids = splits.get("synthetic", [])
    
    valid_pseudo_count = 0
    generated_pids = []
    
    for pid in tqdm(sparse_pids, desc="Generating Pseudo-Labels"):
        case_dir = SPARSE_CACHE_DIR / pid
        if not case_dir.exists():
            continue
            
        data = np.load(case_dir / "data.npy")
        gt_sparse = np.load(case_dir / "gt_sparse.npy") if (case_dir / "gt_sparse.npy").exists() else None
        
        # 1. Predict full volume probabilities
        probs = sliding_window_predict(model, data, device=device, patch_size=PATCH_SIZE)
        fg_prob = 1.0 - probs[0]  # Foreground probability (sum of Left and Right)
        
        # 2. Geodesic Shortest Path Tracking anchored on gt_sparse
        if gt_sparse is not None and gt_sparse.sum() > 0:
            pseudo_mask = reconstruct_geodesic_dense_mask(fg_prob, gt_sparse)
        else:
            # Fallback to direct argmax + bilateral filtering
            raw_classes = np.argmax(probs, axis=0)
            pseudo_mask = filter_bilateral_ian(raw_classes)
            
        # 3. Quality & Topology Verification
        r_sum = (pseudo_mask == 1).sum()
        l_sum = (pseudo_mask == 2).sum()
        
        # A valid IAN canal typically contains between 6,000 and 35,000 voxels per side
        if r_sum > 3000 and l_sum > 3000:
            out_dir = PSEUDO_CACHE_DIR / pid
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "data.npy", data)
            np.save(out_dir / "label_bilateral.npy", pseudo_mask)
            valid_pseudo_count += 1
            generated_pids.append(pid)
            
    print(f"\n✅ Successfully generated {valid_pseudo_count} high-quality bilateral pseudo-labels out of {len(sparse_pids)} sparse scans!")
    with open(PSEUDO_CACHE_DIR / "valid_pseudo_pids.json", "w") as f:
        json.dump(generated_pids, f, indent=2)

if __name__ == "__main__":
    generate_pseudo_labels()
