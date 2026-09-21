import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Ensure root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.default_config import RAW_DATA_DIR, SPLITS_FILE, DENSE_CACHE_DIR, SPARSE_CACHE_DIR, INTENSITY_CLIPPING

def preprocess_and_cache_dataset(max_cases: int = None):
    print("=" * 70)
    print("🦷 PHASE 1: PREPROCESSING & BILATERAL CACHING ENGINE")
    print("=" * 70)
    
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
        
    all_dense_cases = splits.get("train", []) + splits.get("val", []) + splits.get("test", [])
    all_sparse_cases = splits.get("synthetic", [])
    
    print(f"Total Densely Labeled Scans: {len(all_dense_cases)}")
    print(f"Total Sparsely Labeled Scans: {len(all_sparse_cases)}")
    
    # Process Dense Scans
    print("\n📦 Preprocessing Dense Ground Truth Scans...")
    dense_targets = all_dense_cases[:max_cases] if max_cases else all_dense_cases
    
    for pid in tqdm(dense_targets, desc="Caching Dense"):
        raw_pid_dir = RAW_DATA_DIR / pid
        out_pid_dir = DENSE_CACHE_DIR / pid
        out_pid_dir.mkdir(parents=True, exist_ok=True)
        
        # Load raw data
        raw_data = np.load(raw_pid_dir / "data.npy").astype(np.float32)
        
        # 1. Intensity Normalization
        clipped = np.clip(raw_data, INTENSITY_CLIPPING[0], INTENSITY_CLIPPING[1])
        # Z-score based on tissue voxels (> -400 HU)
        tissue_mask = clipped > -400.0
        if tissue_mask.sum() > 1000:
            mean_val = clipped[tissue_mask].mean()
            std_val = clipped[tissue_mask].std() + 1e-5
        else:
            mean_val = clipped.mean()
            std_val = clipped.std() + 1e-5
        normalized = (clipped - mean_val) / std_val
        
        np.save(out_pid_dir / "data.npy", normalized.astype(np.float32))
        
        # 2. Bilateral Mask Generation (Left = 2, Right = 1)
        if (raw_pid_dir / "gt_alpha.npy").exists():
            gt_alpha = np.load(raw_pid_dir / "gt_alpha.npy")
            D, H, W = gt_alpha.shape
            midline_x = W // 2
            
            label_bilateral = np.zeros_like(gt_alpha, dtype=np.uint8)
            # Right IAN: x < midline_x
            label_bilateral[:, :, :midline_x] = np.where(gt_alpha[:, :, :midline_x] > 0, 1, 0)
            # Left IAN: x >= midline_x
            label_bilateral[:, :, midline_x:] = np.where(gt_alpha[:, :, midline_x:] > 0, 2, 0)
            
            np.save(out_pid_dir / "label_bilateral.npy", label_bilateral)
            np.save(out_pid_dir / "gt_alpha.npy", gt_alpha.astype(np.uint8))
            
        if (raw_pid_dir / "gt_sparse.npy").exists():
            gt_sparse = np.load(raw_pid_dir / "gt_sparse.npy").astype(np.uint8)
            np.save(out_pid_dir / "gt_sparse.npy", gt_sparse)
            
    # Process Sparse Scans (Synthetic)
    print("\n📦 Preprocessing Sparsely Labeled Scans...")
    sparse_targets = all_sparse_cases[:max_cases] if max_cases else all_sparse_cases
    
    for pid in tqdm(sparse_targets, desc="Caching Sparse"):
        raw_pid_dir = RAW_DATA_DIR / pid
        out_pid_dir = SPARSE_CACHE_DIR / pid
        out_pid_dir.mkdir(parents=True, exist_ok=True)
        
        raw_data = np.load(raw_pid_dir / "data.npy").astype(np.float32)
        clipped = np.clip(raw_data, INTENSITY_CLIPPING[0], INTENSITY_CLIPPING[1])
        tissue_mask = clipped > -400.0
        if tissue_mask.sum() > 1000:
            mean_val = clipped[tissue_mask].mean()
            std_val = clipped[tissue_mask].std() + 1e-5
        else:
            mean_val = clipped.mean()
            std_val = clipped.std() + 1e-5
        normalized = (clipped - mean_val) / std_val
        
        np.save(out_pid_dir / "data.npy", normalized.astype(np.float32))
        
        if (raw_pid_dir / "gt_sparse.npy").exists():
            gt_sparse = np.load(raw_pid_dir / "gt_sparse.npy").astype(np.uint8)
            np.save(out_pid_dir / "gt_sparse.npy", gt_sparse)

    print("\n Preprocessing and Caching Complete!")

if __name__ == "__main__":
    preprocess_and_cache_dataset()
