import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.default_config import (
    SPLITS_FILE, DENSE_CACHE_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR,
    PATCH_SIZE, NUM_CLASSES
)
from src.models.resnet_unet3d import ResEncoderUNet3D
from src.postprocessing.anatomical_filter import filter_bilateral_ian
from src.evaluation.metrics import evaluate_case

@torch.no_grad()
def sliding_window_predict_fast(model, volume_np, device="cuda", patch_size=(64, 128, 128), overlap=0.5):
    """
    High-speed, memory-safe sliding window inference.
    Accumulates outputs on CPU to guarantee zero GPU memory overflow on 8GB VRAM.
    """
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
                
    return output_probs / np.maximum(count_map, 1e-5)

from src.models.mednext3d import MedNeXt3D

def run_benchmark_evaluation(checkpoint_paths: list = None, split_name: str = "test", min_cc_size: int = 400):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 75)
    print(f"📊 TOOTHFAIRY SOTA BENCHMARK EVALUATION (Split: {split_name})")
    print("=" * 75)
    
    if not checkpoint_paths:
        possible_checkpoints = sorted(list(CHECKPOINTS_DIR.glob("**/best_model.pth")))
        checkpoint_paths = possible_checkpoints
        
        
    print(f"Ensemble Models ({len(checkpoint_paths)}):")
    models = []
    weights = []
    for cp in checkpoint_paths:
        cp_str = str(cp).lower()
        if "mednext" in cp_str:
            print(f"  - [MedNeXt-3D] {cp}")
            m = MedNeXt3D(in_channels=1, num_classes=NUM_CLASSES, feature_dims=[24, 48, 96, 192, 256], kernel_size=5, deep_supervision=False)
            w = 0.20
        elif "student" in cp_str:
            print(f"  - [ResEnc Student] {cp}")
            m = ResEncoderUNet3D(in_channels=1, num_classes=NUM_CLASSES, feature_dims=[24, 48, 96, 192, 256], deep_supervision=False)
            w = 0.30
        else:
            print(f"  - [ResEnc Teacher] {cp}")
            m = ResEncoderUNet3D(in_channels=1, num_classes=NUM_CLASSES, feature_dims=[24, 48, 96, 192, 256], deep_supervision=False)
            w = 0.50
            
        ck = torch.load(cp, map_location=device)
        sd = {k: v for k, v in ck["model_state_dict"].items() if not k.startswith("ds_head")}
        m.load_state_dict(sd, strict=False)
        m = m.to(device).eval()
        models.append(m)
        weights.append(w)
        
    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    print(f"Ensemble Weights: {[round(w, 3) for w in weights]}")
        
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    test_cases = splits.get(split_name, [])
    print(f"\nEvaluating {len(test_cases)} cases on '{split_name}' split...")
    
    results = []
    
    for pid in tqdm(test_cases, desc="Evaluating"):
        case_dir = DENSE_CACHE_DIR / pid
        if not case_dir.exists():
            continue
            
        data = np.load(case_dir / "data.npy")
        gt_alpha = np.load(case_dir / "gt_alpha.npy") if (case_dir / "gt_alpha.npy").exists() else None
        
        # Multi-model weighted ensemble
        ensemble_probs = np.zeros((NUM_CLASSES, *data.shape), dtype=np.float32)
        for model, w in zip(models, weights):
            probs = sliding_window_predict_fast(model, data, device=device, patch_size=PATCH_SIZE)
            ensemble_probs += w * probs
        
        # Argmax class prediction
        raw_classes = np.argmax(ensemble_probs, axis=0)
        
        # Anatomical Post-Processing
        cleaned_mask = filter_bilateral_ian(raw_classes, min_cc_size=min_cc_size)
        
        # Save output prediction
        out_save_dir = OUTPUTS_DIR / pid
        out_save_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_save_dir / "pred_mask.npy", cleaned_mask.astype(np.uint8))
        
        if gt_alpha is not None:
            metrics = evaluate_case(cleaned_mask, gt_alpha)
            metrics["case_id"] = pid
            metrics["raw_voxels"] = int((raw_classes > 0).sum())
            metrics["cleaned_voxels"] = int((cleaned_mask > 0).sum())
            metrics["gt_voxels"] = int((gt_alpha > 0).sum())
            results.append(metrics)
            
    if results:
        df = pd.DataFrame(results)
        mean_dsc = df["dice"].mean()
        mean_hd95 = df["hd95"].mean()
        mean_r_dsc = df["right_dice"].mean()
        mean_l_dsc = df["left_dice"].mean()
        
        print("\n" + "=" * 75)
        print("🏆 FINAL BENCHMARK LEADERBOARD COMPARISON")
        print("=" * 75)
        print(f"{'Metric':<30} | {'Leaderboard #1 (Champion)':<25} | {'Our SOTA Model':<20}")
        print("-" * 80)
        print(f"{'Dice Similarity (DSC)':<30} | {'0.7956':<25} | {f'{mean_dsc:.4f}':<20}")
        print(f"{'95% Hausdorff Dist (HD95)':<30} | {'4.4906 mm':<25} | {f'{mean_hd95:.4f} mm':<20}")
        print(f"{'Right IAN Canal DSC':<30} | {'N/A':<25} | {f'{mean_r_dsc:.4f}':<20}")
        print(f"{'Left IAN Canal DSC':<30} | {'N/A':<25} | {f'{mean_l_dsc:.4f}':<20}")
        print("=" * 75)
        
        csv_path = OUTPUTS_DIR / f"benchmark_results_{split_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nDetailed per-case results saved to: {csv_path}")
        
        import importlib
        generate_report_mod = importlib.import_module("scripts.07_generate_report")
        generate_report_mod.generate_report()

if __name__ == "__main__":
    run_benchmark_evaluation()
