import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.default_config import (
    SPLITS_FILE, DENSE_CACHE_DIR, PSEUDO_CACHE_DIR, CHECKPOINTS_DIR,
    PATCH_SIZE, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, NUM_CLASSES,
    LEARNING_RATE, WEIGHT_DECAY
)
from src.dataset.toothfairy_dataset import ToothFairyPatchDataset, CombinedToothFairyDataset
from src.dataset.transforms import get_training_transforms, get_validation_transforms
from src.models.mednext3d import MedNeXt3D
from src.models.losses import CompoundTopologyLoss
from src.training.trainer import Trainer

def train_mednext(max_epochs: int = 300, lr: float = 1e-2, kernel_size: int = 5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Initializing MedNeXt-3D (Kernel size: {kernel_size}x{kernel_size}x{kernel_size}) Training")
    
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
        
    dense_train_cases = splits["train"]
    val_cases = splits["val"]
    
    pseudo_json = PSEUDO_CACHE_DIR / "valid_pseudo_pids.json"
    if pseudo_json.exists():
        with open(pseudo_json) as f:
            pseudo_cases = json.load(f)
    else:
        pseudo_cases = []
        
    print(f"Train Dataset: {len(dense_train_cases)} Clean Dense + {len(pseudo_cases)} Geodesic-Refined Pseudo Cases")
    print(f"Validation Dataset: {len(val_cases)} Cases")
    
    train_dataset = CombinedToothFairyDataset(
        dense_pids=dense_train_cases,
        pseudo_pids=pseudo_cases,
        patch_size=PATCH_SIZE,
        transforms=get_training_transforms(),
        samples_per_epoch=300
    )
    val_dataset = ToothFairyPatchDataset(
        case_ids=val_cases,
        cache_dir=DENSE_CACHE_DIR,
        patch_size=PATCH_SIZE,
        is_train=False,
        transforms=get_validation_transforms()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # Instantiate MedNeXt-3D model
    model = MedNeXt3D(
        in_channels=1,
        num_classes=NUM_CLASSES,
        feature_dims=[24, 48, 96, 192, 256],
        kernel_size=kernel_size,
        deep_supervision=True
    )
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99, weight_decay=WEIGHT_DECAY, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    loss_fn = CompoundTopologyLoss(num_classes=NUM_CLASSES)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        grad_accum_steps=GRADIENT_ACCUMULATION_STEPS,
        checkpoint_dir=CHECKPOINTS_DIR,
        experiment_name="mednext3d_final"
    )
    
    trainer.train(max_epochs=max_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--kernel_size", type=int, default=5)
    args = parser.parse_args()
    
    train_mednext(max_epochs=args.epochs, lr=args.lr, kernel_size=args.kernel_size)
