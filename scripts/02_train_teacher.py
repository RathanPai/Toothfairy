import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.default_config import (
    SPLITS_FILE, DENSE_CACHE_DIR, CHECKPOINTS_DIR, PATCH_SIZE,
    BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, NUM_CLASSES,
    LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS_TEACHER, NUM_FOLDS
)
from src.dataset.toothfairy_dataset import ToothFairyPatchDataset
from src.dataset.transforms import get_training_transforms, get_validation_transforms
from src.models.resnet_unet3d import ResEncoderUNet3D
from src.models.losses import CompoundTopologyLoss
from src.training.trainer import Trainer

def train_teacher_fold(fold_idx: int = 0, max_epochs: int = MAX_EPOCHS_TEACHER, lr: float = LEARNING_RATE):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔹 Initializing Teacher Model Training - Fold {fold_idx} on {device}")
    
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
        
    train_pool = splits["train"]
    # 5-fold cross validation split
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    splits_generator = list(kf.split(train_pool))
    train_indices, val_indices = splits_generator[fold_idx]
    
    train_cases = [train_pool[i] for i in train_indices]
    val_cases = [train_pool[i] for i in val_indices] + splits.get("val", [])
    
    print(f"Fold {fold_idx}: Train Cases = {len(train_cases)}, Val Cases = {len(val_cases)}")
    
    # Datasets & Dataloaders
    train_dataset = ToothFairyPatchDataset(
        case_ids=train_cases,
        cache_dir=DENSE_CACHE_DIR,
        patch_size=PATCH_SIZE,
        is_train=True,
        fg_probability=0.70,
        transforms=get_training_transforms(),
        samples_per_epoch=200
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
    
    # Model
    model = ResEncoderUNet3D(in_channels=1, num_classes=NUM_CLASSES, feature_dims=[24, 48, 96, 192, 256], deep_supervision=True)
    
    # Optimizer & Scheduler
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
        experiment_name=f"teacher_resenc_fold{fold_idx}"
    )
    
    trainer.train(max_epochs=max_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS_TEACHER)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    
    train_teacher_fold(fold_idx=args.fold, max_epochs=args.epochs, lr=args.lr)
