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
    LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS_STUDENT
)
from src.dataset.toothfairy_dataset import ToothFairyPatchDataset
from src.dataset.transforms import get_training_transforms, get_validation_transforms
from src.models.resnet_unet3d import ResEncoderUNet3D
from src.models.losses import CompoundTopologyLoss
from src.training.trainer import Trainer

class CombinedToothFairyDataset(ToothFairyPatchDataset):
    """
    Combines dense ground-truth dataset with pseudo-labeled dataset seamlessly.
    """
    def __init__(self, dense_pids: list, pseudo_pids: list, patch_size: tuple, transforms=None, samples_per_epoch: int = 300):
        super().__init__(
            case_ids=dense_pids,
            cache_dir=DENSE_CACHE_DIR,
            patch_size=patch_size,
            is_train=True,
            transforms=transforms,
            samples_per_epoch=samples_per_epoch
        )
        self.dense_pids = dense_pids
        self.pseudo_pids = pseudo_pids

    def __getitem__(self, idx: int):
        # 50% probability of sampling from clean dense vs pseudo-labeled
        if torch.rand(1).item() < 0.50 or len(self.pseudo_pids) == 0:
            case_id = self.dense_pids[torch.randint(0, len(self.dense_pids), (1,)).item()]
            cache_dir = DENSE_CACHE_DIR
        else:
            case_id = self.pseudo_pids[torch.randint(0, len(self.pseudo_pids), (1,)).item()]
            cache_dir = PSEUDO_CACHE_DIR
            
        case_dir = cache_dir / case_id
        data = np.load(case_dir / "data.npy", mmap_mode='r')
        label = np.load(case_dir / "label_bilateral.npy", mmap_mode='r')
        
        patch_data, patch_label = self._sample_patch(data, label)
        if self.transforms is not None:
            patch_data, patch_label = self.transforms(patch_data, patch_label)
            
        tensor_data = torch.from_numpy(patch_data).unsqueeze(0).float()
        tensor_label = torch.from_numpy(patch_label).long()
        return tensor_data, tensor_label, case_id

import numpy as np

def train_student(max_epochs: int = MAX_EPOCHS_STUDENT, lr: float = LEARNING_RATE):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Initializing Student Model Training on Combined (Dense + Pseudo) Dataset")
    
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
        
    dense_train_cases = splits["train"]
    val_cases = splits["val"]
    
    # Load valid pseudo cases
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
    
    model = ResEncoderUNet3D(in_channels=1, num_classes=NUM_CLASSES, feature_dims=[24, 48, 96, 192, 256], deep_supervision=True)
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
        experiment_name="student_resenc_final"
    )
    
    trainer.train(max_epochs=max_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS_STUDENT)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    
    train_student(max_epochs=args.epochs, lr=args.lr)
