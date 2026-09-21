import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random
from pathlib import Path
from src.dataset.transforms import Compose3D

class ToothFairyPatchDataset(Dataset):
    """
    ToothFairy 3D Patch Dataset with intelligent foreground-biased patch extraction.
    Supports cached preprocessed .npy files.
    """
    def __init__(
        self,
        case_ids: list,
        cache_dir: str | Path,
        patch_size: tuple = (80, 128, 160),
        is_train: bool = True,
        fg_probability: float = 0.70,
        transforms: Compose3D = None,
        samples_per_epoch: int = 250
    ):
        self.case_ids = case_ids
        self.cache_dir = Path(cache_dir)
        self.patch_size = patch_size
        self.is_train = is_train
        self.fg_probability = fg_probability
        self.transforms = transforms
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        if self.is_train:
            return self.samples_per_epoch
        return len(self.case_ids)

    def _sample_patch(self, data: np.ndarray, label: np.ndarray = None):
        D, H, W = data.shape
        pD, pH, pW = self.patch_size
        
        # Determine if we should center on foreground
        sample_fg = (self.is_train and label is not None and random.random() < self.fg_probability)
        fg_voxels = np.argwhere(label > 0) if (sample_fg and label is not None) else None
        
        if sample_fg and fg_voxels is not None and len(fg_voxels) > 0:
            center_idx = random.choice(fg_voxels)
            cz, cy, cx = center_idx
            
            sz = max(0, min(cz - pD // 2, D - pD))
            sy = max(0, min(cy - pH // 2, H - pH))
            sx = max(0, min(cx - pW // 2, W - pW))
        else:
            sz = random.randint(0, max(0, D - pD)) if D > pD else 0
            sy = random.randint(0, max(0, H - pH)) if H > pH else 0
            sx = random.randint(0, max(0, W - pW)) if W > pW else 0

        # Extract patch with padding if necessary
        ez = min(sz + pD, D)
        ey = min(sy + pH, H)
        ex = min(sx + pW, W)
        
        patch_data = np.zeros((pD, pH, pW), dtype=np.float32)
        patch_data[:ez-sz, :ey-sy, :ex-sx] = data[sz:ez, sy:ey, sx:ex]
        
        patch_label = None
        if label is not None:
            patch_label = np.zeros((pD, pH, pW), dtype=np.int64)
            patch_label[:ez-sz, :ey-sy, :ex-sx] = label[sz:ez, sy:ey, sx:ex]
            
        return patch_data, patch_label

    def __getitem__(self, idx: int):
        if self.is_train:
            case_id = random.choice(self.case_ids)
        else:
            case_id = self.case_ids[idx]
            
        case_dir = self.cache_dir / case_id
        data_path = case_dir / "data.npy"
        label_path = case_dir / "label_bilateral.npy"
        
        # Load memory mapped
        data = np.load(data_path, mmap_mode='r')
        label = np.load(label_path, mmap_mode='r') if label_path.exists() else None
        
        if self.is_train:
            patch_data, patch_label = self._sample_patch(data, label)
            if self.transforms is not None:
                patch_data, patch_label = self.transforms(patch_data, patch_label)
                
            # Add channel dimension (1, D, H, W)
            tensor_data = torch.from_numpy(patch_data).unsqueeze(0).float()
            tensor_label = torch.from_numpy(patch_label).long() if patch_label is not None else torch.zeros(self.patch_size, dtype=torch.long)
            return tensor_data, tensor_label, case_id
        else:
            # Full volume evaluation
            tensor_data = torch.from_numpy(np.array(data)).unsqueeze(0).float()
            tensor_label = torch.from_numpy(np.array(label)).long() if label is not None else torch.zeros_like(tensor_data[0], dtype=torch.long)
            return tensor_data, tensor_label, case_id
