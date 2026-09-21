import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_erode_3d(img: torch.Tensor) -> torch.Tensor:
    """
    Differentiable 3D erosion via min pooling.
    """
    if len(img.shape) != 5:
        raise ValueError(f"Expected 5D tensor (B, C, D, H, W), got {img.shape}")
    
    # 3D 6-neighborhood structuring element
    p1 = -F.max_pool3d(-img, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
    p2 = -F.max_pool3d(-img, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
    p3 = -F.max_pool3d(-img, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
    
    return torch.min(torch.min(p1, p2), p3)

def soft_dilate_3d(img: torch.Tensor) -> torch.Tensor:
    """
    Differentiable 3D dilation via max pooling.
    """
    if len(img.shape) != 5:
        raise ValueError(f"Expected 5D tensor (B, C, D, H, W), got {img.shape}")
    
    p1 = F.max_pool3d(img, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
    p2 = F.max_pool3d(img, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
    p3 = F.max_pool3d(img, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
    
    return torch.max(torch.max(p1, p2), p3)

def soft_open_3d(img: torch.Tensor) -> torch.Tensor:
    """
    Differentiable 3D morphological opening: erosion followed by dilation.
    """
    return soft_dilate_3d(soft_erode_3d(img))

def soft_skeletonize_3d(img: torch.Tensor, num_iters: int = 4) -> torch.Tensor:
    """
    Differentiable 3D soft skeletonization algorithm (Lantuéjoul's formula).
    S = Sum_{k=0}^N (E^k(img) - Open(E^k(img)))
    """
    img_current = img
    skel = F.relu(img_current - soft_open_3d(img_current))
    
    for _ in range(num_iters):
        img_current = soft_erode_3d(img_current)
        opened = soft_open_3d(img_current)
        skel = skel + F.relu(img_current - opened)
        
    return skel
