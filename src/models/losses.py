import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
from src.models.soft_skeleton import soft_skeletonize_3d

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5, apply_softmax: bool = True):
        super().__init__()
        self.smooth = smooth
        self.apply_softmax = apply_softmax

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: (B, C, D, H, W) logits or probs
        target: (B, D, H, W) or (B, 1, D, H, W) class indices
        """
        if self.apply_softmax:
            pred = F.softmax(pred, dim=1)
            
        if target.dim() == 4:
            target = target.unsqueeze(1)
            
        num_classes = pred.shape[1]
        target_one_hot = torch.zeros_like(pred).scatter_(1, target, 1.0)
        
        # Calculate dice per foreground class (exclude background class 0)
        dice_loss = 0.0
        for c in range(1, num_classes):
            p = pred[:, c]
            t = target_one_hot[:, c]
            intersection = (p * t).sum(dim=(1, 2, 3))
            union = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1.0 - dice.mean())
            
        return dice_loss / max(num_classes - 1, 1)

class SoftclDiceLoss(nn.Module):
    def __init__(self, num_iters: int = 4, smooth: float = 1e-5):
        super().__init__()
        self.num_iters = num_iters
        self.smooth = smooth

    def forward(self, pred_probs: torch.Tensor, target_one_hot: torch.Tensor) -> torch.Tensor:
        """
        pred_probs: (B, C, D, H, W) softmax probabilities
        target_one_hot: (B, C, D, H, W) one-hot ground truth
        """
        cldice_loss = 0.0
        num_classes = pred_probs.shape[1]
        
        for c in range(1, num_classes):
            p = pred_probs[:, c:c+1]
            t = target_one_hot[:, c:c+1]
            
            # Extract soft skeletons
            skel_pred = soft_skeletonize_3d(p, num_iters=self.num_iters)
            skel_true = soft_skeletonize_3d(t, num_iters=self.num_iters)
            
            tprec = ((skel_pred * t).sum(dim=(2, 3, 4)) + self.smooth) / (skel_pred.sum(dim=(2, 3, 4)) + self.smooth)
            tsens = ((skel_true * p).sum(dim=(2, 3, 4)) + self.smooth) / (skel_true.sum(dim=(2, 3, 4)) + self.smooth)
            
            cldice = (2.0 * tprec * tsens) / (tprec + tsens + self.smooth)
            cldice_loss += (1.0 - cldice.mean())
            
        return cldice_loss / max(num_classes - 1, 1)

def compute_distance_transform_maps(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Computes signed Euclidean Distance Transform (EDT) maps for Boundary Loss.
    target: (B, D, H, W)
    Returns: (B, C, D, H, W)
    """
    B, D, H, W = target.shape
    dt_maps = np.zeros((B, num_classes, D, H, W), dtype=np.float32)
    target_np = target.cpu().numpy()
    
    for b in range(B):
        for c in range(1, num_classes):
            posmask = (target_np[b] == c)
            if posmask.any():
                negmask = ~posmask
                pos_edt = distance_transform_edt(posmask)
                neg_edt = distance_transform_edt(negmask)
                # Boundary map: normalized signed distance
                dt_map = neg_edt - (pos_edt - 1)
                dt_maps[b, c] = dt_map / (np.max(np.abs(dt_map)) + 1e-5)
            else:
                dt_maps[b, c] = 1.0
                
    return torch.from_numpy(dt_maps).to(target.device)

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_probs: torch.Tensor, dist_maps: torch.Tensor) -> torch.Tensor:
        """
        Boundary loss = Mean( Pred_Probs * Signed_Distance_Maps )
        """
        # Sum over foreground classes
        b_loss = (pred_probs[:, 1:] * dist_maps[:, 1:]).mean()
        return b_loss

class CompoundTopologyLoss(nn.Module):
    """
    Unified Loss combining:
    1. Weighted Cross-Entropy (voxel classification)
    2. Soft Dice Loss (region overlap)
    3. clDice Loss (topological tubular continuity)
    4. Boundary Loss (surface distance minimization for HD95)
    """
    def __init__(self, weights: dict = None, num_classes: int = 3):
        super().__init__()
        self.weights = weights or {"ce": 0.40, "dice": 0.30, "cldice": 0.20, "boundary": 0.10}
        self.num_classes = num_classes
        
        # Foreground class weights for CE: penalize background class less
        ce_class_weights = torch.tensor([0.2, 1.0, 1.0]) if num_classes == 3 else torch.tensor([0.2, 1.0])
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_class_weights)
        self.dice_loss = SoftDiceLoss(apply_softmax=False)
        self.cldice_loss = SoftclDiceLoss(num_iters=3)
        self.boundary_loss = BoundaryLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, dist_maps: torch.Tensor = None) -> dict:
        """
        logits: (B, C, D, H, W)
        targets: (B, D, H, W)
        dist_maps: Optional precomputed distance transforms
        """
        device = logits.device
        if self.ce_loss.weight.device != device:
            self.ce_loss.weight = self.ce_loss.weight.to(device)
            
        probs = F.softmax(logits, dim=1)
        targets_long = targets.long()
        
        # 1. CE Loss
        l_ce = self.ce_loss(logits, targets_long)
        
        # 2. Dice Loss
        l_dice = self.dice_loss(probs, targets_long)
        
        # 3. clDice Loss
        target_one_hot = torch.zeros_like(probs).scatter_(1, targets_long.unsqueeze(1), 1.0)
        l_cldice = self.cldice_loss(probs, target_one_hot)
        
        # 4. Boundary Loss
        if dist_maps is None:
            dist_maps = compute_distance_transform_maps(targets_long, self.num_classes)
        l_boundary = self.boundary_loss(probs, dist_maps)
        
        # Total Weighted Loss
        total_loss = (
            self.weights["ce"] * l_ce +
            self.weights["dice"] * l_dice +
            self.weights["cldice"] * l_cldice +
            self.weights["boundary"] * l_boundary
        )
        
        return {
            "loss": total_loss,
            "l_ce": l_ce.item(),
            "l_dice": l_dice.item(),
            "l_cldice": l_cldice.item(),
            "l_boundary": l_boundary.item()
        }
