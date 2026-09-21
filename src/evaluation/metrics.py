import numpy as np
import scipy.ndimage as ndi
from scipy.spatial.distance import cdist
import SimpleITK as sitk

def compute_dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Computes standard Dice Similarity Coefficient (DSC) for binary / foreground volumes.
    """
    pred_bool = (pred > 0).astype(bool)
    target_bool = (target > 0).astype(bool)
    
    intersection = np.logical_and(pred_bool, target_bool).sum()
    cardinality = pred_bool.sum() + target_bool.sum()
    
    if cardinality == 0:
        return 1.0  # Both are empty
    return float((2.0 * intersection + smooth) / (cardinality + smooth))

def compute_hd95(pred: np.ndarray, target: np.ndarray, voxel_spacing: tuple = (1.0, 1.0, 1.0)) -> float:
    """
    Computes 95th percentile Hausdorff Distance (HD95) in physical units (mm).
    Uses SimpleITK for robust and fast surface distance computation.
    """
    pred_bool = (pred > 0).astype(np.uint8)
    target_bool = (target > 0).astype(np.uint8)
    
    # If either volume has no foreground, return large penalty (e.g. 100 mm)
    if pred_bool.sum() == 0 or target_bool.sum() == 0:
        return 100.0
    
    # Create SimpleITK images with spatial spacing
    pred_itk = sitk.GetImageFromArray(pred_bool)
    pred_itk.SetSpacing((voxel_spacing[2], voxel_spacing[1], voxel_spacing[0]))  # (x, y, z)
    
    target_itk = sitk.GetImageFromArray(target_bool)
    target_itk.SetSpacing((voxel_spacing[2], voxel_spacing[1], voxel_spacing[0]))
    
    # Surface distance filter
    hd_filter = sitk.HausdorffDistanceImageFilter()
    try:
        # Compute contour surfaces
        pred_surface = sitk.LabelContour(pred_itk, fullyConnected=True)
        target_surface = sitk.LabelContour(target_itk, fullyConnected=True)
        
        # Directed distance from pred to target
        dist_map_target = sitk.SignedMaurerDistanceMap(target_surface, squaredDistance=False, useImageSpacing=True)
        dist_map_pred = sitk.SignedMaurerDistanceMap(pred_surface, squaredDistance=False, useImageSpacing=True)
        
        # Sample distances at surface points
        dist_to_target = np.abs(sitk.GetArrayViewFromImage(dist_map_target)[sitk.GetArrayViewFromImage(pred_surface) > 0])
        dist_to_pred = np.abs(sitk.GetArrayViewFromImage(dist_map_pred)[sitk.GetArrayViewFromImage(target_surface) > 0])
        
        if len(dist_to_target) == 0 or len(dist_to_pred) == 0:
            return 100.0
            
        all_dists = np.concatenate([dist_to_target, dist_to_pred])
        return float(np.percentile(all_dists, 95))
    except Exception:
        # Fallback to standard HD if contour extraction fails
        hd_filter.Execute(pred_itk, target_itk)
        return float(hd_filter.GetHausdorffDistance())

def evaluate_case(pred: np.ndarray, target: np.ndarray, voxel_spacing: tuple = (1.0, 1.0, 1.0)) -> dict:
    """
    Evaluates both binary foreground and per-hemisphere metrics.
    """
    dice = compute_dice_score(pred, target)
    hd95 = compute_hd95(pred, target, voxel_spacing=voxel_spacing)
    
    # If target is binary (gt_alpha), compute per-hemisphere by midline
    D, H, W = target.shape
    midline_x = W // 2
    
    if np.array_equal(np.unique(target), np.array([0, 1])):
        target_r = np.zeros_like(target)
        target_r[:, :, :midline_x] = target[:, :, :midline_x]
        target_l = np.zeros_like(target)
        target_l[:, :, midline_x:] = target[:, :, midline_x:]
    else:
        target_r = (target == 1)
        target_l = (target == 2)
        
    pred_r = np.zeros_like(pred)
    pred_r[:, :, :midline_x] = (pred[:, :, :midline_x] > 0)
    pred_l = np.zeros_like(pred)
    pred_l[:, :, midline_x:] = (pred[:, :, midline_x:] > 0)
    
    right_dice = compute_dice_score(pred_r, target_r)
    left_dice = compute_dice_score(pred_l, target_l)
    
    return {
        "dice": dice,
        "hd95": hd95,
        "right_dice": right_dice,
        "left_dice": left_dice
    }
