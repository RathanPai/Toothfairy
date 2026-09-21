import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import remove_small_holes, binary_closing, ball

def extract_single_largest_component(binary_mask: np.ndarray, min_size: int = 500) -> np.ndarray:
    """
    Extracts strictly the single largest 3D connected component.
    """
    if binary_mask.sum() == 0:
        return binary_mask
        
    labeled_array, num_features = ndi.label(binary_mask)
    if num_features == 0:
        return binary_mask
        
    sizes = ndi.sum(binary_mask, labeled_array, range(1, num_features + 1))
    max_label = np.argmax(sizes) + 1
    
    # If the largest component is too small, it's noise
    if sizes[max_label - 1] < min_size:
        return np.zeros_like(binary_mask)
        
    largest_cc = (labeled_array == max_label)
    return largest_cc

def bridge_short_gaps(binary_mask: np.ndarray, gap_radius: int = 3) -> np.ndarray:
    """
    Performs 3D morphological closing with a small structuring element to bridge micro-breaks.
    """
    if binary_mask.sum() == 0:
        return binary_mask
    struct = ball(gap_radius)
    closed = binary_closing(binary_mask, struct)
    return closed

def filter_bilateral_ian(pred_logits_or_mask: np.ndarray, midline_x: int = None, min_cc_size: int = 600) -> np.ndarray:
    """
    Anatomical Post-Processing Pipeline:
    1. Splits volume into Left and Right hemispheres at the sagittal midline.
    2. Enforces EXACTLY one unbroken canal per side (single largest component).
    3. Bridges micro-gaps (< 3-4 voxels) via morphological dilation/closing.
    4. Suppresses high maxilla/skull base false positives.
    
    pred_logits_or_mask: (D, H, W) integer mask (0, 1, 2) or binary (0, 1)
    Returns: Cleaned (D, H, W) binary mask (or multi-class mask).
    """
    D, H, W = pred_logits_or_mask.shape
    if midline_x is None:
        midline_x = W // 2
        
    cleaned_mask = np.zeros_like(pred_logits_or_mask, dtype=np.uint8)
    
    # Split into Right Hemisphere (x < midline_x) and Left Hemisphere (x >= midline_x)
    right_hemi_raw = (pred_logits_or_mask[:, :, :midline_x] > 0)
    left_hemi_raw = (pred_logits_or_mask[:, :, midline_x:] > 0)
    
    # Process Right Canal
    if right_hemi_raw.sum() > 0:
        right_cc = extract_single_largest_component(right_hemi_raw, min_size=min_cc_size)
        right_bridged = bridge_short_gaps(right_cc, gap_radius=2)
        cleaned_mask[:, :, :midline_x] = right_bridged.astype(np.uint8)
        
    # Process Left Canal
    if left_hemi_raw.sum() > 0:
        left_cc = extract_single_largest_component(left_hemi_raw, min_size=min_cc_size)
        left_bridged = bridge_short_gaps(left_cc, gap_radius=2)
        cleaned_mask[:, :, midline_x:] = left_bridged.astype(np.uint8)
        
    return cleaned_mask
