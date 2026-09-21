import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import binary_closing, ball

def extract_mandibular_components(binary_mask: np.ndarray, min_size: int = 400, max_components: int = 2) -> np.ndarray:
    """
    Extracts the major continuous 3D components along the mandibular corridor and bridges them.
    """
    if binary_mask.sum() == 0:
        return binary_mask
        
    labeled_array, num_features = ndi.label(binary_mask)
    if num_features == 0:
        return binary_mask
        
    sizes = ndi.sum(binary_mask, labeled_array, range(1, num_features + 1))
    sorted_indices = np.argsort(sizes)[::-1]
    
    result_mask = np.zeros_like(binary_mask)
    kept_count = 0
    
    for idx in sorted_indices:
        label_id = idx + 1
        comp_size = sizes[idx]
        
        if comp_size >= min_size and kept_count < max_components:
            result_mask[labeled_array == label_id] = True
            kept_count += 1
            
    return result_mask

def bridge_short_gaps(binary_mask: np.ndarray, gap_radius: int = 2) -> np.ndarray:
    """
    Performs 3D morphological closing with a small structuring element to bridge micro-breaks.
    """
    if binary_mask.sum() == 0:
        return binary_mask
    struct = ball(gap_radius)
    closed = binary_closing(binary_mask, struct)
    return closed

def filter_bilateral_ian(pred_logits_or_mask: np.ndarray, midline_x: int = None, min_cc_size: int = 400, top_cutoff_ratio: float = 0.16) -> np.ndarray:
    """
    Advanced Anatomical Post-Processing Pipeline:
    1. Trims cranial/maxillary false positives above the mandibular ramus cutoff.
    2. Splits volume into Left and Right hemispheres at the sagittal midline.
    3. Retains major continuous tubular components per hemisphere and bridges micro-breaks.
    4. Eliminates isolated satellite noise to minimize Hausdorff Distance (HD95).
    """
    D, H, W = pred_logits_or_mask.shape
    if midline_x is None:
        midline_x = W // 2
        
    cleaned_mask = np.zeros_like(pred_logits_or_mask, dtype=np.uint8)
    
    # 1. Mandibular Anatomical Upper Bounding (Zero out top cranial air/skull base)
    top_cutoff = int(D * top_cutoff_ratio)
    bounded_pred = pred_logits_or_mask.copy()
    bounded_pred[:top_cutoff] = 0
    
    # Split into Right Hemisphere (x < midline_x) and Left Hemisphere (x >= midline_x)
    right_hemi_raw = (bounded_pred[:, :, :midline_x] > 0)
    left_hemi_raw = (bounded_pred[:, :, midline_x:] > 0)
    
    # Process Right Canal
    if right_hemi_raw.sum() > 0:
        right_cc = extract_mandibular_components(right_hemi_raw, min_size=min_cc_size)
        right_bridged = bridge_short_gaps(right_cc, gap_radius=2)
        cleaned_mask[:, :, :midline_x] = right_bridged.astype(np.uint8)
        
    # Process Left Canal
    if left_hemi_raw.sum() > 0:
        left_cc = extract_mandibular_components(left_hemi_raw, min_size=min_cc_size)
        left_bridged = bridge_short_gaps(left_cc, gap_radius=2)
        cleaned_mask[:, :, midline_x:] = left_bridged.astype(np.uint8)
        
    return cleaned_mask
