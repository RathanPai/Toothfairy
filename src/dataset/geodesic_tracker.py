import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import skeletonize_3d, binary_dilation, ball
import heapq

def dijkstra_3d_shortest_path(cost_volume: np.ndarray, start_idx: tuple, end_idx: tuple) -> list:
    """
    Computes 3D Dijkstra shortest path between start_idx and end_idx through a continuous cost volume.
    cost_volume: (D, H, W) non-negative cost map (e.g. 1.0 - probability)
    """
    D, H, W = cost_volume.shape
    dist = np.full((D, H, W), np.inf, dtype=np.float32)
    dist[start_idx] = 0.0
    
    predecessor = {}
    visited = set()
    
    # Priority queue: (current_distance, z, y, x)
    pq = [(0.0, start_idx[0], start_idx[1], start_idx[2])]
    
    # 26-connectivity neighborhood
    dz = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    dy = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
    dx = [-1, 0, 1]
    
    neighbors = []
    for z in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            for x in [-1, 0, 1]:
                if not (z == 0 and y == 0 and x == 0):
                    neighbors.append((z, y, x, np.sqrt(z*z + y*y + x*x)))
                    
    while pq:
        d, z, y, x = heapq.heappop(pq)
        curr = (z, y, x)
        
        if curr == end_idx:
            break
            
        if curr in visited:
            continue
        visited.add(curr)
        
        for nz, ny, nx, step_dist in neighbors:
            vz, vy, vx = z + nz, y + ny, x + nx
            if 0 <= vz < D and 0 <= vy < H and 0 <= vx < W:
                v_node = (vz, vy, vx)
                if v_node not in visited:
                    # Edge weight = step_distance * average cost
                    edge_weight = step_dist * (cost_volume[curr] + cost_volume[v_node]) * 0.5
                    new_dist = d + edge_weight
                    if new_dist < dist[v_node]:
                        dist[v_node] = new_dist
                        predecessor[v_node] = curr
                        heapq.heappush(pq, (new_dist, vz, vy, vx))
                        
    # Reconstruct path
    path = []
    curr = end_idx
    while curr in predecessor:
        path.append(curr)
        curr = predecessor[curr]
    if path:
        path.append(start_idx)
        path.reverse()
        
    return path

def reconstruct_geodesic_dense_mask(
    prob_map: np.ndarray,
    sparse_gt: np.ndarray,
    midline_x: int = None,
    dilation_radius: int = 2
) -> np.ndarray:
    """
    Combines Model Probability Map + gt_sparse seed anchors to reconstruct unbroken dense IAN pseudo-masks.
    prob_map: (D, H, W) float in [0, 1]
    sparse_gt: (D, H, W) sparse seed points
    """
    D, H, W = prob_map.shape
    if midline_x is None:
        midline_x = W // 2
        
    dense_reconstructed = np.zeros((D, H, W), dtype=np.uint8)
    cost_map = np.clip(1.0 - prob_map, 0.001, 1.0)
    
    # Process Right (x < midline_x) and Left (x >= midline_x) separately
    for is_right, hemi_slice in [(True, (slice(None), slice(None), slice(0, midline_x))),
                                 (False, (slice(None), slice(None), slice(midline_x, None)))]:
        sparse_hemi = sparse_gt[hemi_slice]
        prob_hemi = prob_map[hemi_slice]
        cost_hemi = cost_map[hemi_slice]
        
        seed_points = np.argwhere(sparse_hemi > 0)
        if len(seed_points) < 5:
            continue
            
        # Sort seed points by z-axis (or principal curve axis)
        seed_points = seed_points[np.argsort(seed_points[:, 0])]
        
        # Connect consecutive seed clusters via shortest path
        centerline_voxels = []
        step_stride = max(1, len(seed_points) // 10)
        key_anchors = seed_points[::step_stride]
        if not np.array_equal(key_anchors[-1], seed_points[-1]):
            key_anchors = np.vstack([key_anchors, seed_points[-1]])
            
        for i in range(len(key_anchors) - 1):
            p_start = tuple(key_anchors[i])
            p_end = tuple(key_anchors[i + 1])
            segment = dijkstra_3d_shortest_path(cost_hemi, p_start, p_end)
            if segment:
                centerline_voxels.extend(segment)
            else:
                centerline_voxels.append(p_start)
                centerline_voxels.append(p_end)
                
        # Create continuous 3D skeleton tube
        hemi_skel = np.zeros(sparse_hemi.shape, dtype=bool)
        for (z, y, x) in centerline_voxels:
            hemi_skel[z, y, x] = True
            
        # Dilate centerline and intersect with confident prob map region
        struct = ball(dilation_radius)
        dilated_tube = binary_dilation(hemi_skel, struct)
        # Refine tube with model prediction (keep voxels where prob > 0.20 or directly on skeleton)
        refined_tube = np.logical_or(hemi_skel, np.logical_and(dilated_tube, prob_hemi > 0.20))
        
        class_val = 1 if is_right else 2
        dense_reconstructed[hemi_slice][refined_tube] = class_val
        
    return dense_reconstructed
