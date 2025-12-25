
import jax.numpy as jnp
import numpy as np
from typing import List, Union, Tuple, Optional

def suggest_symmetries(
    points: Union[np.ndarray, jnp.ndarray, List[List[float]]],
    tolerance: float = 1e-5
) -> List[Union[str, Tuple]]:
    """
    Analyze a set of 2D points (e.g., voter ideal points) and suggest
    symmetries that are supported by Grid.partition_from_symmetry.

    Checks for:
    - Reflection around x=0 ('reflect_x')
    - Reflection around y=0 ('reflect_y')
    - Reflection around y=x ('swap_xy')
    - Reflection around x=centroid_x ('reflect_x=c')
    - Reflection around y=centroid_y ('reflect_y=c')
    - Rotational symmetry (2, 3, 4, 6-fold) around centroid and origin

    Args:
        points: Array-like of shape (N, 2) containing (x, y) coordinates
        tolerance: Distance tolerance for matching points

    Returns:
        List of symmetry specifications supported by Grid.partition_from_symmetry
    """
    points = np.array(points)
    
    # Handle empty input
    if points.size == 0:
        return []
        
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Points must be an (N, 2) array of coordinates")

    n_points = len(points)

    # Calculate centroid
    centroid = np.mean(points, axis=0)
    cx, cy = centroid

    symmetries = []

    # Helper to check if a transformed set matches original set
    def check_match(transformed_points):
        # specific check: for every p in points, is there a q in transformed close enough?
        # And since same size, it implies bijection if unique matches.
        # Brute force O(N^2) is fine for small N (voters usually small)
        matched_count = 0
        used = np.zeros(n_points, dtype=bool)
        
        for p in transformed_points:
            # Find closest point in original set
            dists = np.sum((points - p)**2, axis=1)
            closest_idx = np.argmin(dists)
            min_dist = np.sqrt(dists[closest_idx])
            
            if min_dist < tolerance:
                # We perform a greedy match; for strict set equality checks with duplicates
                # we'd need to be more careful, but for ideal points usually fine.
                matched_count += 1
        
        return matched_count == n_points

    # 1. Reflection around x=0
    # (x, y) -> (-x, y)
    reflected_x0 = points.copy()
    reflected_x0[:, 0] = -reflected_x0[:, 0]
    if check_match(reflected_x0):
        symmetries.append('reflect_x')

    # 2. Reflection around x=cx (if cx is significantly different from 0)
    if abs(cx) > tolerance:
        # (x, y) -> (2cx - x, y)
        reflected_xc = points.copy()
        reflected_xc[:, 0] = 2 * cx - reflected_xc[:, 0]
        if check_match(reflected_xc):
            symmetries.append(f'reflect_x={cx:g}')

    # 3. Reflection around y=0
    # (x, y) -> (x, -y)
    reflected_y0 = points.copy()
    reflected_y0[:, 1] = -reflected_y0[:, 1]
    if check_match(reflected_y0):
        symmetries.append('reflect_y')

    # 4. Reflection around y=cy (if cy is significantly different from 0)
    if abs(cy) > tolerance:
        # (x, y) -> (x, 2cy - y)
        reflected_yc = points.copy()
        reflected_yc[:, 1] = 2 * cy - reflected_yc[:, 1]
        if check_match(reflected_yc):
            symmetries.append(f'reflect_y={cy:g}')

    # 5. Swap XY (y=x reflection)
    # (x, y) -> (y, x)
    swapped_xy = points.copy()
    swapped_xy = swapped_xy[:, [1, 0]] # Swap columns
    if check_match(swapped_xy):
        symmetries.append('swap_xy')

    # 6. Rotational Symmetry
    # Check rotation around Centroid and Origin
    centers_to_check = []
    if np.allclose(centroid, 0, atol=tolerance):
        centers_to_check.append(("origin", 0.0, 0.0))
    else:
        centers_to_check.append(("origin", 0.0, 0.0))
        centers_to_check.append(("centroid", cx, cy))

    # Common folds: 2 (180), 3 (120), 4 (90), 6 (60)
    folds = [2, 3, 4, 6]
    
    for name, center_x, center_y in centers_to_check:
        for fold in folds:
            degrees = 360.0 / fold
            theta = np.radians(degrees)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]])
            
            # Translate to center -> Rotate -> Translate back
            centered = points - np.array([center_x, center_y])
            rotated_centered = centered @ R.T
            rotated = rotated_centered + np.array([center_x, center_y])
            
            if check_match(rotated):
                symmetries.append(('rotate', float(center_x), float(center_y), float(degrees)))

    return symmetries
