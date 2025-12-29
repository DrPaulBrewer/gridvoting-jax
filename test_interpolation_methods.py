"""
Prototype to test three interpolation methods for outline-based solvers.

Tests on BJM spatial triangle g=40 to determine which method is fastest.
"""

import jax
import jax.numpy as jnp
import time
import numpy as np

# Import gridvoting_jax
import sys
sys.path.insert(0, '/home/paul/gridvoting-jax/gridvoting-jax/src')
import gridvoting_jax as gv

print("JAX devices:", jax.devices())
print("JAX version:", jax.__version__)

# Create BJM spatial triangle model with g=40
print("\n" + "="*80)
print("Creating BJM spatial triangle model (g=40, zi=False)")
print("="*80)

model = gv.bjm_spatial_triangle(g=40, zi=False)
print(f"Original grid: {model.grid.len} alternatives")
print(f"Grid bounds: x=[{model.grid.x0}, {model.grid.x1}], y=[{model.grid.y0}, {model.grid.y1}]")
print(f"Grid steps: xstep={model.grid.xstep}, ystep={model.grid.ystep}")

# Create coarsened model
print("\n" + "="*80)
print("Creating coarsened model (2x spacing)")
print("="*80)

coarse_grid = gv.Grid(
    x0=model.grid.x0,
    x1=model.grid.x1,
    xstep=2 * model.grid.xstep,
    y0=model.grid.y0,
    y1=model.grid.y1,
    ystep=2 * model.grid.ystep
)

coarse_model = gv.SpatialVotingModel(
    voter_ideal_points=model.voter_ideal_points,
    grid=coarse_grid,
    number_of_voters=model.number_of_voters,
    majority=model.majority,
    zi=model.zi,
    distance_measure=model.distance_measure
)

print(f"Coarsened grid: {coarse_model.grid.len} alternatives")
print(f"Ratio: {model.grid.len / coarse_model.grid.len:.2f}x")

# Solve coarsened model
print("\n" + "="*80)
print("Solving coarsened model...")
print("="*80)

start = time.time()
coarse_model.analyze(solver="full_matrix_inversion")
solve_time = time.time() - start

print(f"Solved in {solve_time:.3f}s")
print(f"Stationary distribution sum: {coarse_model.stationary_distribution.sum():.10f}")

coarse_dist = coarse_model.stationary_distribution

# ============================================================================
# METHOD A: 4-Nearest Neighbor Averaging
# ============================================================================

print("\n" + "="*80)
print("METHOD A: 4-Nearest Neighbor Averaging")
print("="*80)

def interpolate_4neighbor(fine_grid, coarse_grid, coarse_dist):
    """Interpolate using 4-nearest neighbor averaging."""
    result = jnp.zeros(fine_grid.len)
    
    for i in range(fine_grid.len):
        x, y = fine_grid.x[i], fine_grid.y[i]
        
        # Try to find exact match in coarse grid
        try:
            coarse_idx = coarse_grid.index(x=x, y=y, tolerance=1e-9)
            result = result.at[i].set(coarse_dist[coarse_idx])
        except ValueError:
            # Point not in coarse grid - find neighbors
            # Find nearest coarse grid points
            x_coarse = coarse_grid.x
            y_coarse = coarse_grid.y
            
            # Find neighbors (up to 4)
            neighbors = []
            neighbor_probs = []
            
            # Try all 4 directions
            for dx in [-coarse_grid.xstep, coarse_grid.xstep]:
                try:
                    idx = coarse_grid.index(x=x+dx, y=y, tolerance=1e-9)
                    neighbors.append(idx)
                    neighbor_probs.append(coarse_dist[idx])
                except ValueError:
                    pass
            
            for dy in [-coarse_grid.ystep, coarse_grid.ystep]:
                try:
                    idx = coarse_grid.index(x=x, y=y+dy, tolerance=1e-9)
                    neighbors.append(idx)
                    neighbor_probs.append(coarse_dist[idx])
                except ValueError:
                    pass
            
            if neighbor_probs:
                avg_prob = jnp.mean(jnp.array(neighbor_probs))
                result = result.at[i].set(avg_prob)
    
    # Normalize
    result = result / result.sum()
    return result

start = time.time()
result_a = interpolate_4neighbor(model.grid, coarse_model.grid, coarse_dist)
time_a = time.time() - start

print(f"Time: {time_a:.3f}s")
print(f"Result sum: {result_a.sum():.10f}")
print(f"Non-zero elements: {jnp.count_nonzero(result_a)}")

# ============================================================================
# METHOD B: JAX map_coordinates
# ============================================================================

print("\n" + "="*80)
print("METHOD B: jax.scipy.ndimage.map_coordinates")
print("="*80)

def interpolate_jax(fine_grid, coarse_grid, coarse_dist):
    """Interpolate using JAX's map_coordinates."""
    from jax.scipy import ndimage
    
    # Reshape coarse distribution to 2D grid
    coarse_shape = coarse_grid.shape()
    coarse_2d = coarse_dist.reshape(coarse_shape)
    
    # Create coordinate arrays for fine grid
    # Map fine grid coordinates to coarse grid indices
    fine_coords = []
    for i in range(fine_grid.len):
        x, y = fine_grid.x[i], fine_grid.y[i]
        
        # Convert to coarse grid indices (row, col)
        col = (x - coarse_grid.x0) / coarse_grid.xstep
        row = (coarse_grid.y1 - y) / coarse_grid.ystep
        
        fine_coords.append([row, col])
    
    fine_coords = jnp.array(fine_coords).T  # Shape: (2, fine_grid.len)
    
    # Interpolate
    result = ndimage.map_coordinates(coarse_2d, fine_coords, order=1, mode='nearest')
    
    # Normalize
    result = result / result.sum()
    return result

start = time.time()
result_b = interpolate_jax(model.grid, coarse_model.grid, coarse_dist)
time_b = time.time() - start

print(f"Time: {time_b:.3f}s")
print(f"Result sum: {result_b.sum():.10f}")
print(f"Non-zero elements: {jnp.count_nonzero(result_b)}")

# ============================================================================
# METHOD C: Matrix-based interpolation
# ============================================================================

print("\n" + "="*80)
print("METHOD C: Matrix-based interpolation")
print("="*80)

def create_interpolation_matrix(fine_grid, coarse_grid):
    """
    Create interpolation matrix C of shape (fine_grid.len, coarse_grid.len).
    
    C[i, j] represents the weight of coarse grid point j for fine grid point i.
    Values are 0, 0.25, 0.5, or 1.0 depending on the interpolation scheme.
    """
    n_fine = fine_grid.len
    n_coarse = coarse_grid.len
    
    # Use lists to build sparse matrix
    rows = []
    cols = []
    data = []
    
    for i in range(n_fine):
        x, y = fine_grid.x[i], fine_grid.y[i]
        
        # Try exact match first
        try:
            coarse_idx = coarse_grid.index(x=x, y=y, tolerance=1e-9)
            rows.append(i)
            cols.append(coarse_idx)
            data.append(1.0)
        except ValueError:
            # Find neighbors
            neighbors = []
            
            # Try all 4 directions
            for dx in [-coarse_grid.xstep, coarse_grid.xstep]:
                try:
                    idx = coarse_grid.index(x=x+dx, y=y, tolerance=1e-9)
                    neighbors.append(idx)
                except ValueError:
                    pass
            
            for dy in [-coarse_grid.ystep, coarse_grid.ystep]:
                try:
                    idx = coarse_grid.index(x=x, y=y+dy, tolerance=1e-9)
                    neighbors.append(idx)
                except ValueError:
                    pass
            
            # Average neighbors
            if neighbors:
                weight = 1.0 / len(neighbors)
                for neighbor_idx in neighbors:
                    rows.append(i)
                    cols.append(neighbor_idx)
                    data.append(weight)
    
    # Convert to JAX array (dense for now, could use sparse)
    C = jnp.zeros((n_fine, n_coarse))
    for r, c, d in zip(rows, cols, data):
        C = C.at[r, c].set(C[r, c] + d)
    
    return C

def interpolate_matrix(fine_grid, coarse_grid, coarse_dist):
    """Interpolate using matrix multiplication."""
    C = create_interpolation_matrix(fine_grid, coarse_grid)
    result = C @ coarse_dist
    # Normalize
    result = result / result.sum()
    return result, C

start = time.time()
result_c, C_matrix = interpolate_matrix(model.grid, coarse_model.grid, coarse_dist)
time_c = time.time() - start

print(f"Time (including matrix creation): {time_c:.3f}s")
print(f"Result sum: {result_c.sum():.10f}")
print(f"Non-zero elements: {jnp.count_nonzero(result_c)}")
print(f"Matrix shape: {C_matrix.shape}")
print(f"Matrix non-zero elements: {jnp.count_nonzero(C_matrix)}")
print(f"Matrix sparsity: {100 * (1 - jnp.count_nonzero(C_matrix) / C_matrix.size):.1f}%")

# Test reusing matrix
start = time.time()
result_c_reuse = C_matrix @ coarse_dist
result_c_reuse = result_c_reuse / result_c_reuse.sum()
time_c_reuse = time.time() - start
print(f"Time (reusing matrix): {time_c_reuse:.3f}s")

# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\nMethod A (4-neighbor):     {time_a:.4f}s")
print(f"Method B (JAX):            {time_b:.4f}s")
print(f"Method C (matrix):         {time_c:.4f}s (first time)")
print(f"Method C (matrix reuse):   {time_c_reuse:.4f}s")

print(f"\nFastest: ", end="")
times = {'A': time_a, 'B': time_b, 'C': time_c_reuse}
fastest = min(times, key=times.get)
print(f"Method {fastest} ({times[fastest]:.4f}s)")

# Check if results are similar
print(f"\nL1 difference (A vs B): {jnp.sum(jnp.abs(result_a - result_b)):.6e}")
print(f"L1 difference (A vs C): {jnp.sum(jnp.abs(result_a - result_c)):.6e}")
print(f"L1 difference (B vs C): {jnp.sum(jnp.abs(result_b - result_c)):.6e}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

if fastest == 'C':
    print("Method C (matrix-based) is fastest when reusing the matrix.")
    print("This is ideal for outline-based solvers since the matrix can be")
    print("created once and reused across multiple solver calls.")
elif fastest == 'B':
    print("Method B (JAX map_coordinates) is fastest.")
    print("This uses JAX's optimized interpolation routines.")
else:
    print("Method A (4-neighbor averaging) is fastest.")
    print("This uses simple neighbor averaging with explicit loops.")

print("\nRecommendation: Use Method " + fastest)
