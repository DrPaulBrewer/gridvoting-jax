"""
Test JAX sparse (BCOO) vs dense matrix for interpolation.

Compares performance and memory usage of sparse vs dense interpolation matrices.
"""

import jax
import jax.numpy as jnp
from jax.experimental import sparse
import time
import sys
sys.path.insert(0, '/home/paul/gridvoting-jax/gridvoting-jax/src')
import gridvoting_jax as gv

print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())

# Create BJM model with g=40
print("\n" + "="*80)
print("Creating BJM spatial triangle model (g=40)")
print("="*80)

model = gv.bjm_spatial_triangle(g=40, zi=False)
print(f"Original grid: {model.grid.len} alternatives")

# Create coarsened model
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
print("\nSolving coarsened model...")
coarse_model.analyze(solver="full_matrix_inversion")
coarse_dist = coarse_model.stationary_distribution
print(f"Solved. Distribution sum: {coarse_dist.sum():.10f}")

# ============================================================================
# Create interpolation matrix (dense)
# ============================================================================

print("\n" + "="*80)
print("Creating DENSE interpolation matrix")
print("="*80)

def create_dense_interpolation_matrix(fine_grid, coarse_grid):
    """Create dense interpolation matrix."""
    n_fine = fine_grid.len
    n_coarse = coarse_grid.len
    
    # Use lists to build matrix
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
            
            if neighbors:
                weight = 1.0 / len(neighbors)
                for neighbor_idx in neighbors:
                    rows.append(i)
                    cols.append(neighbor_idx)
                    data.append(weight)
    
    # Convert to dense JAX array
    C_dense = jnp.zeros((n_fine, n_coarse))
    for r, c, d in zip(rows, cols, data):
        C_dense = C_dense.at[r, c].set(C_dense[r, c] + d)
    
    return C_dense, rows, cols, data

start = time.time()
C_dense, rows, cols, data = create_dense_interpolation_matrix(model.grid, coarse_model.grid)
time_create_dense = time.time() - start

print(f"Creation time: {time_create_dense:.3f}s")
print(f"Matrix shape: {C_dense.shape}")
print(f"Non-zero elements: {jnp.count_nonzero(C_dense)}")
print(f"Sparsity: {100 * (1 - jnp.count_nonzero(C_dense) / C_dense.size):.2f}%")
print(f"Memory (approx): {C_dense.nbytes / 1024 / 1024:.2f} MB")

# Test dense multiplication
start = time.time()
result_dense = C_dense @ coarse_dist
result_dense = result_dense / result_dense.sum()
time_mult_dense = time.time() - start

print(f"Multiplication time: {time_mult_dense:.6f}s")
print(f"Result sum: {result_dense.sum():.10f}")

# ============================================================================
# Create interpolation matrix (sparse BCOO)
# ============================================================================

print("\n" + "="*80)
print("Creating SPARSE (BCOO) interpolation matrix")
print("="*80)

def create_sparse_interpolation_matrix(fine_grid, coarse_grid):
    """Create sparse BCOO interpolation matrix."""
    n_fine = fine_grid.len
    n_coarse = coarse_grid.len
    
    # Use lists to build matrix
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
            
            if neighbors:
                weight = 1.0 / len(neighbors)
                for neighbor_idx in neighbors:
                    rows.append(i)
                    cols.append(neighbor_idx)
                    data.append(weight)
    
    # Convert to BCOO sparse matrix
    indices = jnp.column_stack([jnp.array(rows), jnp.array(cols)])
    values = jnp.array(data)
    
    C_sparse = sparse.BCOO((values, indices), shape=(n_fine, n_coarse))
    
    return C_sparse

start = time.time()
C_sparse = create_sparse_interpolation_matrix(model.grid, coarse_model.grid)
time_create_sparse = time.time() - start

print(f"Creation time: {time_create_sparse:.3f}s")
print(f"Matrix shape: {C_sparse.shape}")
print(f"Non-zero elements: {C_sparse.nse}")
print(f"Memory (approx): {(C_sparse.data.nbytes + C_sparse.indices.nbytes) / 1024 / 1024:.2f} MB")

# Test sparse multiplication
start = time.time()
result_sparse = C_sparse @ coarse_dist
result_sparse = result_sparse / result_sparse.sum()
time_mult_sparse = time.time() - start

print(f"Multiplication time: {time_mult_sparse:.6f}s")
print(f"Result sum: {result_sparse.sum():.10f}")

# ============================================================================
# Test JIT compilation
# ============================================================================

print("\n" + "="*80)
print("Testing JIT compilation")
print("="*80)

@jax.jit
def sparse_multiply_jit(C_sparse, vec):
    result = C_sparse @ vec
    return result / result.sum()

@jax.jit
def dense_multiply_jit(C_dense, vec):
    result = C_dense @ vec
    return result / result.sum()

# Warm up JIT
print("Warming up JIT...")
_ = sparse_multiply_jit(C_sparse, coarse_dist)
_ = dense_multiply_jit(C_dense, coarse_dist)

# Time JIT sparse
start = time.time()
result_sparse_jit = sparse_multiply_jit(C_sparse, coarse_dist)
time_mult_sparse_jit = time.time() - start

print(f"Sparse JIT multiplication time: {time_mult_sparse_jit:.6f}s")

# Time JIT dense
start = time.time()
result_dense_jit = dense_multiply_jit(C_dense, coarse_dist)
time_mult_dense_jit = time.time() - start

print(f"Dense JIT multiplication time: {time_mult_dense_jit:.6f}s")

# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\nMatrix Creation:")
print(f"  Dense:  {time_create_dense:.3f}s")
print(f"  Sparse: {time_create_sparse:.3f}s")

print(f"\nMatrix Multiplication (no JIT):")
print(f"  Dense:  {time_mult_dense:.6f}s")
print(f"  Sparse: {time_mult_sparse:.6f}s")

print(f"\nMatrix Multiplication (with JIT):")
print(f"  Dense:  {time_mult_dense_jit:.6f}s")
print(f"  Sparse: {time_mult_sparse_jit:.6f}s")

print(f"\nMemory Usage:")
dense_mem = C_dense.nbytes / 1024 / 1024
sparse_mem = (C_sparse.data.nbytes + C_sparse.indices.nbytes) / 1024 / 1024
print(f"  Dense:  {dense_mem:.2f} MB")
print(f"  Sparse: {sparse_mem:.2f} MB")
print(f"  Savings: {100 * (1 - sparse_mem / dense_mem):.1f}%")

print(f"\nAccuracy:")
print(f"  L1 diff (dense vs sparse): {jnp.sum(jnp.abs(result_dense - result_sparse)):.6e}")
print(f"  L1 diff (dense vs sparse JIT): {jnp.sum(jnp.abs(result_dense - result_sparse_jit)):.6e}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

if time_mult_sparse_jit < time_mult_dense_jit:
    speedup = time_mult_dense_jit / time_mult_sparse_jit
    print(f"Sparse BCOO is {speedup:.1f}x faster than dense (with JIT)")
    print("Recommendation: Use sparse BCOO format")
else:
    slowdown = time_mult_sparse_jit / time_mult_dense_jit
    print(f"Dense is {slowdown:.1f}x faster than sparse BCOO (with JIT)")
    print("Recommendation: Use dense format despite higher memory usage")

print(f"\nMemory savings with sparse: {100 * (1 - sparse_mem / dense_mem):.1f}%")
