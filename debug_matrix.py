"""
Check if BCOO matrix multiplication is the issue.
"""

import jax.numpy as jnp
from jax.experimental import sparse
import sys
sys.path.insert(0, '/home/paul/gridvoting-jax/gridvoting-jax/src')

from gridvoting_jax.models.spatial import create_outline_interpolation_matrix
from gridvoting_jax.spatial import Grid

g = 10
fine_grid = Grid(x0=-g, x1=g, xstep=1, y0=-g, y1=g, ystep=1)
coarse_grid = Grid(x0=-g, x1=g, xstep=2, y0=-g, y1=g, ystep=2)

C = create_outline_interpolation_matrix(fine_grid, coarse_grid)
coarse_dist = jnp.ones(coarse_grid.len) / coarse_grid.len

print(f"C type: {type(C)}")
print(f"C shape: {C.shape}")
print(f"coarse_dist shape: {coarse_dist.shape}")

# Try sparse multiplication
result_sparse = C @ coarse_dist
print(f"\nSparse result sum: {result_sparse.sum()}")

# Try dense multiplication
C_dense = C.todense()
result_dense = C_dense @ coarse_dist
print(f"Dense result sum: {result_dense.sum()}")

# Check if they're the same
print(f"\nResults equal? {jnp.allclose(result_sparse, result_dense)}")

# Check matrix properties
print(f"\nC.nse: {C.nse}")
print(f"C.data shape: {C.data.shape}")
print(f"C.indices shape: {C.indices.shape}")

# Check for duplicate indices
indices_as_tuples = [(int(C.indices[i, 0]), int(C.indices[i, 1])) for i in range(C.nse)]
unique_indices = set(indices_as_tuples)
print(f"\nTotal indices: {len(indices_as_tuples)}")
print(f"Unique indices: {len(unique_indices)}")
if len(indices_as_tuples) != len(unique_indices):
    print("WARNING: Duplicate indices found!")
    from collections import Counter
    counts = Counter(indices_as_tuples)
    duplicates = {k: v for k, v in counts.items() if v > 1}
    print(f"Duplicates: {list(duplicates.items())[:10]}")
