"""
Benchmark actual pattern-based matrix creation time.
"""

import time
import jax.numpy as jnp
import sys
sys.path.insert(0, '/home/paul/gridvoting-jax/gridvoting-jax/src')

import gridvoting_jax as gv
from gridvoting_jax.models.spatial import create_outline_interpolation_matrix

# Test at g=40 (like in benchmarks)
print("Benchmarking pattern-based matrix creation at g=40")
print("="*80)

model = gv.bjm_spatial_triangle(g=40, zi=False)
coarse_grid = gv.Grid(
    x0=model.grid.x0,
    x1=model.grid.x1,
    xstep=2 * model.grid.xstep,
    y0=model.grid.y0,
    y1=model.grid.y1,
    ystep=2 * model.grid.ystep
)

print(f"Fine grid: {model.grid.len} points")
print(f"Coarse grid: {coarse_grid.len} points")

# Time matrix creation
start = time.time()
C = create_outline_interpolation_matrix(model.grid, coarse_grid)
matrix_time = time.time() - start

print(f"\nPattern-based matrix creation: {matrix_time:.3f}s")
print(f"Matrix shape: {C.shape}")
print(f"Non-zero elements: {C.nse}")

# Now time the three solvers
print("\n" + "="*80)
print("Timing all three outline solvers")
print("="*80)

for solver in ["outline_and_fill", "outline_and_power", "outline_and_gmres"]:
    test_model = gv.bjm_spatial_triangle(g=40, zi=False)
    
    start = time.time()
    test_model.analyze(solver=solver, tolerance=1e-6, max_iterations=5000)
    total_time = time.time() - start
    
    print(f"\n{solver}: {total_time:.3f}s")
    print(f"  Distribution sum: {test_model.stationary_distribution.sum():.10f}")
