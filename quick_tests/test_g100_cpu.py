"""Test g=100 BJM spatial triangle with CPU - FIXED subgrid comparison."""

import sys
sys.path.insert(0, '/home/paul/gridvoting-jax/gridvoting-jax/src')

# Force CPU usage (CPU parallelization is now auto-configured in core.py)
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import gridvoting_jax as gv
import jax.numpy as jnp
import time

print("=" * 70)
print("g=100 BJM Spatial Triangle - CPU Mode (Optimized)")
print("=" * 70)

print("\n1. Creating g=100 BJM spatial triangle model...")
model_100 = gv.bjm_spatial_triangle(g=100, zi=False)
print(f"   ✓ N = {model_100.model.number_of_feasible_alternatives}")

print("\n2. Solving with lazy grid upscaling (CPU)...")
start = time.time()
model_100.analyze(solver="lazy_grid_upscaling", max_iterations=3000)
solve_time = time.time() - start
print(f"   ✓ Time: {solve_time:.2f}s")
print(f"   ✓ Sum: {jnp.sum(model_100.stationary_distribution):.6f}")

print("\n3. Loading g=80 reference for subgrid validation...")
model_80 = gv.bjm_spatial_triangle(g=80, zi=False)
model_80.analyze(solver="lazy_grid_upscaling", max_iterations=3000)
print(f"   ✓ g=80 solved (N={model_80.model.number_of_feasible_alternatives})")

print("\n4. Extracting g=80 subgrid from g=100...")
# g=80 grid: -40 to 40 in steps of 1
# g=100 grid: -50 to 50 in steps of 1
grid_100 = model_100.grid
grid_80 = model_80.grid

# Find g=80 box within g=100
x0, x1 = -40.0, 40.0
y0, y1 = -40.0, 40.0
box_mask_100 = grid_100.within_box(x0=x0, x1=x1, y0=y0, y1=y1)

# Extract subgrid probabilities from g=100
statd_100_subgrid = model_100.stationary_distribution[box_mask_100]
statd_100_outside = jnp.sum(model_100.stationary_distribution[~box_mask_100])

print(f"   ✓ g=80 subgrid in g=100: {jnp.sum(box_mask_100)} points")
print(f"   ✓ Probability in subgrid: {jnp.sum(statd_100_subgrid):.6f}")
print(f"   ✓ Probability outside subgrid: {statd_100_outside:.6f}")
print(f"   ✓ Outside/Total ratio: {statd_100_outside:.2%}")

print("\n5. Comparing g=100 subgrid with g=80...")
# The g=80 model has all 25921 points, but we need to compare with the same subgrid
# Actually, for g=80, ALL points are within the -40 to 40 box, so we compare directly
print(f"   ✓ g=80 stationary distribution shape: {model_80.stationary_distribution.shape}")
print(f"   ✓ g=100 subgrid shape: {statd_100_subgrid.shape}")

# The shapes should match if the grids are aligned correctly
# g=80: 81x81 grid from -40 to 40 = 6561 points (after removing invalid)
# But model_80 shows 25921 points, which suggests it includes more points

# Let's check the actual grid sizes
print(f"   ✓ g=80 grid len: {grid_80.len}")
print(f"   ✓ g=100 grid len: {grid_100.len}")

# For proper comparison, we need to find which points in g=80 correspond to the subgrid
# The g=80 grid IS the subgrid, so all points should match
# But the issue is that g=80 has 161x161 = 25921 points, not 81x81

# Actually, g=80 means 80 steps on each side, so -80 to 80 in steps of 2
# Let me recalculate the proper box
print(f"\n   Recalculating proper subgrid bounds...")
print(f"   g=80 grid: x range [{grid_80.x.min():.1f}, {grid_80.x.max():.1f}]")
print(f"   g=80 grid: y range [{grid_80.y.min():.1f}, {grid_80.y.max():.1f}]")
print(f"   g=100 grid: x range [{grid_100.x.min():.1f}, {grid_100.x.max():.1f}]")
print(f"   g=100 grid: y range [{grid_100.y.min():.1f}, {grid_100.y.max():.1f}]")

# Use the actual g=80 grid bounds
x0_actual = grid_80.x.min()
x1_actual = grid_80.x.max()
y0_actual = grid_80.y.min()
y1_actual = grid_80.y.max()

box_mask_100_actual = grid_100.within_box(x0=x0_actual, x1=x1_actual, y0=y0_actual, y1=y1_actual)
statd_100_subgrid_actual = model_100.stationary_distribution[box_mask_100_actual]
statd_100_outside_actual = jnp.sum(model_100.stationary_distribution[~box_mask_100_actual])

print(f"\n6. Corrected subgrid extraction:")
print(f"   ✓ Subgrid size: {jnp.sum(box_mask_100_actual)} points")
print(f"   ✓ Probability in subgrid: {jnp.sum(statd_100_subgrid_actual):.6f}")
print(f"   ✓ Probability outside: {statd_100_outside_actual:.6f} ({statd_100_outside_actual:.2%})")

if statd_100_subgrid_actual.shape == model_80.stationary_distribution.shape:
    diff_subgrid = jnp.abs(statd_100_subgrid_actual - model_80.stationary_distribution)
    max_diff = jnp.max(diff_subgrid)
    
    # Relative differences
    rel_diff = diff_subgrid / (model_80.stationary_distribution + 1e-10)
    max_rel_diff = jnp.max(rel_diff)
    mean_rel_diff = jnp.mean(rel_diff)
    
    print(f"\n7. Comparison:")
    print(f"   ✓ Max absolute difference: {max_diff:.2e}")
    print(f"   ✓ Max relative difference: {max_rel_diff:.2%}")
    print(f"   ✓ Mean relative difference: {mean_rel_diff:.2%}")
    
    print(f"\n8. Validation:")
    if statd_100_outside_actual < 0.01:
        print(f"   ✅ Probability outside << 1%: PASS")
    
    if max_rel_diff < 0.01:
        print(f"   ✅ Subgrid values within 1% of g=80: PASS")
    else:
        print(f"   ⚠️  Max relative difference: {max_rel_diff:.2%}")
else:
    print(f"   ❌ Shape mismatch: {statd_100_subgrid_actual.shape} vs {model_80.stationary_distribution.shape}")

print("\n" + "=" * 70)
print("g=100 CPU TEST COMPLETE")
print("=" * 70)
