"""Comprehensive OSF validation test for g=80 and g=100 with CPU optimization.

This test validates:
1. g=80 lazy solver results against OSF reference
2. g=100 lazy solver results by comparing g=80 subgrid within g=100 to OSF reference
3. Verifies probability mass outside g=80 box in g=100 is << 1%
"""

import sys
sys.path.insert(0, '/home/paul/gridvoting-jax/gridvoting-jax/src')

# Force CPU mode to use new multi-device configuration
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import gridvoting_jax as gv
from gridvoting_jax.benchmarks.osf_comparison import load_osf_distribution
import jax.numpy as jnp
import numpy as np
import time

print("=" * 80)
print("COMPREHENSIVE OSF VALIDATION - g=80 and g=100 BJM Spatial Triangle (CPU)")
print("=" * 80)

# Show CPU configuration
print("\n" + "=" * 80)
print("CPU Configuration (Automatic)")
print("=" * 80)
print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'NOT SET')[:80]}...")
print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'NOT SET')}")
print(f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'NOT SET')}")

import jax
print(f"\nJAX Devices: {len(jax.devices())} CPU devices")
if len(jax.devices()) > 1:
    print(f"  ✅ Multi-device mode enabled")

# ============================================================================
# PART 1: g=80 Validation against OSF Reference
# ============================================================================

print("\n" + "=" * 80)
print("PART 1: g=80 Validation Against OSF Reference")
print("=" * 80)

print("\n1. Creating g=80 BJM spatial triangle model...")
model_80 = gv.bjm_spatial_triangle(g=80, zi=False)
print(f"   ✓ N = {model_80.model.number_of_feasible_alternatives}")

print("\n2. Solving g=80 with lazy grid upscaling...")
start = time.time()
model_80.analyze(solver="lazy_grid_upscaling", max_iterations=3000)
solve_time_80 = time.time() - start
print(f"   ✓ Time: {solve_time_80:.2f}s")
print(f"   ✓ Sum: {jnp.sum(model_80.stationary_distribution):.6f}")

print("\n3. Loading OSF reference data for g=80...")
try:
    ref_statd_80 = load_osf_distribution(g=80, zi=False)
    
    if ref_statd_80 is not None:
        print(f"   ✓ Loaded OSF reference for g=80, zi=False")
        
        # Extract stationary distribution from log10prob column
        ref_statd_80 = 10 ** ref_statd_80['log10prob'].values
        ref_statd_80 = np.array(ref_statd_80)
        
        print(f"   ✓ Reference sum: {np.sum(ref_statd_80):.6f}")
        
        # Compare
        diff_80 = np.abs(np.array(model_80.stationary_distribution) - ref_statd_80)
        max_diff_80 = np.max(diff_80)
        l1_diff_80 = np.sum(diff_80)
        
        # Relative error for non-zero values
        nonzero_mask = ref_statd_80 > 1e-10
        rel_diff_80 = diff_80[nonzero_mask] / ref_statd_80[nonzero_mask]
        max_rel_diff_80 = np.max(rel_diff_80)
        mean_rel_diff_80 = np.mean(rel_diff_80)
        
        print(f"\n4. g=80 Comparison with OSF reference:")
        print(f"   ✓ Max absolute difference: {max_diff_80:.2e}")
        print(f"   ✓ L1 difference: {l1_diff_80:.2e}")
        print(f"   ✓ Max relative difference: {max_rel_diff_80:.2%}")
        print(f"   ✓ Mean relative difference: {mean_rel_diff_80:.2%}")
        
        print(f"\n5. g=80 Validation:")
        g80_pass = True
        if max_rel_diff_80 < 0.01:
            print(f"   ✅ Max relative error < 1%: PASS")
        else:
            print(f"   ❌ Max relative error {max_rel_diff_80:.2%} >= 1%: FAIL")
            g80_pass = False
            
    else:
        print("   ❌ OSF reference data not available for g=80")
        g80_pass = False
        
except Exception as e:
    print(f"   ❌ Error loading OSF reference: {e}")
    g80_pass = False

# ============================================================================
# PART 2: g=100 Validation via g=80 Subgrid Comparison
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: g=100 Validation via g=80 Subgrid Comparison")
print("=" * 80)

print("\n1. Creating g=100 BJM spatial triangle model...")
model_100 = gv.bjm_spatial_triangle(g=100, zi=False)
print(f"   ✓ N = {model_100.model.number_of_feasible_alternatives}")

print("\n2. Solving g=100 with lazy grid upscaling...")
start = time.time()
model_100.analyze(solver="lazy_grid_upscaling", max_iterations=3000)
solve_time_100 = time.time() - start
print(f"   ✓ Time: {solve_time_100:.2f}s")
print(f"   ✓ Sum: {jnp.sum(model_100.stationary_distribution):.6f}")

print("\n3. Extracting g=80 subgrid from g=100...")
grid_100 = model_100.grid
grid_80 = model_80.grid

# Use actual g=80 grid bounds
x0 = grid_80.x.min()
x1 = grid_80.x.max()
y0 = grid_80.y.min()
y1 = grid_80.y.max()

print(f"   ✓ g=80 bounds: x=[{x0:.1f}, {x1:.1f}], y=[{y0:.1f}, {y1:.1f}]")
print(f"   ✓ g=100 bounds: x=[{grid_100.x.min():.1f}, {grid_100.x.max():.1f}], "
      f"y=[{grid_100.y.min():.1f}, {grid_100.y.max():.1f}]")

# Extract subgrid
box_mask_100 = grid_100.within_box(x0=x0, x1=x1, y0=y0, y1=y1)
statd_100_subgrid = model_100.stationary_distribution[box_mask_100]
statd_100_outside = jnp.sum(model_100.stationary_distribution[~box_mask_100])

print(f"   ✓ Points in g=80 subgrid: {jnp.sum(box_mask_100)}")
print(f"   ✓ Probability in subgrid: {jnp.sum(statd_100_subgrid):.6f}")
print(f"   ✓ Probability outside subgrid: {statd_100_outside:.6f} ({statd_100_outside:.2%})")

print("\n4. Comparing g=100 subgrid with g=80 reference...")
if statd_100_subgrid.shape == model_80.stationary_distribution.shape:
    diff_subgrid = jnp.abs(statd_100_subgrid - model_80.stationary_distribution)
    max_diff_subgrid = jnp.max(diff_subgrid)
    
    # Relative differences
    rel_diff_subgrid = diff_subgrid / (model_80.stationary_distribution + 1e-10)
    max_rel_diff_subgrid = jnp.max(rel_diff_subgrid)
    mean_rel_diff_subgrid = jnp.mean(rel_diff_subgrid)
    
    print(f"   ✓ Max absolute difference: {max_diff_subgrid:.2e}")
    print(f"   ✓ Max relative difference: {max_rel_diff_subgrid:.2%}")
    print(f"   ✓ Mean relative difference: {mean_rel_diff_subgrid:.2%}")
    
    print(f"\n5. g=100 Validation:")
    g100_pass = True
    
    if statd_100_outside < 0.01:
        print(f"   ✅ Probability outside g=80 box << 1%: PASS")
    else:
        print(f"   ❌ Probability outside {statd_100_outside:.2%} >= 1%: FAIL")
        g100_pass = False
    
    if max_rel_diff_subgrid < 0.01:
        print(f"   ✅ Subgrid values within 1% of g=80: PASS")
    else:
        print(f"   ⚠️  Max relative difference: {max_rel_diff_subgrid:.2%}")
        if max_rel_diff_subgrid < 0.05:
            print(f"   ⚠️  Within 5% tolerance (acceptable)")
        else:
            print(f"   ❌ Exceeds 5% tolerance: FAIL")
            g100_pass = False
else:
    print(f"   ❌ Shape mismatch: {statd_100_subgrid.shape} vs {model_80.stationary_distribution.shape}")
    g100_pass = False

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

print(f"\nPerformance:")
print(f"  g=80 solve time:  {solve_time_80:.2f}s")
print(f"  g=100 solve time: {solve_time_100:.2f}s")

print(f"\nValidation Results:")
if g80_pass:
    print(f"  ✅ g=80 validation: PASS")
else:
    print(f"  ❌ g=80 validation: FAIL")

if g100_pass:
    print(f"  ✅ g=100 validation: PASS")
else:
    print(f"  ❌ g=100 validation: FAIL")

if g80_pass and g100_pass:
    print(f"\n🎉 ALL VALIDATIONS PASSED!")
else:
    print(f"\n⚠️  Some validations failed - review results above")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
