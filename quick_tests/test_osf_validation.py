"""Validate lazy solvers against OSF reference data for BJM spatial triangle.

This test:
1. Compares g=80 lazy solver results against OSF reference
2. Validates precision of lazy grid upscaling
"""

import sys
sys.path.insert(0, '/home/paul/gridvoting-jax/gridvoting-jax/src')

import gridvoting_jax as gv
from gridvoting_jax.benchmarks.osf_comparison import load_osf_distribution
import jax.numpy as jnp
import numpy as np
import time

print("=" * 70)
print("OSF REFERENCE VALIDATION TEST - g=80 BJM Spatial Triangle")
print("=" * 70)

# Create g=80 BJM spatial triangle model
print("\n1. Creating g=80 BJM spatial triangle model...")
model_80 = gv.bjm_spatial_triangle(g=80, zi=False)
print(f"   ✓ N = {model_80.model.number_of_feasible_alternatives}")

# Solve with lazy grid upscaling
print("\n2. Solving with lazy grid upscaling...")
start = time.time()
model_80.analyze(solver="lazy_grid_upscaling", max_iterations=3000)
solve_time = time.time() - start
print(f"   ✓ Time: {solve_time:.2f}s")
print(f"   ✓ Sum: {jnp.sum(model_80.stationary_distribution):.6f}")

# Load OSF reference
print("\n3. Loading OSF reference data...")
try:
    ref_statd = load_osf_distribution(g=80, zi=False)
    
    if ref_statd is not None:
        print(f"   ✓ Loaded OSF reference for g=80, zi=False")
        print(f"   ✓ Reference shape: {ref_statd.shape}")
        print(f"   ✓ Reference columns: {list(ref_statd.columns)}")
        
        # Extract stationary distribution from log10prob column
        ref_statd = 10 ** ref_statd['log10prob'].values
        ref_statd = np.array(ref_statd)
        
        print(f"   ✓ Extracted probabilities from log10prob")
        print(f"   ✓ Reference sum: {np.sum(ref_statd):.6f}")
        
        # Compare
        diff = np.abs(np.array(model_80.stationary_distribution) - ref_statd)
        max_diff = np.max(diff)
        l1_diff = np.sum(diff)
        mean_diff = np.mean(diff)
        
        # Find where max difference occurs
        max_idx = np.argmax(diff)
        lazy_val = model_80.stationary_distribution[max_idx]
        ref_val = ref_statd[max_idx]
        
        print(f"\n4. Comparison with OSF reference:")
        print(f"   ✓ Max absolute difference: {max_diff:.2e}")
        print(f"   ✓ Max diff location: index {max_idx}")
        print(f"   ✓   Lazy value: {lazy_val:.6e}")
        print(f"   ✓   OSF value: {ref_val:.6e}")
        print(f"   ✓ L1 (total variation) difference: {l1_diff:.2e}")
        print(f"   ✓ Mean absolute difference: {mean_diff:.2e}")
        
        # Relative error for non-zero values
        nonzero_mask = ref_statd > 1e-10
        rel_diff = diff[nonzero_mask] / ref_statd[nonzero_mask]
        max_rel_diff = np.max(rel_diff)
        mean_rel_diff = np.mean(rel_diff)
        
        print(f"   ✓ Max relative difference (nonzero): {max_rel_diff:.2%}")
        print(f"   ✓ Mean relative difference (nonzero): {mean_rel_diff:.2%}")
        
        # Validation thresholds
        print(f"\n5. Validation:")
        if max_diff < 1e-4:
            print(f"   ✅ Max difference < 1e-4: PASS")
        else:
            print(f"   ⚠️  Max difference {max_diff:.2e} >= 1e-4")
        
        if l1_diff < 1e-3:
            print(f"   ✅ L1 difference < 1e-3: PASS")
        else:
            print(f"   ⚠️  L1 difference {l1_diff:.2e} >= 1e-3")
        
        if max_rel_diff < 0.01:
            print(f"   ✅ Max relative error < 1%: PASS")
        else:
            print(f"   ⚠️  Max relative error {max_rel_diff:.2%} >= 1%")
            
    else:
        print("   ❌ OSF reference data not available for g=80, zi=False")
        
except Exception as e:
    print(f"   ❌ Error loading OSF reference: {e}")
    import traceback
    traceback.print_exc()

# Also test standard lazy GMRES for comparison
print("\n" + "=" * 70)
print("BONUS: Lazy GMRES (no initial guess) for comparison")
print("=" * 70)

print("\n1. Solving with lazy GMRES (no initial guess)...")
model_80_nogrid = gv.bjm_spatial_triangle(g=80, zi=False)
start = time.time()
model_80_nogrid.analyze_lazy(force_lazy=True, solver="gmres", max_iterations=3000)
nogrid_time = time.time() - start
print(f"   ✓ Time: {nogrid_time:.2f}s")
print(f"   ✓ Sum: {jnp.sum(model_80_nogrid.stationary_distribution):.6f}")

print(f"\n2. Grid upscaling speedup: {nogrid_time / solve_time:.2f}x")

# Compare grid upscaling vs no grid
diff_methods = jnp.abs(model_80.stationary_distribution - model_80_nogrid.stationary_distribution)
print(f"   ✓ Max difference (grid vs no-grid): {jnp.max(diff_methods):.2e}")

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
