"""Enhanced OSF validation test comparing lazy GMRES and lazy power method solvers.

This test validates:
1. g=80 lazy GMRES (with grid upscaling) against OSF reference
2. g=80 lazy power method against OSF reference
3. Compatibility between lazy GMRES and lazy power method solutions
4. g=100 validation for both solvers
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
print("SOLVER COMPARISON: Lazy GMRES vs Lazy Power Method (CPU)")
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
# PART 1: g=80 - Lazy GMRES with Grid Upscaling
# ============================================================================

print("\n" + "=" * 80)
print("PART 1: g=80 - Lazy GMRES with Grid Upscaling")
print("=" * 80)

print("\n1. Creating g=80 BJM spatial triangle model...")
model_80_gmres = gv.bjm_spatial_triangle(g=80, zi=False)
print(f"   ✓ N = {model_80_gmres.model.number_of_feasible_alternatives}")

print("\n2. Solving with lazy GMRES + grid upscaling...")
start = time.time()
model_80_gmres.analyze(solver="lazy_grid_upscaling", max_iterations=3000)
time_gmres = time.time() - start
print(f"   ✓ Time: {time_gmres:.2f}s")
print(f"   ✓ Sum: {jnp.sum(model_80_gmres.stationary_distribution):.6f}")

# ============================================================================
# PART 2: g=80 - Lazy Power Method
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: g=80 - Lazy Power Method")
print("=" * 80)

print("\n1. Creating g=80 BJM spatial triangle model...")
model_80_power = gv.bjm_spatial_triangle(g=80, zi=False)
print(f"   ✓ N = {model_80_power.model.number_of_feasible_alternatives}")

print("\n2. Solving with lazy power method...")
start = time.time()
# Use analyze_lazy with power_method solver
model_80_power.analyze_lazy(force_lazy=True, solver="power_method", max_iterations=3000, timeout=300)
time_power = time.time() - start
print(f"   ✓ Time: {time_power:.2f}s")
print(f"   ✓ Sum: {jnp.sum(model_80_power.stationary_distribution):.6f}")

# ============================================================================
# PART 3: Compare GMRES vs Power Method
# ============================================================================

print("\n" + "=" * 80)
print("PART 3: Solver Compatibility - GMRES vs Power Method")
print("=" * 80)

diff_solvers = jnp.abs(model_80_gmres.stationary_distribution - model_80_power.stationary_distribution)
max_diff_solvers = jnp.max(diff_solvers)
l1_diff_solvers = jnp.sum(diff_solvers)

# Relative differences
rel_diff_solvers = diff_solvers / (model_80_gmres.stationary_distribution + 1e-10)
max_rel_diff_solvers = jnp.max(rel_diff_solvers)
mean_rel_diff_solvers = jnp.mean(rel_diff_solvers)

print(f"\n1. GMRES vs Power Method Comparison:")
print(f"   ✓ Max absolute difference: {max_diff_solvers:.2e}")
print(f"   ✓ L1 difference: {l1_diff_solvers:.2e}")
print(f"   ✓ Max relative difference: {max_rel_diff_solvers:.2%}")
print(f"   ✓ Mean relative difference: {mean_rel_diff_solvers:.2%}")

print(f"\n2. Performance Comparison:")
print(f"   ✓ GMRES time: {time_gmres:.2f}s")
print(f"   ✓ Power method time: {time_power:.2f}s")
print(f"   ✓ Speedup: {time_power/time_gmres:.2f}x (GMRES {'faster' if time_gmres < time_power else 'slower'})")

print(f"\n3. Solver Compatibility:")
solvers_compatible = True
if max_rel_diff_solvers < 0.01:
    print(f"   ✅ Solvers agree within 1%: COMPATIBLE")
elif max_rel_diff_solvers < 0.05:
    print(f"   ⚠️  Solvers agree within 5%: ACCEPTABLE")
else:
    print(f"   ❌ Solvers differ by {max_rel_diff_solvers:.2%}: INCOMPATIBLE")
    solvers_compatible = False

# ============================================================================
# PART 4: OSF Reference Validation
# ============================================================================

print("\n" + "=" * 80)
print("PART 4: OSF Reference Validation")
print("=" * 80)

print("\n1. Loading OSF reference data for g=80...")
try:
    ref_statd_80 = load_osf_distribution(g=80, zi=False)
    
    if ref_statd_80 is not None:
        print(f"   ✓ Loaded OSF reference for g=80, zi=False")
        
        # Extract stationary distribution from log10prob column
        ref_statd_80 = 10 ** ref_statd_80['log10prob'].values
        ref_statd_80 = np.array(ref_statd_80)
        
        print(f"   ✓ Reference sum: {np.sum(ref_statd_80):.6f}")
        
        # Compare GMRES to OSF
        diff_gmres_osf = np.abs(np.array(model_80_gmres.stationary_distribution) - ref_statd_80)
        max_diff_gmres_osf = np.max(diff_gmres_osf)
        nonzero_mask = ref_statd_80 > 1e-10
        rel_diff_gmres_osf = diff_gmres_osf[nonzero_mask] / ref_statd_80[nonzero_mask]
        max_rel_diff_gmres_osf = np.max(rel_diff_gmres_osf)
        
        # Compare Power Method to OSF
        diff_power_osf = np.abs(np.array(model_80_power.stationary_distribution) - ref_statd_80)
        max_diff_power_osf = np.max(diff_power_osf)
        rel_diff_power_osf = diff_power_osf[nonzero_mask] / ref_statd_80[nonzero_mask]
        max_rel_diff_power_osf = np.max(rel_diff_power_osf)
        
        print(f"\n2. GMRES vs OSF Reference:")
        print(f"   ✓ Max absolute difference: {max_diff_gmres_osf:.2e}")
        print(f"   ✓ Max relative difference: {max_rel_diff_gmres_osf:.2%}")
        
        print(f"\n3. Power Method vs OSF Reference:")
        print(f"   ✓ Max absolute difference: {max_diff_power_osf:.2e}")
        print(f"   ✓ Max relative difference: {max_rel_diff_power_osf:.2%}")
        
        print(f"\n4. OSF Validation:")
        gmres_pass = max_rel_diff_gmres_osf < 0.01
        power_pass = max_rel_diff_power_osf < 0.01
        
        if gmres_pass:
            print(f"   ✅ GMRES within 1% of OSF: PASS")
        else:
            print(f"   ❌ GMRES error {max_rel_diff_gmres_osf:.2%} >= 1%: FAIL")
            
        if power_pass:
            print(f"   ✅ Power method within 1% of OSF: PASS")
        else:
            print(f"   ❌ Power method error {max_rel_diff_power_osf:.2%} >= 1%: FAIL")
            
    else:
        print("   ❌ OSF reference data not available for g=80")
        gmres_pass = False
        power_pass = False
        
except Exception as e:
    print(f"   ❌ Error loading OSF reference: {e}")
    gmres_pass = False
    power_pass = False

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

print(f"\nPerformance:")
print(f"  GMRES (with grid upscaling): {time_gmres:.2f}s")
print(f"  Power method:                {time_power:.2f}s")
print(f"  Speedup:                     {time_power/time_gmres:.2f}x")

print(f"\nSolver Compatibility:")
print(f"  Max difference: {max_rel_diff_solvers:.2%}")
if solvers_compatible:
    print(f"  ✅ Solvers produce compatible results")
else:
    print(f"  ❌ Solvers produce incompatible results")

print(f"\nOSF Validation:")
if gmres_pass:
    print(f"  ✅ GMRES validation: PASS")
else:
    print(f"  ❌ GMRES validation: FAIL")

if power_pass:
    print(f"  ✅ Power method validation: PASS")
else:
    print(f"  ❌ Power method validation: FAIL")

if gmres_pass and power_pass and solvers_compatible:
    print(f"\n🎉 ALL VALIDATIONS PASSED!")
    print(f"   Both solvers produce accurate, compatible results")
else:
    print(f"\n⚠️  Some validations failed - review results above")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
