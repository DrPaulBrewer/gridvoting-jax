"""Test g=100 outside probability with BOTH solvers (float64, CPU).

Compares:
1. GMRES with grid upscaling (concentrated initial guess)
2. Power method (uniform initial guess)

This will reveal if the ~1.5e-09 outside probability is accurate or solver-dependent.
"""

import sys
sys.path.insert(0, '/home/paul/gridvoting-jax/gridvoting-jax/src')

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
jax.config.update('jax_enable_x64', True)

import gridvoting_jax as gv
from gridvoting_jax.benchmarks.osf_comparison import load_osf_distribution
import jax.numpy as jnp
import numpy as np

print("=" * 80)
print("g=100 Outside Probability Comparison - GMRES vs Power Method")
print("=" * 80)

# Expected from exponential decay model
expected_outside = 7.096940e-07

print(f"\nTheoretical estimate (exp decay): {expected_outside:.10e}")

# Test 1: GMRES with grid upscaling
print("\n" + "=" * 80)
print("TEST 1: GMRES with Grid Upscaling")
print("=" * 80)

print("\n1. Solving g=100 with GMRES + grid upscaling...")
model_gmres = gv.bjm_spatial_triangle(g=100, zi=False)
model_gmres.analyze(solver="lazy_grid_upscaling", max_iterations=3000)

# Calculate outside probability
grid_100 = model_gmres.grid
box_mask = grid_100.within_box(x0=-80, x1=80, y0=-80, y1=80)
outside_gmres = float(jnp.sum(model_gmres.stationary_distribution[~box_mask]))

print(f"\n2. GMRES Results:")
print(f"   Probability outside g=80 box: {outside_gmres:.15e}")
print(f"   Ratio to theoretical:         {outside_gmres/expected_outside:.6f}")
print(f"   Probability sum:              {jnp.sum(model_gmres.stationary_distribution):.15f}")

# Test 2: Power method
print("\n" + "=" * 80)
print("TEST 2: Power Method (Uniform Start)")
print("=" * 80)

print("\n1. Solving g=100 with power method...")
model_power = gv.bjm_spatial_triangle(g=100, zi=False)
model_power.analyze_lazy(force_lazy=True, solver="power_method", 
                        max_iterations=10000, timeout=600)

# Calculate outside probability
outside_power = float(jnp.sum(model_power.stationary_distribution[~box_mask]))

print(f"\n2. Power Method Results:")
print(f"   Probability outside g=80 box: {outside_power:.15e}")
print(f"   Ratio to theoretical:         {outside_power/expected_outside:.6f}")
print(f"   Probability sum:              {jnp.sum(model_power.stationary_distribution):.15f}")

# Comparison
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

print(f"\nOutside Probability:")
print(f"  Theoretical (exp decay): {expected_outside:.15e}")
print(f"  GMRES (grid upscaling):  {outside_gmres:.15e}  ({outside_gmres/expected_outside:.4f}x)")
print(f"  Power method (uniform):  {outside_power:.15e}  ({outside_power/expected_outside:.4f}x)")
print(f"\nRatio (power/GMRES): {outside_power/outside_gmres:.4f}x")

# Analyze difference
diff = abs(outside_power - outside_gmres)
print(f"\nAbsolute difference: {diff:.15e}")

if abs(outside_power/outside_gmres - 1.0) < 0.1:
    print("\n✅ Both solvers agree within 10% - values are consistent")
elif outside_power > outside_gmres * 10:
    print(f"\n⚠️  Power method shows {outside_power/outside_gmres:.1f}x more probability outside")
    print("   This suggests grid upscaling suppresses outer regions")
elif outside_gmres > outside_power * 10:
    print(f"\n⚠️  GMRES shows {outside_gmres/outside_power:.1f}x more probability outside")
    print("   Unexpected - power method should explore more")
else:
    print(f"\n⚠️  Solvers differ by {abs(outside_power/outside_gmres - 1.0)*100:.1f}%")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
