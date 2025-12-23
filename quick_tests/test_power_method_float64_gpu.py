"""Test lazy power method with float64 on GPU for g=80.

Based on BJM paper which used float64 and achieved convergence.
"""

import sys
sys.path.insert(0, '/home/paul/gridvoting-jax/gridvoting-jax/src')

import os
# Force GPU
os.environ.pop('JAX_PLATFORMS', None)

import jax
# Enable float64
jax.config.update('jax_enable_x64', True)

import gridvoting_jax as gv
from gridvoting_jax.benchmarks.osf_comparison import load_osf_distribution
import jax.numpy as jnp
import numpy as np

print("=" * 80)
print("Float64 Power Method Test on GPU - g=80")
print("=" * 80)

# Check GPU
print(f"\nDevice: {jax.devices()[0]}")
print(f"Float64 enabled: {jax.config.read('jax_enable_x64')}")

# Load OSF reference
print("\n1. Loading OSF reference data...")
ref_data = load_osf_distribution(g=80, zi=False)
if ref_data is None:
    print("   ❌ Could not load OSF data")
    sys.exit(1)

ref_statd = 10 ** ref_data['log10prob'].values
ref_statd = np.array(ref_statd, dtype=np.float64)
print(f"   ✓ Loaded {len(ref_statd)} points")

# Create model
print("\n2. Creating g=80 model (zi=False)...")
model_80 = gv.bjm_spatial_triangle(g=80, zi=False)
print(f"   ✓ N = {model_80.model.number_of_feasible_alternatives}")

# Solve with power method
print("\n3. Solving with lazy power method (float64, GPU)...")
print("   Timeout: 300s")
print("   Max iterations: 10000")

try:
    model_80.analyze_lazy(force_lazy=True, solver="power_method",
                         max_iterations=10000, timeout=300)
    print(f"   ✓ Solved successfully")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Check convergence
prob_sum = jnp.sum(model_80.stationary_distribution)
print(f"\n4. Results:")
print(f"   Probability sum: {prob_sum:.15f}")

# Compare with OSF
diff = np.abs(np.array(model_80.stationary_distribution) - ref_statd)
max_abs_diff = np.max(diff)
l1_diff = np.sum(diff)

print(f"\n5. Comparison with OSF reference:")
print(f"   Max absolute difference: {max_abs_diff:.10e}")
print(f"   L1 difference: {l1_diff:.10e}")

# Validate
if max_abs_diff < 1e-6:
    print(f"\n✅ PASS: Power method converged with float64!")
    print(f"   Accuracy is excellent (< 1e-6)")
else:
    print(f"\n⚠️  WARNING: Accuracy not as good as expected")
    print(f"   Max diff {max_abs_diff:.2e} >= 1e-6")

if abs(prob_sum - 1.0) > 1e-8:
    print(f"\n⚠️  WARNING: Probability sum {prob_sum:.10f} not close to 1.0")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
