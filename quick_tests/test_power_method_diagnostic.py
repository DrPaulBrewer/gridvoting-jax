"""Diagnostic test to understand why power method check_norm is inf."""

import sys
sys.path.insert(0, '/home/paul/gridvoting-jax/gridvoting-jax/src')

import os
os.environ.pop('JAX_PLATFORMS', None)

import jax
jax.config.update('jax_enable_x64', True)

import gridvoting_jax as gv
import jax.numpy as jnp
import numpy as np

print("=" * 80)
print("Power Method Diagnostic - Why is check_norm=inf?")
print("=" * 80)

# Create small model for testing
print("\n1. Creating g=20 model (small, for debugging)...")
model = gv.bjm_spatial_triangle(g=20, zi=False)
N = model.model.number_of_feasible_alternatives
print(f"   N = {N}")

# Get lazy transition matrix
from gridvoting_jax.dynamics.lazy.base import LazyTransitionMatrix
lazy_P = LazyTransitionMatrix(
    utility_functions=model.model.utility_functions,
    majority=model.model.majority,
    zi=model.model.zi,
    number_of_feasible_alternatives=N
)

print("\n2. Testing rmatvec_batched...")
# Start with uniform distribution
x = jnp.ones(N, dtype=jnp.float64) / N
print(f"   Initial x sum: {jnp.sum(x):.15f}")
print(f"   Initial x min/max: [{jnp.min(x):.10e}, {jnp.max(x):.10e}]")

# Apply one iteration
print("\n3. First iteration...")
Px = lazy_P.rmatvec_batched(x)
print(f"   Px sum: {jnp.sum(Px):.15f}")
print(f"   Px min/max: [{jnp.min(Px):.10e}, {jnp.max(Px):.10e}]")
print(f"   Has inf: {jnp.any(jnp.isinf(Px))}")
print(f"   Has nan: {jnp.any(jnp.isnan(Px))}")

# Calculate check norm
check_norm = float(jnp.sum(jnp.abs(Px - x)))
print(f"\n4. Check norm: {check_norm:.10e}")
print(f"   Is inf: {np.isinf(check_norm)}")

# Try a few more iterations
print("\n5. Running 10 iterations...")
x_iter = x
for i in range(10):
    Px_iter = lazy_P.rmatvec_batched(x_iter)
    check_norm_iter = float(jnp.sum(jnp.abs(Px_iter - x_iter)))
    x_sum = float(jnp.sum(x_iter))
    Px_sum = float(jnp.sum(Px_iter))
    
    print(f"   Iter {i+1}: check_norm={check_norm_iter:.6e}, "
          f"x_sum={x_sum:.10f}, Px_sum={Px_sum:.10f}")
    
    if np.isinf(check_norm_iter) or np.isnan(check_norm_iter):
        print(f"   ❌ FOUND PROBLEM at iteration {i+1}!")
        print(f"      Px has inf: {jnp.any(jnp.isinf(Px_iter))}")
        print(f"      Px has nan: {jnp.any(jnp.isnan(Px_iter))}")
        break
    
    # Normalize for next iteration
    x_iter = Px_iter / jnp.sum(Px_iter)

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
