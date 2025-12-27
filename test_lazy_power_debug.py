#!/usr/bin/env python3
"""
Debug lazy power method poor results in ZI mode.
"""

import gridvoting_jax as gv
import jax.numpy as jnp

def test_lazy_power_zi(g=40):
    """Test lazy power method on ZI g=40."""
    print(f"\n{'='*60}")
    print(f"Testing lazy power method: g={g}, ZI mode")
    print(f"{'='*60}")
    
    model = gv.bjm_spatial_triangle(g=g, zi=True)
    
    # Test lazy power method
    print("\nRunning lazy power method...")
    model.analyze_lazy(solver="power_method", force_lazy=True, max_iterations=5000)
    dist = model.stationary_distribution
    
    print(f"\nResults:")
    print(f"  Sum: {jnp.sum(dist):.10f}")
    print(f"  Has NaN: {jnp.any(jnp.isnan(dist))}")
    print(f"  Has Inf: {jnp.any(jnp.isinf(dist))}")
    print(f"  Min: {jnp.min(dist):.6e}")
    print(f"  Max: {jnp.max(dist):.6e}")
    print(f"  Mean: {jnp.mean(dist):.6e}")
    print(f"  Std: {jnp.std(dist):.6e}")
    print(f"  Unique values: {len(jnp.unique(dist))}")
    
    # Check if nearly uniform
    uniform_val = 1.0 / len(dist)
    max_deviation = jnp.max(jnp.abs(dist - uniform_val))
    print(f"  Max deviation from uniform: {max_deviation:.6e}")
    
    if jnp.std(dist) < 1e-10:
        print(f"\n  ❌ Distribution is essentially uniform (zero variance)!")
        print(f"     This causes division by zero in correlation calculation")

if __name__ == "__main__":
    test_lazy_power_zi(g=40)
