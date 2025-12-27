#!/usr/bin/env python3
"""
Test if lazy power method times out during calibration.
"""

import gridvoting_jax as gv
import jax.numpy as jnp
import time

def test_calibration_timeout(g=40):
    """Test if calibration phase causes timeout."""
    print(f"\n{'='*60}")
    print(f"Testing lazy power method calibration: g={g}, ZI mode")
    print(f"{'='*60}")
    
    model = gv.bjm_spatial_triangle(g=g, zi=True)
    
    # Manually time what happens
    print("\nTiming calibration phase...")
    start = time.time()
    
    # This will trigger the lazy power method
    model.analyze_lazy(solver="power_method", force_lazy=True, max_iterations=5000, timeout=30.0)
    
    elapsed = time.time() - start
    print(f"Total time: {elapsed:.2f}s")
    
    dist = model.stationary_distribution
    is_uniform = jnp.std(dist) < 1e-10
    
    if is_uniform and elapsed < 2.0:
        print(f"\n❌ CONFIRMED: Method returned uniform in {elapsed:.2f}s")
        print(f"   This suggests it timed out during/after calibration")
        print(f"   without running any actual power iterations")
    elif is_uniform:
        print(f"\n⚠ Returned uniform but took {elapsed:.2f}s")
    else:
        print(f"\n✓ Found non-uniform solution in {elapsed:.2f}s")

if __name__ == "__main__":
    test_calibration_timeout(g=40)
