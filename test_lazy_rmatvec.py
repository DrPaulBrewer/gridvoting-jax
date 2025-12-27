#!/usr/bin/env python3
"""
Test if lazy transition matrix rmatvec_batched is working correctly.
"""

import gridvoting_jax as gv
import jax.numpy as jnp

def test_lazy_rmatvec(g=40):
    """Test lazy rmatvec_batched."""
    print(f"\nTesting lazy rmatvec_batched for g={g}, ZI mode\n")
    
    model = gv.bjm_spatial_triangle(g=g, zi=True)
    
    # Create lazy transition matrix
    from gridvoting_jax.dynamics.lazy import LazyTransitionMatrix
    lazy_P = LazyTransitionMatrix(
        utility_functions=model.model.utility_functions,
        majority=model.model.majority,
        zi=model.model.zi,
        number_of_feasible_alternatives=model.model.number_of_feasible_alternatives
    )
    
    N = lazy_P.N
    print(f"N = {N}")
    print(f"Batch size: {lazy_P.num_batches} batches of 128")
    
    # Test with non-uniform vector
    v = jnp.arange(N, dtype=jnp.float32)
    v = v / jnp.sum(v)
    
    print(f"\nInput v:")
    print(f"  std: {jnp.std(v):.6e}")
    print(f"  min: {jnp.min(v):.6e}")
    print(f"  max: {jnp.max(v):.6e}")
    
    # Test batched rmatvec
    result_batched = lazy_P.rmatvec_batched(v)
    
    print(f"\nResult from rmatvec_batched:")
    print(f"  std: {jnp.std(result_batched):.6e}")
    print(f"  min: {jnp.min(result_batched):.6e}")
    print(f"  max: {jnp.max(result_batched):.6e}")
    print(f"  sum: {jnp.sum(result_batched):.6f}")
    
    if jnp.std(result_batched) < 1e-10:
        print(f"\n❌ rmatvec_batched returns UNIFORM output!")
        print(f"   This is the bug - it should return non-uniform")
    else:
        print(f"\n✓ rmatvec_batched returns non-uniform output")
    
    # Compare with non-batched
    result_nonbatched = lazy_P.rmatvec(v)
    
    print(f"\nResult from rmatvec (non-batched):")
    print(f"  std: {jnp.std(result_nonbatched):.6e}")
    
    diff = jnp.max(jnp.abs(result_batched - result_nonbatched))
    print(f"\nDifference between batched and non-batched:")
    print(f"  max abs diff: {diff:.6e}")
    
    if diff > 1e-5:
        print(f"  ❌ MISMATCH! Batched and non-batched give different results")
    else:
        print(f"  ✓ Match")

if __name__ == "__main__":
    test_lazy_rmatvec(g=40)
