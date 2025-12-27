#!/usr/bin/env python3
"""
Test if transition matrix is uniform for ZI mode.
"""

import gridvoting_jax as gv
import jax.numpy as jnp

def test_transition_matrix(g=40):
    """Check if P itself is uniform for ZI."""
    print(f"\nTesting transition matrix for g={g}, ZI mode\n")
    
    model = gv.bjm_spatial_triangle(g=g, zi=True)
    
    # Get transition matrix
    P = model.model.MarkovChain.P
    
    print(f"P shape: {P.shape}")
    print(f"P row 0 (first 10 elements): {P[0, :10]}")
    print(f"P row 0 unique values: {len(jnp.unique(P[0, :]))}")
    print(f"P row 0 all same: {jnp.allclose(P[0, :], P[0, 0])}")
    
    # Check if all rows are uniform
    row_stds = jnp.std(P, axis=1)
    print(f"\nRow standard deviations:")
    print(f"  Min: {jnp.min(row_stds):.6e}")
    print(f"  Max: {jnp.max(row_stds):.6e}")
    print(f"  Mean: {jnp.mean(row_stds):.6e}")
    
    if jnp.max(row_stds) < 1e-10:
        print(f"\n❌ ALL ROWS ARE UNIFORM!")
        print(f"   P is essentially the identity/uniform matrix")
        print(f"   This means stationary dist will be uniform")
    
    # Test rmatvec with non-uniform vector
    v = jnp.arange(P.shape[0], dtype=jnp.float32)
    v = v / jnp.sum(v)
    
    Pv = P.T @ v
    print(f"\nTest P.T @ v where v is non-uniform:")
    print(f"  v std: {jnp.std(v):.6e}")
    print(f"  (P.T @ v) std: {jnp.std(Pv):.6e}")
    
    if jnp.std(Pv) < 1e-10:
        print(f"  ❌ P.T @ v is uniform even though v is not!")

if __name__ == "__main__":
    test_transition_matrix(g=40)
