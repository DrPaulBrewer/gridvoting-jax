#!/usr/bin/env python3
"""
Test if P matrix for ZI has different rows for different grid points.
"""

import gridvoting_jax as gv
import jax.numpy as jnp

def test_p_rows_different(g=40):
    """Test if P matrix rows are different for different grid points."""
    print(f"\nTesting P matrix rows for g={g}, ZI mode\n")
    
    model = gv.bjm_spatial_triangle(g=g, zi=True)
    
    # Just use indices 0 and 1000 to test different rows
    idx_1 = 0
    idx_2 = 1000
    
    print(f"Testing row {idx_1} vs row {idx_2}")
    
    # Compute transition rows using lazy implementation
    from gridvoting_jax.dynamics.lazy import LazyTransitionMatrix
    lazy_P = LazyTransitionMatrix(
        utility_functions=model.model.utility_functions,
        majority=model.model.majority,
        zi=model.model.zi,
        number_of_feasible_alternatives=model.model.number_of_feasible_alternatives
    )
    
    # Get full P matrix
    P = lazy_P.todense()
    
    row_1 = P[idx_1, :]
    row_2 = P[idx_2, :]
    
    print(f"\nRow {idx_1}:")
    print(f"  First 10 elements: {row_1[:10]}")
    print(f"  Std: {jnp.std(row_1):.6e}")
    print(f"  Min: {jnp.min(row_1):.6e}")
    print(f"  Max: {jnp.max(row_1):.6e}")
    print(f"  Unique values: {len(jnp.unique(row_1))}")
    
    print(f"\nRow {idx_2}:")
    print(f"  First 10 elements: {row_2[:10]}")
    print(f"  Std: {jnp.std(row_2):.6e}")
    print(f"  Min: {jnp.min(row_2):.6e}")
    print(f"  Max: {jnp.max(row_2):.6e}")
    print(f"  Unique values: {len(jnp.unique(row_2))}")
    
    # Compare rows
    diff = jnp.max(jnp.abs(row_1 - row_2))
    print(f"\nDifference between rows:")
    print(f"  Max abs diff: {diff:.6e}")
    
    if diff < 1e-10:
        print(f"\n  ❌ Rows are IDENTICAL")
    else:
        print(f"\n  ✓ Rows are DIFFERENT")
    
    # Check if rows are uniform
    if jnp.std(row_1) < 1e-10:
        print(f"\n  ❌ Row {idx_1} is uniform (all elements equal)")
    else:
        print(f"\n  ✓ Row {idx_1} is non-uniform")
    
    if jnp.std(row_2) < 1e-10:
        print(f"\n  ❌ Row {idx_2} is uniform (all elements equal)")
    else:
        print(f"\n  ✓ Row {idx_2} is non-uniform")

if __name__ == "__main__":
    test_p_rows_different(g=40)
