#!/usr/bin/env python3
"""Debug script to analyze entropy calculation differences between dense and lazy matrices."""

import jax.numpy as jnp
import gridvoting_jax as gv
from gridvoting_jax.stochastic.utils import entropy_in_bits
from gridvoting_jax.core.constants import EPSILON

print("=" * 80)
print("ENTROPY CALCULATION ANALYSIS")
print("=" * 80)

for g in [20, 40]:
    print(f"\n{'='*80}")
    print(f"Testing g={g}")
    print(f"{'='*80}")
    
    model = gv.bjm_spatial_triangle(g=g, zi=False)
    P_lazy = model.model.transition_matrix()
    P_dense = P_lazy.to_dense()
    
    # Compute entropies both ways
    entropy_dense = entropy_in_bits(P_dense)
    entropy_lazy = P_lazy.row_entropies()
    
    # Check differences
    diff = jnp.abs(entropy_dense - entropy_lazy)
    l1_diff = jnp.sum(diff)
    max_diff = jnp.max(diff)
    
    print(f"\nResults:")
    print(f"  L1 difference:        {l1_diff:.15e}")
    print(f"  Max difference:       {max_diff:.15e}")
    print(f"  EPSILON:              {EPSILON:.15e}")
    print(f"  Tolerance (5*EPSILON): {5*EPSILON:.15e}")
    print(f"  L1 / Tolerance:       {l1_diff / (5*EPSILON):.2f}x")
    print(f"  Dtype:                {entropy_dense.dtype}")
    
    # Show a few examples where they differ most
    indices = jnp.argsort(diff)[-5:]
    print(f"\n  Top 5 rows with largest differences:")
    for idx, i in enumerate(indices):
        i_int = int(i)
        print(f"    {idx+1}. Row {i_int:4d}: dense={entropy_dense[i]:.10f}, lazy={entropy_lazy[i]:.10f}, diff={diff[i]:.2e}")
        
        # Show details for this row
        winners = int(jnp.sum(P_lazy.mask[i_int]))
        sq_val = float(P_lazy.status_quo_values[i_int])
        ch_val = float(P_lazy.challenger_values[i_int])
        
        print(f"              winners={winners}, status_quo={sq_val:.10f}, challenger={ch_val:.10f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
