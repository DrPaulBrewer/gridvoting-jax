"""Debug what's happening in rmatvec_batched."""

import jax.numpy as jnp
import gridvoting_jax as gv
from gridvoting_jax.dynamics.lazy.base import LazyTransitionMatrix

# Use condorcet cycle
vm = gv.condorcet_cycle(zi=True)
N = 3

# Create lazy P
lazy_P = LazyTransitionMatrix(
    utility_functions=vm.utility_functions,
    majority=vm.majority,
    zi=vm.zi,
    number_of_feasible_alternatives=N
)

print(f"N = {N}")
print(f"num_batches = {lazy_P.num_batches}")

# Check batch indices
for i in range(lazy_P.num_batches):
    batch_inds = lazy_P.batch_indices[i]
    valid_mask = batch_inds < N
    valid_inds = batch_inds[valid_mask]
    
    print(f"\nBatch {i}:")
    print(f"  batch_inds (first 10): {batch_inds[:10]}")
    print(f"  valid_mask (first 10): {valid_mask[:10]}")
    print(f"  valid_inds: {valid_inds}")
    print(f"  Number of valid: {len(valid_inds)}")
