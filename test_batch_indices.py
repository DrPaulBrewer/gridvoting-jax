"""Debug batch indices to see if there's duplication."""

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
print(f"BATCH_SIZE = {lazy_P.batch_size}")
print(f"num_batches = {lazy_P.num_batches}")
print(f"\nBatch indices:")
for i in range(lazy_P.num_batches):
    batch_inds = lazy_P.batch_indices[i]
    print(f"  Batch {i}: {batch_inds}")
    
# Check for duplicates
all_indices = []
for i in range(lazy_P.num_batches):
    batch_inds = lazy_P.batch_indices[i]
    all_indices.extend(batch_inds.tolist())

print(f"\nAll indices collected: {all_indices}")
print(f"Unique indices: {set([x for x in all_indices if x < N])}")
print(f"Total count: {len(all_indices)}")
print(f"Unique count: {len(set([x for x in all_indices if x < N]))}")

if len(all_indices) > len(set([x for x in all_indices if x < N])):
    print("⚠️  DUPLICATES FOUND!")
