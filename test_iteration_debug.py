"""Debug lazy power method iteration to see where it goes wrong."""

import jax.numpy as jnp
import gridvoting_jax as gv
from gridvoting_jax.dynamics.lazy.base import LazyTransitionMatrix

# Use condorcet cycle
vm = gv.condorcet_cycle(zi=True)
N = 3

print(f"Testing Condorcet Cycle, ZI mode, N={N}")

# Get dense P for reference
vm.analyze(solver="full_matrix_inversion")
P_dense = vm.MarkovChain.P
print(f"\nDense P matrix:")
print(P_dense)

# Create lazy P
lazy_P = LazyTransitionMatrix(
    utility_functions=vm.utility_functions,
    majority=vm.majority,
    zi=vm.zi,
    number_of_feasible_alternatives=N
)

# Start with uniform distribution
x = jnp.ones(N) / N
print(f"\nInitial x: {x}")

# Manually iterate a few times
for i in range(5):
    x_new_dense = x @ P_dense
    x_new_lazy = lazy_P.rmatvec_batched(x)
    
    print(f"\nIteration {i+1}:")
    print(f"  Dense x @ P:     {x_new_dense}")
    print(f"  Lazy rmatvec(x): {x_new_lazy}")
    print(f"  Difference:      {jnp.abs(x_new_dense - x_new_lazy).sum():.10e}")
    
    x = x_new_lazy

print(f"\nFinal lazy result after 5 iterations: {x}")
print(f"Is uniform: {jnp.allclose(x, 1.0/N)}")
