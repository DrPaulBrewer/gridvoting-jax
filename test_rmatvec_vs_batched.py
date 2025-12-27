"""Compare rmatvec vs rmatvec_batched."""

import jax.numpy as jnp
import gridvoting_jax as gv
from gridvoting_jax.dynamics.lazy.base import LazyTransitionMatrix

# Use condorcet cycle
vm = gv.condorcet_cycle(zi=True)
N = 3

# Get dense P
vm.analyze(solver="full_matrix_inversion")
P_dense = vm.MarkovChain.P
print(f"Dense P matrix:")
print(P_dense)

# Create lazy P
lazy_P = LazyTransitionMatrix(
    utility_functions=vm.utility_functions,
    majority=vm.majority,
    zi=vm.zi,
    number_of_feasible_alternatives=N
)

# Test vector
x = jnp.ones(N) / N
print(f"\nTest vector x: {x}")

# Dense result
result_dense = x @ P_dense
print(f"\nDense x @ P: {result_dense}")

# Lazy rmatvec (non-batched)
result_lazy_rmatvec = lazy_P.rmatvec(x)
print(f"Lazy rmatvec(x): {result_lazy_rmatvec}")
print(f"Difference from dense: {jnp.abs(result_dense - result_lazy_rmatvec).sum():.10e}")

# Lazy rmatvec_batched
result_lazy_batched = lazy_P.rmatvec_batched(x)
print(f"\nLazy rmatvec_batched(x): {result_lazy_batched}")
print(f"Difference from dense: {jnp.abs(result_dense - result_lazy_batched).sum():.10e}")

# Compare the two lazy methods
print(f"\nrmatvec vs rmatvec_batched difference: {jnp.abs(result_lazy_rmatvec - result_lazy_batched).sum():.10e}")
