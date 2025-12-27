"""Test to compare dense vs lazy rmatvec for ZI mode."""

import jax.numpy as jnp
import gridvoting_jax as gv

# Use a simple example
vm = gv.condorcet_cycle(zi=True)
N = 3  # Condorcet cycle has 3 states
print(f"Testing Condorcet Cycle, ZI mode, N={N}")

# Get dense P matrix
vm.analyze(solver="full_matrix_inversion")
P_dense = vm.MarkovChain.P
print(f"\nDense P matrix shape: {P_dense.shape}")
print(f"Dense P matrix:")
print(P_dense)
print(f"\nDense P row 0: {P_dense[0]}")
print(f"Dense P is uniform: {jnp.allclose(P_dense[0], 1.0/N)}")

# Create lazy P
from gridvoting_jax.dynamics.lazy.base import LazyTransitionMatrix

lazy_P = LazyTransitionMatrix(
    utility_functions=vm.utility_functions,
    majority=vm.majority,
    zi=vm.zi,
    number_of_feasible_alternatives=N
)

# Test rmatvec with a simple vector
v = jnp.ones(N) / N  # Uniform distribution

# Dense: P.T @ v
result_dense = P_dense.T @ v
print(f"\nDense P.T @ v:")
print(f"  Result: {result_dense}")
print(f"  Sum: {result_dense.sum()}")
print(f"  Is uniform: {jnp.allclose(result_dense, 1.0/N)}")

# Lazy: rmatvec(v)
result_lazy = lazy_P.rmatvec(v)
print(f"\nLazy rmatvec(v):")
print(f"  Result: {result_lazy}")
print(f"  Sum: {result_lazy.sum()}")
print(f"  Is uniform: {jnp.allclose(result_lazy, 1.0/N)}")

# Compare
diff = jnp.abs(result_dense - result_lazy).sum()
print(f"\nL1 difference: {diff:.10e}")

if diff > 1e-5:
    print("⚠️  MISMATCH FOUND!")
    print(f"Dense result: {result_dense}")
    print(f"Lazy result:  {result_lazy}")
else:
    print("✓ Results match!")

# Now test with a non-uniform vector
print("\n" + "="*60)
print("Testing with non-uniform vector (all mass on state 0)")
v2 = jnp.zeros(N).at[0].set(1.0)  # All mass on state 0

result_dense2 = P_dense.T @ v2
result_lazy2 = lazy_P.rmatvec(v2)

print(f"\nDense P.T @ e_0: {result_dense2}")
print(f"Lazy rmatvec(e_0): {result_lazy2}")
diff2 = jnp.abs(result_dense2 - result_lazy2).sum()
print(f"L1 difference: {diff2:.10e}")

if diff2 > 1e-5:
    print("⚠️  MISMATCH FOUND!")
else:
    print("✓ Results match!")

# Also test compute_rows directly
print("\n" + "="*60)
print("Testing compute_rows directly")
lazy_rows = lazy_P.compute_rows(jnp.array([0, 1, 2]))
print(f"\nLazy row 0: {lazy_rows[0]}")
print(f"Dense row 0: {P_dense[0]}")
print(f"Match: {jnp.allclose(lazy_rows[0], P_dense[0])}")
