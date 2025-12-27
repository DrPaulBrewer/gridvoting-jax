"""Test lazy power method vs dense power method for ZI mode."""

import jax.numpy as jnp
import gridvoting_jax as gv

# Use condorcet cycle (small, simple)
vm = gv.condorcet_cycle(zi=True)
N = 3

print(f"Testing Condorcet Cycle, ZI mode, N={N}")

# Dense power method
vm.analyze(solver="power_method", max_iterations=100)
pi_dense = vm.MarkovChain.stationary_distribution
print(f"\nDense power method result:")
print(f"  π = {pi_dense}")
print(f"  Sum = {pi_dense.sum()}")

# Lazy power method
vm.analyze_lazy(solver="power_method", force_lazy=True, max_iterations=100)
pi_lazy = vm.MarkovChain.stationary_distribution
print(f"\nLazy power method result:")
print(f"  π = {pi_lazy}")
print(f"  Sum = {pi_lazy.sum()}")

# Compare
diff = jnp.abs(pi_dense - pi_lazy).sum()
print(f"\nL1 difference: {diff:.10e}")

if diff > 1e-3:
    print("⚠️  MISMATCH!")
else:
    print("✓ Results match!")

# Also check if either is uniform
is_uniform_dense = jnp.allclose(pi_dense, 1.0/N)
is_uniform_lazy = jnp.allclose(pi_lazy, 1.0/N)
print(f"\nDense is uniform: {is_uniform_dense}")
print(f"Lazy is uniform: {is_uniform_lazy}")
