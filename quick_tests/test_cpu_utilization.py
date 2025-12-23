"""Test CPU utilization with the new multi-device configuration."""

import sys
sys.path.insert(0, '/home/paul/gridvoting-jax/gridvoting-jax/src')

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import gridvoting_jax as gv
import jax
import jax.numpy as jnp
import time

print("=" * 70)
print("CPU Utilization Test with Multi-Device Configuration")
print("=" * 70)

print(f"\n1. JAX Devices: {len(jax.devices())} devices")
for i, dev in enumerate(jax.devices()):
    print(f"   Device {i}: {dev}")

print(f"\n2. Environment variables:")
print(f"   XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'NOT SET')}")
print(f"   OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'NOT SET')}")
print(f"   MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'NOT SET')}")

print(f"\n3. Testing with g=60 BJM spatial triangle (GMRES solver)...")
model = gv.bjm_spatial_triangle(g=60, zi=False)
print(f"   ✓ Model created (N={model.model.number_of_feasible_alternatives})")

print("\n   Running GMRES solver (watch CPU usage in 'top' or 'htop')...")
start = time.time()
model.analyze(solver="lazy_grid_upscaling", max_iterations=2000)
elapsed = time.time() - start

print(f"\n4. Results:")
print(f"   ✓ Solved in {elapsed:.2f}s")
print(f"   ✓ Sum: {jnp.sum(model.stationary_distribution):.6f}")

print("\n5. Testing power method solver...")
model2 = gv.bjm_spatial_triangle(g=40, zi=False)
print(f"   ✓ Model created (N={model2.model.number_of_feasible_alternatives})")

print("\n   Running power method solver (watch CPU usage)...")
start = time.time()
# Use power_method directly
from gridvoting_jax.dynamics.lazy import LazyMarkovChain
lazy_chain = LazyMarkovChain(model2.model.transition_matrix_lazy)
lazy_chain.find_unique_stationary_distribution(solver="power_method", max_iterations=2000)
elapsed = time.time() - start

print(f"\n6. Power method results:")
print(f"   ✓ Solved in {elapsed:.2f}s")
print(f"   ✓ Sum: {jnp.sum(lazy_chain.stationary_distribution):.6f}")

print("\n" + "=" * 70)
print("CPU Utilization Test COMPLETE")
print("Monitor CPU usage during the tests to see if all cores are utilized")
print("=" * 70)
