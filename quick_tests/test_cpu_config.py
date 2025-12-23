"""Test that CPU configuration is automatically detected and applied."""

import sys
sys.path.insert(0, '/home/paul/gridvoting-jax/gridvoting-jax/src')

import os

print("=" * 70)
print("Testing CPU Configuration in core.py")
print("=" * 70)

print("\n1. Environment variables BEFORE importing gridvoting_jax:")
print(f"   XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'NOT SET')}")
print(f"   OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'NOT SET')}")

print("\n2. Importing gridvoting_jax...")
import gridvoting_jax as gv

print("\n3. Environment variables AFTER importing gridvoting_jax:")
print(f"   XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'NOT SET')}")
print(f"   OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'NOT SET')}")

print("\n4. Detected CPU count:")
cpu_count = os.cpu_count()
print(f"   CPU cores: {cpu_count}")

print("\n5. JAX device info:")
import jax
devices = jax.devices()
print(f"   Default device: {devices[0]}")
print(f"   Platform: {devices[0].platform}")

print("\n6. Testing with a simple model:")
model = gv.bjm_spatial_triangle(g=20, zi=False)
print(f"   ✓ Model created (N={model.model.number_of_feasible_alternatives})")

model.analyze(solver="lazy_grid_upscaling", max_iterations=1000)
print(f"   ✓ Model solved")
print(f"   ✓ Sum: {model.stationary_distribution.sum():.6f}")

print("\n" + "=" * 70)
print("CPU Configuration Test COMPLETE")
print("=" * 70)
