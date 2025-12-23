"""Test CPU configuration with forced CPU mode."""

import sys
sys.path.insert(0, '/home/paul/gridvoting-jax/gridvoting-jax/src')

import os

# Force CPU mode BEFORE importing gridvoting_jax
os.environ['JAX_PLATFORMS'] = 'cpu'

print("=" * 70)
print("Testing CPU Configuration (FORCED CPU MODE)")
print("=" * 70)

print("\n1. Environment variables BEFORE importing gridvoting_jax:")
print(f"   JAX_PLATFORMS: {os.environ.get('JAX_PLATFORMS', 'NOT SET')}")
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
import time
model = gv.bjm_spatial_triangle(g=40, zi=False)
print(f"   ✓ Model created (N={model.model.number_of_feasible_alternatives})")

start = time.time()
model.analyze(solver="lazy_grid_upscaling", max_iterations=1000)
elapsed = time.time() - start
print(f"   ✓ Model solved in {elapsed:.2f}s")
print(f"   ✓ Sum: {model.stationary_distribution.sum():.6f}")

print("\n" + "=" * 70)
print("CPU Configuration Test (CPU MODE) COMPLETE")
print("=" * 70)
