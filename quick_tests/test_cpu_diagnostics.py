"""Diagnostic script to check JAX CPU configuration."""

import sys
sys.path.insert(0, '/home/paul/gridvoting-jax/gridvoting-jax/src')

import os

print("=" * 70)
print("JAX CPU Configuration Diagnostics")
print("=" * 70)

# Force CPU mode
os.environ['JAX_PLATFORMS'] = 'cpu'

print("\n1. BEFORE importing gridvoting_jax:")
print(f"   XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'NOT SET')}")
print(f"   OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'NOT SET')}")
print(f"   JAX_PLATFORMS: {os.environ.get('JAX_PLATFORMS', 'NOT SET')}")

import gridvoting_jax as gv

print("\n2. AFTER importing gridvoting_jax:")
print(f"   XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'NOT SET')}")
print(f"   OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'NOT SET')}")

print("\n3. CPU count detection:")
print(f"   os.cpu_count(): {os.cpu_count()}")

print("\n4. JAX configuration:")
import jax
print(f"   JAX devices: {jax.devices()}")
print(f"   JAX default backend: {jax.default_backend()}")

# Check if we can see XLA thread settings
print("\n5. Testing actual parallelism with a simple operation:")
import jax.numpy as jnp
import time

# Create a moderately large matrix operation
n = 5000
A = jnp.ones((n, n))
B = jnp.ones((n, n))

print(f"   Matrix size: {n}x{n}")
print("   Running matrix multiplication...")
start = time.time()
C = jnp.dot(A, B)
C.block_until_ready()  # Ensure computation completes
elapsed = time.time() - start
print(f"   Time: {elapsed:.3f}s")

print("\n" + "=" * 70)
print("Check 'top' or 'htop' during the matrix multiplication above")
print("to see actual CPU utilization")
print("=" * 70)
