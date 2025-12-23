"""Test that user-provided environment variables are respected."""

import sys
sys.path.insert(0, '/home/paul/gridvoting-jax/gridvoting-jax/src')

import os

# Set custom values BEFORE importing gridvoting_jax
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4'
os.environ['OMP_NUM_THREADS'] = '4'

print("=" * 70)
print("Testing CPU Configuration (USER-PROVIDED VALUES)")
print("=" * 70)

print("\n1. User-provided environment variables:")
print(f"   XLA_FLAGS: {os.environ.get('XLA_FLAGS')}")
print(f"   OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")

print("\n2. Importing gridvoting_jax...")
import gridvoting_jax as gv

print("\n3. Environment variables AFTER importing (should be unchanged):")
print(f"   XLA_FLAGS: {os.environ.get('XLA_FLAGS')}")
print(f"   OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")

expected_xla = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4'
expected_omp = '4'

if os.environ.get('XLA_FLAGS') == expected_xla:
    print("\n   ✅ XLA_FLAGS preserved (not overwritten)")
else:
    print(f"\n   ❌ XLA_FLAGS was overwritten!")
    
if os.environ.get('OMP_NUM_THREADS') == expected_omp:
    print("   ✅ OMP_NUM_THREADS preserved (not overwritten)")
else:
    print(f"   ❌ OMP_NUM_THREADS was overwritten!")

print("\n" + "=" * 70)
print("User-Provided Values Test COMPLETE")
print("=" * 70)
