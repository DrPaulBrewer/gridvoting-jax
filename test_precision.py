#!/usr/bin/env python3
"""Test precision differences between float32 and float64"""
import jax.numpy as jnp
import jax

print("Float32 test:")
print("=" * 50)
vec32 = jnp.full(101, 1/101, dtype=jnp.float32)
sum32 = jnp.sum(vec32)
diff32 = abs(sum32 - 1.0)
print(f"  Sum: {sum32}")
print(f"  Diff from 1.0: {diff32:.15e}")
print()

# Enable float64
jax.config.update('jax_enable_x64', True)

print("Float64 test:")
print("=" * 50)
vec64 = jnp.full(101, 1/101, dtype=jnp.float64)
sum64 = jnp.sum(vec64)
diff64 = abs(sum64 - 1.0)
print(f"  Sum: {sum64}")
print(f"  Diff from 1.0: {diff64:.15e}")
print()

print("Comparison:")
print("=" * 50)
print(f"Float32 error is {diff32/diff64:.1f}x larger than float64 error")
