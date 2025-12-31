"""Tests for normalize_if_needed utility function."""

import jax.numpy as jnp
import pytest
from gridvoting_jax.core import normalize_if_needed


def test_normalize_if_needed_no_op():
    """Test that normalized vectors are unchanged."""
    v = jnp.array([0.25, 0.25, 0.25, 0.25])
    result = normalize_if_needed(v)
    assert jnp.allclose(result, v)
    assert jnp.allclose(jnp.sum(result), 1.0)


def test_normalize_if_needed_corrects():
    """Test that unnormalized vectors are corrected."""
    v = jnp.array([0.5, 0.5, 0.5, 0.5])  # sum = 2.0
    result = normalize_if_needed(v)
    assert jnp.allclose(jnp.sum(result), 1.0)
    assert jnp.allclose(result, jnp.array([0.25, 0.25, 0.25, 0.25]))


def test_normalize_if_needed_threshold_small_deviation():
    """Test that small deviations within threshold are ignored."""
    n = 1000
    eps = jnp.finfo(jnp.float32).eps
    v = jnp.ones(n, dtype=jnp.float32) / n
    
    # Add deviation smaller than threshold (4·N·eps)
    small_deviation = eps  # Much smaller than 4·N·eps
    v = v.at[0].add(small_deviation)
    
    result = normalize_if_needed(v)
    # Should be unchanged (deviation < 4·N·eps)
    assert jnp.array_equal(result, v)


def test_normalize_if_needed_threshold_large_deviation():
    """Test that large deviations beyond threshold trigger normalization."""
    n = 100
    eps = jnp.finfo(jnp.float32).eps
    v = jnp.ones(n, dtype=jnp.float32) / n
    
    # Add deviation larger than threshold (4·N·eps)
    large_deviation = 10.0 * n * eps  # Much larger than 4·N·eps
    v = v.at[0].add(large_deviation)
    
    result = normalize_if_needed(v)
    # Should be normalized
    assert not jnp.array_equal(result, v)
    assert jnp.allclose(jnp.sum(result), 1.0)


def test_normalize_if_needed_preserves_shape():
    """Test that function preserves vector shape."""
    for n in [10, 100, 1000]:
        v = jnp.ones(n) / n
        result = normalize_if_needed(v)
        assert result.shape == v.shape


def test_normalize_if_needed_jit_compatible():
    """Test that function is JIT-compatible."""
    import jax
    
    jitted_normalize = jax.jit(normalize_if_needed)
    v = jnp.array([0.5, 0.5, 0.5, 0.5])
    result = jitted_normalize(v)
    assert jnp.allclose(jnp.sum(result), 1.0)
