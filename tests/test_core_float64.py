"""Test float64 precision support"""
import pytest
import jax.numpy as jnp
import gridvoting_jax as gv


def test_enable_float64():
    """Test that enable_float64() enables 64-bit precision"""
    # Enable float64
    gv.enable_float64()
    
    # Test precision with sum that should equal 1.0
    # Create vector of 101 elements, each 1/101
    vec = jnp.full(101, 1/101)
    total = jnp.sum(vec)
    diff = abs(total - 1.0)
    
    # With float64, difference should be very small (< 1e-10)
    # With float32, difference would be ~2.4e-7
    assert diff < 1e-10, f"Float64 precision not enabled: diff={diff}"
    
    # Verify dtype is float64
    assert vec.dtype == jnp.float64, f"Expected float64, got {vec.dtype}"
    
    # Verify constants are updated
    from gridvoting_jax.core import constants
    assert constants.DTYPE_FLOAT == jnp.float64, f"DTYPE_FLOAT not updated: {constants.DTYPE_FLOAT}"
    assert constants.TOLERANCE == 1e-10, f"TOLERANCE not updated: {constants.TOLERANCE}"
    assert constants.EPSILON == float(jnp.finfo(jnp.float64).eps), f"EPSILON not updated: {constants.EPSILON}"
