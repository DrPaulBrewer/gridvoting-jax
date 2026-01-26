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

    # test that normalize_if_needed returns float64
    vec = jnp.full(101, 2.0, dtype=jnp.float64)
    assert gv.stochastic.utils.normalize_if_needed(vec).dtype == jnp.float64, f"normalize_if_needed did not return float64"

    # Test that it works with a voting model
    # because thats where the bug was
    svm = gv.models.examples.shapes.ring(g=10,r=7,zi=False,voters=3,polar=True,thetastep=60)
    assert svm.model.transition_matrix().dtype == jnp.float64, f"Transition matrix dtype is not float64: {svm.model.transition_matrix().dtype}"
    for solver in ['full_matrix_inversion', 'gmres_matrix_inversion','power_method','bifurcated_power_method']:
        svm.analyze(solver=solver)
        assert svm.stationary_distribution.dtype == jnp.float64, f"Stationary distribution dtype is not float64: {svm.stationary_distribution.dtype} for solver={solver}"

    #  test that it works with lumping
    for solver in ['full_matrix_inversion', 'gmres_matrix_inversion','power_method','bifurcated_power_method']:
        svm.analyze(solver=solver, partitions=svm.grid.partition_from_rotation(angle=120))
        assert svm.stationary_distribution.dtype == jnp.float64, f"Stationary distribution dtype is not float64: {svm.stationary_distribution.dtype} for solver={solver}"
        