"""
Regression tests for GMRES initial_guess propagation.

These tests ensure that initial_guess is properly passed to GMRES solvers
and actually affects the computation. This prevents regressions where
initial_guess might be silently ignored or not propagated through the
solver dispatch chain.

Tests run at float32 precision for consistency with typical usage.
"""

import pytest
import jax
import jax.numpy as jnp
import gridvoting_jax as gv
from unittest.mock import patch

pytestmark = pytest.mark.essential


@pytest.fixture(autouse=True)
def force_float32():
    """Force float32 precision for all tests in this module."""
    original_value = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', False)
    yield
    jax.config.update('jax_enable_x64', original_value)


def test_gmres_respects_initial_guess(bmj_g20_zi):
    """
    Test 1: Output Differentiation
    
    Verify that GMRES produces different outputs when given different
    initial guesses. This proves that initial_guess is not being ignored.
    
    Uses BJM spatial triangle (g=20, zi=True) as canonical example.
    """
    # Create model
    model = bmj_g20_zi
    
    # Get uniform initial guess
    n = model.model.number_of_feasible_alternatives
    uniform_guess = jnp.ones(n) / n
    
    # Create concentrated initial guess (all mass on first point)
    concentrated_guess = jnp.zeros(n).at[0].set(1.0)
    
    # Run GMRES with uniform initial guess
    model_uniform = bmj_g20_zi
    model_uniform.model.analyze(
        solver="gmres_matrix_inversion",
        initial_guess=uniform_guess,
        max_iterations=2000
    )
    dist_uniform = model_uniform.stationary_distribution
    
    # Run GMRES with concentrated initial guess
    model_concentrated = bmj_g20_zi
    model_concentrated.model.analyze(
        solver="gmres_matrix_inversion",
        initial_guess=concentrated_guess,
        max_iterations=2000
    )
    dist_concentrated = model_concentrated.stationary_distribution
    
    # Verify outputs are different (L1 norm > 1e-6)
    l1_diff = float(jnp.linalg.norm(dist_uniform - dist_concentrated, ord=1))
    
    print(f"L1 difference between uniform and concentrated initial guess: {l1_diff:.2e}")
    
    # If initial_guess is being ignored, outputs would be identical (L1 ~ 0)
    # If initial_guess is respected, outputs should differ
    assert l1_diff > 1e-6, (
        f"GMRES outputs are too similar (L1={l1_diff:.2e}), "
        "suggesting initial_guess is being ignored"
    )




if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
