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


def test_gmres_respects_initial_guess():
    """
    Test 1: Output Differentiation
    
    Verify that GMRES produces different outputs when given different
    initial guesses. This proves that initial_guess is not being ignored.
    
    Uses BJM spatial triangle (g=20, zi=True) as canonical example.
    """
    # Create model
    model = gv.bjm_spatial_triangle(g=20, zi=True)
    
    # Get uniform initial guess
    n = model.model.number_of_feasible_alternatives
    uniform_guess = jnp.ones(n) / n
    
    # Create concentrated initial guess (all mass on first point)
    concentrated_guess = jnp.zeros(n).at[0].set(1.0)
    
    # Run GMRES with uniform initial guess
    model_uniform = gv.bjm_spatial_triangle(g=20, zi=True)
    model_uniform.model.analyze(
        solver="gmres_matrix_inversion",
        initial_guess=uniform_guess,
        max_iterations=2000
    )
    dist_uniform = model_uniform.stationary_distribution
    
    # Run GMRES with concentrated initial guess
    model_concentrated = gv.bjm_spatial_triangle(g=20, zi=True)
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


def test_gmres_initial_guess_propagation():
    """
    Test 3: Direct Instrumentation
    
    Monkey-patch JAX's GMRES to verify that initial_guess is actually
    passed as the x0 argument. This directly verifies the code path.
    
    Tests grid_upscaling solver which should pass upscaled distribution
    as initial_guess to GMRES.
    """
    captured_x0 = {'value': None}
    
    # Store original gmres function
    original_gmres = jax.scipy.sparse.linalg.gmres
    
    def patched_gmres(A, b, x0=None, **kwargs):
        """Capture x0 argument and call original."""
        captured_x0['value'] = x0
        return original_gmres(A, b, x0=x0, **kwargs)
    
    # Monkey-patch gmres
    with patch('jax.scipy.sparse.linalg.gmres', side_effect=patched_gmres):
        # Run grid_upscaling which should pass initial_guess to GMRES
        model = gv.bjm_spatial_triangle(g=20, zi=True)
        model.analyze(solver="grid_upscaling", max_iterations=2000)
    
    # Verify x0 was captured
    assert captured_x0['value'] is not None, (
        "GMRES was called with x0=None, initial_guess was not propagated"
    )
    
    # Verify x0 is not uniform (should be upscaled distribution)
    x0 = captured_x0['value']
    n = len(x0)
    uniform = jnp.ones(n) / n
    l1_from_uniform = float(jnp.linalg.norm(x0 - uniform, ord=1))
    
    print(f"L1 distance from uniform: {l1_from_uniform:.2e}")
    
    assert l1_from_uniform > 1e-6, (
        f"initial_guess is too close to uniform (L1={l1_from_uniform:.2e}), "
        "suggesting grid upscaling is not working correctly"
    )


def test_lazy_gmres_initial_guess():
    """
    Test 4: Lazy Solver Direct Instrumentation
    
    Verify that the lazy GMRES path actually passes initial_guess to GMRES.
    Uses direct instrumentation (monkey-patching) to verify the code path.
    
    Note: We use instrumentation rather than output differentiation because
    GMRES converges to the same solution regardless of initial_guess (it only
    affects convergence speed, not the final result).
    """
    captured_x0 = {'value': None}
    
    # Store original gmres function
    original_gmres = jax.scipy.sparse.linalg.gmres
    
    def patched_gmres(A, b, x0=None, **kwargs):
        """Capture x0 argument and call original."""
        captured_x0['value'] = x0
        return original_gmres(A, b, x0=x0, **kwargs)
    
    # Monkey-patch gmres
    with patch('jax.scipy.sparse.linalg.gmres', side_effect=patched_gmres):
        # Run lazy GMRES with custom initial_guess
        model = gv.bjm_spatial_triangle(g=20, zi=True)
        n = model.model.number_of_feasible_alternatives
        custom_guess = jnp.zeros(n).at[0].set(1.0)
        
        model.model.analyze_lazy(
            solver="gmres",
            force_lazy=True,
            initial_guess=custom_guess,
            max_iterations=2000
        )
    
    # Verify x0 was captured and is not None
    assert captured_x0['value'] is not None, (
        "Lazy GMRES was called with x0=None, initial_guess was not propagated"
    )
    
    # Verify x0 matches our custom guess (concentrated on first point)
    x0 = captured_x0['value']
    
    # Check that x0 is concentrated (not uniform)
    n = len(x0)
    uniform = jnp.ones(n) / n
    l1_from_uniform = float(jnp.linalg.norm(x0 - uniform, ord=1))
    
    print(f"Lazy GMRES initial_guess L1 distance from uniform: {l1_from_uniform:.2e}")
    
    assert l1_from_uniform > 0.1, (
        f"Lazy GMRES initial_guess is too close to uniform (L1={l1_from_uniform:.2e}), "
        "suggesting custom initial_guess was not propagated correctly"
    )


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
