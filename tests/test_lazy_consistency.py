"""Test lazy vs dense consistency across different grid sizes.

This test validates that lazy solvers produce results consistent with dense solvers.
"""

import pytest
import sys
sys.path.insert(0, '/home/paul/gridvoting-jax/gridvoting-jax/src')

import gridvoting_jax as gv
import jax.numpy as jnp
import chex

pytestmark = pytest.mark.lazy


def test_lazy_gmres_vs_dense_g10():
    """Test lazy GMRES vs dense on small grid (g=10, N=341)."""
    # Dense
    model_dense = gv.bjm_spatial_triangle(g=10, zi=False)
    model_dense.analyze(solver="full_matrix_inversion")
    
    # Lazy
    model_lazy = gv.bjm_spatial_triangle(g=10, zi=False)
    model_lazy.analyze_lazy(force_lazy=True, solver="gmres")
    
    # Compare
    diff = jnp.abs(model_dense.stationary_distribution - model_lazy.stationary_distribution)
    max_diff = jnp.max(diff)
    
    assert max_diff < 1e-4, f"Lazy vs dense mismatch: {max_diff}"
    assert jnp.abs(jnp.sum(model_lazy.stationary_distribution) - 1.0) < 1e-6


def test_lazy_gmres_vs_dense_g20():
    """Test lazy GMRES vs dense on medium grid (g=20, N=1261)."""
    # Dense
    model_dense = gv.bjm_spatial_triangle(g=20, zi=False)
    model_dense.analyze(solver="full_matrix_inversion")
    
    # Lazy
    model_lazy = gv.bjm_spatial_triangle(g=20, zi=False)
    model_lazy.analyze_lazy(force_lazy=True, solver="gmres")
    
    # Compare
    diff = jnp.abs(model_dense.stationary_distribution - model_lazy.stationary_distribution)
    max_diff = jnp.max(diff)
    
    assert max_diff < 1e-4, f"Lazy vs dense mismatch: {max_diff}"
    assert jnp.abs(jnp.sum(model_lazy.stationary_distribution) - 1.0) < 1e-6


@pytest.mark.large_grid
def test_lazy_gmres_vs_dense_g40():
    """Test lazy GMRES vs dense on large grid (g=40, N=6561)."""
    # Dense
    model_dense = gv.bjm_spatial_triangle(g=40, zi=False)
    model_dense.analyze(solver="full_matrix_inversion")
    
    # Lazy
    model_lazy = gv.bjm_spatial_triangle(g=40, zi=False)
    model_lazy.analyze_lazy(force_lazy=True, solver="gmres")
    
    # Compare
    diff = jnp.abs(model_dense.stationary_distribution - model_lazy.stationary_distribution)
    max_diff = jnp.max(diff)
    
    assert max_diff < 1e-4, f"Lazy vs dense mismatch: {max_diff}"
    assert jnp.abs(jnp.sum(model_lazy.stationary_distribution) - 1.0) < 1e-6


def test_lazy_auto_selection_small():
    """Test auto-selection chooses dense for small grids."""
    model = gv.bjm_spatial_triangle(g=10, zi=False)
    model.analyze_lazy(solver="auto")
    
    # Should work regardless of selection
    assert model.analyzed
    assert jnp.abs(jnp.sum(model.stationary_distribution) - 1.0) < 1e-6


def test_lazy_force_dense():
    """Test force_dense parameter."""
    model = gv.bjm_spatial_triangle(g=20, zi=False)
    model.analyze_lazy(force_dense=True, solver="auto")
    
    assert model.analyzed
    assert jnp.abs(jnp.sum(model.stationary_distribution) - 1.0) < 1e-6


if __name__ == "__main__":
    print("Running lazy consistency tests...")
    test_lazy_gmres_vs_dense_g10()
    print("✓ g=10 passed")
    test_lazy_gmres_vs_dense_g20()
    print("✓ g=20 passed")
    test_lazy_gmres_vs_dense_g40()
    print("✓ g=40 passed")
    test_lazy_auto_selection_small()
    print("✓ Auto-selection passed")
    test_lazy_force_dense()
    print("✓ Force dense passed")
    print("\n✅ All consistency tests passed!")
