"""Test lazy solver auto-selection and OOM prevention.

This test validates memory-based auto-selection and that lazy solvers
work on grids that would OOM with dense construction.
"""

import pytest
import sys
sys.path.insert(0, '/home/paul/gridvoting-jax/gridvoting-jax/src')

import gridvoting_jax as gv
import jax.numpy as jnp
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.lazy


def test_auto_selection_mechanism():
    """Test that auto-selection works correctly."""
    # Small grid - should work with auto
    model_small = gv.bjm_spatial_triangle(g=10, zi=False)
    model_small.analyze_lazy(solver="auto")
    assert model_small.analyzed
    
    # Medium grid - should work with auto
    model_medium = gv.bjm_spatial_triangle(g=20, zi=False)
    model_medium.analyze_lazy(solver="auto")
    assert model_medium.analyzed


def test_force_lazy_small_grid():
    """Test that force_lazy works even on small grids."""
    model = gv.bjm_spatial_triangle(g=10, zi=False)
    model.analyze_lazy(force_lazy=True, solver="gmres")
    
    assert model.analyzed
    assert jnp.abs(jnp.sum(model.stationary_distribution) - 1.0) < 1e-6


@pytest.mark.slow
def test_lazy_g60():
    """Test lazy solver on g=60 (N=14641)."""
    model = gv.bjm_spatial_triangle(g=60, zi=False)
    model.analyze_lazy(force_lazy=True, solver="gmres", max_iterations=3000)
    
    assert model.analyzed
    assert jnp.abs(jnp.sum(model.stationary_distribution) - 1.0) < 1e-6


@pytest.mark.slow
def test_lazy_g80_oom_prevention():
    """
    Test lazy solver on g=80 (N=25921) - the critical OOM prevention case.
    
    This grid size causes GPU OOM with dense construction on 1080 Ti (11GB VRAM),
    but lazy solver should work.
    """
    model = gv.bjm_spatial_triangle(g=80, zi=False)
    model.analyze_lazy(force_lazy=True, solver="gmres", max_iterations=3000)
    
    assert model.analyzed
    assert jnp.abs(jnp.sum(model.stationary_distribution) - 1.0) < 1e-6
    
    # Verify it's actually using lazy construction
    # (FlexMarkovChain with force_lazy should create LazyMarkovChain)
    assert hasattr(model.MarkovChain, 'lazy_P') or not hasattr(model.MarkovChain, 'P')


def test_lazy_vs_dense_memory_usage():
    """
    Verify that lazy construction uses less memory than dense.
    
    Note: This is a qualitative test - we just verify both work,
    actual memory profiling would require external tools.
    """
    # Dense on small grid
    model_dense = gv.bjm_spatial_triangle(g=20, zi=False)
    model_dense.analyze(solver="full_matrix_inversion")
    
    # Lazy on same grid
    model_lazy = gv.bjm_spatial_triangle(g=20, zi=False)
    model_lazy.analyze_lazy(force_lazy=True, solver="gmres")
    
    # Both should work
    assert model_dense.analyzed
    assert model_lazy.analyzed
    
    # Results should match
    diff = jnp.abs(model_dense.stationary_distribution - model_lazy.stationary_distribution)
    assert jnp.max(diff) < 1e-4


if __name__ == "__main__":
    print("Running lazy auto-selection tests...")
    test_auto_selection_mechanism()
    print("✓ Auto-selection mechanism passed")
    test_force_lazy_small_grid()
    print("✓ Force lazy on small grid passed")
    test_lazy_g60()
    print("✓ g=60 passed")
    test_lazy_g80_oom_prevention()
    print("✓ g=80 OOM prevention passed")
    test_lazy_vs_dense_memory_usage()
    print("✓ Memory usage comparison passed")
    print("\n✅ All auto-selection tests passed!")
