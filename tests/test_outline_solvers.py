"""
Test outline-based solvers and interpolation matrix creation.
"""

import pytest
import jax.numpy as jnp
from jax.experimental import sparse
import sys
sys.path.insert(0, '/home/paul/gridvoting-jax/gridvoting-jax/src')

import gridvoting_jax as gv
from gridvoting_jax.models.spatial import create_outline_interpolation_matrix
from gridvoting_jax.spatial import Grid


class TestOutlineInterpolationMatrix:
    """Test interpolation matrix creation."""
    
    def test_matrix_properties(self):
        """Test basic matrix properties."""
        fine_grid = Grid(x0=0, x1=10, xstep=1, y0=0, y1=10, ystep=1)
        coarse_grid = Grid(x0=0, x1=10, xstep=2, y0=0, y1=10, ystep=2)
        
        C = create_outline_interpolation_matrix(fine_grid, coarse_grid)
        
        # Each row should sum to 1.0 (probability conservation)
        row_sums = C.todense().sum(axis=1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-10), "Rows don't sum to 1.0"
        
        # Matrix should be very sparse (>90%)
        sparsity = 1.0 - (C.nse / (fine_grid.len * coarse_grid.len))
        assert sparsity > 0.90, f"Matrix not sparse enough: {sparsity:.2%}"
        
        # Check shape
        assert C.shape == (fine_grid.len, coarse_grid.len)
    
    @pytest.mark.parametrize("g", [6, 10])
    def test_interpolation_correctness(self, g):
        """Test that interpolation produces valid results that can be normalized."""
        fine_grid = Grid(x0=-g, x1=g, xstep=1, y0=-g, y1=g, ystep=1)
        coarse_grid = Grid(x0=-g, x1=g, xstep=2, y0=-g, y1=g, ystep=2)
        
        C = create_outline_interpolation_matrix(fine_grid, coarse_grid)
        
        # Create a simple test distribution on coarse grid (uniform)
        coarse_dist = jnp.ones(coarse_grid.len) / coarse_grid.len
        
        # Interpolate
        fine_dist = C @ coarse_dist
        
        # Check it's valid (non-negative, can be normalized)
        assert jnp.all(fine_dist >= 0), "Negative probabilities"
        assert fine_dist.sum() > 0, "Sum is zero"
        
        # Normalize and check
        fine_dist_normalized = fine_dist / fine_dist.sum()
        assert jnp.isclose(fine_dist_normalized.sum(), 1.0, atol=1e-10), \
            f"Normalized sum = {fine_dist_normalized.sum()}, not 1.0"


class TestOutlineSolvers:
    """Test outline-based solvers."""
    
    @pytest.mark.parametrize("solver", [
        "outline_and_fill",
        "outline_and_power",
        "outline_and_gmres"
    ])
    def test_solver_completes(self, solver):
        """Test that each solver completes successfully."""
        model = gv.bjm_spatial_triangle(g=20, zi=False)
        model.analyze(solver=solver, tolerance=1e-6, max_iterations=5000)
        
        assert model.analyzed, f"{solver} did not mark model as analyzed"
        assert model.stationary_distribution is not None
        assert abs(model.stationary_distribution.sum() - 1.0) < 1e-4, \
            f"{solver} distribution doesn't sum to 1.0"
    
    def test_outline_and_fill_no_refinement(self):
        """Test that outline_and_fill returns raw interpolated solution."""
        model = gv.bjm_spatial_triangle(g=20, zi=False)
        model.analyze(solver="outline_and_fill")
        
        # Should complete quickly (no refinement)
        assert model.analyzed
        assert model.stationary_distribution.sum() > 0.99
    
    def test_interpolation_matrix_parameter(self):
        """Test that pre-computed interpolation matrix can be passed."""
        model = gv.bjm_spatial_triangle(g=20, zi=False)
        
        # Pre-compute matrix
        coarse_grid = Grid(
            x0=model.grid.x0,
            x1=model.grid.x1,
            xstep=2 * model.grid.xstep,
            y0=model.grid.y0,
            y1=model.grid.y1,
            ystep=2 * model.grid.ystep
        )
        C = create_outline_interpolation_matrix(model.grid, coarse_grid)
        
        # Use pre-computed matrix
        model.analyze(solver="outline_and_fill", interpolation_matrix=C)
        
        assert model.analyzed
        assert abs(model.stationary_distribution.sum() - 1.0) < 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
