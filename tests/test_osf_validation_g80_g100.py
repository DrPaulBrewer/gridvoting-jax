"""OSF validation tests for g=80 and g=100 BJM spatial triangle.

Tests validate lazy grid upscaling solver against OSF reference data.
Includes both float32 (default) and float64 (high precision) versions.

Note: Float64 tests run on CPU to avoid GPU memory issues.
"""

import pytest
import os
import sys
import numpy as np
import jax.numpy as jnp
import chex

pytestmark = pytest.mark.benchmarks

# Force CPU for float64 tests to avoid GPU memory issues
# This is set before importing gridvoting_jax
if 'GV_ENABLE_FLOAT64' in os.environ or any('Float64' in arg for arg in sys.argv):
    os.environ['JAX_PLATFORMS'] = 'cpu'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import gridvoting_jax as gv
from gridvoting_jax.benchmarks.osf_comparison import load_osf_distribution


class TestOSFValidation:
    """Test suite for OSF reference validation."""
    
    @pytest.fixture(scope="class")
    def osf_g80_reference(self):
        """Load OSF reference data for g=80."""
        ref_data = load_osf_distribution(g=80, zi=False)
        if ref_data is None:
            pytest.skip("OSF reference data not available for g=80")
        
        # Extract stationary distribution from log10prob
        ref_statd = 10 ** ref_data['log10prob'].values
        return np.array(ref_statd)
    
    def test_g80_lazy_grid_upscaling_vs_osf(self, osf_g80_reference):
        """Test g=80 lazy grid upscaling against OSF reference (float32)."""
        # Create and solve g=80 model
        model_80 = gv.bjm_spatial_triangle(g=80, zi=False)
        model_80.analyze(solver="lazy_grid_upscaling", max_iterations=3000)
        
        # Compare with OSF reference
        diff = np.abs(np.array(model_80.stationary_distribution) - osf_g80_reference)
        max_abs_diff = np.max(diff)
        l1_diff = np.sum(diff)
        
        # Validate using absolute error (not relative, which is misleading for small values)
        assert max_abs_diff < 1e-6, f"Max absolute difference {max_abs_diff:.2e} exceeds 1e-6"
        assert l1_diff < 1e-5, f"L1 difference {l1_diff:.2e} exceeds 1e-5"
        
        # Check probability sum
        prob_sum = jnp.sum(model_80.stationary_distribution)
        assert abs(prob_sum - 1.0) < 1e-5, f"Probability sum {prob_sum:.6f} not close to 1.0"
    
    @pytest.mark.slow
    @pytest.mark.large_grid
    def test_g100_subgrid_validation(self, osf_g80_reference):
        """Test g=100 lazy grid upscaling with g=80 subgrid validation (float32)."""
        # Create and solve g=80 for reference
        model_80 = gv.bjm_spatial_triangle(g=80, zi=False)
        model_80.analyze(solver="lazy_grid_upscaling", max_iterations=3000)
        
        # Create and solve g=100
        model_100 = gv.bjm_spatial_triangle(g=100, zi=False)
        model_100.analyze(solver="lazy_grid_upscaling", max_iterations=3000)
        
        # Extract g=80 subgrid from g=100
        grid_80 = model_80.grid
        grid_100 = model_100.grid
        
        x0, x1 = grid_80.x.min(), grid_80.x.max()
        y0, y1 = grid_80.y.min(), grid_80.y.max()
        
        box_mask = grid_100.within_box(x0=x0, x1=x1, y0=y0, y1=y1)
        statd_100_subgrid = model_100.stationary_distribution[box_mask]
        statd_100_outside = jnp.sum(model_100.stationary_distribution[~box_mask])
        
        # Validate subgrid matches g=80
        assert statd_100_subgrid.shape == model_80.stationary_distribution.shape, \
            "Subgrid shape mismatch"
        
        diff_subgrid = jnp.abs(statd_100_subgrid - model_80.stationary_distribution)
        max_abs_diff = jnp.max(diff_subgrid)
        
        assert max_abs_diff < 1e-6, \
            f"Subgrid max absolute difference {max_abs_diff:.2e} exceeds 1e-6"
        
        # Probability outside should be very small
        # Based on exponential decay analysis: expected ~7e-7
        # Allow up to 1e-5 for float32 precision
        assert statd_100_outside < 1e-5, \
            f"Probability outside g=80 box {statd_100_outside:.2e} exceeds 1e-5"
        
        # Check total probability
        prob_sum = jnp.sum(model_100.stationary_distribution)
        assert abs(prob_sum - 1.0) < 1e-5, \
            f"Probability sum {prob_sum:.6f} not close to 1.0"


class TestOSFValidationFloat64:
    """High-precision float64 tests for OSF validation."""
    
    @pytest.fixture(scope="class", autouse=True)
    def enable_float64(self):
        """Enable float64 precision for these tests."""
        import jax
        original_state = jax.config.read('jax_enable_x64')
        jax.config.update('jax_enable_x64', True)
        yield
        jax.config.update('jax_enable_x64', original_state)
    
    @pytest.fixture(scope="class")
    def osf_g80_reference(self):
        """Load OSF reference data for g=80."""
        ref_data = load_osf_distribution(g=80, zi=False)
        if ref_data is None:
            pytest.skip("OSF reference data not available for g=80")
        
        ref_statd = 10 ** ref_data['log10prob'].values
        return np.array(ref_statd, dtype=np.float64)
    

    
    @pytest.mark.slow
    @pytest.mark.large_grid
    def test_g80_float64_vs_osf(self, osf_g80_reference):
        """Test g=80 with float64 precision against OSF reference."""
        model_80 = gv.bjm_spatial_triangle(g=80, zi=False)
        model_80.analyze(solver="lazy_grid_upscaling", max_iterations=3000)
        
        # Compare with OSF reference
        diff = np.abs(np.array(model_80.stationary_distribution) - osf_g80_reference)
        max_abs_diff = np.max(diff)
        l1_diff = np.sum(diff)
        
        # With float64, expect even better precision
        assert max_abs_diff < 1e-8, \
            f"Float64: Max absolute difference {max_abs_diff:.2e} exceeds 1e-8"
        assert l1_diff < 1e-7, \
            f"Float64: L1 difference {l1_diff:.2e} exceeds 1e-7"
        
        prob_sum = jnp.sum(model_80.stationary_distribution)
        assert abs(prob_sum - 1.0) < 1e-10, \
            f"Float64: Probability sum {prob_sum:.12f} not close to 1.0"
    
    @pytest.mark.slow
    @pytest.mark.large_grid
    def test_g100_outside_probability_float64(self, osf_g80_reference):
        """Test g=100 with float64 to measure probability outside g=80 box.
        
        Validates that outside probability is small but non-zero (~1.5e-09).
        """
        # Solve g=80 for reference
        model_80 = gv.bjm_spatial_triangle(g=80, zi=False)
        model_80.analyze(solver="lazy_grid_upscaling", max_iterations=3000)
        
        # Solve g=100 with float64 precision
        model_100 = gv.bjm_spatial_triangle(g=100, zi=False)
        model_100.analyze(solver="lazy_grid_upscaling", max_iterations=3000)
        
        # Extract g=80 subgrid
        grid_80 = model_80.grid
        grid_100 = model_100.grid
        
        x0, x1 = grid_80.x.min(), grid_80.x.max()
        y0, y1 = grid_80.y.min(), grid_80.y.max()
        
        box_mask = grid_100.within_box(x0=x0, x1=x1, y0=y0, y1=y1)
        statd_100_subgrid = model_100.stationary_distribution[box_mask]
        statd_100_outside = float(jnp.sum(model_100.stationary_distribution[~box_mask]))
        
        # Report the actual value
        print(f"\n  Probability outside g=80 box: {statd_100_outside:.10e}")
        
        # Validate subgrid accuracy
        diff_subgrid = jnp.abs(statd_100_subgrid - model_80.stationary_distribution)
        max_abs_diff = float(jnp.max(diff_subgrid))
        
        assert max_abs_diff < 1e-8, \
            f"Float64: Subgrid max absolute difference {max_abs_diff:.2e} exceeds 1e-8"
        
        # Validate outside probability
        # Empirical criterion: should be small but non-zero
        # Based on actual solver results: ~1.5e-09 for both GMRES and power method
        if statd_100_outside > 0:
            # Should be less than 1e-07 (very small)
            assert statd_100_outside < 1e-07, \
                f"Outside probability {statd_100_outside:.2e} exceeds 1e-07"
        else:
            # Should not be exactly zero (would indicate numerical underflow)
            pytest.fail(
                f"Probability outside g=80 box is exactly 0. Expected small but non-zero value (~1e-09)."
            )
        
        # Total probability should still sum to 1
        prob_sum = jnp.sum(model_100.stationary_distribution)
        assert abs(prob_sum - 1.0) < 1e-10, \
            f"Float64: Probability sum {prob_sum:.12f} not close to 1.0"
        
        # Additional analysis: show distribution by distance
        x_100 = np.array(grid_100.x)
        y_100 = np.array(grid_100.y)
        r_100 = np.sqrt(x_100**2 + y_100**2)
        
        outside_mask = ~np.array(box_mask)
        r_outside = r_100[outside_mask]
        prob_outside = np.array(model_100.stationary_distribution[outside_mask])
        
        # Show distribution by distance bins
        bins = [80, 90, 100, 110, 120, 150]
        print(f"\n  Distribution by distance:")
        for i in range(len(bins)-1):
            bin_mask = (r_outside >= bins[i]) & (r_outside < bins[i+1])
            if np.sum(bin_mask) > 0:
                bin_prob = np.sum(prob_outside[bin_mask])
                bin_count = np.sum(bin_mask)
                pct = bin_prob / statd_100_outside * 100 if statd_100_outside > 0 else 0
                print(f"    r ∈ [{bins[i]:3d}, {bins[i+1]:3d}): {bin_count:5d} points, "
                      f"prob={bin_prob:.6e} ({pct:5.1f}%)")


class TestPowerMethodGPU:
    """GPU tests for lazy power method solver."""
    
    @pytest.fixture(scope="class")
    def osf_g80_reference(self):
        """Load OSF reference data for g=80."""
        ref_data = load_osf_distribution(g=80, zi=False)
        if ref_data is None:
            pytest.skip("OSF reference data not available for g=80")
        
        ref_statd = 10 ** ref_data['log10prob'].values
        return np.array(ref_statd)
    

    
    @pytest.fixture(scope="class", autouse=True)
    def check_gpu_available(self):
        """Skip tests if GPU not available."""
        import jax
        if jax.devices()[0].platform != 'gpu':
            pytest.skip("GPU not available")
    
    @pytest.mark.slow
    @pytest.mark.large_grid
    def test_g80_power_method_gpu(self, osf_g80_reference):
        """Test g=80 with lazy power method on GPU (float32).
        
        Note: zi=False provides faster power method convergence than zi=True.
        """
        model_80 = gv.bjm_spatial_triangle(g=80, zi=False)
        
        # Use lazy power method
        model_80.analyze_lazy(force_lazy=True, solver="power_method", 
                             max_iterations=5000, timeout=120)
        
        # Compare with OSF reference
        diff = np.abs(np.array(model_80.stationary_distribution) - osf_g80_reference)
        max_abs_diff = np.max(diff)
        l1_diff = np.sum(diff)
        
        # Power method should also be accurate
        assert max_abs_diff < 1e-6, \
            f"Power method: Max absolute difference {max_abs_diff:.2e} exceeds 1e-6"
        assert l1_diff < 1e-5, \
            f"Power method: L1 difference {l1_diff:.2e} exceeds 1e-5"
        
        prob_sum = jnp.sum(model_80.stationary_distribution)
        assert abs(prob_sum - 1.0) < 1e-5, \
            f"Power method: Probability sum {prob_sum:.6f} not close to 1.0"
    
    @pytest.mark.slow
    @pytest.mark.large_grid
    def test_g100_power_method_outside_probability_gpu(self, osf_g80_reference):
        """Test g=100 with lazy power method on GPU to compare outside probability.
        
        Power method starts with uniform distribution, so it should explore
        the full space more thoroughly than grid upscaling.
        
        Note: zi=False provides faster power method convergence than zi=True.
        """
        # Solve g=80 for reference
        model_80 = gv.bjm_spatial_triangle(g=80, zi=False)
        model_80.analyze_lazy(force_lazy=True, solver="power_method", 
                             max_iterations=5000, timeout=120)
        
        # Solve g=100 with power method
        model_100 = gv.bjm_spatial_triangle(g=100, zi=False)
        model_100.analyze_lazy(force_lazy=True, solver="power_method",
                              max_iterations=5000, timeout=300)
        
        # Extract g=80 subgrid
        grid_80 = model_80.grid
        grid_100 = model_100.grid
        
        x0, x1 = grid_80.x.min(), grid_80.x.max()
        y0, y1 = grid_80.y.min(), grid_80.y.max()
        
        box_mask = grid_100.within_box(x0=x0, x1=x1, y0=y0, y1=y1)
        statd_100_subgrid = model_100.stationary_distribution[box_mask]
        statd_100_outside = float(jnp.sum(model_100.stationary_distribution[~box_mask]))
        
        # Report the actual value
        print(f"\n  Power Method Results:")
        print(f"  Probability outside g=80 box: {statd_100_outside:.10e}")
        
        # Compare with GMRES result
        gmres_outside = 1.51e-09  # From float64 CPU test
        print(f"\n  Comparison with GMRES grid upscaling:")
        print(f"  GMRES result (CPU):  {gmres_outside:.10e}")
        print(f"  Power method (GPU):  {statd_100_outside:.10e}")
        print(f"  Ratio (power/GMRES): {statd_100_outside/gmres_outside:.2f}x")
        
        # Validate subgrid accuracy
        diff_subgrid = jnp.abs(statd_100_subgrid - model_80.stationary_distribution)
        max_abs_diff = float(jnp.max(diff_subgrid))
        
        assert max_abs_diff < 1e-6, \
            f"Power method: Subgrid max absolute difference {max_abs_diff:.2e} exceeds 1e-6"
        
        # Validate outside probability
        # Empirical criterion: should be small but non-zero (~1.5e-09)
        if statd_100_outside > 0:
            assert statd_100_outside < 1e-07, \
                f"Outside probability {statd_100_outside:.2e} exceeds 1e-07"
        else:
            pytest.fail(
                f"Probability outside g=80 box is exactly 0. Expected small but non-zero value (~1e-09)."
            )
        
        # Total probability should sum to 1
        prob_sum = jnp.sum(model_100.stationary_distribution)
        assert abs(prob_sum - 1.0) < 1e-5, \
            f"Power method: Probability sum {prob_sum:.6f} not close to 1.0"
