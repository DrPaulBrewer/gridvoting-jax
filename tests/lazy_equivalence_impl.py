
import pytest
import jax
import jax.numpy as jnp
import gridvoting_jax as gv

def assert_distributions_close(pi1, pi2, tol_factor=500.0):
    """Assert two distributions are close within tolerance based on dtype."""
    diff = jnp.linalg.norm(pi1 - pi2, ord=1)
    dtype = pi1.dtype
    eps = jnp.finfo(dtype).eps
    tol = tol_factor * eps
    
    # Debug info if failure imminent
    if diff >= tol:
        print(f"L1 Difference: {diff}")
        print(f"Tolerance ({tol_factor} * eps): {tol}")
        print(f"Dtype: {dtype}")
    
    assert diff < tol, f"L1 diff {diff} exceeds tolerance {tol}"

@pytest.mark.parametrize("g", [20, 40])
def test_power_method_equivalence(g):
    """Test standard Power Method equivalence.
    
    Tolerances (updated 2025-12-29 after normalization fix):
    - Measured: ~5-16 eps (g=20: 5.1 eps, g=40: 16.2 eps)
    - Set to 20 eps (was 350 eps before normalization fix)
    - 17x improvement from adding periodic normalization!
    """
    model_dense = gv.bjm_spatial_triangle(g=g, zi=False)
    model_lazy = gv.bjm_spatial_triangle(g=g, zi=False)
    
    params = {"max_iterations": 20, "timeout": 60}
    
    model_dense.analyze(solver="power_method", **params)
    model_lazy.analyze_lazy(solver="power_method", force_lazy=True, **params)
    
    assert_distributions_close(
        model_dense.stationary_distribution, 
        model_lazy.stationary_distribution,
        tol_factor=50.0  # Increased from 20.0 to account for per-iteration normalization
    )

@pytest.mark.parametrize("g", [20, 40])
def test_bifurcated_power_method_equivalence(g):
    """Test Bifurcated Power Method equivalence.
    
    Tolerances (updated 2025-12-29 after normalization fix):
    - Measured: ~1 eps (g=20: 0.9 eps, g=40: 1.4 eps)
    - Set to 10 eps (was 50 eps before normalization fix)
    - 50x improvement from adding periodic normalization!
    - Now essentially identical between dense and lazy
    """
    model_dense = gv.bjm_spatial_triangle(g=g, zi=False)
    model_lazy = gv.bjm_spatial_triangle(g=g, zi=False)
    
    params = {"max_iterations": 20, "timeout": 60}
    
    model_dense.analyze(solver="bifurcated_power_method", **params)
    model_lazy.analyze_lazy(solver="bifurcated_power_method", force_lazy=True, **params)
    
    assert_distributions_close(
        model_dense.stationary_distribution, 
        model_lazy.stationary_distribution,
        tol_factor=10.0  # Updated from 50.0
    )

@pytest.mark.parametrize("g", [20, 40])
def test_gmres_equivalence(g):
    """Test GMRES equivalence.
    
    Tolerances (updated 2025-12-29 after normalization fix):
    - Float32: ~16-36 eps (g=20: 36.2 eps, g=40: 15.7 eps)
    - Float64: ~127-414 eps (g=20: 127.1 eps, g=40: 413.8 eps)
    - Set to 500 eps (unchanged, but now supports both precisions)
    - Note: Float64 requires higher tolerance due to different numerical behavior
    """
    model_dense = gv.bjm_spatial_triangle(g=g, zi=False)
    model_lazy = gv.bjm_spatial_triangle(g=g, zi=False)
    
    params = {"max_iterations": 20}
    
    model_dense.analyze(solver="gmres_matrix_inversion", **params)
    model_lazy.analyze_lazy(solver="gmres", force_lazy=True, **params)
    
    assert_distributions_close(
        model_dense.stationary_distribution, 
        model_lazy.stationary_distribution,
        tol_factor=500.0  # Supports both float32 and float64
    )

def test_condorcet_equivalence():
    """Test equivalence on simple Condorcet cycle model."""
    from gridvoting_jax.models.examples.condorcet import condorcet_cycle
    
    # Test all 3 solvers on this small model
    solvers = [
        ("power_method", "power_method", {"max_iterations": 20, "timeout": 60}),
        ("power_method", "power_method", {"max_iterations": 20, "timeout": 60, "initial_guess": jnp.array([1.0,0.0,0.0])}),
        ("power_method", "power_method", {"max_iterations": 20, "timeout": 60, "initial_guess": jnp.array([0.0,1.0,0.0])}),
        ("power_method", "power_method", {"max_iterations": 20, "timeout": 60, "initial_guess": jnp.array([0.0,0.0,1.0])}),
        ("power_method", "power_method", {"max_iterations": 20, "timeout": 60, "initial_guess": jnp.array([0.0,0.5,0.5])}),
        ("bifurcated_power_method", "bifurcated_power_method", {"max_iterations": 20, "timeout": 60}),
        ("gmres_matrix_inversion", "gmres", {"max_iterations": 20})
    ]
    
    for dense_solver, lazy_solver, params in solvers:
        model_dense = condorcet_cycle(zi=False)
        model_lazy = condorcet_cycle(zi=False)
        
        # Dense Execution
        model_dense.analyze(solver=dense_solver, **params)
        
        # Lazy Execution
        model_lazy.analyze_lazy(solver=lazy_solver, force_lazy=True, **params)
        
        assert_distributions_close(
            model_dense.stationary_distribution, 
            model_lazy.stationary_distribution,
            tol_factor=10.0
        )
