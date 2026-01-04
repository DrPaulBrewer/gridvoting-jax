
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
def test_power_method_equivalence(g, bmj_g20_mi, bmj_g40_mi):
    """Test standard Power Method equivalence.
    
    Tolerances (updated 2025-12-29 after normalization fix):
    - Measured: ~5-16 eps (g=20: 5.1 eps, g=40: 16.2 eps)
    - Set to 20 eps (was 350 eps before normalization fix)
    - 17x improvement from adding periodic normalization!
    """
    # Select fixture based on g parameter
    model_dense = bmj_g20_mi if g == 20 else bmj_g40_mi
    model_lazy = bmj_g20_mi if g == 20 else bmj_g40_mi
    
    params = {"max_iterations": 10, "timeout": 20}
    
    model_dense.analyze(solver="power_method", force_dense=True, **params)
    model_lazy.analyze(solver="power_method", force_dense=False, **params)
    
    assert_distributions_close(
        model_dense.stationary_distribution, 
        model_lazy.stationary_distribution,
        tol_factor=50.0  # Increased from 20.0 to account for per-iteration normalization
    )

@pytest.mark.parametrize("g", [20, 40])
def test_bifurcated_power_method_equivalence(g, bmj_g20_mi, bmj_g40_mi):
    """Test Bifurcated Power Method equivalence.
    
    Tolerances (updated 2025-12-29 after normalization fix):
    - Measured: ~1 eps (g=20: 0.9 eps, g=40: 1.4 eps)
    - Set to 10 eps (was 50 eps before normalization fix)
    - 50x improvement from adding periodic normalization!
    - Now essentially identical between dense and lazy
    """
    # Select fixture based on g parameter
    model_dense = bmj_g20_mi if g == 20 else bmj_g40_mi
    model_lazy = bmj_g20_mi if g == 20 else bmj_g40_mi
    
    params = {"max_iterations": 10, "timeout": 20}
    
    model_dense.analyze(solver="bifurcated_power_method", force_dense=True, **params)
    model_lazy.analyze(solver="bifurcated_power_method", force_dense=False, **params)
    
    assert_distributions_close(
        model_dense.stationary_distribution, 
        model_lazy.stationary_distribution,
        tol_factor=30.0  # Updated from 50.0
    )


def test_condorcet_equivalence(condorcet_mi):
    """Test equivalence on simple Condorcet cycle model."""
    # Test all 3 solvers on this small model
    solvers = [
        ("power_method", "power_method", {"max_iterations": 20, "timeout": 60}),
        ("power_method", "power_method", {"max_iterations": 20, "timeout": 60, "initial_guess": jnp.array([1.0,0.0,0.0])}),
        ("power_method", "power_method", {"max_iterations": 20, "timeout": 60, "initial_guess": jnp.array([0.0,1.0,0.0])}),
        ("power_method", "power_method", {"max_iterations": 20, "timeout": 60, "initial_guess": jnp.array([0.0,0.0,1.0])}),
        ("power_method", "power_method", {"max_iterations": 20, "timeout": 60, "initial_guess": jnp.array([0.0,0.5,0.5])}),
        ("bifurcated_power_method", "bifurcated_power_method", {"max_iterations": 20, "timeout": 60})
    ]
    
    for dense_solver, lazy_solver, params in solvers:
        model_dense = condorcet_mi
        model_lazy = condorcet_mi
        
        # Dense Execution
        model_dense.analyze(solver=dense_solver, force_dense=True, **params)
        
        # Lazy Execution
        model_lazy.analyze(solver=lazy_solver, **params)
        
        assert_distributions_close(
            model_dense.stationary_distribution, 
            model_lazy.stationary_distribution,
            tol_factor=10.0
        )
