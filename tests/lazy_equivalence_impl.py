
import pytest
import jax
import jax.numpy as jnp
import gridvoting_jax as gv
from copy import deepcopy
from gridvoting_jax.stochastic.markov import iterate_power_method, iterate_bifurcated_power_method
from gridvoting_jax.stochastic.utils import matrix_is_dense

def assert_distributions_close(pi1, pi2, tol_factor):
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


def _iterator_equivalence(*,P_lazy,iterator,tol_factor):
    """Test equivalence of iterator function on lazy and dense versions of a model's Markov Chain."""
    assert not matrix_is_dense(P_lazy), "lazy P matrix should not be dense"
    P_dense = P_lazy.to_dense()
    assert matrix_is_dense(P_dense), "dense P matrix should be dense"
    dense_power_method = iterator(P=P_dense, initial_guess=None, iterations=1)
    lazy_power_method = iterator(P=P_lazy, initial_guess=None, iterations=1)
    assert_distributions_close(
        dense_power_method, 
        lazy_power_method,
        tol_factor=tol_factor
    )    

@pytest.mark.parametrize("g", [20, 40])
def test_power_method_equivalence(g, bmj_g20_mi, bmj_g40_mi):
    """Test standard Power Method equivalence.
    """
    _iterator_equivalence(
        P_lazy=bmj_g20_mi.model.transition_matrix() if g == 20 else bmj_g40_mi.model.transition_matrix(),
        iterator=iterate_power_method,
        tol_factor=5.0
    )

@pytest.mark.parametrize("g", [20, 40])
def test_bifurcated_power_method_equivalence(g, bmj_g20_mi, bmj_g40_mi):
    """Test Bifurcated Power Method equivalence."""    
    _iterator_equivalence(
        P_lazy=bmj_g20_mi.model.transition_matrix() if g == 20 else bmj_g40_mi.model.transition_matrix(),
        iterator=iterate_bifurcated_power_method,
        tol_factor=5.0
    )


def test_condorcet_equivalence(condorcet_mi):
    """ Test lazy and dense equivalence on simple Condorcet cycle model. """
    # Test both power_method and bifurcated_power_method solvers on this small model
    solvers = [
        (iterate_power_method, {"iterations": 20, "initial_guess": None}),
        (iterate_power_method, {"iterations": 20, "initial_guess": jnp.array([1.0,0.0,0.0])}),
        (iterate_power_method, {"iterations": 20, "initial_guess": jnp.array([0.0,1.0,0.0])}),
        (iterate_power_method, {"iterations": 20, "initial_guess": jnp.array([0.0,0.0,1.0])}),
        (iterate_power_method, {"iterations": 20, "initial_guess": jnp.array([0.0,0.5,0.5])}),
        (iterate_bifurcated_power_method, {"iterations": 20, "initial_guess": None})
    ]
    
    for iterator, params in solvers:
        P_lazy = condorcet_mi.transition_matrix()
        P_dense = P_lazy.to_dense()
        
        result_dense = iterator(P=P_dense, **params)
        result_lazy = iterator(P=P_lazy, **params)
        
        assert_distributions_close(
            result_dense, 
            result_lazy,
            tol_factor=5.0
        )
