"""Test stochastic lazy classes"""
import pytest
import jax.numpy as jnp
from gridvoting_jax.core.constants import EPSILON
from gridvoting_jax.stochastic.lazy_stochastic import LazyStochasticMatrix
from gridvoting_jax.stochastic.lazy_stochastic import LazyStochasticMatrixTranspose
from gridvoting_jax.stochastic.lazy_weighted_stochastic import LazyWeightedStochasticMatrix
from gridvoting_jax.stochastic.lazy_weighted_stochastic import LazyWeightedStochasticMatrixTranspose
from gridvoting_jax.stochastic.lazy_q import LazyQMatrix
from gridvoting_jax.stochastic.lazy_q import LazyQMatrixTranspose

M1 =dict(
    mask=jnp.array(
        [[False,False,True],
         [True,False,False],
         [True,True,False]]
    ),
    status_quo_values=jnp.array(
        [0.5,1.0,0.1]
        )
)

D1 = jnp.array([[0.5,0.0,0.5],
               [0.0,1.0,0.0],
               [0.45,0.45,0.1]])

Q1 = jnp.array([[1.0,1.0,1.0],
               [0.0,0.0,0.45],
               [0.5,0.0,-0.9]])

# Weight configurations for LazyWeightedStochasticMatrix tests
W_UNIFORM_1 = jnp.array([1.0, 1.0, 1.0])  # Should match unweighted behavior
W_UNIFORM_2 = jnp.array([2.0, 2.0, 2.0])  # Should match unweighted behavior
W_VARIABLE = jnp.array([1.0, 3.0, 5.0])   # Variable weights for true weighted behavior

# Expected dense matrix for variable weights [1.0, 3.0, 5.0]
# Row 0: mask=[F,F,T], sq=0.5, masked weights sum=5.0
#   M[0,0] = 0.5 (diagonal)
#   M[0,2] = 0.5 * (5.0/5.0) = 0.5
# Row 1: mask=[T,F,F], sq=1.0, masked weights sum=1.0
#   M[1,1] = 1.0 (diagonal)
#   M[1,0] = 0.0 * (1.0/1.0) = 0.0
# Row 2: mask=[T,T,F], sq=0.1, masked weights sum=1.0+3.0=4.0
#   M[2,0] = 0.9 * (1.0/4.0) = 0.225
#   M[2,1] = 0.9 * (3.0/4.0) = 0.675
#   M[2,2] = 0.1 (diagonal)
D1_WEIGHTED = jnp.array([[0.5, 0.0, 0.5],
                         [0.0, 1.0, 0.0],
                         [0.225, 0.675, 0.1]])

@pytest.mark.parametrize("params, expected", [
    (
    M1,D1
    ),
])
def test_lazy_stochastic_matrix_dense(params, expected):
    lazy_stochastic_matrix = LazyStochasticMatrix(**params)
    assert jnp.allclose(lazy_stochastic_matrix.to_dense(), expected, atol=2*EPSILON)

@pytest.mark.parametrize("params, expected", [
    (
    M1,D1
    ),
])
def test_lazy_stochastic_matrix_transpose(params, expected):
    lazy_stochastic_matrix  = LazyStochasticMatrix(**params)
    lazy_stochastic_matrix_transpose = LazyStochasticMatrixTranspose(lazy_stochastic_matrix)
    assert jnp.allclose(lazy_stochastic_matrix_transpose.to_dense(), expected.T, atol=2*EPSILON)

@pytest.mark.parametrize("params, expected", [
    (
    M1,D1
    ),
])
def test_lazy_stochastic_matrix_left_mult_identity(params, expected):
    lazy_stochastic_matrix  = LazyStochasticMatrix(**params)
    assert jnp.allclose(jnp.eye(3) @ lazy_stochastic_matrix , expected, atol=2*EPSILON)

@pytest.mark.parametrize("params, expected", [
    (
    M1,D1
    ),
])
def test_lazy_stochastic_matrix_right_mult_identity(params, expected):
    lazy_stochastic_matrix  = LazyStochasticMatrix(**params)
    assert jnp.allclose(lazy_stochastic_matrix @ jnp.eye(3) , expected, atol=2*EPSILON)

@pytest.mark.parametrize("params, expected", [
    (
    M1,jnp.ones(3)
    ),
])
def test_lazy_stochastic_matrix_right_mult_ones(params, expected):
    lazy_stochastic_matrix  = LazyStochasticMatrix(**params)
    assert jnp.allclose(lazy_stochastic_matrix @ jnp.ones(3) , expected, atol=2*EPSILON)

@pytest.mark.parametrize("params, expected", [
    (
    M1,Q1
    ),
])
def test_lazy_q_matrix_dense(params, expected):
    P = LazyStochasticMatrix(**params)
    Q = LazyQMatrix(P)
    assert jnp.allclose(Q.to_dense(), expected, atol=2*EPSILON)

@pytest.mark.parametrize("params, expected", [
    (
    M1,Q1
    ),
])
def test_lazy_q_matrix_transpose(params, expected):
    P = LazyStochasticMatrix(**params)
    Q = LazyQMatrix(P)
    Q_transpose = LazyQMatrixTranspose(Q)
    assert jnp.allclose(Q_transpose.to_dense(), expected.T, atol=2*EPSILON)

@pytest.mark.parametrize("params, expected", [
    (
    M1,Q1
    ),
])
def test_lazy_q_matrix_left_mult_identity(params, expected):
    P = LazyStochasticMatrix(**params)
    Q = LazyQMatrix(P)
    assert jnp.allclose(jnp.eye(3) @ Q , expected, atol=2*EPSILON)

@pytest.mark.parametrize("params, expected", [
    (
    M1,Q1
    ),
])
def test_lazy_q_matrix_right_mult_identity(params, expected):
    P = LazyStochasticMatrix(**params)
    Q = LazyQMatrix(P)
    assert jnp.allclose(Q @ jnp.eye(3) , expected, atol=2*EPSILON)

# ============================================================================
# LazyWeightedStochasticMatrix Tests
# ============================================================================

# Test dense conversion with uniform weights (all 1.0)
@pytest.mark.parametrize("params, weights, expected", [
    (M1, W_UNIFORM_1, D1),
])
def test_lazy_weighted_stochastic_matrix_dense_uniform_1(params, weights, expected):
    """Test dense conversion with uniform weights=1.0 (should match unweighted)"""
    params_with_weights = {**params, 'weights': weights}
    lazy_weighted = LazyWeightedStochasticMatrix(**params_with_weights)
    assert jnp.allclose(lazy_weighted.to_dense(), expected, atol=2*EPSILON)

# Test dense conversion with uniform weights (all 2.0)
@pytest.mark.parametrize("params, weights, expected", [
    (M1, W_UNIFORM_2, D1),
])
def test_lazy_weighted_stochastic_matrix_dense_uniform_2(params, weights, expected):
    """Test dense conversion with uniform weights=2.0 (should match unweighted)"""
    params_with_weights = {**params, 'weights': weights}
    lazy_weighted = LazyWeightedStochasticMatrix(**params_with_weights)
    assert jnp.allclose(lazy_weighted.to_dense(), expected, atol=2*EPSILON)

# Test dense conversion with variable weights
@pytest.mark.parametrize("params, weights, expected", [
    (M1, W_VARIABLE, D1_WEIGHTED),
])
def test_lazy_weighted_stochastic_matrix_dense_variable(params, weights, expected):
    """Test dense conversion with variable weights [1.0, 3.0, 5.0]"""
    params_with_weights = {**params, 'weights': weights}
    lazy_weighted = LazyWeightedStochasticMatrix(**params_with_weights)
    assert jnp.allclose(lazy_weighted.to_dense(), expected, atol=2*EPSILON)

# Test transpose with uniform weights (all 1.0)
@pytest.mark.parametrize("params, weights, expected", [
    (M1, W_UNIFORM_1, D1),
])
def test_lazy_weighted_stochastic_matrix_transpose_uniform_1(params, weights, expected):
    """Test transpose with uniform weights=1.0 (should match unweighted)"""
    params_with_weights = {**params, 'weights': weights}
    lazy_weighted = LazyWeightedStochasticMatrix(**params_with_weights)
    lazy_weighted_transpose = LazyWeightedStochasticMatrixTranspose(lazy_weighted)
    assert jnp.allclose(lazy_weighted_transpose.to_dense(), expected.T, atol=2*EPSILON)

# Test transpose with uniform weights (all 2.0)
@pytest.mark.parametrize("params, weights, expected", [
    (M1, W_UNIFORM_2, D1),
])
def test_lazy_weighted_stochastic_matrix_transpose_uniform_2(params, weights, expected):
    """Test transpose with uniform weights=2.0 (should match unweighted)"""
    params_with_weights = {**params, 'weights': weights}
    lazy_weighted = LazyWeightedStochasticMatrix(**params_with_weights)
    lazy_weighted_transpose = LazyWeightedStochasticMatrixTranspose(lazy_weighted)
    assert jnp.allclose(lazy_weighted_transpose.to_dense(), expected.T, atol=2*EPSILON)

# Test transpose with variable weights
@pytest.mark.parametrize("params, weights, expected", [
    (M1, W_VARIABLE, D1_WEIGHTED),
])
def test_lazy_weighted_stochastic_matrix_transpose_variable(params, weights, expected):
    """Test transpose with variable weights [1.0, 3.0, 5.0]"""
    params_with_weights = {**params, 'weights': weights}
    lazy_weighted = LazyWeightedStochasticMatrix(**params_with_weights)
    lazy_weighted_transpose = LazyWeightedStochasticMatrixTranspose(lazy_weighted)
    assert jnp.allclose(lazy_weighted_transpose.to_dense(), expected.T, atol=2*EPSILON)

# Test left multiplication with identity (uniform weights 1.0)
@pytest.mark.parametrize("params, weights, expected", [
    (M1, W_UNIFORM_1, D1),
])
def test_lazy_weighted_stochastic_matrix_left_mult_identity_uniform_1(params, weights, expected):
    """Test left multiplication with identity, uniform weights=1.0"""
    params_with_weights = {**params, 'weights': weights}
    lazy_weighted = LazyWeightedStochasticMatrix(**params_with_weights)
    assert jnp.allclose(jnp.eye(3) @ lazy_weighted, expected, atol=2*EPSILON)

# Test left multiplication with identity (uniform weights 2.0)
@pytest.mark.parametrize("params, weights, expected", [
    (M1, W_UNIFORM_2, D1),
])
def test_lazy_weighted_stochastic_matrix_left_mult_identity_uniform_2(params, weights, expected):
    """Test left multiplication with identity, uniform weights=2.0"""
    params_with_weights = {**params, 'weights': weights}
    lazy_weighted = LazyWeightedStochasticMatrix(**params_with_weights)
    assert jnp.allclose(jnp.eye(3) @ lazy_weighted, expected, atol=2*EPSILON)

# Test left multiplication with identity (variable weights)
@pytest.mark.parametrize("params, weights, expected", [
    (M1, W_VARIABLE, D1_WEIGHTED),
])
def test_lazy_weighted_stochastic_matrix_left_mult_identity_variable(params, weights, expected):
    """Test left multiplication with identity, variable weights [1.0, 3.0, 5.0]"""
    params_with_weights = {**params, 'weights': weights}
    lazy_weighted = LazyWeightedStochasticMatrix(**params_with_weights)
    assert jnp.allclose(jnp.eye(3) @ lazy_weighted, expected, atol=2*EPSILON)

# Test right multiplication with identity (uniform weights 1.0)
@pytest.mark.parametrize("params, weights, expected", [
    (M1, W_UNIFORM_1, D1),
])
def test_lazy_weighted_stochastic_matrix_right_mult_identity_uniform_1(params, weights, expected):
    """Test right multiplication with identity, uniform weights=1.0"""
    params_with_weights = {**params, 'weights': weights}
    lazy_weighted = LazyWeightedStochasticMatrix(**params_with_weights)
    assert jnp.allclose(lazy_weighted @ jnp.eye(3), expected, atol=2*EPSILON)

# Test right multiplication with identity (uniform weights 2.0)
@pytest.mark.parametrize("params, weights, expected", [
    (M1, W_UNIFORM_2, D1),
])
def test_lazy_weighted_stochastic_matrix_right_mult_identity_uniform_2(params, weights, expected):
    """Test right multiplication with identity, uniform weights=2.0"""
    params_with_weights = {**params, 'weights': weights}
    lazy_weighted = LazyWeightedStochasticMatrix(**params_with_weights)
    assert jnp.allclose(lazy_weighted @ jnp.eye(3), expected, atol=2*EPSILON)

# Test right multiplication with identity (variable weights)
@pytest.mark.parametrize("params, weights, expected", [
    (M1, W_VARIABLE, D1_WEIGHTED),
])
def test_lazy_weighted_stochastic_matrix_right_mult_identity_variable(params, weights, expected):
    """Test right multiplication with identity, variable weights [1.0, 3.0, 5.0]"""
    params_with_weights = {**params, 'weights': weights}
    lazy_weighted = LazyWeightedStochasticMatrix(**params_with_weights)
    assert jnp.allclose(lazy_weighted @ jnp.eye(3), expected, atol=2*EPSILON)

# Test right multiplication with ones vector (uniform weights 1.0)
@pytest.mark.parametrize("params, weights, expected", [
    (M1, W_UNIFORM_1, jnp.ones(3)),
])
def test_lazy_weighted_stochastic_matrix_right_mult_ones_uniform_1(params, weights, expected):
    """Test right multiplication with ones vector, uniform weights=1.0"""
    params_with_weights = {**params, 'weights': weights}
    lazy_weighted = LazyWeightedStochasticMatrix(**params_with_weights)
    assert jnp.allclose(lazy_weighted @ jnp.ones(3), expected, atol=2*EPSILON)

# Test right multiplication with ones vector (uniform weights 2.0)
@pytest.mark.parametrize("params, weights, expected", [
    (M1, W_UNIFORM_2, jnp.ones(3)),
])
def test_lazy_weighted_stochastic_matrix_right_mult_ones_uniform_2(params, weights, expected):
    """Test right multiplication with ones vector, uniform weights=2.0"""
    params_with_weights = {**params, 'weights': weights}
    lazy_weighted = LazyWeightedStochasticMatrix(**params_with_weights)
    assert jnp.allclose(lazy_weighted @ jnp.ones(3), expected, atol=2*EPSILON)

# Test right multiplication with ones vector (variable weights)
@pytest.mark.parametrize("params, weights, expected", [
    (M1, W_VARIABLE, jnp.ones(3)),
])
def test_lazy_weighted_stochastic_matrix_right_mult_ones_variable(params, weights, expected):
    """Test right multiplication with ones vector, variable weights [1.0, 3.0, 5.0]"""
    params_with_weights = {**params, 'weights': weights}
    lazy_weighted = LazyWeightedStochasticMatrix(**params_with_weights)
    assert jnp.allclose(lazy_weighted @ jnp.ones(3), expected, atol=2*EPSILON)

# Test equivalence to unweighted LazyStochasticMatrix
def test_lazy_weighted_equivalence_to_unweighted():
    """Verify LazyWeightedStochasticMatrix with uniform weights produces identical results to LazyStochasticMatrix"""
    lazy_unweighted = LazyStochasticMatrix(**M1)
    
    # Test with uniform weights = 1.0
    lazy_weighted_1 = LazyWeightedStochasticMatrix(**M1, weights=W_UNIFORM_1)
    assert jnp.allclose(lazy_unweighted.to_dense(), lazy_weighted_1.to_dense(), atol=2*EPSILON)
    assert jnp.allclose(lazy_unweighted.T.to_dense(), lazy_weighted_1.T.to_dense(), atol=2*EPSILON)
    assert jnp.allclose(lazy_unweighted @ jnp.ones(3), lazy_weighted_1 @ jnp.ones(3), atol=2*EPSILON)
    assert jnp.allclose(jnp.eye(3) @ lazy_unweighted, jnp.eye(3) @ lazy_weighted_1, atol=2*EPSILON)
    
    # Test with uniform weights = 2.0
    lazy_weighted_2 = LazyWeightedStochasticMatrix(**M1, weights=W_UNIFORM_2)
    assert jnp.allclose(lazy_unweighted.to_dense(), lazy_weighted_2.to_dense(), atol=2*EPSILON)
    assert jnp.allclose(lazy_unweighted.T.to_dense(), lazy_weighted_2.T.to_dense(), atol=2*EPSILON)
    assert jnp.allclose(lazy_unweighted @ jnp.ones(3), lazy_weighted_2 @ jnp.ones(3), atol=2*EPSILON)
    assert jnp.allclose(jnp.eye(3) @ lazy_unweighted, jnp.eye(3) @ lazy_weighted_2, atol=2*EPSILON)
