"""Test stochastic lazy classes"""
import pytest
import jax.numpy as jnp
from gridvoting_jax.core.constants import EPSILON
from gridvoting_jax.stochastic.lazy_stochastic import LazyStochasticMatrix
from gridvoting_jax.stochastic.lazy_stochastic import LazyStochasticMatrixTranspose
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
