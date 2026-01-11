"""Test stochastic/utils.py functions"""
from gridvoting_jax.core.constants import EPSILON
import pytest
import jax.numpy as jnp
from gridvoting_jax.stochastic.utils import _move_neg_prob_to_max, entropy_in_bits, normalize_if_needed

def test_move_neg_prob_to_max():
    pvector = jnp.array([0.5, -0.01, 0.51])
    pvector = _move_neg_prob_to_max(pvector)
    assert pvector[0]==0.5
    assert pvector[1]==0.0
    assert abs(pvector[2]-0.5)<2*EPSILON

@pytest.mark.parametrize("pvector, expected", [
    (jnp.array([0.5, 0.5]), 1.0),
    (jnp.array([0.25, 0.25, 0.25, 0.25]), 2.0),
    (jnp.array([1.0,0.0]), 0.0),
    (jnp.array([0.0,1.0]), 0.0),
    (jnp.array([0.0,0.0,1.0]), 0.0),
    (jnp.array([0.0,0.0,0.0,1.0]), 0.0),
    (jnp.array([0.0,0.0,0.0,0.0,1.0]), 0.0),
    (jnp.array([0.0,0.0,0.0,0.0,0.0,1.0]), 0.0),
    (jnp.array([[0.5,0.5],[1.0,0.0]]), jnp.array([1.0,0.0])),
    (jnp.array([[0.25,0.25,0.25,0.25],[0.25,0.25,0.25,0.25], [0.0,0.5,0.0,0.5]]), jnp.array([2.0,2.0,1.0]))
])
def test_entropy_in_bits(pvector, expected):
    assert jnp.allclose(entropy_in_bits(pvector), expected, atol=2*EPSILON)


@pytest.mark.parametrize("parray, expected", [
    (jnp.array([0.5, 0.5]), jnp.array([0.5, 0.5])),
    (jnp.array([0.25, 0.25, 0.25, 0.25]), jnp.array([0.25, 0.25, 0.25, 0.25])),
    (jnp.array([1.0,0.0]), jnp.array([1.0,0.0])),
    (jnp.array([0.0,1.0]), jnp.array([0.0,1.0])),
    (jnp.array([0.0,0.0,1.0]), jnp.array([0.0,0.0,1.0])),
    (jnp.array([0.0,0.0,0.0,1.0]), jnp.array([0.0,0.0,0.0,1.0])),
    (jnp.array([0.0,0.0,0.0,0.0,1.0]), jnp.array([0.0,0.0,0.0,0.0,1.0])),
    (jnp.array([0.0,0.0,0.0,0.0,0.0,1.0]), jnp.array([0.0,0.0,0.0,0.0,0.0,1.0])),
    (jnp.array([[0.5,0.5],[1.0,0.0]]), jnp.array([[0.5,0.5],[1.0,0.0]])),
    (jnp.array([[0.25,0.25,0.25,0.25],[0.25,0.25,0.25,0.25], [0.0,0.5,0.0,0.5]]), jnp.array([[0.25,0.25,0.25,0.25],[0.25,0.25,0.25,0.25], [0.0,0.5,0.0,0.5]])),
    (jnp.array([100,100,100,100]), jnp.array([0.25,0.25,0.25,0.25])),
    (jnp.array([[100,100,100,100],[100,100,100,100],[100,100,100,100]]), jnp.array([[0.25,0.25,0.25,0.25],[0.25,0.25,0.25,0.25],[0.25,0.25,0.25,0.25]])),
    (jnp.array([[10,10,10,10],[0,1,0,0],[0,0,4,4]]), jnp.array([[0.25,0.25,0.25,0.25],[0.0,1.0,0.0,0.0],[0.0,0.0,0.5,0.5]]))
    ])
def test_normalize_if_needed(parray, expected):
    assert jnp.allclose(normalize_if_needed(parray), expected, atol=2*EPSILON)
    
