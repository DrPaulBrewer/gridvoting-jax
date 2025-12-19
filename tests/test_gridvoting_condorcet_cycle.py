import pytest
import chex
import jax.numpy as jnp


@pytest.mark.parametrize("zi,correct_P", [
    (True, [
        [2./3.,0.,1./3.],
        [1./3.,2./3.,0.],
        [0., 1./3., 2./3.]
    ]),
    (False,[
        [ 1./2., 0, 1./2.],
        [ 1./2., 1./2., 0],
        [ 0,  1./2., 1./2.]
    ])
])
def test_condorcet(zi, correct_P):
    import gridvoting_jax as gv
    import numpy as np
    condorcet_model =  gv.CondorcetCycle(zi=zi)
    assert not condorcet_model.analyzed
    condorcet_model.analyze()
    assert condorcet_model.analyzed
    mc = condorcet_model.MarkovChain
    chex.assert_trees_all_close(
        mc.P,
        jnp.array(correct_P),
        atol=1e-6,
        rtol=0
    )
    chex.assert_trees_all_close(
        condorcet_model.stationary_distribution,
        jnp.array([1.0/3.0, 1.0/3.0, 1.0/3.0]),
        atol=1e-6,
        rtol=0
    )
    mc=condorcet_model.MarkovChain
    alt = mc.solve_for_unit_eigenvector()
    chex.assert_trees_all_close(
        alt,
        jnp.array([1.0/3.0, 1.0/3.0, 1.0/3.0]),
        atol=1e-6,
        rtol=0
    )

