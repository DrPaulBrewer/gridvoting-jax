import pytest


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
    np.testing.assert_array_almost_equal(
        np.array(mc.P),  # Convert JAX to NumPy
        np.array(correct_P),
        decimal=6
    )
    np.testing.assert_array_almost_equal(
        np.array(condorcet_model.stationary_distribution),  # Convert JAX to NumPy
        np.array([1.0/3.0,1.0/3.0,1.0/3.0]),
        decimal=6
    )
    mc=condorcet_model.MarkovChain
    alt = mc.solve_for_unit_eigenvector()
    np.testing.assert_array_almost_equal(
        np.array(alt),  # Convert JAX to NumPy
        np.array([1.0/3.0,1.0/3.0,1.0/3.0]),
        decimal=6
    )

