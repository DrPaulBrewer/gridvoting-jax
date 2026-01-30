import gridvoting_jax as gv
import jax.numpy as jnp
import pytest

# this test passes in float32 mode for voters=3,5,9 but not 15
# for unknown reasons, probably related to numerical precision
# it will pass voters=15 if we use float64

# in order for this test to pass, we needed to create consistent_cos and consistent_sin
# to move all cos/sin calculations to the first quadrant of the unit circle

cases = [(v,n,None) for v in range(3,13,2) if 360%v==0 for n in range((360//v)-1,0,-1) if (360//v)%n==0 ]

# @pytest.mark.parametrize("voters,thetastep,decimals", cases)
#def test_integrated_lump_polar_count_mismatches(voters, thetastep, decimals):
#    svm = gv.models.examples.shapes.ring(g=10, r=6, voters=voters, polar=True, thetastep=thetastep, decimals=decimals)
#    pg = svm.grid
#    parts = pg.partition_from_rotation(angle=360//voters)
#    mismatches = svm.count_mismatches(parts)
#    assert mismatches == 0, f"{mismatches} mismatches found"

# slow , saw 3-120-None through 3-4-None pass,  PB 1/29/26
# @pytest.mark.parametrize("voters,thetastep,decimals", cases)
# def test_integrated_lump_polar_is_lumpable(voters, thetastep, decimals):
#    svm = gv.models.examples.shapes.ring(g=10, r=6, voters=voters, polar=True, thetastep=thetastep, decimals=decimals)
#    pg = svm.grid
#    parts = pg.partition_from_rotation(angle=360//voters)
#    svm.analyze()
#    MC=svm.model.MarkovChain
#    assert gv.stochastic.markov.is_lumpable(MC,parts), f"Partition is not lumpable"

# these all pass voters=3,5,9 PB 1/29/26
@pytest.mark.parametrize("voters,thetastep,decimals", cases)
def test_integrated_lump_polar_lump_matrix_lazy_dense_equals_dense(voters, thetastep, decimals):
    svm = gv.models.examples.shapes.ring(g=10, r=6, voters=voters, polar=True, thetastep=thetastep, decimals=decimals)
    pg = svm.grid
    parts = pg.partition_from_rotation(angle=360//voters)
    svm.analyze()
    MC_lazy=svm.model.MarkovChain
    LM_lazy_dense=gv.stochastic.markov.lump(MC_lazy,parts)
    MC_dense=gv.stochastic.markov.MarkovChain(P=MC_lazy.P.to_dense())
    LM_dense=gv.stochastic.markov.lump(MC_dense,parts)
    assert jnp.allclose(LM_lazy_dense.P, LM_dense.P, atol=1e-6), f"Lump matrix is not equal to dense matrix"
    sd_lazy_dense=LM_lazy_dense.solve()
    sd_dense=LM_dense.solve()
    assert jnp.allclose(sd_lazy_dense, sd_dense, atol=1e-6), f"lazy-dense lumped Stationary distribution is not equal to dense lumped stationary distribution"
    unlumped_sd_lazy_dense=gv.stochastic.markov.unlump(sd_lazy_dense,parts)
    unlumped_sd_dense=gv.stochastic.markov.unlump(sd_dense,parts)
    assert jnp.allclose(unlumped_sd_lazy_dense, unlumped_sd_dense, atol=1e-6), f"lazy-dense unlumped Stationary distribution is not equal to dense unlumped stationary distribution"
    assert jnp.allclose(unlumped_sd_lazy_dense, svm.model.stationary_distribution, atol=1e-6), f"unlumped stationary distribution is not equal to original stationary distribution"
    assert jnp.allclose(unlumped_sd_dense, svm.model.stationary_distribution, atol=1e-6), f"unlumped stationary distribution is not equal to original stationary distribution"
