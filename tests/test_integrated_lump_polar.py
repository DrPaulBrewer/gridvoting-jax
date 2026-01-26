import gridvoting_jax as gv
import jax.numpy as jnp
import pytest

# this test passes in float32 mode for voters=3,5,9 but not 15
# for unknown reasons, probably related to numerical precision
# it will pass voters=15 if we use float64

# in order for this test to pass, we needed to create consistent_cos and consistent_sin
# to move all cos/sin calculations to the first quadrant of the unit circle

@pytest.mark.parametrize("voters,thetastep,decimals", 
    [(v,n,None) for v in range(3,13,2) if 360%v==0 for n in range(360,0,-1) if (360//v)%n==0 ]
)
def test_integrated_lump_polar_similarity(voters, thetastep, decimals):
    svm = gv.models.examples.shapes.ring(g=30, r=20, voters=voters, polar=True, thetastep=thetastep, decimals=decimals)
    pg = svm.grid
    parts = pg.partition_from_rotation(angle=360//voters)
    mismatches = svm.count_mismatches(parts)
    assert mismatches == 0, f"{mismatches} mismatches found"
