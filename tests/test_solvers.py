import pytest
import jax.numpy as jnp
import gridvoting_jax as gv
import numpy as np
pytestmark = pytest.mark.essential
import matplotlib.pyplot as plt
from gridvoting_jax.core import TOLERANCE
from gridvoting_jax.spatial import Grid
from gridvoting_jax import VotingModel, SpatialVotingModel
from gridvoting_jax.models.examples.bjm_spatial import BJM_TRIANGLE_VOTER_IDEAL_POINTS

def test_solvers_consistency():
    """Verify all solvers produce consistent results on a small grid."""
    
    # Use canonical BJM spatial triangle example (g=20, step=1.0)
    grid = Grid(x0=-20, x1=20, y0=-20, y1=20)  # 41x41 = 1681 points
    
    # BJM Triangle 1 configuration
    voter_ideal_points = jnp.array(BJM_TRIANGLE_VOTER_IDEAL_POINTS)
    
    utils = grid.spatial_utilities(voter_ideal_points=voter_ideal_points)
    
    # Base Model (Full Matrix Inversion)
    model_base = VotingModel(
        utility_functions=utils,
        number_of_voters=3,
        number_of_feasible_alternatives=len(grid.points),
        majority=2,
        zi=False # MI
    )
    model_base.analyze(solver="full_matrix_inversion")
    base_dist = model_base.stationary_distribution
    assert base_dist is not None

    # Test GMRES
    model_gmres = VotingModel(
        utility_functions=utils,
        number_of_voters=3,
        number_of_feasible_alternatives=len(grid.points),
        majority=2,
        zi=False
    )
    model_gmres.analyze(solver="gmres_matrix_inversion", tolerance=1e-6)
    dist_gmres = model_gmres.stationary_distribution
    
    # L1 Match Check
    l1_gmres = float(jnp.linalg.norm(base_dist - dist_gmres, ord=1))
    print(f"GMRES L1 Diff: {l1_gmres}")
    assert l1_gmres < 1e-4

    # Test Power Method (Dual-Start Entropy)
    model_power = VotingModel(
        utility_functions=utils,
        number_of_voters=3,
        number_of_feasible_alternatives=len(grid.points),
        majority=2,
        zi=False
    )
    # Using looser tolerance for power method in test to ensure convergence speed
    model_power.analyze(solver="power_method", tolerance=1e-5, max_iterations=5000)
    dist_power = model_power.stationary_distribution
    
    l1_power = float(jnp.linalg.norm(base_dist - dist_power, ord=1))
    print(f"Power Method L1 Diff: {l1_power}")
    assert l1_power < 1e-4


    # Test Grid Upscaling (now uses SpatialVotingModel)
    model_upscale = SpatialVotingModel(
        voter_ideal_points=voter_ideal_points,
        grid=grid,
        number_of_voters=3,
        majority=2,
        zi=False
    )
    model_upscale.analyze(solver="grid_upscaling", tolerance=1e-5)
    dist_upscale = model_upscale.stationary_distribution
    
    l1_upscale = float(jnp.linalg.norm(base_dist - dist_upscale, ord=1))
    print(f"Upscaling L1 Diff: {l1_upscale}")
    assert l1_upscale < 1e-4


def test_solver_invalid_arg():
    grid = Grid(x0=-1.0, x1=1.0, xstep=0.2, y0=-1.0, y1=1.0, ystep=0.2)
    
    # Use triangular configuration to ensure NO Core (cycle)
    # This forces analyze() to call the solver
    voter_ideal_points = jnp.array([
        [0.0, 0.8],
        [-0.7, -0.4],
        [0.7, -0.4]
    ])
    utils = grid.spatial_utilities(voter_ideal_points=voter_ideal_points)
    
    model = VotingModel(
        utility_functions=utils,
        number_of_voters=3,
        number_of_feasible_alternatives=len(grid.points),
        majority=2,
        zi=False
    )
    with pytest.raises(ValueError, match="Unknown solver"):
        model.analyze(solver="fake_solver")


# test_upscaling_missing_args removed - grid_upscaling is now in SpatialVotingModel
# which requires grid and voter_ideal_points in __init__, so this test is obsolete


if __name__ == "__main__":
    test_solvers_consistency()
