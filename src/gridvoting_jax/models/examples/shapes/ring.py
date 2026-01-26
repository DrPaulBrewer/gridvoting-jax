"""Ring (circular) spatial voting examples."""

from ...spatial import SpatialVotingModel
from ....geometry import Grid, PolarGrid, consistent_cos, consistent_sin
from ....core import constants
import jax.numpy as jnp


def ring(g=20, r=10, voters=5, round_ideal_points=False, majority=None, zi=False, polar=False, thetastep=1, decimals=None):
    """
    Voters uniformly distributed on a ring (circle).
    
    Args:
        g: Grid size
        r: Ring radius
        voters: Number of voters (MUST be odd)
        round_ideal_points: Round x,y to integers
        majority: Majority threshold (default: (voters+1)//2)
        zi: Zero Intelligence mode
        polar: Use polar grid
        thetastep: Step size for theta (polar grid only)
        decimals: Number of decimals to round ideal points to
    
    Returns:
        SpatialVotingModel with voters on circle
    
    Raises:
        ValueError: If voters is not odd
    """
    if voters % 2 == 0:
        raise ValueError(f"voters must be odd, got {voters}")
    
    # Uniform angles around circle
    angles = jnp.linspace(0.0, 360.0, voters, endpoint=False, dtype=constants.DTYPE_FLOAT)
    rs = jnp.full(voters, r, dtype=constants.DTYPE_FLOAT)
    if polar:
        voter_ideal_points = jnp.column_stack((rs,angles)) 
    else:
        x = rs * consistent_cos(angles)
        y = rs * consistent_sin(angles)
        voter_ideal_points = jnp.column_stack((x,y))
    
    if round_ideal_points:
        voter_ideal_points = jnp.round(voter_ideal_points)
    
    # Default majority
    if majority is None:
        majority = (voters + 1) // 2
    
    grid = PolarGrid(radius=g, thetastep=thetastep) if polar else Grid(x0=-g, x1=g, y0=-g, y1=g)
    
    return SpatialVotingModel(
        voter_ideal_points=voter_ideal_points,
        grid=grid, number_of_voters=voters, majority=majority, zi=zi, decimals=decimals
    )
