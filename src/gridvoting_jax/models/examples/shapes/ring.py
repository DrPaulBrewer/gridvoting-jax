"""Ring (circular) spatial voting examples."""

from ...spatial import SpatialVotingModel
from ....spatial import Grid
import jax.numpy as jnp


def ring(g=20, r=10, voters=5, round_ideal_points=True, majority=None, zi=False):
    """
    Voters uniformly distributed on a ring (circle).
    
    Args:
        g: Grid size
        r: Ring radius
        voters: Number of voters (MUST be odd)
        round_ideal_points: Round x,y to integers
        majority: Majority threshold (default: (voters+1)//2)
        zi: Zero Intelligence mode
    
    Returns:
        SpatialVotingModel with voters on circle
    
    Raises:
        ValueError: If voters is not odd
    """
    if voters % 2 == 0:
        raise ValueError(f"voters must be odd, got {voters}")
    
    # Uniform angles around circle
    angles = jnp.linspace(0, 2 * jnp.pi, voters, endpoint=False)
    x = r * jnp.cos(angles)
    y = r * jnp.sin(angles)
    voter_ideal_points = jnp.column_stack([x, y])
    
    if round_ideal_points:
        voter_ideal_points = jnp.round(voter_ideal_points)
    
    # Default majority
    if majority is None:
        majority = (voters + 1) // 2
    
    grid = Grid(x0=-g, x1=g, y0=-g, y1=g)
    
    return SpatialVotingModel(
        voter_ideal_points=voter_ideal_points,
        grid=grid, number_of_voters=voters, majority=majority, zi=zi
    )
