"""BJM Research spatial voting example (Triangle 1 from OSF data)."""

from ..spatial import SpatialVotingModel
from ...spatial import Grid
import jax.numpy as jnp


def bjm_spatial_triangle(g=20, zi=False):
    """
    BJM spatial voting: Triangle 1 configuration for OSF validation.
    
    Voter ideal points: [[-15, -9], [0, 17], [15, -9]]
    Used in OSF benchmark validation.
    
    Args:
        g: Grid size (default 20)
        zi: Zero Intelligence mode (default False for MI)
    
    Returns:
        SpatialVotingModel instance
    """
    grid = Grid(x0=-g, x1=g, y0=-g, y1=g)
    return SpatialVotingModel(
        voter_ideal_points=jnp.array([[-15, -9], [0, 17], [15, -9]]),
        grid=grid,
        number_of_voters=3,
        majority=2,
        zi=zi
    )
