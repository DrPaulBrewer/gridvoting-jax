"""BJM Research spatial voting example (Triangle 1 from OSF data)."""

from ..spatial import SpatialVotingModel
from ...spatial import Grid
import jax.numpy as jnp



BJM_TRIANGLE_VOTER_IDEAL_POINTS = [[-15, -9], [0, 17], [15, -9]]


def bjm_spatial_triangle(g=20, zi=False):
    """
    BJM spatial voting: Near equilateral triangle configuration
    This is the configuration used in the research paper by Brewer, Juybari, and Moberly   

    Brewer, P., Juybari, J. & Moberly, R. 
       A comparison of zero- and minimal-intelligence agendas in majority-rule voting models. 
       J Econ Interact Coord 19, 403–437 (2024). 
       
       https://doi.org/10.1007/s11403-023-00387-8    (open access)

    Data from this research was deposited at OSF.  An "OSF benchmark" tests your implementation
    against the data from this research.


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
        voter_ideal_points=jnp.array(BJM_TRIANGLE_VOTER_IDEAL_POINTS),
        grid=grid,
        number_of_voters=3,
        majority=2,
        zi=zi
    )
