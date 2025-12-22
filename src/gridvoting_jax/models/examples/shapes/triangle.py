"""Random triangle spatial voting examples."""

from ...spatial import SpatialVotingModel
from ....spatial import Grid
import jax.numpy as jnp
import jax


def random_triangle(g=20, within=10, zi=False, key=None):
    """
    Create spatial voting model with random non-degenerate triangle.
    
    Args:
        g: Grid size
        within: Bounds for random ideal points (±within)
        zi: Zero Intelligence mode
        key: JAX random key (if None, creates new key)
    
    Returns:
        SpatialVotingModel with 3 voters at random triangle vertices
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # Generate random points within ±within
    keys = jax.random.split(key, 3)
    voter_ideal_points = jnp.array([
        jax.random.uniform(keys[0], shape=(2,), minval=-within, maxval=within),
        jax.random.uniform(keys[1], shape=(2,), minval=-within, maxval=within),
        jax.random.uniform(keys[2], shape=(2,), minval=-within, maxval=within)
    ])
    
    # Check for degeneracy (collinear points)
    # Cross product of vectors (p1-p0) and (p2-p0)
    v1 = voter_ideal_points[1] - voter_ideal_points[0]
    v2 = voter_ideal_points[2] - voter_ideal_points[0]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    
    # If degenerate, perturb slightly
    if jnp.abs(cross) < 1e-6:
        voter_ideal_points = voter_ideal_points.at[2, 0].add(0.5)
    
    grid = Grid(x0=-g, x1=g, y0=-g, y1=g)
    return SpatialVotingModel(
        voter_ideal_points=voter_ideal_points,
        grid=grid,
        number_of_voters=3,
        majority=2,
        zi=zi
    )
