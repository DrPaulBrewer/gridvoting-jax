"""Plott's Theorem examples: core existence conditions."""

from ..spatial import SpatialVotingModel
from ...spatial import Grid
import jax.numpy as jnp


def core1(g=20, zi=False):
    """Core exists: 5 voters on horizontal line. Core: [2,0]"""
    grid = Grid(x0=-g, x1=g, y0=-g, y1=g)
    return SpatialVotingModel(
        voter_ideal_points=jnp.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]),
        grid=grid, number_of_voters=5, majority=3, zi=zi
    )


def core2(g=20, zi=True):
    """Core exists: 5 voters on vertical line. Core: [0,2]"""
    grid = Grid(x0=-g, x1=g, y0=-g, y1=g)
    return SpatialVotingModel(
        voter_ideal_points=jnp.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]]),
        grid=grid, number_of_voters=5, majority=3, zi=zi
    )


def core3(g=20, zi=False):
    """Core exists: 5 voters on diagonal. Core: [0,0]"""
    grid = Grid(x0=-g, x1=g, y0=-g, y1=g)
    return SpatialVotingModel(
        voter_ideal_points=jnp.array([[-2, -2], [-1, -1], [0, 0], [1, 1], [2, 2]]),
        grid=grid, number_of_voters=5, majority=3, zi=zi
    )


def core4(g=20, zi=True):
    """Core exists: 4 corners + center. Core: [0,0]"""
    grid = Grid(x0=-g, x1=g, y0=-g, y1=g)
    return SpatialVotingModel(
        voter_ideal_points=jnp.array([[-10, -10], [-10, 10], [10, -10], [10, 10], [0, 0]]),
        grid=grid, number_of_voters=5, majority=3, zi=zi
    )


def nocore_triangle(g=20, zi=False):
    """No core: Equilateral triangle. Cycling expected."""
    grid = Grid(x0=-g, x1=g, y0=-g, y1=g)
    return SpatialVotingModel(
        voter_ideal_points=jnp.array([[-10, 0], [10, 0], [0, 10]]),
        grid=grid, number_of_voters=3, majority=2, zi=zi
    )


def ring_with_central_core(g=20, r=10, voters=7, round_ideal_points=False, zi=False):
    """
    Core exists: Central voter at origin + even number of voters on ring.
    
    Creates a Plott theorem core existence example. The central voter is the
    median in all directions, guaranteeing a core at the origin.
    
    Args:
        g: Grid size
        r: Ring radius
        voters: Total number of voters (MUST be odd, default 7)
        round_ideal_points: Round x,y to integers (default False to guarantee core)
        zi: Zero Intelligence mode
    
    Returns:
        SpatialVotingModel with core at (0,0)
    
    Raises:
        ValueError: If voters is not odd
    
    Note:
        voters = total number of voters (odd)
        voters_on_ring = voters - 1 (even)
        Central voter is placed at index 0
    """
    if voters % 2 == 0:
        raise ValueError(f"voters (total) must be odd, got {voters}")
    
    voters_on_ring = voters - 1
    
    # Place voters uniformly on ring
    angles = jnp.linspace(0, 2 * jnp.pi, voters_on_ring, endpoint=False)
    x = r * jnp.cos(angles)
    y = r * jnp.sin(angles)
    ring_points = jnp.column_stack([x, y])
    
    if round_ideal_points:
        ring_points = jnp.round(ring_points)
    
    # Central voter at origin (index 0)
    central_point = jnp.array([[0.0, 0.0]])
    voter_ideal_points = jnp.vstack([central_point, ring_points])
    
    majority = (voters + 1) // 2
    grid = Grid(x0=-g, x1=g, y0=-g, y1=g)
    
    return SpatialVotingModel(
        voter_ideal_points=voter_ideal_points,
        grid=grid,
        number_of_voters=voters,
        majority=majority,
        zi=zi
    )
