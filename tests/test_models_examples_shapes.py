"""Tests for shape examples."""

import pytest
import jax.numpy as jnp
from gridvoting_jax.models.examples.shapes import random_triangle, ring


def test_random_triangle():
    """Test random triangle generation."""
    model = random_triangle(g=20, within=10, zi=False)
    assert model.number_of_voters == 3
    assert model.majority == 2
    # Check points within bounds
    assert jnp.all(jnp.abs(model.voter_ideal_points) <= 10)


def test_ring_odd_voters():
    """Test ring requires odd voters."""
    model = ring(g=20, r=10, voters=5)
    assert model.number_of_voters == 5
    assert model.majority == 3
    
    # Even voters should raise error
    with pytest.raises(ValueError):
        ring(g=20, r=10, voters=4)


def test_ring_rounding():
    """Test ring ideal point rounding."""
    model_rounded = ring(g=20, r=10, voters=5, round_ideal_points=True)
    model_float = ring(g=20, r=10, voters=5, round_ideal_points=False)
    
    # Rounded should have integer coordinates
    assert jnp.all(model_rounded.voter_ideal_points == 
                   jnp.round(model_rounded.voter_ideal_points))


def test_ring_adjustable_majority():
    """Test ring with custom majority."""
    model = ring(g=20, r=10, voters=7, majority=5)
    assert model.majority == 5
    
    # Default majority
    model_default = ring(g=20, r=10, voters=7)
    assert model_default.majority == 4  # (7+1)//2
