"""Tests for Plott theorem examples."""

import pytest
import jax.numpy as jnp
from gridvoting_jax.models.examples.plott_theorem import (
    core1, core2, core3, core4, nocore_triangle, ring_with_central_core
)
from gridvoting_jax.models.examples.shapes import random_triangle, ring


def test_plott_core_existence():
    """Test that Plott theorem examples have cores at expected locations."""
    # core1: horizontal line, core at [2,0]
    model1 = core1(g=20, zi=False)
    model1.analyze()
    assert model1.core_exists
    # Core should be at (2,0)
    core_points = model1.grid.points[model1.core_points]
    assert jnp.any(jnp.all(core_points == jnp.array([2, 0]), axis=1))
    
    # ring_with_central_core: core at (0,0)
    model_ring = ring_with_central_core(g=20, r=10, voters=7)
    model_ring.analyze()
    assert model_ring.core_exists
    # Core should be at (0,0)
    core_points_ring = model_ring.grid.points[model_ring.core_points]
    assert jnp.any(jnp.all(core_points_ring == jnp.array([0, 0]), axis=1))


def test_no_core_examples():
    """
    Test that examples expected to have no core indeed have no core.
    """
    # nocore_triangle: equilateral triangle, no core
    model_tri = nocore_triangle(g=20, zi=False)
    model_tri.analyze()
    assert not model_tri.core_exists
    
    # random_triangle: test 10 triangles with area >= 50
    # Most random triangles won't have a core
    import jax.random as random
    
    def triangle_area(points):
        """Calculate area of triangle using cross product."""
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        return 0.5 * jnp.abs(v1[0] * v2[1] - v1[1] * v2[0])
    
    key = random.PRNGKey(42)
    no_core_count = 0
    
    for i in range(100):  # Try up to 100 to get 10 with area >= 50
        key, subkey = random.split(key)
        model_rand = random_triangle(g=20, within=15, zi=False, key=subkey)
        
        # Check area
        area = triangle_area(model_rand.voter_ideal_points)
        
        if area >= 50:
            model_rand.analyze()
            if not model_rand.core_exists:
                no_core_count += 1
            
            if no_core_count >= 10:
                break
    
    # At least 10 triangles with area >= 50 should have no core
    assert no_core_count >= 10, \
        f"Only {no_core_count}/10 random triangles (area>=50) had no core"
    
    # ring with odd voters: no core (cycling)
    model_ring = ring(g=20, r=10, voters=5, zi=False)
    model_ring.analyze()
    assert not model_ring.core_exists


def test_ring_with_central_core_validation():
    """Test ring_with_central_core parameter validation."""
    # Should work with odd voters
    model = ring_with_central_core(g=20, r=10, voters=7)
    assert model.number_of_voters == 7
    
    # Should fail with even voters
    with pytest.raises(ValueError):
        ring_with_central_core(g=20, r=10, voters=6)
