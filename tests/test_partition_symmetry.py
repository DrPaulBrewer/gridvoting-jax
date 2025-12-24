import jax.numpy as jnp
import pytest
from gridvoting_jax import Grid


def test_partition_spatial_reflect_x():
    """Test reflection symmetry around x=0."""
    grid = Grid(x0=-2, x1=2, y0=-1, y1=1)
    partition = grid.partition_from_symmetry( ['reflect_x'])
    
    # Should have some grouping (points symmetric around x=0)
    assert len(partition) > 0
    assert len(partition) < grid.len  # Some grouping occurred
    
    # Verify partition is valid
    all_states = [s for group in partition for s in group]
    assert set(all_states) == set(range(grid.len))


def test_partition_spatial_reflect_y():
    """Test reflection symmetry around y=0."""
    grid = Grid(x0=-1, x1=1, y0=-2, y1=2)
    partition = grid.partition_from_symmetry( ['reflect_y'])
    
    # Should have some grouping
    assert len(partition) > 0
    assert len(partition) < grid.len
    
    # Verify partition is valid
    all_states = [s for group in partition for s in group]
    assert set(all_states) == set(range(grid.len))


def test_partition_spatial_swap_xy():
    """Test (x,y) <-> (y,x) symmetry."""
    grid = Grid(x0=-2, x1=2, y0=-2, y1=2)
    partition = grid.partition_from_symmetry( ['swap_xy'])
    
    # Should have some grouping
    assert len(partition) > 0
    assert len(partition) < grid.len
    
    # Diagonal points (x,x) should be in singleton groups
    # Off-diagonal points should be paired
    for group in partition:
        if len(group) == 1:
            idx = group[0]
            # Check if it's on diagonal
            assert abs(grid.x[idx] - grid.y[idx]) < 1e-6
    
    # Verify partition is valid
    all_states = [s for group in partition for s in group]
    assert set(all_states) == set(range(grid.len))


def test_partition_spatial_rotation_120():
    """Test 120° rotation for BJM spatial triangle."""
    # Small grid for testing
    grid = Grid(x0=-5, x1=5, y0=-5, y1=5)
    
    # 120° rotation around origin with tolerance
    partition = grid.partition_from_symmetry(
        [('rotate', 0, 0, 120)], tolerance=1.0
    )
    
    # Should group points that are approximately 120° rotations
    assert len(partition) > 0
    assert len(partition) <= grid.len
    
    # Verify partition is valid
    all_states = [s for group in partition for s in group]
    assert set(all_states) == set(range(grid.len))


def test_partition_spatial_multiple_symmetries():
    """Test combining multiple symmetries."""
    grid = Grid(x0=-2, x1=2, y0=-2, y1=2)
    
    # Reflect around both axes
    partition = grid.partition_from_symmetry( ['reflect_x', 'reflect_y'])
    
    # Should have significant grouping
    assert len(partition) > 0
    assert len(partition) < grid.len / 2  # At least 2x reduction
    
    # Verify partition is valid
    all_states = [s for group in partition for s in group]
    assert set(all_states) == set(range(grid.len))


def test_partition_spatial_identity():
    """Test with no symmetries (identity partition)."""
    grid = Grid(x0=-1, x1=1, y0=-1, y1=1)
    partition = grid.partition_from_symmetry( [])
    
    # Should have one state per group (no grouping)
    assert len(partition) == grid.len
    for group in partition:
        assert len(group) == 1


def test_partition_spatial_reflect_x_custom_axis():
    """Test reflection around custom x=c axis."""
    grid = Grid(x0=-2, x1=4, y0=-1, y1=1)
    partition = grid.partition_from_symmetry( ['reflect_x=1'])
    
    # Should group points symmetric around x=1
    assert len(partition) > 0
    assert len(partition) < grid.len
    
    # Verify partition is valid
    all_states = [s for group in partition for s in group]
    assert set(all_states) == set(range(grid.len))


def test_partition_spatial_rotation_90():
    """Test 90° rotation symmetry."""
    grid = Grid(x0=-2, x1=2, y0=-2, y1=2)
    
    # 90° rotation around origin
    partition = grid.partition_from_symmetry(
        [('rotate', 0, 0, 90)], tolerance=0.1
    )
    
    # Should group points in sets of up to 4 (90° rotations)
    assert len(partition) > 0
    assert len(partition) <= grid.len
    
    # Verify partition is valid
    all_states = [s for group in partition for s in group]
    assert set(all_states) == set(range(grid.len))
