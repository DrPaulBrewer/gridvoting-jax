import jax.numpy as jnp
import pytest
from gridvoting_jax import Grid


def test_partition_spatial_reflect_x():
    """Test reflection symmetry around x=0."""
    grid = Grid(x0=-2, x1=2, y0=-1, y1=1)
    partition = grid.partition_from_symmetry(['reflect_x'])
    
    # Should have some grouping (points symmetric around x=0)
    num_groups = int(partition.max()) + 1
    assert num_groups > 0
    assert num_groups < grid.len  # Some grouping occurred
    
    # Verify partition is valid (correct length)
    assert len(partition) == grid.len


def test_partition_spatial_reflect_y():
    """Test reflection symmetry around y=0."""
    grid = Grid(x0=-1, x1=1, y0=-2, y1=2)
    partition = grid.partition_from_symmetry(['reflect_y'])
    
    # Should have some grouping
    num_groups = int(partition.max()) + 1
    assert num_groups > 0
    assert num_groups < grid.len
    
    # Verify partition is valid
    assert len(partition) == grid.len


def test_partition_spatial_swap_xy():
    """Test (x,y) <-> (y,x) symmetry."""
    grid = Grid(x0=-2, x1=2, y0=-2, y1=2)
    partition = grid.partition_from_symmetry(['swap_xy'])
    
    # Should have some grouping
    num_groups = int(partition.max()) + 1
    assert num_groups > 0
    assert num_groups < grid.len
    
    # Diagonal points (x,x) should be in singleton groups
    # Check a few diagonal points
    for i in range(grid.len):
        if abs(grid.x[i] - grid.y[i]) < 1e-6:
            # Diagonal point - check if paired with itself only
            group_id = partition[i]
            group_size = jnp.sum(partition == group_id)
            assert group_size == 1  # Singleton
    
    # Verify partition is valid
    assert len(partition) == grid.len


def test_partition_spatial_rotation_120():
    """Test 120° rotation for BJM spatial triangle."""
    # Small grid for testing
    grid = Grid(x0=-5, x1=5, y0=-5, y1=5)
    
    # 120° rotation around origin with tolerance
    partition = grid.partition_from_symmetry(
        [('rotate', 0, 0, 120)], tolerance=1.0
    )
    
    # Should group points that are approximately 120° rotations
    num_groups = int(partition.max()) + 1
    assert num_groups > 0
    assert num_groups <= grid.len
    
    # Verify partition is valid
    assert len(partition) == grid.len


def test_partition_spatial_multiple_symmetries():
    """Test combining multiple symmetries."""
    grid = Grid(x0=-2, x1=2, y0=-2, y1=2)
    
    # Reflect around both axes
    partition = grid.partition_from_symmetry(['reflect_x', 'reflect_y'])
    
    # Should have significant grouping
    num_groups = int(partition.max()) + 1
    assert num_groups > 0
    assert num_groups < grid.len / 2  # At least 2x reduction
    
    # Verify partition is valid
    assert len(partition) == grid.len


def test_partition_spatial_identity():
    """Test with no symmetries (identity partition)."""
    grid = Grid(x0=-1, x1=1, y0=-1, y1=1)
    partition = grid.partition_from_symmetry([])
    
    # Should have one state per group (no grouping)
    num_groups = int(partition.max()) + 1
    assert num_groups == grid.len
    # Each state in its own group
    assert jnp.all(partition == jnp.arange(grid.len))


def test_partition_spatial_reflect_x_custom_axis():
    """Test reflection around custom x=c axis."""
    grid = Grid(x0=-2, x1=4, y0=-1, y1=1)
    partition = grid.partition_from_symmetry(['reflect_x=1'])
    
    # Should group points symmetric around x=1
    num_groups = int(partition.max()) + 1
    assert num_groups > 0
    assert num_groups < grid.len
    
    # Verify partition is valid
    assert len(partition) == grid.len


def test_partition_spatial_rotation_90():
    """Test 90° rotation symmetry."""
    grid = Grid(x0=-2, x1=2, y0=-2, y1=2)
    
    # 90° rotation around origin
    partition = grid.partition_from_symmetry(
        [('rotate', 0, 0, 90)], tolerance=0.1
    )
    
    # Should group points in sets of up to 4 (90° rotations)
    num_groups = int(partition.max()) + 1
    assert num_groups > 0
    assert num_groups <= grid.len
    
    # Verify partition is valid
    assert len(partition) == grid.len


# ============================================================================
# Permutation Symmetry Tests
# ============================================================================

def test_partition_permutation_identity():
    """Test identity permutation (no grouping)."""
    from gridvoting_jax import partition_from_permutation_symmetry
    
    # 3 states with distinct labels
    state_labels = [(0,), (1,), (2,)]
    # Empty permutation group (identity only)
    partition = partition_from_permutation_symmetry(3, state_labels, [])
    
    # Should have one state per group
    num_groups = int(partition.max()) + 1
    assert num_groups == 3
    # Each state in its own group
    assert jnp.all(partition == jnp.arange(3))


def test_partition_permutation_z2():
    """Test Z2 symmetry (swap two elements)."""
    from gridvoting_jax import partition_from_permutation_symmetry
    
    # 6 states representing permutations of 3 voters
    state_labels = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
    
    # Z2: swap voters 0 and 1
    z2_group = [((0,1),)]
    partition = partition_from_permutation_symmetry(6, state_labels, z2_group)
    
    # Should have 3 groups of 2
    num_groups = int(partition.max()) + 1
    assert num_groups == 3
    # Check all groups have size 2
    for g in range(num_groups):
        group_size = jnp.sum(partition == g)
        assert group_size == 2
    
    # Verify partition is valid
    assert len(partition) == 6


def test_partition_permutation_s3():
    """Test S3 symmetry (all voters interchangeable)."""
    from gridvoting_jax import partition_from_permutation_symmetry
    
    # 6 states representing permutations of 3 voters
    state_labels = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
    
    # S3 generators: (0,1) swap and (0,1,2) 3-cycle
    s3_group = [((0,1),), ((0,1,2),)]
    partition = partition_from_permutation_symmetry(6, state_labels, s3_group)
    
    # All states should be in one group (full symmetry)
    num_groups = int(partition.max()) + 1
    assert num_groups == 1
    assert jnp.all(partition == 0)
    
    # Verify partition is valid
    assert len(partition) == 6


def test_partition_permutation_multiple_cycles():
    """Test permutation with multiple disjoint cycles."""
    from gridvoting_jax import partition_from_permutation_symmetry
    
    # 4 voters
    state_labels = [(0,1,2,3), (1,0,3,2), (2,3,0,1), (3,2,1,0)]
    
    # Swap (0,1) and (2,3) simultaneously
    perm_group = [((0,1), (2,3))]
    partition = partition_from_permutation_symmetry(4, state_labels, perm_group)
    
    # Should group states that are swapped
    num_groups = int(partition.max()) + 1
    assert num_groups == 2
    for g in range(num_groups):
        group_size = jnp.sum(partition == g)
        assert group_size == 2


def test_partition_permutation_3cycle():
    """Test 3-cycle permutation."""
    from gridvoting_jax import partition_from_permutation_symmetry
    
    # 3 states with cyclic labels
    state_labels = [(0,1,2), (1,2,0), (2,0,1)]
    
    # 3-cycle: 0→1→2→0
    perm_group = [((0,1,2),)]
    partition = partition_from_permutation_symmetry(3, state_labels, perm_group)
    
    # All should be in one group (cyclic symmetry)
    num_groups = int(partition.max()) + 1
    assert num_groups == 1
    assert jnp.all(partition == 0)


# ============================================================================
# Model Integration Tests
# ============================================================================

def test_budget_voting_model_s3_symmetry():
    """Test BudgetVotingModel.get_permutation_symmetry_partition() with S3."""
    from gridvoting_jax import bjm_budget_triangle
    
    # Small budget for testing
    model = bjm_budget_triangle(budget=5, zi=False)
    
    # Get S3 partition (default)
    partition = model.get_permutation_symmetry_partition()
    
    # Should group symmetric allocations
    num_groups = int(partition.max()) + 1
    assert num_groups > 0
    assert num_groups < model.number_of_alternatives
    
    # Verify partition is valid
    assert len(partition) == model.number_of_alternatives


def test_spatial_voting_model_symmetry():
    """Test SpatialVotingModel.get_spatial_symmetry_partition()."""
    from gridvoting_jax import Grid, SpatialVotingModel
    import pytest
    import jax.numpy as jnp

    pytestmark = pytest.mark.lumping
    
    # Small grid
    grid = Grid(x0=-2, x1=2, y0=-2, y1=2)
    voter_ideal_points = jnp.array([[0, 0], [1, 1], [-1, -1]])
    
    model = SpatialVotingModel(
        voter_ideal_points=voter_ideal_points,
        grid=grid,
        number_of_voters=3,
        majority=2,
        zi=False
    )
    
    # Get reflection partition
    partition = model.get_spatial_symmetry_partition(['reflect_x'])
    
    # Should group symmetric points
    num_groups = int(partition.max()) + 1
    assert num_groups > 0
    assert num_groups < grid.len
    
    # Verify partition is valid
    assert len(partition) == grid.len
