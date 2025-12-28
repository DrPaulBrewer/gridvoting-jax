import jax.numpy as jnp
import pytest
import chex

pytestmark = pytest.mark.lumping

from gridvoting_jax import MarkovChain, lump, unlump, is_lumpable, condorcet_cycle


def test_lump_preserves_markov_property():
    """Test lumping that preserves Markov property (strong lumpability)."""
    # Use a chain where lumping won't create absorbing states
    P = jnp.array([
        [0.4, 0.4, 0.2],
        [0.4, 0.4, 0.2],
        [0.1, 0.1, 0.8]
    ])
    mc = MarkovChain(P=P)
    
    # Valid partition: states 0,1 have identical transitions
    partition = [[0, 1], [2]]
    assert is_lumpable(mc, partition)
    
    # Test lump-unlump roundtrip
    mc.find_unique_stationary_distribution()
    lumped = lump(mc, partition)
    lumped.find_unique_stationary_distribution()
    
    pi_unlumped = unlump(lumped.stationary_distribution, partition)
    
    # Reaggregate and compare
    pi_reaggregated = jnp.array([
        pi_unlumped[0] + pi_unlumped[1],
        pi_unlumped[2]
    ])
    assert jnp.allclose(pi_reaggregated, lumped.stationary_distribution)


def test_lump_violates_markov_property():
    """Test lumping that violates Markov property."""
    P = jnp.array([
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.1, 0.1, 0.8]
    ])
    mc = MarkovChain(P=P)
    
    # Invalid partition: states 0,2 have different transitions
    partition = [[0, 2], [1]]
    assert not is_lumpable(mc, partition)
    
    # Stationary distributions won't match after lump-unlump
    mc.find_unique_stationary_distribution()
    original_pi = mc.stationary_distribution
    
    lumped = lump(mc, partition)
    lumped.find_unique_stationary_distribution()
    pi_unlumped = unlump(lumped.stationary_distribution, partition)
    
    # Should NOT match (violated Markov property)
    assert not jnp.allclose(pi_unlumped, original_pi, atol=1e-3)
    
    # But it should still sum to 1
    assert jnp.allclose(jnp.sum(pi_unlumped), 1.0)


def test_swap_partition():
    """Test partition that swaps two states."""
    P = jnp.array([[0.7, 0.3], [0.4, 0.6]])
    mc = MarkovChain(P=P)
    
    # Swap states 0 and 1
    partition = [[1], [0]]
    lumped = lump(mc, partition)
    
    # Expected: rows and columns swapped
    P_expected = jnp.array([[0.6, 0.4], [0.3, 0.7]])
    assert jnp.allclose(lumped.P, P_expected)


def test_lump_simple_2state_to_1state():
    """Lump 2-state chain into 1-state chain."""
    P = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    mc = MarkovChain(P=P)
    
    lumped = lump(mc, partition=[[0, 1]])
    
    assert lumped.P.shape == (1, 1)
    assert jnp.allclose(lumped.P, jnp.array([[1.0]]))


def test_lump_4state_to_2state():
    """Lump 4-state chain into 2-state chain."""
    P = jnp.array([
        [0.5, 0.3, 0.1, 0.1],
        [0.2, 0.6, 0.1, 0.1],
        [0.1, 0.1, 0.5, 0.3],
        [0.1, 0.1, 0.2, 0.6]
    ])
    mc = MarkovChain(P=P)
    
    # Lump [0,1] -> 0, [2,3] -> 1
    lumped = lump(mc, partition=[[0, 1], [2, 3]])
    
    assert lumped.P.shape == (2, 2)
    # Check row sums = 1
    assert jnp.allclose(jnp.sum(lumped.P, axis=1), 1.0)


def test_lump_identity_partition():
    """Partition where each state is its own group (identity)."""
    P = jnp.array([[0.5, 0.5], [0.3, 0.7]])
    mc = MarkovChain(P=P)
    
    lumped = lump(mc, partition=[[0], [1]])
    
    assert jnp.allclose(lumped.P, P)


def test_lump_invalid_empty_group():
    """Empty group should raise ValueError immediately."""
    P = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    mc = MarkovChain(P=P)
    
    with pytest.raises(ValueError, match="group .* is empty"):
        lump(mc, partition=[[], [0, 1]])


def test_lump_invalid_missing_state():
    """Partition missing a state should raise ValueError."""
    P = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    mc = MarkovChain(P=P)
    
    with pytest.raises(ValueError, match="missing states"):
        lump(mc, partition=[[0]])  # Missing state 1


def test_lump_invalid_duplicate():
    """Partition with duplicate states should raise ValueError."""
    P = jnp.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.1, 0.1, 0.8]])
    mc = MarkovChain(P=P)
    
    with pytest.raises(ValueError, match="duplicate"):
        lump(mc, partition=[[0, 1], [1, 2]])  # State 1 appears twice


def test_lump_invalid_index():
    """Invalid state index should raise ValueError."""
    P = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    mc = MarkovChain(P=P)
    
    with pytest.raises(ValueError, match="Invalid state index"):
        lump(mc, partition=[[0, 5]])  # State 5 doesn't exist


def test_unlump_simple():
    """Test unlumping a simple distribution."""
    partition = [[0, 1], [2, 3]]
    lumped_pi = jnp.array([0.4, 0.6])
    
    pi = unlump(lumped_pi, partition)
    
    # Each aggregate's probability distributed uniformly
    expected = jnp.array([0.2, 0.2, 0.3, 0.3])
    assert jnp.allclose(pi, expected)


def test_unlump_unequal_groups():
    """Test unlumping with unequal group sizes."""
    partition = [[0, 1], [2, 3, 4]]
    lumped_pi = jnp.array([0.4, 0.6])
    
    pi = unlump(lumped_pi, partition)
    
    # 0.4 split between 2 states, 0.6 split between 3 states
    expected = jnp.array([0.2, 0.2, 0.2, 0.2, 0.2])
    assert jnp.allclose(pi, expected)


def test_unlump_invalid_size():
    """Test unlump with mismatched distribution size."""
    partition = [[0, 1], [2, 3]]
    lumped_pi = jnp.array([0.3, 0.4, 0.3])  # Wrong size
    
    with pytest.raises(ValueError, match="doesn't match"):
        unlump(lumped_pi, partition)


def test_condorcet_cycle_symmetry_mi():
    """Test CondorcetCycle MI has cyclic symmetry."""
    model = condorcet_cycle(zi=False)
    model.analyze()  # This creates MarkovChain
    mc = model.MarkovChain
    
    # Identity partition (baseline)
    partition = [[0], [1], [2]]
    assert is_lumpable(mc, partition)
    
    # All states equivalent under full symmetry
    partition_full = [[0, 1, 2]]
    # This should be lumpable for symmetric cycle
    # (all states have same outgoing probabilities)
    assert is_lumpable(mc, partition_full)


def test_condorcet_cycle_symmetry_zi():
    """Test CondorcetCycle ZI has symmetry."""
    model = condorcet_cycle(zi=True)
    model.analyze()  # This creates MarkovChain
    mc = model.MarkovChain
    
    # All states equivalent under full symmetry
    partition = [[0, 1, 2]]
    assert is_lumpable(mc, partition)


def test_is_lumpable_valid():
    """Test is_lumpable returns True for valid partition."""
    P = jnp.array([
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.1, 0.1, 0.8]
    ])
    mc = MarkovChain(P=P)
    
    partition = [[0, 1], [2]]
    assert is_lumpable(mc, partition)


def test_is_lumpable_invalid():
    """Test is_lumpable returns False for invalid partition."""
    # Use the same P as test_lump_preserves_markov_property but invalid partition
    P = jnp.array([
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.1, 0.1, 0.8]
    ])
    mc = MarkovChain(P=P)
    
    # States 0,2 have different transitions (invalid lumping)
    partition = [[0, 2], [1]]
    assert not is_lumpable(mc, partition)




def test_lump_preserves_tolerance():
    """Test that lumped chain preserves tolerance from original."""
    P = jnp.array([[0.5, 0.5], [0.3, 0.7]])
    mc = MarkovChain(P=P, tolerance=1e-8)
    
    lumped = lump(mc, partition=[[0], [1]])
    
    assert lumped.tolerance == 1e-8
