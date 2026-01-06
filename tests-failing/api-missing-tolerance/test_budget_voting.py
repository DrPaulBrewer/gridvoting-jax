"""Tests for budget voting model."""

import pytest
import jax.numpy as jnp
import gridvoting_jax as gv
from gridvoting_jax.models.budget import BudgetVotingModel

pytestmark = pytest.mark.budget


def test_budget_voting_basic():
    """Test basic budget voting model creation."""
    model = BudgetVotingModel(budget=10, zi=False)
    assert model.number_of_alternatives == (10 + 1) * (10 + 2) // 2  # 66
    assert model.U.shape == (3, 66)


def test_budget_voting_triangle_constraint():
    """Verify triangle constraint: x + y <= budget."""
    model = BudgetVotingModel(budget=10, zi=False)
    assert jnp.all(model.u1 + model.u2 <= 10)
    assert jnp.all(model.u1 >= 0)
    assert jnp.all(model.u2 >= 0)


def test_budget_voting_utilities():
    """Test utility functions: u1=x, u2=y, u3=budget-x-y."""
    model = BudgetVotingModel(budget=10, zi=False)
    assert jnp.allclose(model.u3, 10 - model.u1 - model.u2)


def test_budget_voting_symmetry():
    """
    Test symmetry property: π[x,y] ≈ π[y,x].
    
    CAREFUL: This test deals with multiple coordinate systems:
    1. Grid coordinates: (grid.x, grid.y) - full square grid
    2. Valid indices: model.valid - boolean mask for triangle
    3. Triangle coordinates: (model.u1, model.u2) - coordinates within valid region
    4. Stationary distribution indices: index into model.stationary_distribution
    
    The symmetry property states that for any point (x,y) in the triangle,
    the probability at (x,y) should equal the probability at (y,x).
    """
    model = BudgetVotingModel(budget=10, zi=False)
    model.analyze()
    
    # model.u1 and model.u2 are the x,y coordinates of valid (triangle) points
    # model.stationary_distribution[i] is the probability at the i-th valid point
    # where the i-th valid point has coordinates (model.u1[i], model.u2[i])
    
    # For each valid point (x,y), find the corresponding (y,x) point
    for i in range(len(model.u1)):
        x, y = model.u1[i], model.u2[i]
        
        # Find index j where (model.u1[j], model.u2[j]) == (y, x)
        # Note: We're looking for the swapped coordinates
        swapped_mask = (model.u1 == y) & (model.u2 == x)
        swapped_indices = jnp.where(swapped_mask)[0]
        
        if len(swapped_indices) > 0:
            j = int(swapped_indices[0])
            # Check symmetry: probability at (x,y) ≈ probability at (y,x)
            prob_xy = model.stationary_distribution[i]
            prob_yx = model.stationary_distribution[j]
            assert jnp.abs(prob_xy - prob_yx) < 1e-5, \
                f"Symmetry violated at ({x},{y}): π[{x},{y}]={prob_xy:.6f} != π[{y},{x}]={prob_yx:.6f}"


def test_budget_voting_zi_modes():
    """
    Test ZI vs MI modes produce different maxima.
    
    Using budget=30 for clean even splits:
    - ZI (Zero Intelligence): Equal 3-way split at (10,10) → voter 3 gets 10
    - MI (Minimal Intelligence): 2-way splits at (0,30), (30,0), or (0,0)
    """
    model_zi = BudgetVotingModel(budget=30, zi=True)
    model_mi = BudgetVotingModel(budget=30, zi=False)
    
    model_zi.analyze()
    model_mi.analyze()
    
    # ZI: argmax should be at equal 3-way split (10,10)
    # This means voter 1 gets 10, voter 2 gets 10, voter 3 gets 10
    argmax_zi = jnp.argmax(model_zi.stationary_distribution)
    # Verify argmax is a scalar (single element)
    assert argmax_zi.shape == (), f"argmax_zi should be scalar, got shape {argmax_zi.shape}"
    x_zi, y_zi = model_zi.u1[argmax_zi], model_zi.u2[argmax_zi]
    # Should be close to (10, 10) for budget=30
    assert jnp.abs(x_zi - 10) < 3 and jnp.abs(y_zi - 10) < 3, \
        f"ZI argmax at ({x_zi},{y_zi}), expected near (10,10)"
    
    # MI: argmax at 2-way split
    # Due to symmetry, there may be 3 locations with equal max probability:
    # (0,15): voters 2,3 split, voter 1 gets 0
    # (15,0): voters 1,2 split, voter 3 gets 0
    # (15,15): voters 1,3 split, voter 2 gets 0
    # Test only the first one found by argmax
    argmax_mi_item = jnp.argmax(model_mi.stationary_distribution).item()
    x_mi, y_mi = model_mi.u1[argmax_mi_item], model_mi.u2[argmax_mi_item]
    
    # MI should favor 2-way splits where one voter gets 0
    # Check that at least one coordinate is small (< 5)
    # This allows for various 2-way split configurations
    assert (x_mi < 5 or y_mi < 5), \
        f"MI argmax at ({x_mi},{y_mi}), expected a 2-way split with one voter getting ~0"


def test_voter_utility_distribution():
    """Test voter utility probability distributions."""
    model = BudgetVotingModel(budget=10, zi=False)
    model.analyze()
    
    # Get distribution for voter 1
    utility_values, probabilities = model.voter_utility_distribution(voter_index=0)
    
    # Check shape
    assert len(utility_values) == 11  # 0, 1, 2, ..., 10
    assert len(probabilities) == 11
    
    # Probabilities should sum to 1
    assert jnp.abs(jnp.sum(probabilities) - 1.0) < 1e-6
    
    # All probabilities should be non-negative
    assert jnp.all(probabilities >= 0)
    
    # Test all three voters
    for voter_idx in [0, 1, 2]:
        _, probs = model.voter_utility_distribution(voter_idx)
        assert jnp.abs(jnp.sum(probs) - 1.0) < 1e-6


def test_giniss_distribution():
    """Test GiniSS inequality index distribution."""
    model = BudgetVotingModel(budget=10, zi=False)
    model.analyze()
    
    # Test with default granularity (0.10 = 11 bins)
    gini_values, gini_probabilities = model.giniss_distribution()
    assert len(gini_values) == 11
    assert len(gini_probabilities) == 11
    
    # Test with custom granularity (0.01 = 101 bins)
    gini_values_fine, gini_probabilities_fine = model.giniss_distribution(granularity=0.01)
    assert len(gini_values_fine) == 101
    assert len(gini_probabilities_fine) == 101
    
    # Probabilities should sum to 1
    assert jnp.abs(jnp.sum(gini_probabilities) - 1.0) < 1e-6
    assert jnp.abs(jnp.sum(gini_probabilities_fine) - 1.0) < 1e-6
    
    # All probabilities should be non-negative
    assert jnp.all(gini_probabilities >= 0)
    assert jnp.all(gini_probabilities_fine >= 0)
