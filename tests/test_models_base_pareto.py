
import pytest
import jax.numpy as jnp
import gridvoting_jax as gv
from gridvoting_jax.models.examples.shapes.triangle import random_triangle as triangle

def test_unanimize_logic():
    """Test that unanimize creates a correct independent copy."""
    # Setup simple model (random_triangle creates 3 voters by default)
    model = triangle(g=10)
    
    # Create unanimized copy
    u_model = model.model.unanimize()
    
    # Verify majority
    assert u_model.majority == 3
    assert u_model.number_of_voters == 3
    
    # Verify independence
    assert u_model is not model.model
    assert u_model.analyzed is False
    
    # Verify original is untouched (default majority is 2 for 3 voters)
    assert model.model.majority == 2

def test_pareto_triangle():
    """
    Test that the Pareto set of a spatial triangle matches the geometry.
    This test uses a random triangle. For 3 voters with convex preferences (euclidean), the Pareto set 
    is the convex hull of ideal points (the triangle itself).
    """
    g = 40
    # Create triangle model: voters at corners
    model = triangle(g=g)
        
    # Calculate Pareto set
    pareto_mask = model.Pareto
    
    # Check against geometric ground truth
    # grid.within_triangle takes triangle vertices
    # We need to get the exact vertices used in the model
    vertices = model.voter_ideal_points
    
    geometric_pareto = model.grid.within_triangle(points=vertices)
    
    # Compare
    # Note: Discrete grid approximations might have slight boundary mismatches
    # but for a convex hull, they should be very close.
    # Let's assert overlap is high > 95%
    
    intersection = pareto_mask & geometric_pareto
    
    # 1. Verification: The pareto set should contain the entire geometric triangle
    # (Recall should be ~100%)
    recall = intersection.sum() / geometric_pareto.sum()
    assert recall > 0.99, f"Pareto set should contain geometric triangle. Recall: {recall}"
    
    # 2. Verification: The pareto set should not be 'too much' larger than the triangle
    # Due to grid discretization, points just outside the hull may not be dominated 
    # by any *grid* point, so the set will be larger.
    # We assert reasonable precision > 0.5
    precision = intersection.sum() / pareto_mask.sum()
    assert precision > 0.5, f"Pareto set is too large (Precision {precision} < 0.5)"
    
    # Also verify that cached result works
    pareto_mask_2 = model.Pareto
    assert pareto_mask_2 is pareto_mask # Identity check for cache

def test_pareto_budget():
    """Test Pareto set for budget voting (should be entire simplex)."""
    from gridvoting_jax.models.budget import BudgetVotingModel
    
    # Budget model feasible set is the simplex x+y<=budget
    # For budget voting, utilities are linear/monotonic in x, y, z
    # Usually the entire simplex is Pareto optimal?
    # Actually, with linear utilities, yes.
    
    model = BudgetVotingModel(budget=10)
    pareto_mask = model.Pareto
    
    # Budget model 'valid' mask defines the simplex
    # So Pareto set should equal the valid set
    
    # We need to access the internal mask of the budget model
    # base.py doesn't strictly know about 'valid', but BudgetVotingModel has model.number_of_alternatives
    # The Pareto mask returned is size N (alternatives), not grid size.
    # Wait, BudgetVotingModel uses a VotingModel initialized with 'valid' points only?
    # No, BudgetVotingModel wraps VotingModel. 
    # Let's check BudgetVotingModel implementation in budget.py
    
    # In budget.py: 
    # self.U = jnp.array(...) shape (3, num_alternatives)
    # self.model = VotingModel(..., number_of_feasible_alternatives=self.number_of_alternatives)
    
    # So model.Pareto returns array of shape (num_alternatives,)
    # It should be all True (ones) because every point in the simplex is Pareto optimal 
    # for these linear utilities (sum(u) = const).
    
    assert jnp.all(pareto_mask), "For linear budget model, entire simplex should be Pareto optimal"
