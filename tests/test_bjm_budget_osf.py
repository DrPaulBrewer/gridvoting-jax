"""OSF validation tests for budget voting (all solvers)."""

import pytest
import jax.numpy as jnp
import gridvoting_jax as gv
pytestmark = pytest.mark.budget
from gridvoting_jax.models.examples import bjm_budget_triangle


# Note: This test requires OSF data loading functionality
# The actual OSF data loader needs to be implemented in benchmarks module
# For now, this is a placeholder structure

@pytest.mark.parametrize("solver", [
    "full_matrix_inversion",
    "gmres_matrix_inversion", 
    "power_method"
])
def test_bjm_budget_osf_validation(solver):
    """
    Validate budget voting against OSF data for all solvers.
    
    Note: This test requires OSF budget data to be available.
    The load_osf_budget_distribution function needs to be implemented.
    """
    # TODO: Implement load_osf_budget_distribution in benchmarks module
    # For now, just test that the model runs with all solvers
    
    model = bjm_budget_triangle(budget=100, zi=False)
    model.analyze(solver=solver)
    
    # Basic sanity checks
    assert model.stationary_distribution is not None
    assert jnp.abs(jnp.sum(model.stationary_distribution) - 1.0) < 1e-5
    assert jnp.all(model.stationary_distribution >= 0)
    
    # TODO: When OSF data is available, add:
    # osf_data = load_osf_budget_distribution(budget=100, zi=False)
    # l1_diff = jnp.sum(jnp.abs(model.stationary_distribution - osf_data))
    # assert l1_diff < 1e-4, f"Solver {solver}: L1 diff = {l1_diff:.6f}"
    # correlation = jnp.corrcoef(model.stationary_distribution, osf_data)[0, 1]
    # assert correlation > 0.999, f"Solver {solver}: correlation = {correlation:.6f}"
