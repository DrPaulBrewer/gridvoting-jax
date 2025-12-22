"""BJM Research budget voting example."""

from ..budget import BudgetVotingModel


def bjm_budget_triangle(budget=100, zi=False):
    """
    BJM budget voting: Fixed budget=100 for OSF validation.
    
    Args:
        budget: Total budget to allocate (default 100)
        zi: Zero Intelligence mode (default False for MI)
    
    Returns:
        BudgetVotingModel instance
    """
    return BudgetVotingModel(budget=budget, zi=zi)
