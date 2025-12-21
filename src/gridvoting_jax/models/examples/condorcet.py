"""CondorcetCycle: Classic 3-voter, 3-alternative cycle example."""
import jax.numpy as jnp


def condorcet_cycle(zi=False):
    """
    Condorcet cycle: A>B>C>A with 3 voters.
    
    Args:
        zi: bool, whether to use fully random agenda (True) or 
            intelligent challengers (False)
    
    Returns:
        VotingModel instance
    """
    from ..base import VotingModel
    
    return VotingModel(
        zi=zi,
        number_of_voters=3,
        majority=2,
        number_of_feasible_alternatives=3,
        utility_functions=jnp.array([
            [3, 2, 1],  # Voter 1: A>B>C
            [1, 3, 2],  # Voter 2: B>C>A
            [2, 1, 3],  # Voter 3: C>A>B
        ]),
    )
