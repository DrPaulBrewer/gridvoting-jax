"""Winner determination for voting models.

This module contains voting-specific logic for determining pairwise winners
based on voter utilities and majority thresholds.

This is VOTING-SPECIFIC domain logic, not general Markov chain logic.
"""

import jax
import jax.numpy as jnp


@jax.jit
def compute_winner_matrix_jit(utility_functions, majority, status_quo_indices):
    """
    Compute winner matrix for pairwise comparisons.
    
    Determines which alternatives beat which status quo alternatives based on
    majority voting.
    
    Args:
        utility_functions: (V, N) array of voter utilities
        majority: int, votes needed to win
        status_quo_indices: (B,) array of status quo state indices
    
    Returns:
        cV_batch: (B, N) winner matrix where cV[b, j] = 1 if j beats status_quo_indices[b]
    """
    cU = jnp.asarray(utility_functions)
    batch_size = status_quo_indices.shape[0]
    N = cU.shape[1]
    
    # Get utilities for status quo alternatives in this batch
    # U_sq shape: (V, B)
    U_sq = cU[:, status_quo_indices]
    
    # Generate preferences: CH (all N) vs SQ (batch)
    # LHS: cU -> (V, N) -> reshape to (V, 1, N) for broadcasting
    # RHS: U_sq -> (V, B) -> reshape to (V, B, 1)
    # Result: (V, B, N) where [v, b, j] = "does voter v prefer j over status_quo_indices[b]?"
    prefs = jnp.greater(cU[:, jnp.newaxis, :], U_sq[:, :, jnp.newaxis])
    
    # Sum votes -> (B, N)
    # dtype=bool is allowed with sum
    votes = prefs.sum(axis=0)
    
    # Determine winners: cV[b, j] = 1 if j beats status_quo_indices[b]
    cV_batch = jnp.greater_equal(votes, majority)
    
    return cV_batch
