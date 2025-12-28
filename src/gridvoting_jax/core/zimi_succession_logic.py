"""ZI/MI succession logic for voting models.

This module contains the single source of truth for Zero Intelligence (ZI) and
Majority Intelligence (MI) succession logic - how states succeed each other in
the Markov chain based on voting outcomes.

This is VOTING-SPECIFIC domain logic, not general Markov chain logic.
"""

import jax
import jax.numpy as jnp


from . import DTYPE_FLOAT

@jax.jit
def finalize_transition_matrix_zi_jit(cV, status_quo_indices, eligibility_mask=None):
    """
    Zero Intelligence (ZI) succession logic.
    
    Uniform random selection over ALL eligible alternatives.
    
    Mathematical formulation:
        - If j beats i: prob(i→j) = 1/N (or 1/eligible_count if masked)
        - If j loses to i: prob(i→j) = 0
        - prob(i→i) = (N - row_sum)/N
    
    Args:
        cV: (B, N) winner matrix where cV[b, j] = 1 if j beats status_quo_indices[b]
        nfa: int, number of feasible alternatives
        status_quo_indices: (B,) array of status quo state indices
        eligibility_mask: Optional (B, N) boolean mask for eligible challengers
    
    Returns:
        cP: (B, N) transition probability matrix
    """
    batch_size = cV.shape[0]
    
# Future:  the issue with this implementation is that it assumes a fixed eligibility mask
# but we need to be able to update the eligibility mask in place
# so we need to do it in a different way.  One of the more likely candidates for eligibility mask
# is nearest neighbors, which is a kind of convolution, a boolean version of jax.scipy.signal.convolve2d ?

    if eligibility_mask is not None:
        cV = cV * eligibility_mask
        eligible_count = eligibility_mask.sum(axis=1).astype(DTYPE_FLOAT)
    else:
        nfa = cV.shape[1]
        eligible_count = jnp.full(batch_size, nfa, dtype=DTYPE_FLOAT)
    
    # Count winning alternatives for each status quo
    row_sums = cV.sum(axis=1)
    
    # Start with cV (winners get 1, losers get 0)
    # Use same float precision as eligible_count
    cP = cV.astype(DTYPE_FLOAT)
    
    # Add diagonal: status quo gets (eligible_count - row_sum)
    diag_values = eligible_count - row_sums
    cP = cP.at[jnp.arange(batch_size), status_quo_indices].add(diag_values)
    
    # Divide by eligible count
    cP = cP / eligible_count[:, jnp.newaxis]
    
    return cP


@jax.jit
def finalize_transition_matrix_mi_jit(cV, status_quo_indices, eligibility_mask=None):
    """
    Majority Intelligence (MI) succession logic.
    
    Uniform random selection over eligible winners ∪ {status quo}.
    
    Mathematical formulation:
        - Set size = |{j : j beats i}| + 1
        - If j beats i: prob(i→j) = 1/set_size
        - If j loses to i and j ≠ i: prob(i→j) = 0
        - prob(i→i) = 1/set_size
    
    Args:
        cV: (B, N) winner matrix where cV[b, j] = 1 if j beats status_quo_indices[b]
        nfa: int, number of feasible alternatives
        status_quo_indices: (B,) array of status quo state indices
        eligibility_mask: Optional (B, N) boolean mask for eligible challengers
    
    Returns:
        cP: (B, N) transition probability matrix
    """

    batch_size = cV.shape[0]
    
    if eligibility_mask is not None:
        cV = cV * eligibility_mask
    
    # Count winning alternatives for each status quo
    row_sums = cV.sum(axis=1)
    
    # Set size = winners + status quo
    set_sizes = row_sums + 1
    
    # Probability for winners
    cP = cV.astype(DTYPE_FLOAT) / set_sizes[:, jnp.newaxis]
    
    # Add status quo probability
    sq_probs = 1.0 / set_sizes
    cP = cP.at[jnp.arange(batch_size), status_quo_indices].add(sq_probs)
    
    return cP


def finalize_transition_matrix(cV, zi, status_quo_indices, eligibility_mask=None):
    """
    Dispatch to ZI or MI succession logic.
    
    This is the main entry point for converting a winner matrix to a transition matrix.
    
    Args:
        cV: (B, N) winner matrix where cV[b, j] = 1 if j beats status_quo_indices[b]
        zi: bool, True for Zero Intelligence, False for Majority Intelligence
        status_quo_indices: (B,) array of status quo state indices
        eligibility_mask: Optional (B, N) boolean mask for eligible challengers
    
    Returns:
        cP: (B, N) transition probability matrix
    """

    if zi:
        return finalize_transition_matrix_zi_jit(cV, status_quo_indices, eligibility_mask)
    else:
        return finalize_transition_matrix_mi_jit(cV, status_quo_indices, eligibility_mask)
