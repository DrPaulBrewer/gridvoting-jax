"""Lazy matrix operators for memory-efficient large-scale models.

This module provides JIT-compiled helper functions for computing transition
matrix rows and columns on-demand, without materializing the full matrix.
"""

import jax
import jax.numpy as jnp


@jax.jit
def _compute_transition_rows_jit(utility_functions, majority, zi, status_quo_indices):
    """
    Compute multiple rows of transition matrix P.
    
    Each row i represents transitions FROM status quo i TO all challengers.
    P[i, j] = probability of transitioning from state i to state j.
    
    Args:
        utility_functions: (V, N) array of voter utilities
        majority: int, votes needed to win
        zi: bool, True for fully random agenda, False for intelligent challengers
        status_quo_indices: (B,) array of status quo state indices
    
    Returns:
        (B, N) array where result[b, j] = P[status_quo_indices[b], j]
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
    votes = prefs.astype("int32").sum(axis=0)
    
    # Determine winners: cV[b, j] = 1 if j beats status_quo_indices[b]
    cV_batch = jnp.greater_equal(votes, majority).astype("int32")
    
    # Convert winner matrix to transition matrix
    # Use jax.lax.cond instead of if/else for JIT compatibility
    def zi_true_fn():
        # ZI: Uniform random over ALL alternatives
        # If ch beats sq: move to ch (prob 1/N)
        # If ch loses to sq: stay at sq
        # Plus picked sq itself: stay at sq
        # So prob(move i->j) = 1/N if j beats i
        # prob(stay i) = (1/N) * (count(j that lose to i) + 1)
        #              = (1/N) * ((N - count(win) - 1) + 1)
        #              = (N - row_sum)/N
        
        # Count winning alternatives for each status quo
        row_sums = cV_batch.sum(axis=1)  # (B,)
        
        # Start with cV_batch (winners get 1, losers get 0)
        cP = cV_batch.astype(cU.dtype)
        
        # Add diagonal: status quo gets (N - row_sum)
        # For each batch element b, add (N - row_sum[b]) to position status_quo_indices[b]
        diag_values = N - row_sums  # (B,)
        cP = cP.at[jnp.arange(batch_size), status_quo_indices].add(diag_values.astype(cU.dtype))
        
        # Divide everything by N
        cP = cP / N
        
        return cP
    
    def zi_false_fn():
        # Intelligent challengers: random over {j : j beats i} ∪ {i}
        # winning_set_size[b] = number of alternatives that beat status_quo_indices[b]
        winning_set_sizes = cV_batch.sum(axis=1)  # (B,)
        
        # Add 1 for status quo (always in the set)
        set_sizes = winning_set_sizes + 1  # (B,)
        
        # Probability for each alternative
        # If j beats i: prob = 1 / set_size
        # If j doesn't beat i and j != i: prob = 0
        # If j == i: prob = 1 / set_size
        cP = cV_batch.astype(cU.dtype) / set_sizes[:, jnp.newaxis]
        
        # Add status quo probability
        # For each batch element b, add 1/set_size to position status_quo_indices[b]
        sq_probs = 1.0 / set_sizes  # (B,)
        cP = cP.at[jnp.arange(batch_size), status_quo_indices].add(sq_probs)
        return cP
    
    cP_batch = jax.lax.cond(zi, zi_true_fn, zi_false_fn)
    
    return cP_batch


def estimate_memory_for_dense_matrix(N, dtype):
    """
    Estimate memory needed for N×N dense transition matrix.
    
    Args:
        N: int, number of states
        dtype: jax dtype
    
    Returns:
        int, estimated memory in bytes
    """
    bytes_per_element = jnp.dtype(dtype).itemsize
    return N * N * bytes_per_element


def should_use_lazy(N, dtype, available_memory_bytes, threshold=0.75):
    """
    Decide whether to use lazy construction based on memory.
    
    Args:
        N: int, number of states
        dtype: jax dtype
        available_memory_bytes: int, available memory in bytes
        threshold: float, use lazy if estimated_memory > threshold * available_memory
    
    Returns:
        bool, True if should use lazy construction
    """
    if available_memory_bytes is None:
        # Can't determine memory, default to dense for small N, lazy for large N
        return N > 10000
    
    estimated_memory = estimate_memory_for_dense_matrix(N, dtype)
    return estimated_memory > threshold * available_memory_bytes
