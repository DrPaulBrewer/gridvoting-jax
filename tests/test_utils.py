"""Test utilities for gridvoting-jax test suite.

Shared helper functions used across multiple test files.
"""

import jax.numpy as jnp
import gridvoting_jax as gv


def finalize_transition_matrix(vm: gv.VotingModel, cV):
    """Shared logic to convert winner matrix cV to transition matrix cP"""
    nfa = vm.number_of_feasible_alternatives
    zi = vm.zi
    cV_sum_of_row = cV.sum(axis=1)  # number of winning alternatives for each SQ
        
    # set up the ZI and MI transition matrices
    if zi:
        # ZI: Uniform random over ALL alternatives.
        # If ch beats sq: move to ch (prob 1/N)
        # If ch loses to sq: stay at sq
        # Plus picked sq itself: stay at sq
        # So prob(move i->j) = 1/N if j beats i
        # prob(stay i) = (1/N) * (count(j that lose to i) + 1)
        #              = (1/N) * ((N - count(win) - 1) + 1)
        #              = (N - row_sum)/N
        # logic in code: cV + diag(N - row_sum) / N
        cP = jnp.divide(
            jnp.add(cV, jnp.diag(jnp.subtract(nfa, cV_sum_of_row))), 
            nfa
            )
    else:
        # MI: Uniform random over Winning Set(i) U {i}
        # Size of set = row_sum + 1
        # Prob(move i->j) = 1/(row_sum+1) if j beats i
        # Prob(stay i) = 1/(row_sum+1)
        # logic in code: (cV + I) / (1 + row_sum)
        cP = jnp.divide(
            jnp.add(cV, jnp.eye(nfa)), 
            (1 + cV_sum_of_row)[:, jnp.newaxis]
            )
    return cP


def get_transition_matrix_vectorized(vm: gv.VotingModel):
    """Adapted from v0.9.1: Original fully vectorized implementation. O(V * N^2) memory."""
    utility_functions = vm.utility_functions
    majority = vm.majority
    cU = jnp.asarray(utility_functions) 
    
    # Vectorized computation: compare all alternatives at once
    # cU shape: (n_voters, nfa)
    # cU[:, :, jnp.newaxis] shape: (n_voters, nfa, 1) to broadcast vs challengers (rows)
    # cU[:, jnp.newaxis, :] shape: (n_voters, 1, nfa) to broadcast vs status quo (cols) 
    # Note: Previous implementation comment had axes swapped in explanation but logic was correct for outcome.
    # Let's align with the standard logic:
    # P[i, j] is prob of moving i -> j.
    # i is Status Quo (SQ), j is Challenger (CH).
    # We need votes for CH against SQ.
    # Utility for SQ: cU[:, i] (column i)
    # Utility for CH: cU[:, j] (column j)
    # pref = u(CH) > u(SQ)
    
    # In the original code:
    # preferences = jnp.greater(cU[:, jnp.newaxis, :], cU[:, :, jnp.newaxis])
    # LHS: cU[:, 1, N] -> varying last dim is COLUMNS (CH)
    # RHS: cU[:, N, 1] -> varying middle dim is ROWS (SQ)
    # Result: (V, SQ, CH).  [v, i, j] is "does v prefer j over i?"
    # Correct.
    
    preferences = jnp.greater(cU[:, jnp.newaxis, :], cU[:, :, jnp.newaxis])
    
    # Sum votes across voters: shape (nfa, nfa) -> (SQ, CH)
    total_votes = preferences.astype("int32").sum(axis=0)
    
    # Determine winners: 1 if challenger gets majority, 0 otherwise
    # cV[i, j] = 1 if j beats i
    cV = jnp.greater_equal(total_votes, majority)
    
    return finalize_transition_matrix(vm, cV)
