"""Lazy transition matrix with hybrid batching strategy.

This module provides LazyTransitionMatrix with two matvec implementations:
- Non-batched: For GMRES (avoids nested JIT issues)
- Batched: For power method (more memory efficient)

The class automatically selects the appropriate implementation.
"""

import jax
import jax.numpy as jnp
from ...core.winner_determination import compute_winner_matrix_jit
from ...core.zimi_succession_logic import finalize_transition_matrix
from ...core import DTYPE_FLOAT

# Fixed batch size for memory-efficient computation
BATCH_SIZE = 128


class LazyTransitionMatrix:
    """
    Lazy transition matrix with hybrid batching strategy.
    
    Provides both batched and non-batched matvec implementations:
    - Use batched version for power method (memory efficient)
    - Use non-batched version for GMRES (avoids nested JIT issues)
    """
    
    def __init__(self, utility_functions, majority, zi, number_of_feasible_alternatives):
        """
        Initialize lazy transition matrix.
        
        Args:
            utility_functions: (V, N) array of voter utilities
            majority: int, votes needed to win
            zi: bool, True for fully random agenda, False for intelligent challengers
            number_of_feasible_alternatives: int, number of states N
        """
        self.utility_functions = jnp.asarray(utility_functions,dtype=DTYPE_FLOAT)
        self.majority = majority
        self.zi = zi
        self.N = number_of_feasible_alternatives
        self.shape = (self.N, self.N)
        
        # Pre-compute batch structure for batched operations
        self.num_batches = (self.N + BATCH_SIZE - 1) // BATCH_SIZE
        total_size = self.num_batches * BATCH_SIZE
        
        # Create padded indices array
        # Use self.N as padding value (not 0) so mask comparison works correctly
        indices = jnp.arange(total_size)
        indices = jnp.where(indices < self.N, indices, self.N)
        self.batch_indices = indices.reshape(self.num_batches, BATCH_SIZE)
    
    def rmatvec(self, v):
        """
        Compute P.T @ v without materializing P (non-batched for GMRES compatibility).
        
        Args:
            v: (N,) vector
        
        Returns:
            (N,) vector, result of P.T @ v
        """
        v = jnp.asarray(v)
        
        # Non-batched: works with GMRES
        all_indices = jnp.arange(self.N)
        cV = compute_winner_matrix_jit(
            self.utility_functions, self.majority, all_indices
        )
        # this recreates the transition matrix on every call to rmatvec
        # so it is computationally expensive but slightly more memory efficient because P can be deallocated
        # also compatible with GMRES

        P = finalize_transition_matrix(cV, self.zi, all_indices)

        return jnp.sum(P * v[:, jnp.newaxis], axis=0)

    
    def rmatvec_batched(self, v):
        """
        Compute P.T @ v with batching (for power method).
        
        More memory efficient than rmatvec, but incompatible with GMRES.
        Use this for power method iterations.
        
        Args:
            v: (N,) vector
        
        Returns:
            (N,) vector, result of P.T @ v
        """
        v = jnp.asarray(v)
        
        result = jnp.zeros(self.N, dtype=DTYPE_FLOAT)
        
        # Process batches with Python loop (not JIT, so no nested issues)
        for batch_idx in range(self.num_batches):
            batch_inds = self.batch_indices[batch_idx]
            
            # Create mask for valid indices (not padding)
            valid_mask = batch_inds < self.N
            
            # Only process valid indices to avoid duplicate diagonal additions
            valid_inds = batch_inds[valid_mask]
            
            # Compute rows for valid indices only
            cV_batch = compute_winner_matrix_jit(
                self.utility_functions, self.majority, valid_inds
            )
            batch_rows = finalize_transition_matrix(cV_batch, self.zi, valid_inds)
            
            # For P.T @ v, weight each row i by v[valid_inds[i]]
            # batch_rows has shape (num_valid, N)
            # We want: sum over i of (v[i] * P[i, :])
            v_weights = v[valid_inds]  # Get v values for valid indices
            
            weighted = batch_rows * v_weights[:, jnp.newaxis]
            result = result + jnp.sum(weighted, axis=0)
        
        return result
    
    def matvec(self, v):
        """
        Compute P @ v without materializing P (non-batched for GMRES compatibility).
        
        Args:
            v: (N,) vector
        
        Returns:
            (N,) vector, result of P @ v
        """
        v = jnp.asarray(v)
        
        # Non-batched:
        # Compute all rows
        all_indices = jnp.arange(self.N)
        cV = compute_winner_matrix_jit(
            self.utility_functions, self.majority, all_indices
        )
        P = finalize_transition_matrix(cV, self.zi, all_indices)
        
        return jnp.sum(P * v[jnp.newaxis, :], axis=1)
    
    def todense(self):
        """
        Materialize the full matrix (for testing/comparison).
        
        Returns:
            (N, N) dense transition matrix
        """
        all_indices = jnp.arange(self.N)
        cV = compute_winner_matrix_jit(
            self.utility_functions, self.majority, all_indices
        )
        return finalize_transition_matrix(cV, self.zi, all_indices)

    def compute_rows(self, indices):
        """
        Compute specific rows of P matrix.
        
        Required for bifurcated power method entropy initialization.
        
        Args:
            indices: (k,) array of row indices to compute
            
        Returns:
            (k, N) array of transition probabilities
        """
        indices = jnp.asarray(indices)
        cV = compute_winner_matrix_jit(
            self.utility_functions, self.majority, indices
        )
        return finalize_transition_matrix(cV, self.zi, indices)

