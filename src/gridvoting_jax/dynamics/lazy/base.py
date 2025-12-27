"""Lazy transition matrix with hybrid batching strategy.

This module provides LazyTransitionMatrix with two matvec implementations:
- Non-batched: For GMRES (avoids nested JIT issues)
- Batched: For power method (more memory efficient)

The class automatically selects the appropriate implementation.
"""

import jax
import jax.numpy as jnp
from .operators import _compute_transition_rows_jit

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
        self.utility_functions = jnp.asarray(utility_functions)
        self.majority = majority
        self.zi = zi
        self.N = number_of_feasible_alternatives
        self.shape = (self.N, self.N)
        self.dtype = self.utility_functions.dtype
        
        # Pre-compute batch structure for batched operations
        self.num_batches = (self.N + BATCH_SIZE - 1) // BATCH_SIZE
        total_size = self.num_batches * BATCH_SIZE
        
        # Create padded indices array
        indices = jnp.arange(total_size)
        indices = jnp.where(indices < self.N, indices, 0)
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
        P = _compute_transition_rows_jit(
            self.utility_functions, self.majority, self.zi, all_indices
        )
        
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
        
        result = jnp.zeros(self.N, dtype=self.dtype)
        
        # Process batches with Python loop (not JIT, so no nested issues)
        for batch_idx in range(self.num_batches):
            batch_inds = self.batch_indices[batch_idx]
            
            # Create mask for valid indices (not padding)
            valid_mask = batch_inds < self.N
            
            # Compute rows for this batch
            # batch_rows[i] is row batch_inds[i] of P
            batch_rows = _compute_transition_rows_jit(
                self.utility_functions, self.majority, self.zi, batch_inds
            )
            
            # For P.T @ v, weight each row i by v[batch_inds[i]]
            # batch_rows has shape (BATCH_SIZE, N)
            # We want: sum over i of (v[i] * P[i, :])
            v_weights = v[batch_inds]  # Get v values for this batch
            
            # Mask out padded entries to avoid double-counting
            v_weights = jnp.where(valid_mask, v_weights, 0.0)
            
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
        
        # Non-batched: works with GMRES
        all_indices = jnp.arange(self.N)
        P = _compute_transition_rows_jit(
            self.utility_functions, self.majority, self.zi, all_indices
        )
        
        return jnp.sum(P * v[jnp.newaxis, :], axis=1)
    
    def todense(self):
        """
        Materialize the full matrix (for testing/comparison).
        
        Returns:
            (N, N) dense transition matrix
        """
        all_indices = jnp.arange(self.N)
        return _compute_transition_rows_jit(
            self.utility_functions, self.majority, self.zi, all_indices
        )
