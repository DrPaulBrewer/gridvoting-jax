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
        
        # Create padded indices and valid mask for JAX-native loops
        indices = jnp.arange(total_size)
        self.valid_mask = (indices < self.N).reshape(self.num_batches, BATCH_SIZE)
        # Use index 0 for padding (will be masked out by weight * 0)
        self.batch_indices = jnp.where(indices < self.N, indices, 0).reshape(self.num_batches, BATCH_SIZE)
    
    def rmatvec(self, v):
        """
        Compute P.T @ v without materializing P (JAX-native batched for GMRES).
        
        Uses an unrolled batch loop to ensure complete compatibility with JAX AD
        (custom_linear_solve) while maintaining memory efficiency.
        """
        v = jnp.asarray(v, dtype=DTYPE_FLOAT)
        
        # Pad v to the full execution size (num_batches * BATCH_SIZE) to avoid OOB
        v_full = jnp.zeros(self.num_batches * BATCH_SIZE, dtype=DTYPE_FLOAT)
        v_full = v_full.at[:self.N].set(v)
        
        contributions = []
        for i in range(self.num_batches):
            batch_inds = self.batch_indices[i]
            mask = self.valid_mask[i]
            
            cV_batch = compute_winner_matrix_jit(
                self.utility_functions, self.majority, batch_inds
            )
            batch_rows = finalize_transition_matrix(cV_batch, self.zi, batch_inds)
            
            # Weighted rows: sum_i (v_i * Row_i)
            batch_rows = batch_rows.astype(DTYPE_FLOAT)
            v_weights = (v_full[batch_inds] * mask).astype(DTYPE_FLOAT)
            contributions.append(jnp.sum(batch_rows * v_weights[:, jnp.newaxis], axis=0))
            
        return jnp.sum(jnp.stack(contributions), axis=0)

    
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
        v = jnp.asarray(v, dtype=DTYPE_FLOAT)
        
        result = jnp.zeros(self.N, dtype=DTYPE_FLOAT)
        
        # Process batches with Python loop (not JIT, so no nested issues)
        for batch_idx in range(self.num_batches):
            batch_inds = self.batch_indices[batch_idx]
            # Use pre-computed mask (index-based masking broke when padding changed to 0)
            mask = self.valid_mask[batch_idx]
            
            # Only process valid indices
            valid_inds = batch_inds[mask]
            
            # Compute rows for valid indices only
            cV_batch = compute_winner_matrix_jit(
                self.utility_functions, self.majority, valid_inds
            )
            batch_rows = finalize_transition_matrix(cV_batch, self.zi, valid_inds)
            
            # For P.T @ v, weight each row i by v[valid_inds[i]]
            v_weights = v[valid_inds]
            
            weighted = batch_rows * v_weights[:, jnp.newaxis]
            result = result + jnp.sum(weighted, axis=0)
        
        return result
    
    def matvec(self, v):
        """
        Compute P @ v without materializing P (JAX-native batched for GMRES).
        
        Uses an unrolled batch loop for JAX AD compatibility. 
        """
        v = jnp.asarray(v, dtype=DTYPE_FLOAT)
        
        batch_results = []
        for i in range(self.num_batches):
            batch_inds = self.batch_indices[i]
            mask = self.valid_mask[i]
            
            cV_batch = compute_winner_matrix_jit(
                self.utility_functions, self.majority, batch_inds
            )
            batch_rows = finalize_transition_matrix(cV_batch, self.zi, batch_inds)
            
            # Row-wise dot product: sum_j (P[batch_inds, j] * v[j])
            batch_rows = batch_rows.astype(DTYPE_FLOAT)
            row_results = jnp.sum(batch_rows * v[jnp.newaxis, :], axis=1)
            
            # Mask out contribution of padding rows
            batch_results.append(row_results * mask.astype(DTYPE_FLOAT))
            
        # Concatenate and truncate
        flat_results = jnp.concatenate(batch_results)
        return flat_results[:self.N]
            
    
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

