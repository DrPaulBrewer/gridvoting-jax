"""Lazy weighted stochastic matrix implementation.

This module provides memory-efficient lazy evaluation of stochastic matrices
where diagonal and off-diagonal elements follow a specific pattern.

The LazyWeightedStochasticMatrix represents a matrix M where:
- M[i,i] = status_quo_values[i] (diagonal)
- M[i,j] = mask[i,j] * (1.0-M[i,i])*weight[j]/(sum_j weight[j]) (off-diagonal, i≠j)

where challenger_values[i] is calculated as (1-status_quo_values[i]) / number_of_winners_in_row[i]
"""

from typing import Union, Tuple
import jax.numpy as jnp
from jax import Array

from ..core import constants


class LazyWeightedStochasticMatrix:
    """Lazy evaluation of stochastic matrices with structured diagonal/off-diagonal patterns.
    
    This class provides memory-efficient matrix operations without materializing
    the full matrix. It's particularly useful for large transition matrices in
    Markov chain analysis.
    
    Attributes:
        mask: 2D boolean array defining which off-diagonal elements are non-zero
        status_quo_values: 1D array of diagonal values M[i,i]
        weights: 1D array of weights for each state
        shape: Tuple of matrix dimensions (n, n)
        ndim: Number of dimensions (always 2)
        dtype: Data type of matrix elements
    """
    
    def __init__(
        self, 
        mask: Array, 
        status_quo_values: Array,
        weights: Array
    ) -> None:
        """Initialize a LazyWeightedStochasticMatrix.

        Args:
            mask: 2D square boolean array defining non-zero off-diagonal positions
            status_quo_values: 1D array of diagonal values
            weights: 1D array of weights for each state
        """
        self.mask = mask
        self.status_quo_values = jnp.array(status_quo_values, dtype=constants.DTYPE_FLOAT)
        self.weights = jnp.array(weights, dtype=constants.DTYPE_FLOAT)
        assert mask.dtype == jnp.bool_, "Mask must be boolean"
        assert mask.shape[0] == mask.shape[1], "Mask must be square"
        assert mask.shape[0] == status_quo_values.shape[0], "Mask and status quo values must have same shape"
        assert mask.shape[0] == weights.shape[0], "Mask and weights must have same shape"
        assert jnp.all(status_quo_values>=0.0) & jnp.all(status_quo_values<=1.0), "Status quo values must be in [0,1]"
        assert jnp.all(weights>=0.0), "Weights must be non-negative"
        assert jnp.all(jnp.logical_not(jnp.diagonal(mask))), "Mask diagonal must be all False"
        winners_per_row = jnp.sum(mask, axis=1)
        assert not jnp.any(jnp.where(winners_per_row==0, status_quo_values!=1.0, False)), "No winners in row but non-1 status quo value"
        # Compute per-row weight sums (sum of weights for masked positions in each row)
        # masked_weights[i,j] = mask[i,j] * weights[j]
        masked_weights = mask * weights[None, :]
        row_weight_sums = jnp.sum(masked_weights, axis=1)
        # Normalize weights per row: normalized_weights[i,j] = weights[j] / row_weight_sums[i]
        # Safe division: if row_weight_sums[i] == 0, set to 0
        safe_row_weight_sums = jnp.where(row_weight_sums == 0, 1.0, row_weight_sums)
        self.row_normalized_weights = weights[None, :] / safe_row_weight_sums[:, None]
        # challenger_scale[i] = (1 - status_quo_values[i])
        self.challenger_scale = 1.0 - status_quo_values
        self.ndim = 2
        self.shape = mask.shape
        self.dtype = constants.DTYPE_FLOAT

    def __matmul__(self, other: Array) -> Array:
        """Right multiplication: self @ other (M * v or M * V).
        
        Args:
            other: Vector (1D) or matrix (2D) to multiply
            
        Returns:
            Result of matrix-vector or matrix-matrix multiplication
        """
        other = jnp.asarray(other, dtype=constants.DTYPE_FLOAT)
        # M @ v: result[i] = status_quo[i]*v[i] + challenger_scale[i] * sum_j(mask[i,j] * row_normalized_weights[i,j] * v[j])
        
        if other.ndim == 1:
            # (n,n) @ (n,) -> (n,)
            weighted_other = self.row_normalized_weights * other[None, :]  # (n, n) element-wise
            off_diag_contrib = jnp.sum(self.mask * weighted_other, axis=1)  # (n,)
            return self.status_quo_values * other + self.challenger_scale * off_diag_contrib
        else:
            # (n,n) @ (n, k) -> (n, k)
            # result[i,k] = status_quo[i]*other[i,k] + challenger_scale[i] * sum_j(mask[i,j] * row_normalized_weights[i,j] * other[j,k])
            # This is: (mask * row_normalized_weights) @ other, scaled by challenger_scale
            off_diag_contrib = (self.mask * self.row_normalized_weights) @ other  # (n, k)
            return self.status_quo_values[:, None] * other + self.challenger_scale[:, None] * off_diag_contrib

    def __rmatmul__(self, other: Array) -> Array:
        """Left multiplication: other @ self (v * M or V * M).
        
        Args:
            other: Vector (1D) or matrix (2D) to multiply
            
        Returns:
            Result of vector-matrix or matrix-matrix multiplication
        """
        other = jnp.asarray(other, dtype=constants.DTYPE_FLOAT)
        # v @ M: result[j] = status_quo[j]*v[j] + sum_i(mask[i,j] * challenger_scale[i] * row_normalized_weights[i,j] * v[i])
        
        # Compute the weighted contribution: mask.T * row_normalized_weights.T gives us the effective weights
        # Then we need to multiply by challenger_scale and the input vector
        if other.ndim == 1:
            # (n,) @ (n,n) -> (n,)
            # Compute: sum_i(mask[i,j] * row_normalized_weights[i,j] * challenger_scale[i] * v[i])
            weighted_input = other * self.challenger_scale  # (n,)
            off_diag_contrib = (self.mask.T * self.row_normalized_weights.T) @ weighted_input  # (n,)
            return other * self.status_quo_values + off_diag_contrib
        else:
            # (k, n) @ (n, n) -> (k, n)
            # For batch: result[k,j] = other[k,j]*status_quo[j] + sum_i(mask[i,j] * row_normalized_weights[i,j] * challenger_scale[i] * other[k,i])
            weighted_input = other * self.challenger_scale[None, :]  # (k, n)
            # We need: for each k, compute (mask.T * row_normalized_weights.T) @ weighted_input[k]
            # This is equivalent to: weighted_input @ (mask * row_normalized_weights)
            off_diag_contrib = weighted_input @ (self.mask * self.row_normalized_weights)  # (k, n)
            return other * self.status_quo_values[None, :] + off_diag_contrib

    def __getitem__(self, key: Union[int, Tuple[int, int], Tuple[int, slice], Tuple[slice, int]]) -> Array:
        """Element access and basic slicing.
        
        Args:
            key: Index or slice specification
            
        Returns:
            Requested element, row, or column
            
        Raises:
            NotImplementedError: For advanced indexing patterns
            
        Note:
            Does NOT support advanced indexing (fancy indexing).
        """
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            if isinstance(i, int) and isinstance(j, int):
                # Scalar access: M[i,j] = status_quo[i] if i==j, else mask[i,j] * challenger_scale[i] * row_normalized_weights[i,j]
                if i == j:
                    return self.status_quo_values[i]
                else:
                    return self.mask[i, j] * self.challenger_scale[i] * self.row_normalized_weights[i, j]
            
            # Row access M[i, :]
            if isinstance(i, int) and isinstance(j, slice) and j == slice(None):
                row = self.mask[i, :] * self.challenger_scale[i] * self.row_normalized_weights[i, :]
                return row.at[i].set(self.status_quo_values[i])
            
            # Column access M[:, j]
            if isinstance(i, slice) and i == slice(None) and isinstance(j, int):
                col = self.mask[:, j] * self.challenger_scale * self.row_normalized_weights[:, j]
                return col.at[j].set(self.status_quo_values[j])
        elif isinstance(key, int):
            # row access M[i]
            row = self.mask[key, :] * self.challenger_scale[key] * self.row_normalized_weights[key, :]
            return row.at[key].set(self.status_quo_values[key])

        raise NotImplementedError("Advanced indexing/slicing not supported for LazyWeightedStochasticMatrix, got:"+str(type(key))+' '+str(key))

    @property
    def T(self) -> 'LazyWeightedStochasticMatrixTranspose':
        """Return the transpose of the matrix.
        
        Returns:
            LazyWeightedStochasticMatrixTranspose wrapper providing transposed view
        """
        return LazyWeightedStochasticMatrixTranspose(self)

    def diagonal(self) -> Array:
        """Return the diagonal elements.
        
        Returns:
            1D array of diagonal values M[i,i]
        """
        return self.status_quo_values

    def row_entropies(self) -> Array:
        """Return the entropies of each row.
        
        Returns:
            1D array of row entropies
        """
        # For weighted stochastic matrix, entropy calculation is more complex
        # H[i] = -sum_j M[i,j] * log2(M[i,j])
        # = -status_quo[i]*log2(status_quo[i]) - sum_{j!=i, mask[i,j]} (challenger_scale[i]*row_normalized_weights[i,j]) * log2(challenger_scale[i]*row_normalized_weights[i,j])
        
        # Compute entropy contribution from diagonal
        safe_sq = jnp.where(self.status_quo_values == 0, 1.0, self.status_quo_values)
        diagonal_entropy = -self.status_quo_values * jnp.log2(safe_sq)
        
        # Compute entropy contribution from off-diagonal elements
        # For each row i: sum over j where mask[i,j] is True
        off_diag_probs = self.challenger_scale[:, None] * self.row_normalized_weights * self.mask
        safe_off_diag = jnp.where(off_diag_probs == 0, 1.0, off_diag_probs)
        off_diagonal_entropy = -jnp.sum(off_diag_probs * jnp.log2(safe_off_diag), axis=1)
        
        return diagonal_entropy + off_diagonal_entropy
    
    def to_dense(self) -> Array:
        """Materialize the full matrix.
        
        Returns:
            Dense JAX array representation of the matrix
            
        Warning:
            This creates an n×n dense matrix in memory. Use only when necessary.
        """
        # M[i,j] = status_quo[i] if i==j, else mask[i,j] * challenger_scale[i] * row_normalized_weights[i,j]
        return jnp.diag(self.status_quo_values) + self.challenger_scale[:, None] * self.mask * self.row_normalized_weights


class LazyWeightedStochasticMatrixTranspose:
    """Transpose wrapper for LazyWeightedStochasticMatrix.
    
    Provides a transposed view without materializing the full matrix.
    
    Attributes:
        original: The original LazyWeightedStochasticMatrix
        shape: Transposed shape (original.shape[1], original.shape[0])
        ndim: Number of dimensions (always 2)
        dtype: Data type of matrix elements
    """
    
    def __init__(self, original: LazyWeightedStochasticMatrix) -> None:
        """Initialize transpose wrapper.
        
        Args:
            original: The LazyWeightedStochasticMatrix to transpose
        """
        self.original = original
        self.shape = (original.shape[1], original.shape[0])
        self.ndim = original.ndim
        self.dtype = original.dtype

    def __matmul__(self, other: Array) -> Array:
        """Right multiplication: (M^T) @ other.
        
        Args:
            other: Vector or matrix to multiply
            
        Returns:
            Result of transposed matrix multiplication
        """
        # (M^T) @ v = (v^T @ M)^T
        return self.original.__rmatmul__(other.T).T if other.ndim > 1 else self.original.__rmatmul__(other)

    def __rmatmul__(self, other: Array) -> Array:
        """Left multiplication: other @ (M^T).
        
        Args:
            other: Vector or matrix to multiply
            
        Returns:
            Result of transposed matrix multiplication
        """
        # v @ (M^T) = (M @ v^T)^T
        return self.original.__matmul__(other.T).T if other.ndim > 1 else self.original.__matmul__(other)

    @property
    def T(self) -> LazyWeightedStochasticMatrix:
        """Return the transpose of the transpose (i.e., the original matrix).
        
        Returns:
            The original LazyWeightedStochasticMatrix
        """
        return self.original

    def diagonal(self) -> Array:
        """Return the diagonal elements.
        
        Returns:
            1D array of diagonal values (same as original since diagonal is symmetric)
        """
        return self.original.diagonal()

    def to_dense(self) -> Array:
        """Materialize the full transposed matrix.
        
        Returns:
            Dense JAX array representation of the transposed matrix
        """
        return self.original.to_dense().T
