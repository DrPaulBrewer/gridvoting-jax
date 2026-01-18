"""Lazy stochastic matrix implementation.

This module provides memory-efficient lazy evaluation of stochastic matrices
where diagonal and off-diagonal elements follow a specific pattern.

The LazyStochasticMatrix represents a matrix M where:
- M[i,i] = status_quo_values[i] (diagonal)
- M[i,j] = mask[i,j] * challenger_values[i] (off-diagonal, i≠j)

where challenger_values[i] is calculated as (1-status_quo_values[i]) / number_of_winners_in_row[i]
"""

from typing import Union, Tuple
import jax.numpy as jnp
from jax import Array

from ..core import constants


class LazyStochasticMatrix:
    """Lazy evaluation of stochastic matrices with structured diagonal/off-diagonal patterns.
    
    This class provides memory-efficient matrix operations without materializing
    the full matrix. It's particularly useful for large transition matrices in
    Markov chain analysis.
    
    Attributes:
        mask: 2D boolean array defining which off-diagonal elements are non-zero
        status_quo_values: 1D array of diagonal values M[i,i]
        challenger_values: computed 1D array of off-diagonal values (constant per row)
        shape: Tuple of matrix dimensions (n, n)
        ndim: Number of dimensions (always 2)
        dtype: Data type of matrix elements
    """
    
    def __init__(
        self, 
        mask: Array, 
        status_quo_values: Array
    ) -> None:
        """Initialize a LazyStochasticMatrix.

        Args:
            mask: 2D square boolean array defining non-zero off-diagonal positions
            status_quo_values: 1D array of diagonal values
        """
        self.mask = mask
        self.status_quo_values = status_quo_values
        assert mask.shape[0] == mask.shape[1], "Mask must be square"
        assert mask.shape[0] == status_quo_values.shape[0], "Mask and status quo values must have same shape"
        assert jnp.all(status_quo_values>=0.0) & jnp.all(status_quo_values<=1.0), "Status quo values must be in [0,1]"
        assert jnp.all(jnp.logical_not(jnp.diagonal(mask))), "Mask diagonal must be all False"
        winners_per_row = jnp.sum(mask, axis=1)
        assert not jnp.any(jnp.where(winners_per_row==0, status_quo_values!=1.0, False)), "No winners in row but non-1 status quo value"
        safe_inverse_winners_per_row = jnp.where(winners_per_row == 0, 0.0, 1.0 / winners_per_row)
        # challenger values are constant per row, dividing remaining probability by number of winners
        self.challenger_values = (1.0-status_quo_values) * safe_inverse_winners_per_row
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
        other = jnp.asarray(other, dtype=self.dtype)
        if other.ndim == 1:
            return self.status_quo_values * other + self.challenger_values * (self.mask @ other)
        else:
            # batch of column vectors (n, k)
            return self.status_quo_values[:, None] * other + self.challenger_values[:, None] * (self.mask @ other)

    def __rmatmul__(self, other: Array) -> Array:
        """Left multiplication: other @ self (v * M or V * M).
        
        Args:
            other: Vector (1D) or matrix (2D) to multiply
            
        Returns:
            Result of vector-matrix or matrix-matrix multiplication
        """
        other = jnp.asarray(other, dtype=self.dtype)
        # broadcasting works correctly for (n,) * (n,) or (k, n) * (n,)
        weighted_other = other * self.challenger_values
        return other * self.status_quo_values + (weighted_other @ self.mask)

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
                # Scalar access
                if i == j:
                    return self.status_quo_values[i]
                else:
                    return self.mask[i, j] * self.challenger_values[i]
            
            # Row access M[i, :]
            if isinstance(i, int) and isinstance(j, slice) and j == slice(None):
                row = self.mask[i, :] * self.challenger_values[i]
                return row.at[i].set(self.status_quo_values[i])
            
            # Column access M[:, j]
            if isinstance(i, slice) and i == slice(None) and isinstance(j, int):
                col = self.mask[:, j] * self.challenger_values
                return col.at[j].set(self.status_quo_values[j])
        elif isinstance(key, int):
            # row access M[i]
            row = self.mask[key, :] * self.challenger_values[key]
            return row.at[key].set(self.status_quo_values[key])

        raise NotImplementedError("Advanced indexing/slicing not supported for LazyStochasticMatrix, got:"+str(type(key))+' '+str(key))

    @property
    def T(self) -> 'LazyStochasticMatrixTranspose':
        """Return the transpose of the matrix.
        
        Returns:
            LazyStochasticMatrixTranspose wrapper providing transposed view
        """
        return LazyStochasticMatrixTranspose(self)

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
        winners_per_row = jnp.sum(self.mask, axis=1)
        # if winner_per_row == 0, entropy is 0 and status quo value is 1 (zero entropy)
        # otherwise, entropy is the sum of two terms:
        #        -winners_per_row[i]*(challenger_value[i] * log2(challenger_value[i])) 
        #        -status_quo[i]*log2(status_quo[i])
        # where challenger_value[i] is the challenger value for row i
        # and status_quo[i] is the status quo value for row i
        # 
        safe_values = jnp.where(self.challenger_values == 0, 1.0, self.challenger_values)   
        return -winners_per_row*(safe_values * jnp.log2(safe_values)) - \
            self.status_quo_values * jnp.log2(self.status_quo_values)
    
    def to_dense(self) -> Array:
        """Materialize the full matrix.
        
        Returns:
            Dense JAX array representation of the matrix
            
        Warning:
            This creates an n×n dense matrix in memory. Use only when necessary.
        """
        # diag(S) + diag(C) * mask
        return jnp.diag(self.status_quo_values) + self.challenger_values[:, None] * self.mask


class LazyStochasticMatrixTranspose:
    """Transpose wrapper for LazyStochasticMatrix.
    
    Provides a transposed view without materializing the full matrix.
    
    Attributes:
        original: The original LazyStochasticMatrix
        shape: Transposed shape (original.shape[1], original.shape[0])
        ndim: Number of dimensions (always 2)
        dtype: Data type of matrix elements
    """
    
    def __init__(self, original: LazyStochasticMatrix) -> None:
        """Initialize transpose wrapper.
        
        Args:
            original: The LazyStochasticMatrix to transpose
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
    def T(self) -> LazyStochasticMatrix:
        """Return the transpose of the transpose (i.e., the original matrix).
        
        Returns:
            The original LazyStochasticMatrix
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
