"""Lazy Q matrix implementation for Markov chain analysis.

This module provides memory-efficient lazy evaluation of Q matrices used
in solving for stationary distributions of Markov chains.

The LazyQMatrix represents a matrix Q derived from a transition matrix P where:
- Q[0,:] = 1.0 (first row is all ones)
- Q[i,i] = P[i,i] - 1.0 for i > 0 (diagonal shifted by -1)
- Q[i,j] = P^T[i,j] for i > 0, j ≠ i (transpose of P for non-diagonal)
"""

from typing import Union, Tuple
import jax.numpy as jnp
from jax import Array
from .utils import matrix_is_dense
from .lazy_stochastic import LazyStochasticMatrix


class LazyQMatrix:
    """Lazy evaluation of Q matrices for Markov chain stationary distribution solving.
    
    The Q matrix is constructed from a transition matrix P and is used in the
    linear system Q @ π = b to find the stationary distribution π.
    
    Attributes:
        P: The underlying LazyStochasticMatrix transition matrix
        shape: Matrix dimensions (n, n)
        ndim: Number of dimensions (always 2)
        dtype: Data type of matrix elements
    """
    
    def __init__(self, P: LazyStochasticMatrix) -> None:
        """Initialize a LazyQMatrix from a transition matrix.
        
        Args:
            P: LazyStochasticMatrix representing the Markov chain transition matrix
        """
        self.P = P
        self.shape = P.shape
        self.ndim = P.ndim
        self.dtype = P.dtype

    def __matmul__(self, other: Array) -> Array:
        """Right multiplication: self @ other (Q * v or Q * V).
        
        Args:
            other: Vector (1D) or matrix (2D) to multiply
            
        Returns:
            Result of Q @ other
        """
        other = jnp.asarray(other, dtype=self.dtype)
        # Q[1:, :] is (P.T - I)[1:, :]
        # (P.T - I) @ other = (other.T @ (P - I)).T = (other.T @ P).T - other
        res = (other.T @ self.P).T - other
        # Replace row 0 with sum(other)
        if other.ndim == 1:
            return res.at[0].set(jnp.sum(other))
        else:
            return res.at[0, :].set(jnp.sum(other, axis=0))

    def __rmatmul__(self, other: Array) -> Array:
        """Left multiplication: other @ self (v * Q or V * Q).
        
        Args:
            other: Vector (1D) or matrix (2D) to multiply
            
        Returns:
            Result of other @ Q
        """
        other = jnp.asarray(other, dtype=self.dtype)
        # v @ Q = v[0] * row_0(Q) + sum_{i>0} v[i] * row_i(Q)
        # Row 0 of Q is all ones. Row i > 0 is Row i of (P^T - I).
        # v @ Q = v[0] * ones + v_prime @ (P^T - I)  where v_prime = [0, v1, v2, ...]
        # v_prime @ P^T = (P @ v_prime.T).T
        v_prime = other.at[..., 0].set(0.0)
        if other.ndim == 1:
            res = (self.P @ v_prime) - v_prime
            return res + other[0]
        else:
            res = (self.P @ v_prime.T).T - v_prime
            return res + other[..., 0:1]

    def __getitem__(
        self, 
        key: Union[int, Tuple[int, int], Tuple[int, slice], Tuple[slice, int]]
    ) -> Array:
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
                if i == 0:
                    return jnp.array(1.0, dtype=self.dtype)
                if i == j:
                    return self.P.status_quo_values[i] - 1.0
                return self.P[j, i]  # Q[i, j] = P^T[i, j] = P[j, i]
            
            # Row access Q[i, :]
            if isinstance(i, int) and isinstance(j, slice) and j == slice(None):
                if i == 0:
                    return jnp.ones(self.shape[0], dtype=self.dtype)
                # Row i of P^T - I is Column i of P - I
                return self.P[:, i].at[i].add(-1.0)
            
            # Column access Q[:, j]
            if isinstance(i, slice) and i == slice(None) and isinstance(j, int):
                # Column j of Q is Row j of P, with Q[0,j]=1 and diagonal shift
                return self.P[j, :].at[j].add(-1.0).at[0].set(1.0)

        raise NotImplementedError("Advanced indexing/slicing not supported for LazyQMatrix")

    def diagonal(self) -> Array:
        """Return the diagonal elements of Q.
        
        Returns:
            1D array where diag[0]=1.0 and diag[i]=P[i,i]-1.0 for i>0
        """
        diagP = self.P.diagonal()
        return diagP.at[0].set(1.0).at[1:].add(-1.0)

    def to_dense(self) -> Array:
        """Materialize the full Q matrix.
        
        Returns:
            Dense JAX array representation of Q
            
        Warning:
            This creates an n×n dense matrix in memory. Use only when necessary.
        """
        n = self.shape[0]
        # P.T - I
        if matrix_is_dense(self.P):
            Q = self.P.T - jnp.eye(n, dtype=self.dtype)
        else:
            Q = self.P.to_dense().T - jnp.eye(n, dtype=self.dtype)
        # overwrite first row
        return Q.at[0, :].set(1.0)

    @property
    def T(self) -> 'LazyQMatrixTranspose':
        """Return the transpose of the matrix.
        
        Returns:
            LazyQMatrixTranspose wrapper providing transposed view
        """
        return LazyQMatrixTranspose(self)


class LazyQMatrixTranspose:
    """Transpose wrapper for LazyQMatrix.
    
    Provides a transposed view without materializing the full matrix.
    
    Attributes:
        original: The original LazyQMatrix
        shape: Matrix dimensions (same as original for square matrices)
        ndim: Number of dimensions (always 2)
        dtype: Data type of matrix elements
    """
    
    def __init__(self, original: LazyQMatrix) -> None:
        """Initialize transpose wrapper.
        
        Args:
            original: The LazyQMatrix to transpose
        """
        self.original = original
        self.shape = original.shape
        self.ndim = original.ndim
        self.dtype = original.dtype

    def __matmul__(self, other: Array) -> Array:
        """Right multiplication: (Q^T) @ other.
        
        Args:
            other: Vector or matrix to multiply
            
        Returns:
            Result of transposed matrix multiplication
        """
        return self.original.__rmatmul__(other.T).T if other.ndim > 1 else self.original.__rmatmul__(other)

    def __rmatmul__(self, other: Array) -> Array:
        """Left multiplication: other @ (Q^T).
        
        Args:
            other: Vector or matrix to multiply
            
        Returns:
            Result of transposed matrix multiplication
        """
        return self.original.__matmul__(other.T).T if other.ndim > 1 else self.original.__matmul__(other)

    @property
    def T(self) -> LazyQMatrix:
        """Return the transpose of the transpose (i.e., the original matrix).
        
        Returns:
            The original LazyQMatrix
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
            Dense JAX array representation of Q^T
        """
        return self.original.to_dense().T
