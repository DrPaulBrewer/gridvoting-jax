"""Utility functions for probability vectors and matrices.

This module provides utilities for:
- Normalizing probability vectors
- Fixing negative probability components
- Computing entropy
- Checking matrix density (dense vs lazy)
"""

from typing import Union
import jax
import jax.numpy as jnp
from jax import Array

from ..core import constants


def _move_neg_prob_to_max(pvector: Array) -> Array:
    """Fix negative probability components by moving mass to maximum values.
    
    Redistributes the total mass from negative components equally among
    all indices that share the maximum value (within TOLERANCE).
    
    This function is NOT decorated with @jax.jit to avoid nested JIT issues
    when called from GMRES (which internally JIT-compiles). JAX will still
    JIT-compile this function when called from JIT-compiled contexts.
    
    Args:
        pvector: JAX array that may contain small negative values
        
    Returns:
        JAX array with negative values zeroed and mass redistributed 
        equally to all maximum-value indices
        
    Example:
        >>> pvector = jnp.array([0.5, -0.01, 0.51])
        >>> fixed = _move_neg_prob_to_max(pvector)
        >>> # Negative mass moved to maximum value(s)
    """
    # Identify negative components and calculate mass to redistribute
    # Use jnp.where to avoid boolean indexing which is incompatible with JIT
    to_zero = pvector < 0.0
    mass_destroyed = jnp.where(to_zero, pvector, 0.0).sum()
    
    # Zero out negative components
    fixed_pvector = jnp.where(to_zero, 0.0, pvector)
    
    # Find ALL indices with maximum value (within 2*constants.EPSILON)
    max_val = fixed_pvector.max()
    is_max = jnp.abs(fixed_pvector - max_val) <= 2*constants.EPSILON
    num_max_indices = is_max.sum()
    
    # Distribute mass equally among all maximum indices
    mass_per_index = mass_destroyed / num_max_indices
    fixed_pvector = jnp.where(is_max, fixed_pvector + mass_per_index, fixed_pvector)
    
    return fixed_pvector


def entropy_in_bits(v: Array) -> float:
    """Compute Shannon entropy of a probability vector in bits.
    
    Args:
        v: Probability vector (should sum to 1.0)
        
    Returns:
        Entropy in bits (using log base 2) per row
        
    Example:
        >>> v = jnp.array([0.5, 0.5])
        >>> entropy_in_bits(v)  # Entropy for 2 states = 1 bit
        1.0
    """
    safe = jnp.where(v > 0, v, 1.0)
    return -jnp.sum(safe * jnp.log2(safe), axis=-1)


def matrix_is_dense(M) -> bool:
    """Check if matrix is dense (JAX array) vs lazy (LazyStochasticMatrix, LazyQMatrix).
    
    Args:
        M: Matrix to check (can be JAX array or lazy matrix object)
        
    Returns:
        True if matrix is a dense JAX array, False if it's a lazy matrix
        
    Note:
        Lazy matrices have a `to_dense()` method, dense matrices do not.
    """
    return not hasattr(M, 'to_dense')


def _normalize_row_if_needed(v: Array) -> Array:
    """Normalize probability vector only if sum deviates beyond accumulation error.
    
    This function attempts to renormalize v to have a sum closer to 1.0.
    If renormalization doesn't improve the deviation, it returns the original vector.
    
    This function is NOT decorated with @jax.jit to avoid nested JIT issues
    when called from GMRES (which internally JIT-compiles). JAX will still
    JIT-compile this function when called from JIT-compiled contexts.
         
    Args:
        v: Probability vector (1D JAX array)
    
    Returns:
        Normalized vector (or original if sum ≈ 1.0 within threshold)
    
    Examples:
        >>> v = jnp.array([0.25, 0.25, 0.25, 0.25])
        >>> v_norm = _normalize_row_if_needed(v)  # No-op, sum already ≈ 1.0
        
        >>> v = jnp.array([0.5, 0.5, 0.5, 0.5])  # sum = 2.0
        >>> v_norm = _normalize_row_if_needed(v)  # Normalizes to sum = 1.0

    Notes:
        - JIT-compatible: uses jnp.where instead of Python conditionals
        - Adaptive threshold based on vector size to handle accumulation error
    """
    # To avoid nested jit, the big and little sums are calculated explicitly here 
    # and again below, instead of in a helper function
    big_sum = jnp.sum(jnp.where(v >= 2*constants.EPSILON, v, 0.0))
    little_sum = jnp.sum(jnp.where(v < 2*constants.EPSILON, v, 0.0))
    s = big_sum + little_sum
    sinv = 1.0/s
    deviation = jnp.abs(s - 1.0)
    n = v.shape[0]
    threshold = constants.EPSILON * jnp.where(n > 1280, (n//128), 10)
    
    v_renorm = jnp.where(
        deviation > threshold,
        v * sinv,
        v
    )
    
    renorm_big_sum = jnp.sum(jnp.where(v_renorm >= 2*constants.EPSILON, v_renorm, 0.0))
    renorm_little_sum = jnp.sum(jnp.where(v_renorm < 2*constants.EPSILON, v_renorm, 0.0))
    renorm_s = renorm_big_sum + renorm_little_sum
    renorm_deviation = jnp.abs(renorm_s - 1.0)
    
    v_final = jnp.where(
        renorm_deviation < deviation,
        v_renorm,
        v
    )
    return v_final


def normalize_if_needed(v: Array) -> Array:
    """Normalize probability vector(s) only if sum deviates beyond accumulation error.
    
    Handles both 1D vectors and 2D batches of vectors.
    
    Args:
        v: Probability vector (1D) or batch of vectors (2D)
        
    Returns:
        Normalized vector(s) with sum closer to 1.0
        
    Examples:
        >>> v = jnp.array([0.3, 0.3, 0.4])
        >>> normalize_if_needed(v)  # 1D vector
        
        >>> V = jnp.array([[0.3, 0.3, 0.4], [0.5, 0.5, 0.0]])
        >>> normalize_if_needed(V)  # 2D batch
    """
    if jnp.ndim(v) == 1:
        return _normalize_row_if_needed(v)
    else:
        return jax.vmap(_normalize_row_if_needed)(v)
