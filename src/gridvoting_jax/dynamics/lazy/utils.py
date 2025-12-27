"""Utilities for lazy matrix construction.

This module contains general-purpose utilities for deciding when to use lazy
matrix construction based on memory constraints.
"""

import jax.numpy as jnp


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
