"""Core utility modules for gridvoting-jax.

This module provides:
- Configuration (JAX setup, device detection)
- Constants (tolerances, precision settings)
- Utilities (probability vector normalization, entropy)
- Lazy matrices (memory-efficient matrix operations)
"""

# Configuration must be imported first to set up JAX
from .config import (
    enable_float64,
    use_accelerator,
    device_type,
    get_available_memory_bytes
)

# Constants
from .constants import (
    TOLERANCE,
    DTYPE_FLOAT,
    BAD_STATIONARY_TOLERANCE,
    EPSILON,
    GEOMETRY_EPSILON,
    NEGATIVE_PROBABILITY_TOLERANCE,
    PLOT_LOG_BIAS
)

# Utilities
from .utils import (
    _move_neg_prob_to_max,
    _normalize_row_if_needed,
    entropy_in_bits,
    matrix_is_dense,
    normalize_if_needed
)

# Lazy matrices
from .lazy_stochastic import LazyStochasticMatrix, LazyStochasticMatrixTranspose
from .lazy_q import LazyQMatrix, LazyQMatrixTranspose

__all__ = [
    # Configuration
    'enable_float64',
    'use_accelerator',
    'device_type',
    'get_available_memory_bytes',
    # Constants
    'TOLERANCE',
    'DTYPE_FLOAT',
    'BAD_STATIONARY_TOLERANCE',
    'EPSILON',
    'GEOMETRY_EPSILON',
    'NEGATIVE_PROBABILITY_TOLERANCE',
    'PLOT_LOG_BIAS',
    # Utilities
    '_move_neg_prob_to_max',
    '_normalize_row_if_needed',
    'entropy_in_bits',
    'matrix_is_dense',
    'normalize_if_needed',
    # Lazy matrices
    'LazyStochasticMatrix',
    'LazyStochasticMatrixTranspose',
    'LazyQMatrix',
    'LazyQMatrixTranspose',
]
