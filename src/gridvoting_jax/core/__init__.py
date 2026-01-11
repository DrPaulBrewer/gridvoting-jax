"""Core utility modules for gridvoting-jax.

This module provides:
- Configuration (JAX setup, device detection)
- Constants (tolerances, precision settings)
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
]
