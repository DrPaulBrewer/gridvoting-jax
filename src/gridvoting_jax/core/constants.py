"""Global constants and tolerances for gridvoting-jax.

This module defines numerical tolerances and constants used throughout
the package for solver convergence, geometric tests, and plotting.
"""

import os
import jax.numpy as jnp

# ============================================================================
# Precision and Tolerance Configuration
# ============================================================================

# Check for Float64 override via environment
# This allows JAX to start in float64 mode and sets tighter tolerances
if os.environ.get("GV_ENABLE_FLOAT64") == "1" or os.environ.get("JAX_ENABLE_X64") in ["1", "True", "true"]:
    import jax
    jax.config.update("jax_enable_x64", True)
    TOLERANCE: float = 1e-10
    DTYPE_FLOAT = jnp.float64
else:
    TOLERANCE: float = 1e-5
    DTYPE_FLOAT = jnp.float32

# ============================================================================
# Solver Tolerances
# ============================================================================

# Bad Stationary Tolerance above which we throw a RuntimeError
BAD_STATIONARY_TOLERANCE: float = 1e-3

# Tolerance for negative probabilities in Markov Chain
# Previously hardcoded as -1e-5 in solve_for_unit_eigenvector
NEGATIVE_PROBABILITY_TOLERANCE: float = -1e-5

# ============================================================================
# Numerical Precision Constants
# ============================================================================

# Floating point epsilon for the current dtype
EPSILON: float = float(jnp.finfo(DTYPE_FLOAT).eps)

# Epsilon for geometric tests (e.g. point in triangle) to handle numerical noise
# Previously hardcoded as 1e-10 in _is_in_triangle_single, Grid.extremes
GEOMETRY_EPSILON: float = 1e-10

# ============================================================================
# Plotting Constants
# ============================================================================

# Log bias for plotting log-scale distributions to avoid log(0)
# Previously hardcoded as 1e-100 in Grid.plot
PLOT_LOG_BIAS: float = 1e-100
