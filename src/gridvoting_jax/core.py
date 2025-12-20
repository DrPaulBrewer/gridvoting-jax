
import os
import jax
import jax.numpy as jnp
import chex
from warnings import warn

# Default tolerances
# 5e-5 for float32 (default), 1e-10 for float64 (after calling enable_float64())
TOLERANCE = 5e-5

# Epsilon for geometric tests (e.g. point in triangle) to handle numerical noise
# Previously hardcoded as 1e-10 in _is_in_triangle_single, Grid.extremes
GEOMETRY_EPSILON = 1e-10

# Tolerance for negative probabilities in Markov Chain
# Previously hardcoded as -1e-5 in solve_for_unit_eigenvector
NEGATIVE_PROBABILITY_TOLERANCE = -1e-5

# Log bias for plotting log-scale distributions to avoid log(0)
# Previously hardcoded as 1e-100 in Grid.plot
PLOT_LOG_BIAS = 1e-100

def enable_float64():
    """Enable 64-bit floating point precision in JAX.
    
    By default, JAX uses 32-bit floats for better GPU performance.
    Call this function to enable 64-bit precision for higher accuracy.
    
    This is a global configuration that affects all subsequent JAX operations.
    See: https://docs.jax.dev/en/latest/default_dtypes.html
    
    Example:
        >>> import gridvoting_jax as gv
        >>> gv.enable_float64()
        >>> # All subsequent JAX operations will use float64
    """
    jax.config.update("jax_enable_x64", True)

# Device detection with GV_FORCE_CPU override
use_accelerator = False
device_type = 'cpu'

# We perform device detection at module load time
if os.environ.get('GV_FORCE_CPU', '0') != '1':
    # Check for available accelerators (TPU > GPU > CPU)
    try:
        devices = jax.devices()
        if devices:
            default_device = devices[0]
            device_type = default_device.platform
            if device_type in ['gpu', 'tpu']:
                use_accelerator = True
                warn(f"JAX using {device_type.upper()}: {default_device}")
            else:
                warn("JAX using CPU (no GPU/TPU detected)")
    except RuntimeError:
         # Fallback if JAX cannot find backend or other init error
         warn("JAX initialization failed to detect devices, falling back to CPU")
else:
    warn("GV_FORCE_CPU=1: JAX forced to CPU-only mode")


@chex.chexify
@jax.jit
def assert_valid_transition_matrix(P, *, decimal=6):
    """asserts that JAX array is square and that each row sums to 1.0
    with default tolerance of 6 decimal places (float32) or 10 decimal places (float64)"""
    P = jnp.asarray(P)
    rows, cols = P.shape
    chex.assert_shape(P, (rows, rows))  # Ensure square matrix
    row_sums = P.sum(axis=1)
    expected = jnp.ones(rows)
    # Using 1.1 multiplier as seen in original code for slightly loose tolerance
    tolerance = 10 ** (-decimal) * 1.1  
    chex.assert_trees_all_close(row_sums, expected, atol=tolerance, rtol=0)

@chex.chexify
@jax.jit
def assert_zero_diagonal_int_matrix(M):
    """asserts that JAX array is square and the diagonal is 0.0"""
    M = jnp.asarray(M)
    rows, cols = M.shape
    chex.assert_shape(M, (rows, rows))  # Ensure square matrix
    diagonal = jnp.diag(M)
    expected = jnp.zeros(rows, dtype=int)    
    chex.assert_trees_all_equal(diagonal, expected)

@jax.jit
def _move_neg_prob_to_max(pvector):
    """Fix negative probability components by moving mass to maximum values.
    
    Redistributes the total mass from negative components equally among
    all indices that share the maximum value (within TOLERANCE).
    
    Args:
        pvector: JAX array that may contain small negative values
        
    Returns:
        fixed_pvector: JAX array with negative values zeroed and mass 
                      redistributed equally to all maximum-value indices
    """
    # Identify negative components and calculate mass to redistribute
    # Use jnp.where to avoid boolean indexing which is incompatible with JIT
    to_zero = pvector < 0.0
    mass_destroyed = jnp.where(to_zero, pvector, 0.0).sum()
    
    # Zero out negative components
    fixed_pvector = jnp.where(to_zero, 0.0, pvector)
    
    # Find ALL indices with maximum value (within TOLERANCE)
    max_val = fixed_pvector.max()
    is_max = jnp.abs(fixed_pvector - max_val) < TOLERANCE
    num_max_indices = is_max.sum()
    
    # Distribute mass equally among all maximum indices
    mass_per_index = mass_destroyed / num_max_indices
    fixed_pvector = jnp.where(is_max, fixed_pvector + mass_per_index, fixed_pvector)
    
    return fixed_pvector
