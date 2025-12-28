import os
from warnings import warn

# ============================================================================
# CPU Configuration - Must be set BEFORE importing JAX
# ============================================================================

# Detect number of CPU cores for optimal parallelization
cpu_count = os.cpu_count()
if cpu_count is None:
    cpu_count = 1  # Fallback if detection fails
    warn("Could not detect CPU count, defaulting to 1 thread")

# Configure JAX CPU parallelization (only if not already set by user)
if 'XLA_FLAGS' not in os.environ:
    # Enable multi-threaded Eigen operations and set parallelism threads
    # intra_op: parallelism within a single operation (e.g., matrix multiply)
    # inter_op: parallelism across independent operations
    # xla_force_host_platform_device_count: exposes CPU cores as separate devices
    #   This is critical for parallelizing iterative solvers like GMRES and power method
    xla_flags = (
        f'--xla_cpu_multi_thread_eigen=true '
        f'--xla_force_host_platform_device_count={cpu_count} '
        f'intra_op_parallelism_threads={cpu_count} '
        f'inter_op_parallelism_threads={cpu_count}'
    )
    os.environ['XLA_FLAGS'] = xla_flags

if 'OMP_NUM_THREADS' not in os.environ:
    # Set OpenMP threads for CPU operations
    os.environ['OMP_NUM_THREADS'] = str(cpu_count)

if 'MKL_NUM_THREADS' not in os.environ:
    # Set Intel MKL threads (if MKL is being used by JAX)
    os.environ['MKL_NUM_THREADS'] = str(cpu_count)

# ============================================================================
# JAX Import - Now with optimized CPU settings
# ============================================================================

import jax
import jax.numpy as jnp
import chex


# ============================================================================
# Default tolerances
# ============================================================================

# Check for Float64 override via environment
# This allows JAX to start in float64 mode and sets tighter tolerances
if os.environ.get("GV_ENABLE_FLOAT64") == "1" or os.environ.get("JAX_ENABLE_X64") in ["1", "True", "true"]:
    jax.config.update("jax_enable_x64", True)
    TOLERANCE = 1e-10
    DTYPE_FLOAT = jnp.float64
    warn("GV_ENABLE_FLOAT64=1: JAX float64 enabled, TOLERANCE set to 1e-10")
else:
    TOLERANCE = 5e-5
    DTYPE_FLOAT = jnp.float32

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
    global TOLERANCE
    jax.config.update("jax_enable_x64", True)
    TOLERANCE = 1e-10
    # Note: If TOLERANCE was imported by other modules using 'from ...', 
    # they will hold the old value. Use 'import core; core.TOLERANCE' or set env var.
    warn("enable_float64 called: JAX float64 enabled, TOLERANCE set to 1e-10")

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
                # Set GPU allocator to reduce fragmentation issues
                if device_type == 'gpu' and 'TF_GPU_ALLOCATOR' not in os.environ:
                    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
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
def assert_valid_transition_matrix(P):
    """asserts that JAX array is square and that each row sums to 1.0
    with default tolerance of 2 * P.rows * jnp.finfo.eps  (float64)"""
    P = jnp.asarray(P)
    rows, cols = P.shape
    chex.assert_shape(P, (rows, rows))  # Ensure square matrix
    row_sums = P.sum(axis=1)
    expected = jnp.ones(rows)
    # epsilon jnp.finfo(dtype).eps is the smallest number that  FP can add to 1.0
    # this allows every row to be 2 epsilon off before assert fails
    tolerance = 2 * rows * jnp.finfo(P.dtype).eps
    chex.assert_trees_all_close(row_sums, expected, atol=tolerance, rtol=0)

@chex.chexify
@jax.jit
def assert_zero_diagonal_matrix(M):
    """asserts that JAX array is square with exactly zero diagonal"""
    M = jnp.asarray(M)
    rows, cols = M.shape
    chex.assert_shape(M, (rows, rows))  # Ensure square matrix
    diagonal = jnp.diag(M)
    expected = jnp.zeros(rows, dtype=M.dtype)
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

def get_available_memory_bytes():
    """ Estimate available memory in bytes on the active device.
    
    Returns:
        int or None: Available memory in bytes, or None if undetermined.
    """
    global use_accelerator
    
    # 1. GPU/TPU Memory via JAX
    if use_accelerator:
        try:
            # Stats for the default device
            stats = jax.devices()[0].memory_stats()
            if 'bytes_limit' in stats and 'bytes_in_use' in stats:
                return stats['bytes_limit'] - stats['bytes_in_use']
        except Exception:
            pass # Fallback to system memory if device stats fail

    # 2. System Memory (CPU)
    
    # Try psutil (most robust cross-platform)
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass

    # Try /proc/meminfo (Linux)
    try:
        with open('/proc/meminfo', 'r') as f:
            mem_info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(':')
                    value = int(parts[1]) * 1024 # kB to bytes
                    mem_info[key] = value
            
            # Available is ideal, falling back to free + buffers + cached
            if 'MemAvailable' in mem_info:
                return mem_info['MemAvailable']
            elif 'MemFree' in mem_info:
                return mem_info['MemFree'] + mem_info.get('Buffers', 0) + mem_info.get('Cached', 0)
    except Exception:
        pass

    # Note: macOS 'vm_stat' parsing is complex without external tools, 
    # skipping here to avoid fragility. psutil is recommended for Mac.
    
    return None
"""Core voting logic modules.

This package contains shared voting-specific logic used across the codebase:
- zimi_succession_logic: ZI/MI succession rules
- winner_determination: Pairwise winner computation
"""

from .zimi_succession_logic import (
    finalize_transition_matrix,
    finalize_transition_matrix_zi_jit,
    finalize_transition_matrix_mi_jit,
)

from .winner_determination import (
    compute_winner_matrix_jit,
)

__all__ = [
    'finalize_transition_matrix',
    'finalize_transition_matrix_zi_jit',
    'finalize_transition_matrix_mi_jit',
    'compute_winner_matrix_jit',
]
