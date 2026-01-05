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
    TOLERANCE = 1e-8
    DTYPE_FLOAT = jnp.float64
    warn("GV_ENABLE_FLOAT64=1: JAX float64 enabled, default solver TOLERANCE set to 1e-8")
else:
    TOLERANCE = 1e-4
    DTYPE_FLOAT = jnp.float32
    warn("GV_ENABLE_FLOAT64=0: JAX float32 enabled, default solver TOLERANCE set to 1e-4")

# Floating point epsilon
EPSILON = jnp.finfo(DTYPE_FLOAT).eps

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


def _move_neg_prob_to_max(pvector):
    """Fix negative probability components by moving mass to maximum values.
    
    Redistributes the total mass from negative components equally among
    all indices that share the maximum value (within TOLERANCE).
    
    This function is NOT decorated with @jax.jit to avoid nested JIT issues
    when called from GMRES (which internally JIT-compiles). JAX will still
    JIT-compile this function when called from JIT-compiled contexts.
    
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
    
    # Find ALL indices with maximum value (within 2*EPSILON)
    max_val = fixed_pvector.max()
    is_max = jnp.abs(fixed_pvector - max_val) <= 2*EPSILON
    num_max_indices = is_max.sum()
    
    # Distribute mass equally among all maximum indices
    mass_per_index = mass_destroyed / num_max_indices
    fixed_pvector = jnp.where(is_max, fixed_pvector + mass_per_index, fixed_pvector)
    
    return fixed_pvector

def entropy_in_bits(v):
    safe = jnp.where(v>0, v, 1.0)
    return -jnp.sum(safe * jnp.log2(safe))

def matrix_is_dense(M):
    """Check if matrix is dense (JAX array) vs lazy (LazyLeftGVMatrix/LazyRightGVMatrix).
    
    JAX arrays don't have get_row/get_col methods, while lazy matrices do.
    We can't use type(M) == jnp.ndarray because JAX arrays are ArrayImpl instances.

    LazyLeftGVMatrix and LazyRightGVMatrix have a to_dense method, 
    and if a matrix has a to_dense method, it is not dense.
    """
    return not hasattr(M, 'to_dense')

def normalize_if_needed(v):
    """Normalize probability vector only if sum deviates beyond accumulation error.
    
    This function attempts to renormalize to v to have a sum closer to 1.0.
    If it fails to do so, it returns the original vector.
    
    This function is NOT decorated with @jax.jit to avoid nested JIT issues
    when called from GMRES (which internally JIT-compiles). JAX will still
    JIT-compile this function when called from JIT-compiled contexts.
         
    Args:
        v: Probability vector (1D JAX array)
    
    Returns:
        Normalized vector (or original if sum ≈ 1.0 within threshold)
    
    Examples:
        >>> v = jnp.array([0.25, 0.25, 0.25, 0.25])
        >>> v_norm = normalize_if_needed(v)  # No-op, sum already ≈ 1.0
        
        >>> v = jnp.array([0.5, 0.5, 0.5, 0.5])  # sum = 2.0
        >>> v_norm = normalize_if_needed(v)  # Normalizes to sum = 1.0

    Notes:
        - JIT-compatible: uses jnp.where instead of Python conditionals
    """
    # to avoid nested jit, the big and little sums are calculated explicitly here and again below, instead of in a helper function
    big_sum = jnp.sum(jnp.where(v>=2*EPSILON, v, 0.0))
    little_sum = jnp.sum(jnp.where(v<2*EPSILON, v, 0.0))
    s = big_sum + little_sum
    # sum is in s
    sinv = 1.0/s
    deviation = jnp.abs(s - 1.0)
    n = v.shape[0]
    threshold = EPSILON*jnp.where(n>1280, (n//128), 10)
    v_renorm =  jnp.where(
        deviation > threshold,
        v * sinv,
        v
    )
    # again we need big and little sums to get around machine epsilon
    renorm_big_sum = jnp.sum(jnp.where(v_renorm>=2*EPSILON, v_renorm, 0.0))
    renorm_little_sum = jnp.sum(jnp.where(v_renorm<2*EPSILON, v_renorm, 0.0))
    renorm_s = renorm_big_sum + renorm_little_sum
    # sum is in renorm_s
    renorm_deviation = jnp.abs(renorm_s - 1.0)
    return jnp.where(
        renorm_deviation < deviation,
        v_renorm,
        v
    )

class LazyLeftGVMatrix():
    def __init__(self, *, n, get_row):
        self.n = n
        self.shape = (n,n)
        self.get_row = get_row
        self.dtype = DTYPE_FLOAT
        self._diagonal_cache = None

    def __rmatmul__(self, v):
        """Compute v @ M using get_row"""
        def scan_rows(carry, i):
            return (jnp.add(carry, v[...,i]*self.get_row(i)), None)
        
        result = jnp.zeros(v.shape, dtype=DTYPE_FLOAT)
        return jax.lax.scan(scan_rows, result, xs=None, length=self.n)[0]
    
    def __matmul__(self, v):
        """Not supported for LazyLeftGVMatrix (use M.T @ v instead)"""
        return NotImplemented

    def diagonal(self):
        """Compute diagonal of matrix using get_row (memoized)"""
        if self._diagonal_cache is None:
            self._diagonal_cache = jax.lax.map(lambda i: self.get_row(i)[i], jnp.arange(self.n))
        return self._diagonal_cache

    def to_dense(self):
        """Convert to dense matrix by stacking all rows"""
        return jax.lax.map(self.get_row, jnp.arange(self.n))

    @property
    def T(self):
        """Return transpose as LazyRightGVMatrix"""
        return LazyRightGVMatrix(n=self.n, get_col=self.get_row)

class LazyRightGVMatrix():
    def __init__(self, *, n, get_col):
        self.n = n
        self.shape = (n,n)
        self.get_col = get_col
        self.dtype=DTYPE_FLOAT
        self._diagonal_cache = None
    
    def __matmul__(self, v):
        """Compute M @ v using get_col"""
        def scan_cols(carry, i):
            return (jnp.add(carry, self.get_col(i)*v[i,...]), None)
        
        result = jnp.zeros(v.shape, dtype=DTYPE_FLOAT)
        return jax.lax.scan(scan_cols, result, xs=None, length=self.n)[0]

    
    def __rmatmul__(self, v):
        """Not supported for LazyRightGVMatrix (use v @ M.T instead)"""
        return NotImplemented

    def diagonal(self):
        """Compute diagonal of matrix using get_col (memoized)"""
        if self._diagonal_cache is None:
            self._diagonal_cache = jax.lax.map(lambda i: self.get_col(i)[i], jnp.arange(self.n))
        return self._diagonal_cache

    def to_dense(self):
        """Convert to dense matrix by stacking all columns as rows, then transpose"""
        # Get all columns and stack them as rows
        cols_as_rows = jax.lax.map(self.get_col, jnp.arange(self.n))
        # Transpose to get columns in correct orientation
        return cols_as_rows.T
    
    @property
    def T(self):
        """Return transpose as LazyLeftGVMatrix"""
        return LazyLeftGVMatrix(n=self.shape[0], get_row=self.get_col)


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
"""Core utility modules."""

__all__ = [
    'TOLERANCE',
    'DTYPE_FLOAT',
    'enable_float64',
    'device_type',
    'use_accelerator',
    'assert_valid_transition_matrix',
    'assert_zero_diagonal_matrix',
    'normalize_if_needed'
]

