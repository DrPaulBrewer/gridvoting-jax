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
    warn("GV_ENABLE_FLOAT64=1: JAX float64 enabled, default solver TOLERANCE set to 1e-8")
else:
    TOLERANCE = 1e-5
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
    """Check if matrix is dense (JAX array) vs lazy (LazyStochasticMatrix, LazyQMatrix).
    
    If a matrix has a to_dense method, it is not dense.
    """
    return not hasattr(M, 'to_dense')

def _normalize_row_if_needed(v):
    """Normalize probability vector(s) only if sum deviates beyond accumulation error.
    
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
    sinv = 1.0/s
    deviation = jnp.abs(s - 1.0)
    n = v.shape[0]
    threshold = EPSILON*jnp.where(n>1280, (n//128), 10)
    v_renorm =  jnp.where(
        deviation > threshold,
        v * sinv,
        v
    )
    renorm_big_sum = jnp.sum(jnp.where(v_renorm>=2*EPSILON, v_renorm, 0.0))
    renorm_little_sum = jnp.sum(jnp.where(v_renorm<2*EPSILON, v_renorm, 0.0))
    renorm_s = renorm_big_sum + renorm_little_sum
    renorm_deviation = jnp.abs(renorm_s - 1.0)
    v_final = jnp.where(
        renorm_deviation < deviation,
        v_renorm,
        v
    )
    return v_final

def normalize_if_needed(v):
    if jnp.ndim(v) == 1:
        return _normalize_row_if_needed(v)
    else:
        return jax.vmap(_normalize_row_if_needed)(v)

class LazyStochasticMatrix:
    def __init__(self, mask, status_quo_values, challenger_values):
        """
        Initializes a LazyStochasticMatrix with a mask, status quo values, and challenger values.

        The code represents a matrix M with the following properties:
        - M is square
        - the diagonal emtries M[i,i] are given by status_quo_values[i]
        - the off-diagonal entries M[i,j] (i!=j) are given by mask[i,j]*challenger_values[i]

        Args:
            mask: A 2D square boolean array
            status_quo_values: A 1D array of array diagonal values
            challenger_values: A 1D array of array off-diagonal values, constant for each row, if mask is True
        """
        self.mask = mask
        self.status_quo_values = jnp.asarray(status_quo_values, dtype=DTYPE_FLOAT)
        self.challenger_values = jnp.asarray(challenger_values, dtype=DTYPE_FLOAT)
        self.ndim = 2
        self.shape = mask.shape
        self.dtype = DTYPE_FLOAT

    def __matmul__(self, other):
        """Right multiplication: self @ other (M * v or M * V)"""
        other = jnp.asarray(other, dtype=self.dtype)
        if other.ndim == 1:
            return self.status_quo_values * other + self.challenger_values * (self.mask @ other)
        else:
            # batch of column vectors (n, k)
            return self.status_quo_values[:, None] * other + self.challenger_values[:, None] * (self.mask @ other)

    def __rmatmul__(self, other):
        """Left multiplication: other @ self (v * M or V * M)"""
        other = jnp.asarray(other, dtype=self.dtype)
        # broadcasting works correctly for (n,) * (n,) or (k, n) * (n,)
        weighted_other = other * self.challenger_values
        return other * self.status_quo_values + (weighted_other @ self.mask)

    def __getitem__(self, key):
        """Element access and basic slicing. Does NOT support advanced indexing."""
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

        raise NotImplementedError("Advanced indexing/slicing not supported for LazyStochasticMatrix")

    @property
    def T(self):
        """Returns the transpose of the matrix as a LazyStochasticMatrixTranspose wrapper."""
        return LazyStochasticMatrixTranspose(self)

    def diagonal(self):
        """Returns the diagonal elements S_i."""
        return self.status_quo_values
    
    def to_dense(self):
        """Materializes the full matrix."""
        # diag(S) + diag(C) * mask
        return jnp.diag(self.status_quo_values) + self.challenger_values[:, None] * self.mask

class LazyStochasticMatrixTranspose:
    """A wrapper for the transpose of a LazyStochasticMatrix."""
    def __init__(self, original):
        self.original = original
        self.shape = (original.shape[1], original.shape[0])
        self.ndim = original.ndim
        self.dtype = original.dtype

    def __matmul__(self, other):
        # (M^T) @ v = (v^T @ M)^T
        return self.original.__rmatmul__(other.T).T if other.ndim > 1 else self.original.__rmatmul__(other)

    def __rmatmul__(self, other):
        # v @ (M^T) = (M @ v^T)^T
        return self.original.__matmul__(other.T).T if other.ndim > 1 else self.original.__matmul__(other)

    @property
    def T(self):
        return self.original

    def diagonal(self):
        return self.original.diagonal()

    def to_dense(self):
        return self.original.to_dense().T

class LazyQMatrix:
    def __init__(self, P: LazyStochasticMatrix):
        """
        LazyQMatrix represents a matrix Q with the following properties:
          - Q is square, with Q.shape=P.shape = (n,n)
          - the entire first row Q[0,:] is 1.0
          - the diagonal from 1 to (n-1) is P.diagonal-1.0
          - the non-diagonal elements (except for row 0) are given by P.T
        """
        self.P = P
        self.shape = P.shape
        self.ndim = P.ndim
        self.dtype = P.dtype

    def __matmul__(self, other):
        """Right multiplication: self @ other (Q * v or Q * V)"""
        other = jnp.asarray(other, dtype=self.dtype)
        # Q[1:, :] is (P.T - I)[1:, :]
        # (P.T - I) @ other = (other.T @ (P - I)).T = (other.T @ P).T - other
        res = (other.T @ self.P).T - other
        # Replace row 0 with sum(other)
        if other.ndim == 1:
            return res.at[0].set(jnp.sum(other))
        else:
            return res.at[0, :].set(jnp.sum(other, axis=0))

    def __rmatmul__(self, other):
        """Left multiplication: other @ self (v * Q or V * Q)"""
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

    def __getitem__(self, key):
        """Element access and basic slicing. Does NOT support advanced indexing."""
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            if isinstance(i, int) and isinstance(j, int):
                if i == 0:
                    return jnp.array(1.0, dtype=self.dtype)
                if i == j:
                    return self.P.status_quo_values[i] - 1.0
                return self.P[j, i] # Q[i, j] = P^T[i, j] = P[j, i]
            
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

    def diagonal(self):
        """Returns the diagonal elements of Q."""
        diagP = self.P.diagonal()
        return diagP.at[0].set(1.0).at[1:].add(-1.0)

    def to_dense(self):
        """Materializes the full Q matrix."""
        n = self.shape[0]
        # P.T - I
        Q = self.P.to_dense().T - jnp.eye(n, dtype=self.dtype)
        # overwrite first row
        return Q.at[0, :].set(1.0)

    @property
    def T(self):
        """Returns the transpose of the matrix as a LazyQMatrixTranspose wrapper."""
        return LazyQMatrixTranspose(self)

class LazyQMatrixTranspose:
    """A wrapper for the transpose of a LazyQMatrix."""
    def __init__(self, original):
        self.original = original
        self.shape = original.shape
        self.ndim = original.ndim
        self.dtype = original.dtype

    def __matmul__(self, other):
        return self.original.__rmatmul__(other.T).T if other.ndim > 1 else self.original.__rmatmul__(other)

    def __rmatmul__(self, other):
        return self.original.__matmul__(other.T).T if other.ndim > 1 else self.original.__matmul__(other)

    @property
    def T(self):
        return self.original

    def diagonal(self):
        return self.original.diagonal()

    def to_dense(self):
        return self.original.to_dense().T
        

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

