"""JAX and system configuration for gridvoting-jax.

This module handles:
- CPU/GPU/TPU device detection and configuration
- JAX threading and parallelization settings
- Float64 precision mode
- Memory availability detection

This module must be imported first to properly configure JAX before any
JAX operations are performed.
"""

import os
from typing import Optional

# ============================================================================
# CPU Configuration - Must be set BEFORE importing JAX
# ============================================================================

# Detect number of CPU cores for optimal parallelization
cpu_count = os.cpu_count()
if cpu_count is None:
    cpu_count = 1  # Fallback if detection fails

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


def enable_float64() -> None:
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
    # Import here to avoid circular dependency
    from . import constants
    
    jax.config.update("jax_enable_x64", True)
    constants.TOLERANCE = 1e-10
    constants.DTYPE_FLOAT = jnp.float64
    constants.EPSILON = float(jnp.finfo(jnp.float64).eps)


# ============================================================================
# Device Detection
# ============================================================================

use_accelerator: bool = False
device_type: str = 'cpu'

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
    except RuntimeError:
         # Fallback if JAX cannot find backend or other init error
         pass

def get_available_memory_bytes() -> Optional[int]:
    """Estimate available memory in bytes on the active device.
    
    Returns:
        Available memory in bytes, or None if undetermined.
        
    Notes:
        - For GPU/TPU: Uses JAX device memory stats
        - For CPU: Tries psutil, then /proc/meminfo (Linux)
        - Returns None if memory cannot be determined
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
            pass  # Fallback to system memory if device stats fail

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
                    value = int(parts[1]) * 1024  # kB to bytes
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
