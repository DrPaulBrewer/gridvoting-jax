# CPU Configuration Auto-Detection

## Overview

The `gridvoting-jax` library now automatically detects and configures optimal CPU parallelization settings when imported. This eliminates the need for manual configuration in most cases.

## What's Configured

When you `import gridvoting_jax`, the library automatically:

1. **Detects CPU cores** using `os.cpu_count()`
2. **Sets JAX XLA flags** for optimal CPU parallelization:
   - `--xla_cpu_multi_thread_eigen=true` - Enables multi-threaded Eigen operations
   - `intra_op_parallelism_threads=<cpu_count>` - Sets parallelism to match CPU cores
3. **Sets OpenMP threads** via `OMP_NUM_THREADS` environment variable

## Implementation Details

The configuration happens in `src/gridvoting_jax/core.py` **before** JAX is imported, ensuring the settings take effect when JAX initializes.

```python
# Detect number of CPU cores
cpu_count = os.cpu_count()

# Configure JAX CPU parallelization (only if not already set by user)
if 'XLA_FLAGS' not in os.environ:
    xla_flags = (
        f'--xla_cpu_multi_thread_eigen=true '
        f'--xla_force_host_platform_device_count={cpu_count} '
        f'intra_op_parallelism_threads={cpu_count} '
        f'inter_op_parallelism_threads={cpu_count}'
    )
    os.environ['XLA_FLAGS'] = xla_flags

if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = str(cpu_count)

if 'MKL_NUM_THREADS' not in os.environ:
    os.environ['MKL_NUM_THREADS'] = str(cpu_count)
```

### XLA Flags Explained

1. **`--xla_cpu_multi_thread_eigen=true`**: Enables multi-threaded Eigen operations for linear algebra
2. **`--xla_force_host_platform_device_count=<N>`**: **Critical for parallelization!** 
   - Exposes CPU cores as separate JAX devices
   - Enables JAX to parallelize iterative solvers like GMRES and power method
   - Without this, JAX treats the entire CPU as a single device
3. **`intra_op_parallelism_threads=<N>`**: Parallelism within a single operation (e.g., matrix multiply)
4. **`inter_op_parallelism_threads=<N>`**: Parallelism across independent operations

### Threading Environment Variables

- **`OMP_NUM_THREADS`**: Controls OpenMP threading (used by many BLAS libraries)
- **`MKL_NUM_THREADS`**: Controls Intel MKL threading (if JAX is using MKL for linear algebra)

## User Override

The auto-configuration **respects user-provided values**. If you set these environment variables before importing `gridvoting_jax`, your values will be preserved:

```python
import os
# Custom configuration
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4'
os.environ['OMP_NUM_THREADS'] = '4'

# Your settings will NOT be overwritten
import gridvoting_jax as gv
```

## Benefits

1. **Automatic optimization** - No manual configuration needed for most users
2. **Better CPU performance** - Utilizes all available CPU cores by default
3. **User control** - Advanced users can still override settings
4. **Consistent behavior** - Same configuration across all scripts and tests

## Testing

Three test scripts verify the functionality:

1. **`test_cpu_config.py`** - Tests auto-detection with default settings
2. **`test_cpu_config_forced.py`** - Tests with forced CPU mode (`JAX_PLATFORMS='cpu'`)
3. **`test_cpu_config_user.py`** - Verifies user-provided values are respected

## Migration

Scripts that previously set these variables manually can now remove them:

**Before:**
```python
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=12'
os.environ['OMP_NUM_THREADS'] = '12'

import gridvoting_jax as gv
```

**After:**
```python
import os
os.environ['JAX_PLATFORMS'] = 'cpu'  # Only needed to force CPU mode

import gridvoting_jax as gv  # CPU parallelization auto-configured
```

## Example: test_g100_cpu.py

The `test_g100_cpu.py` script has been updated to remove manual CPU configuration, demonstrating the simplified usage.
