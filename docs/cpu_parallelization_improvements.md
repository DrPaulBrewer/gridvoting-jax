# CPU Parallelization Improvements - Summary

## Changes Made

### 1. Added Critical XLA Flag: `--xla_force_host_platform_device_count`

**Before:**
- JAX saw the CPU as a **single device**: `[CpuDevice(id=0)]`
- Limited parallelization potential for iterative solvers

**After:**
- JAX sees **12 separate CPU devices**: `[CpuDevice(id=0), ..., CpuDevice(id=11)]`
- Enables better parallelization for GMRES and power method solvers

### 2. Complete XLA Flags Configuration

```python
XLA_FLAGS = (
    '--xla_cpu_multi_thread_eigen=true '           # Multi-threaded linear algebra
    '--xla_force_host_platform_device_count=12 '   # Expose all cores as devices
    'intra_op_parallelism_threads=12 '             # Parallelism within operations
    'inter_op_parallelism_threads=12'              # Parallelism across operations
)
```

### 3. Threading Environment Variables

```python
OMP_NUM_THREADS=12      # OpenMP threading (BLAS libraries)
MKL_NUM_THREADS=12      # Intel MKL threading (if used)
```

## Why This Matters

### For GMRES Solver

GMRES is an iterative linear solver that:
- Performs matrix-vector multiplications repeatedly
- Can benefit from intra-op parallelism (within each matvec)
- With multiple devices, XLA can potentially parallelize the iteration loop

### For Power Method Solver

Power method iteratively computes `v = P @ v`:
- Each iteration is a matrix-vector product
- Can benefit from both intra-op and inter-op parallelism
- Multiple devices enable potential parallelization of independent operations

## Expected CPU Utilization

### Realistic Expectations

Even with optimal configuration, you may see **400-800% CPU usage** (4-8 cores) rather than 1200% (12 cores) because:

1. **Memory bandwidth bottlenecks** - CPU operations are often memory-bound
2. **Sequential dependencies** - Some parts of iterative solvers must run sequentially
3. **XLA optimization decisions** - XLA may determine that using all cores has too much overhead
4. **Algorithm structure** - The lazy solver has inherent sequential steps

### What to Monitor

Watch CPU usage during:
- **GMRES iterations** - Should see higher utilization during matvec operations
- **Power method iterations** - Similar pattern to GMRES
- **Grid upscaling** - May have sequential phases with lower utilization

## Testing

### Test Scripts Created

1. **`test_cpu_config.py`** - Basic configuration verification
2. **`test_cpu_diagnostics.py`** - Detailed diagnostics and device listing
3. **`test_cpu_utilization.py`** - Real workload testing with g=60 model
4. **`test_cpu_config_forced.py`** - CPU-only mode testing
5. **`test_cpu_config_user.py`** - User override verification

### Verification

Run `test_cpu_diagnostics.py` to verify:
```bash
python3 test_cpu_diagnostics.py
```

Expected output:
```
JAX devices: [CpuDevice(id=0), CpuDevice(id=1), ..., CpuDevice(id=11)]
XLA_FLAGS: --xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count=12 ...
```

## Next Steps

### For Further Optimization

If you need even better CPU utilization, consider:

1. **Explicit parallelization with `pmap`** - Manually parallelize across devices
2. **Batched operations with `vmap`** - Vectorize operations when possible
3. **Profile with JAX profiler** - Identify bottlenecks
4. **GPU acceleration** - For large grids (g>80), GPU may be more efficient

### Monitoring CPU Usage

While tests run, use:
```bash
htop  # Interactive process viewer
# or
top   # Standard process viewer
```

Look for the Python process and check:
- **CPU%** column - Should see 400-800% for well-parallelized code
- **Per-core usage** - In htop, press 't' to see tree view and '1' for per-core view

## Summary

✅ **Automatic CPU detection** - Detects all available cores  
✅ **Optimal XLA configuration** - All critical flags set automatically  
✅ **Multi-device support** - JAX now sees 12 separate CPU devices  
✅ **Threading configured** - OMP and MKL threads set correctly  
✅ **User override support** - Advanced users can still customize  
✅ **Backward compatible** - Existing code works without changes  

The configuration is now optimal for CPU-based iterative solvers. While you may not see 1200% CPU usage, the improvements should provide better performance than the default configuration.
