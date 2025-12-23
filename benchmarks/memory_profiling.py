"""Memory profiling for lazy solvers using JAX built-in tracking"""
import jax
import jax.numpy as jnp
import gridvoting_jax as gv
import subprocess

def profile_gpu_memory(g, solver_name):
    """Profile GPU memory usage using nvidia-smi and JAX stats"""
    device = jax.devices()[0]
    
    if device.platform != 'gpu':
        print("GPU not available")
        return None
    
    # Clear JAX cache
    jax.clear_caches()
    
    # Get initial GPU memory
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', 
                           '--format=csv,noheader,nounits'],
                          capture_output=True, text=True)
    initial_mem = int(result.stdout.strip())
    
    # Run solver
    model = gv.bjm_spatial_triangle(g=g, zi=False)
    model.analyze(solver=solver_name, max_iterations=1000)
    
    # Get JAX memory stats
    try:
        stats = device.memory_stats()
        jax_peak_mb = stats.get('peak_bytes_in_use', 0) / 1024 / 1024
    except:
        jax_peak_mb = None
    
    # Get nvidia-smi peak
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used',
                           '--format=csv,noheader,nounits'],
                          capture_output=True, text=True)
    peak_mem = int(result.stdout.strip())
    
    used_mb = peak_mem - initial_mem
    
    print(f"{solver_name} (g={g}):")
    print(f"  nvidia-smi: {used_mb} MB")
    if jax_peak_mb:
        print(f"  JAX stats:  {jax_peak_mb:.1f} MB")
    
    return {'nvidia_smi_mb': used_mb, 'jax_stats_mb': jax_peak_mb}

def profile_cpu_memory(g, solver_name):
    """Profile CPU memory using JAX tracking (if available)"""
    device = jax.devices()[0]
    
    if device.platform != 'cpu':
        print("Not on CPU")
        return None
    
    jax.clear_caches()
    
    # Run solver
    model = gv.bjm_spatial_triangle(g=g, zi=False)
    model.analyze(solver=solver_name, max_iterations=1000)
    
    print(f"{solver_name} (g={g}): CPU memory tracking not available via JAX")
    print("  Use system tools like 'top' or 'htop' for CPU memory monitoring")
    
    return None

if __name__ == "__main__":
    import sys
    g = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    
    device = jax.devices()[0]
    
    if device.platform == 'gpu':
        print(f"Profiling GPU memory (g={g})...\n")
        profile_gpu_memory(g, "lazy_power_method")
        print()
        profile_gpu_memory(g, "lazy_grid_upscaling")
    else:
        print(f"Profiling CPU memory (g={g})...\n")
        profile_cpu_memory(g, "lazy_power_method")
        print()
        profile_cpu_memory(g, "lazy_grid_upscaling")
