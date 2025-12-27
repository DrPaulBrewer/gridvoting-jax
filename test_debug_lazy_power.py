#!/usr/bin/env python3
"""
Add debug logging to understand lazy power method behavior.
"""

import gridvoting_jax as gv
import jax.numpy as jnp

# Monkey-patch to add logging
original_solve = gv.dynamics.lazy.lazy_markov.LazyMarkovChain._solve_power_method_lazy

def debug_solve(self, tolerance, max_iterations, initial_guess=None, timeout=30.0):
    print(f"\n=== Lazy Power Method Debug ===")
    print(f"  max_iterations: {max_iterations}")
    print(f"  timeout: {timeout}")
    print(f"  tolerance: {tolerance}")
    
    import time
    n = self.lazy_P.N
    
    if initial_guess is not None:
        x = jnp.asarray(initial_guess)
        x = x / jnp.sum(x)
    else:
        x = jnp.ones(n, dtype=self.lazy_P.dtype) / n
    
    print(f"  Initial x: uniform={jnp.std(x) < 1e-10}")
    
    start_time = time.time()
    
    # Calibration
    calibration_iters = 10
    print(f"\n  Running {calibration_iters} calibration iterations...")
    calibration_start = time.perf_counter_ns()
    
    x_calib = x
    for i in range(calibration_iters):
        x_calib = self.lazy_P.rmatvec_batched(x_calib)
    
    calibration_time_ns = time.perf_counter_ns() - calibration_start
    calibration_time_s = calibration_time_ns / 1e9
    time_per_iter_s = calibration_time_s / calibration_iters
    
    print(f"  Calibration took: {calibration_time_s:.3f}s")
    print(f"  Time per iteration: {time_per_iter_s:.6f}s")
    print(f"  x_calib after calibration: uniform={jnp.std(x_calib) < 1e-10}")
    
    # Check timeout after calibration
    elapsed = time.time() - start_time
    time_remaining = timeout - elapsed
    print(f"\n  After calibration:")
    print(f"    Elapsed: {elapsed:.3f}s")
    print(f"    Time remaining: {time_remaining:.3f}s")
    
    if time_remaining <= 0:
        print(f"  ❌ TIMEOUT during calibration! Returning initial guess.")
        return x
    
    # Calculate first batch size
    iters_for_half_time = int((time_remaining / 2.0) / time_per_iter_s)
    iters_for_half_time = max(1, min(iters_for_half_time, max_iterations))
    print(f"    First batch size: {iters_for_half_time} iterations")
    
    # Call original
    result = original_solve(self, tolerance, max_iterations, initial_guess, timeout)
    
    print(f"\n  Final result: uniform={jnp.std(result) < 1e-10}")
    print(f"=== End Debug ===\n")
    
    return result

gv.dynamics.lazy.lazy_markov.LazyMarkovChain._solve_power_method_lazy = debug_solve

# Test
model = gv.bjm_spatial_triangle(g=40, zi=True)
model.analyze_lazy(solver="power_method", force_lazy=True, max_iterations=5000, timeout=30.0)
