"""Test lazy grid upscaling on large grids (g=80, g=100)."""

import sys
sys.path.insert(0, '/home/paul/gridvoting-jax/gridvoting-jax/src')

import gridvoting_jax as gv
import jax.numpy as jnp
import time

print("=" * 70)
print("LAZY GRID UPSCALING TEST: g=80 and g=100")
print("=" * 70)

# Test g=80
print("\n" + "=" * 70)
print("Testing g=80 (N=25,921)")
print("=" * 70)

print("\n1. Lazy grid upscaling solver...")
start = time.time()
model_upscaling_80 = gv.bjm_spatial_triangle(g=80, zi=False)
model_upscaling_80.analyze(solver="lazy_grid_upscaling", max_iterations=3000)
upscaling_time_80 = time.time() - start
print(f"   ✓ Time: {upscaling_time_80:.2f}s")
print(f"   ✓ Sum: {jnp.sum(model_upscaling_80.stationary_distribution):.6f}")

print("\n2. Lazy GMRES (no initial guess)...")
start = time.time()
model_lazy_80 = gv.bjm_spatial_triangle(g=80, zi=False)
model_lazy_80.analyze_lazy(force_lazy=True, solver="gmres", max_iterations=3000)
lazy_time_80 = time.time() - start
print(f"   ✓ Time: {lazy_time_80:.2f}s")
print(f"   ✓ Sum: {jnp.sum(model_lazy_80.stationary_distribution):.6f}")

print("\n3. Comparing results...")
diff_80 = jnp.abs(model_upscaling_80.stationary_distribution - model_lazy_80.stationary_distribution)
max_diff_80 = jnp.max(diff_80)
print(f"   ✓ Max difference: {max_diff_80:.2e}")
print(f"   ✓ Speedup: {lazy_time_80 / upscaling_time_80:.2f}x")

if max_diff_80 < 1e-4:
    print("   ✅ Results match!")

# Test g=100
print("\n" + "=" * 70)
print("Testing g=100 (N=40,401)")
print("=" * 70)

print("\n1. Lazy grid upscaling solver...")
start = time.time()
model_upscaling_100 = gv.bjm_spatial_triangle(g=100, zi=False)
model_upscaling_100.analyze(solver="lazy_grid_upscaling", max_iterations=3000)
upscaling_time_100 = time.time() - start
print(f"   ✓ Time: {upscaling_time_100:.2f}s")
print(f"   ✓ Sum: {jnp.sum(model_upscaling_100.stationary_distribution):.6f}")

print("\n2. Lazy GMRES (no initial guess)...")
start = time.time()
model_lazy_100 = gv.bjm_spatial_triangle(g=100, zi=False)
model_lazy_100.analyze_lazy(force_lazy=True, solver="gmres", max_iterations=3000)
lazy_time_100 = time.time() - start
print(f"   ✓ Time: {lazy_time_100:.2f}s")
print(f"   ✓ Sum: {jnp.sum(model_lazy_100.stationary_distribution):.6f}")

print("\n3. Comparing results...")
diff_100 = jnp.abs(model_upscaling_100.stationary_distribution - model_lazy_100.stationary_distribution)
max_diff_100 = jnp.max(diff_100)
print(f"   ✓ Max difference: {max_diff_100:.2e}")
print(f"   ✓ Speedup: {lazy_time_100 / upscaling_time_100:.2f}x")

if max_diff_100 < 1e-4:
    print("   ✅ Results match!")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"g=80:  Lazy grid upscaling: {upscaling_time_80:.2f}s, Lazy GMRES: {lazy_time_80:.2f}s, Speedup: {lazy_time_80/upscaling_time_80:.2f}x")
print(f"g=100: Lazy grid upscaling: {upscaling_time_100:.2f}s, Lazy GMRES: {lazy_time_100:.2f}s, Speedup: {lazy_time_100/upscaling_time_100:.2f}x")
print("=" * 70)
