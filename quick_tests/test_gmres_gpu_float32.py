"""Quick test: Does g=100 GMRES with grid upscaling work on GPU with float32?"""

import sys
sys.path.insert(0, '/home/paul/gridvoting-jax/gridvoting-jax/src')

import os
# Force GPU
os.environ.pop('JAX_PLATFORMS', None)

import jax
print(f"Device: {jax.devices()[0]}")
print(f"Float64 enabled: {jax.config.read('jax_enable_x64')}")

import gridvoting_jax as gv
import jax.numpy as jnp

print("\n" + "=" * 80)
print("Testing g=100 GMRES + Grid Upscaling on GPU (float32)")
print("=" * 80)

try:
    print("\n1. Creating g=100 model...")
    model = gv.bjm_spatial_triangle(g=100, zi=False)
    print(f"   N = {model.model.number_of_feasible_alternatives}")
    
    print("\n2. Solving with GMRES + grid upscaling...")
    model.analyze(solver="lazy_grid_upscaling", max_iterations=3000)
    
    print(f"\n3. Success!")
    print(f"   Probability sum: {jnp.sum(model.stationary_distribution):.15f}")
    
    # Check outside probability
    grid = model.grid
    box_mask = grid.within_box(x0=-80, x1=80, y0=-80, y1=80)
    outside = float(jnp.sum(model.stationary_distribution[~box_mask]))
    print(f"   Outside g=80 box: {outside:.15e}")
    
    print("\n✅ GMRES works on GPU with float32!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    if "RESOURCE_EXHAUSTED" in str(e) or "Out of memory" in str(e):
        print("   GPU ran out of memory")
    
print("\n" + "=" * 80)
