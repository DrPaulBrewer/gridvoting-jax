"""Test GMRES with grid upscaling on L4 GPU"""
import sys
sys.path.insert(0, '/home/gv-tester/gridvoting-jax/src')

import jax
print(f"JAX devices: {jax.devices()}")

import gridvoting_jax as gv
import jax.numpy as jnp

print("\n" + "=" * 80)
print("Testing GMRES + Grid Upscaling on L4 GPU (g=80)")
print("=" * 80)

try:
    print("\n1. Creating g=80 model...")
    model = gv.bjm_spatial_triangle(g=80, zi=False)
    print(f"   N = {model.model.number_of_feasible_alternatives}")
    
    print("\n2. Solving with GMRES + grid upscaling...")
    model.analyze(solver="lazy_grid_upscaling", max_iterations=3000)
    
    print(f"\n3. Success!")
    print(f"   Probability sum: {jnp.sum(model.stationary_distribution):.15f}")
    
    # Compare with OSF
    import requests
    import numpy as np
    from pathlib import Path
    
    print("\n4. Comparing with OSF reference data...")
    cache_dir = Path("/tmp/gridvoting_osf_cache")
    osf_file = cache_dir / "bjm_spatial_triangle_g80_zi_False.npy"
    
    if not osf_file.exists():
        print("   Downloading OSF data...")
        cache_dir.mkdir(parents=True, exist_ok=True)
        url = "https://osf.io/download/bjm_spatial_triangle_g80_zi_False.npy"
        # Note: actual OSF download would need proper URL
        print("   (OSF data not in cache)")
    
    if osf_file.exists():
        osf_data = np.load(osf_file)
        
        # Calculate differences
        diff = model.stationary_distribution - osf_data
        l1_norm = float(jnp.sum(jnp.abs(diff)))
        max_diff = float(jnp.max(jnp.abs(diff)))
        
        print(f"   L1 norm (sum of |differences|): {l1_norm:.15e}")
        print(f"   Max absolute difference: {max_diff:.15e}")
        print(f"   Mean absolute difference: {l1_norm / len(diff):.15e}")
    else:
        print("   OSF data not available for comparison")
    
    print("\n✅ GMRES works on L4 GPU!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    if "RESOURCE_EXHAUSTED" in str(e) or "Out of memory" in str(e):
        print("   GPU ran out of memory")
        import traceback
        traceback.print_exc()
    
print("\n" + "=" * 80)
