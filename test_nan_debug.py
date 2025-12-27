#!/usr/bin/env python3
"""
Test script to reproduce NaN issue in grid_upscaling_lazy_gmres for ZI mode.
Run with: ./test_docker.sh --dev --cpu --command="python3 /workspace/test_nan_debug.py"
"""

import gridvoting_jax as gv
import jax.numpy as jnp

def test_zi_lazy_gmres(g=20):
    """Test grid_upscaling_lazy_gmres on ZI mode."""
    print(f"\n{'='*60}")
    print(f"Testing g={g}, ZI mode with grid_upscaling_lazy_gmres")
    print(f"{'='*60}")
    
    # Create model
    model = gv.bjm_spatial_triangle(g=g, zi=True)
    
    # Test 1: Direct lazy GMRES (no grid upscaling)
    print("\n1. Testing direct lazy GMRES...")
    try:
        model.analyze_lazy(solver="gmres", force_lazy=True)
        dist1 = model.stationary_distribution
        print(f"   Result: sum={jnp.sum(dist1):.6f}, has_nan={jnp.any(jnp.isnan(dist1))}")
        if jnp.any(jnp.isnan(dist1)):
            print("   ❌ Direct lazy GMRES produces NaN!")
        else:
            print("   ✓ Direct lazy GMRES works")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Grid upscaling with lazy GMRES
    print("\n2. Testing grid_upscaling_lazy_gmres...")
    try:
        model.analyze(solver="grid_upscaling_lazy_gmres")
        dist2 = model.stationary_distribution
        print(f"   Result: sum={jnp.sum(dist2):.6f}, has_nan={jnp.any(jnp.isnan(dist2))}")
        if jnp.any(jnp.isnan(dist2)):
            print("   ❌ grid_upscaling_lazy_gmres produces NaN!")
        else:
            print("   ✓ grid_upscaling_lazy_gmres works")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Grid upscaling with lazy power method (for comparison)
    print("\n3. Testing grid_upscaling_lazy_power (for comparison)...")
    try:
        model.analyze(solver="grid_upscaling_lazy_power")
        dist3 = model.stationary_distribution
        print(f"   Result: sum={jnp.sum(dist3):.6f}, has_nan={jnp.any(jnp.isnan(dist3))}")
        if jnp.any(jnp.isnan(dist3)):
            print("   ❌ grid_upscaling_lazy_power produces NaN!")
        else:
            print("   ✓ grid_upscaling_lazy_power works (but may have poor accuracy)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Dense GMRES (for comparison)
    print("\n4. Testing dense GMRES (for comparison)...")
    try:
        model.analyze(solver="gmres_matrix_inversion")
        dist4 = model.stationary_distribution
        print(f"   Result: sum={jnp.sum(dist4):.6f}, has_nan={jnp.any(jnp.isnan(dist4))}")
        print("   ✓ Dense GMRES works")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def test_mi_lazy_gmres(g=20):
    """Test grid_upscaling_lazy_gmres on MI mode (should work)."""
    print(f"\n{'='*60}")
    print(f"Testing g={g}, MI mode with grid_upscaling_lazy_gmres")
    print(f"{'='*60}")
    
    model = gv.bjm_spatial_triangle(g=g, zi=False)
    
    try:
        model.analyze(solver="grid_upscaling_lazy_gmres")
        dist = model.stationary_distribution
        print(f"Result: sum={jnp.sum(dist):.6f}, has_nan={jnp.any(jnp.isnan(dist))}")
        if jnp.any(jnp.isnan(dist)):
            print("❌ MI mode also produces NaN!")
        else:
            print("✓ MI mode works correctly")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("NaN Debug Test for grid_upscaling_lazy_gmres")
    print("=" * 60)
    
    # Test ZI mode (should produce NaN)
    test_zi_lazy_gmres(g=20)
    
    # Test MI mode (should work)
    test_mi_lazy_gmres(g=20)
    
    print(f"\n{'='*60}")
    print("Test complete")
    print(f"{'='*60}")
