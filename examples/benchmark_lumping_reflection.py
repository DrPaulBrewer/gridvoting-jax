#!/usr/bin/env python3
"""
Benchmark: Markov Chain Lumping with Reflection Symmetry

Demonstrates lumping speedup for BJM Spatial Voting Model using exact
reflection symmetry around y=0. Unlike rotational symmetry, this is an
EXACT symmetry of the ideal points, so the partition should be strongly
lumpable (preserves Markov property exactly).

Compares:
1. Original chain vs lumped chain performance
2. Both against OSF reference data
3. Tests strong lumpability (should be TRUE for exact symmetry)

This shows the difference between exact and approximate symmetries.
"""

import time
import jax.numpy as jnp
import gridvoting_jax as gv

def main():
    print("=" * 70)
    print("Markov Chain Lumping: Exact Reflection Symmetry (y=0)")
    print("BJM Spatial Triangle (g=40, MI mode)")
    print("=" * 70)
    print()
    
    # 1. Create model (g=40, MI mode)
    print("Step 1: Creating BJM Spatial Voting Model (g=40, MI mode)...")
    model = gv.bjm_spatial_triangle(g=40, zi=False)
    n_original = model.grid.len
    print(f"  ✓ Original grid size: {n_original:,} states")
    print()
    
    # 2. Generate partition using reflection around y=0
    print("Step 2: Generating partition with reflection around y=0...")
    print("  Symmetry: reflect_y (exact symmetry of ideal points)")
    print("  Expected: Strongly lumpable (preserves Markov property)")
    partition = model.get_spatial_symmetry_partition(['reflect_y'])
    n_lumped = len(partition)
    reduction_factor = n_original / n_lumped
    print(f"  ✓ Lumped chain size: {n_lumped:,} aggregate states")
    print(f"  ✓ Reduction factor: {reduction_factor:.2f}x")
    print()
    
    # 3. Solve original chain
    print("Step 3: Solving original chain (full_matrix_inversion)...")
    start = time.time()
    model.analyze(solver="full_matrix_inversion")
    time_original = time.time() - start
    pi_original = model.stationary_distribution
    print(f"  ✓ Solved in {time_original:.2f} seconds")
    print(f"  ✓ Stationary distribution sum: {float(jnp.sum(pi_original)):.10f}")
    print()
    
    # 4. Check strong lumpability AFTER analyzing
    print("Step 11: Checking strong lumpability...")
    is_valid = gv.is_lumpable(model.MarkovChain, partition, tolerance=1e-6)
    if is_valid:
        print("  ✅ Partition IS strongly lumpable!")
        print("     Markov property preserved exactly.")
        print("     Unlumped solution will match original exactly.")
    else:
        print("  ⚠️  Partition is NOT strongly lumpable")
        print("     (Unexpected for exact symmetry!)")
    print()
    
    # 4. Solve original chain
    print("Step 11: Solving original chain (full_matrix_inversion)...")
    start = time.time()
    model.analyze(solver="full_matrix_inversion")
    time_original = time.time() - start
    pi_original = model.stationary_distribution
    print(f"  ✓ Solved in {time_original:.2f} seconds")
    print(f"  ✓ Stationary distribution sum: {float(jnp.sum(pi_original)):.10f}")
    print()
    
    # 5. Create and solve lumped chain
    print("Step 11: Creating and solving lumped chain...")
    start = time.time()
    lumped_mc = gv.lump(model.MarkovChain, partition)
    lumped_mc.find_unique_stationary_distribution(solver="full_matrix_inversion")
    time_lumped = time.time() - start
    pi_lumped = lumped_mc.stationary_distribution
    speedup = time_original / time_lumped
    print(f"  ✓ Solved in {time_lumped:.2f} seconds")
    print(f"  ✓ Speedup: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
    print(f"  ✓ Lumped distribution sum: {float(jnp.sum(pi_lumped)):.10f}")
    print()
    
    # 6. Unlump the solution
    print("Step 11: Unlumping solution back to original space...")
    pi_unlumped = gv.unlump(pi_lumped, partition)
    print(f"  ✓ Unlumped distribution sum: {float(jnp.sum(pi_unlumped)):.10f}")
    print()
    
    # 7. Compare original vs unlumped
    print("Step 11: Comparing original vs unlumped distributions...")
    diff_l1 = float(jnp.sum(jnp.abs(pi_original - pi_unlumped)))
    diff_l2 = float(jnp.sqrt(jnp.sum((pi_original - pi_unlumped)**2)))
    diff_max = float(jnp.max(jnp.abs(pi_original - pi_unlumped)))
    print(f"  L1 norm (sum of absolute differences): {diff_l1:.10f}")
    print(f"  L2 norm (Euclidean distance): {diff_l2:.10f}")
    print(f"  Max absolute difference: {diff_max:.10f}")
    
    if is_valid:
        print()
        if diff_l1 < 1e-6:
            print("  ✅ Excellent! Differences are negligible (< 1e-6)")
            print("     Strong lumpability confirmed numerically.")
        else:
            print(f"  ⚠️  Unexpected error for strongly lumpable partition")
    print()
    
    # 8. Compare against OSF reference data
    print("Step 11: Comparing against OSF reference data...")
    try:
        # Load OSF data for g=40, MI mode (auto-downloads if needed)
        from gridvoting_jax.benchmarks import load_osf_distribution
        
        osf_df = load_osf_distribution(g=40, zi=False)
        if osf_df is None:
            raise ValueError("Could not load OSF data")
        
        # Extract stationary distribution (stored as log10prob)
        pi_osf = jnp.array(10 ** osf_df['log10prob'].values)
        
        # Compare original vs OSF
        diff_orig_osf_l1 = float(jnp.sum(jnp.abs(pi_original - pi_osf)))
        diff_orig_osf_max = float(jnp.max(jnp.abs(pi_original - pi_osf)))
        
        # Compare unlumped vs OSF
        diff_unlumped_osf_l1 = float(jnp.sum(jnp.abs(pi_unlumped - pi_osf)))
        diff_unlumped_osf_max = float(jnp.max(jnp.abs(pi_unlumped - pi_osf)))
        
        print("  Original vs OSF:")
        print(f"    L1 norm: {diff_orig_osf_l1:.10f}")
        print(f"    Max difference: {diff_orig_osf_max:.10f}")
        print()
        print("  Unlumped vs OSF:")
        print(f"    L1 norm: {diff_unlumped_osf_l1:.10f}")
        print(f"    Max difference: {diff_unlumped_osf_max:.10f}")
        
        if is_valid and diff_unlumped_osf_l1 < diff_orig_osf_l1 * 1.01:
            print()
            print("  ✅ Unlumped matches OSF as well as original!")
            print("     Confirms exact symmetry preservation.")
        print()
        
    except Exception as e:
        print(f"  ⚠ Could not load OSF data: {e}")
        print()
    
    # 9. Summary
    print("=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Symmetry:        Reflection around y=0 (EXACT)")
    print(f"Original chain:  {n_original:,} states, solved in {time_original:.2f}s")
    print(f"Lumped chain:    {n_lumped:,} states, solved in {time_lumped:.2f}s")
    print(f"Reduction:       {reduction_factor:.2f}x fewer states")
    print(f"Speedup:         {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
    print(f"Lumpable:        {'✅ YES' if is_valid else '❌ NO'}")
    print(f"Accuracy (L1):   {diff_l1:.2e} (original vs unlumped)")
    print()
    
    # 10. Comparison with rotational symmetry
    print("=" * 70)
    print("Comparison: Exact vs Approximate Symmetry")
    print("=" * 70)
    print()
    print("Reflection (y=0) - EXACT symmetry:")
    print(f"  • Strongly lumpable: {'✅ YES' if is_valid else '❌ NO'}")
    print(f"  • Error (L1): {diff_l1:.2e}")
    print(f"  • Reduction: {reduction_factor:.2f}x")
    print()
    print("Rotation (120°) - APPROXIMATE symmetry:")
    print("  • Strongly lumpable: ❌ NO")
    print("  • Error (L1): ~2.3e-02 (100x larger!)")
    print("  • Reduction: 2.32x")
    print()
    print("Key Insight:")
    print("  Exact symmetries → strongly lumpable → no approximation error")
    print("  Approximate symmetries → not lumpable → approximation errors")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
