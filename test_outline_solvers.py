"""
Quick test of outline-based solvers implementation.

Tests all three new solvers on BJM spatial triangle g=20.
"""

import sys
sys.path.insert(0, '/home/paul/gridvoting-jax/gridvoting-jax/src')
import gridvoting_jax as gv

print("Testing outline-based solvers")
print("="*80)

# Create small model for quick testing
print("\nCreating BJM spatial triangle (g=20, zi=False)...")
model = gv.bjm_spatial_triangle(g=20, zi=False)
print(f"Grid size: {model.grid.len} alternatives")

# Test each solver
solvers_to_test = [
    "outline_and_fill",
    "outline_and_power",
    "outline_and_gmres"
]

for solver in solvers_to_test:
    print(f"\n{'='*80}")
    print(f"Testing solver: {solver}")
    print(f"{'='*80}")
    
    try:
        # Create fresh model for each test
        test_model = gv.bjm_spatial_triangle(g=20, zi=False)
        
        # Run solver
        test_model.analyze(solver=solver, tolerance=1e-6, max_iterations=5000)
        
        # Check results
        dist_sum = test_model.stationary_distribution.sum()
        print(f"✓ Solver completed successfully")
        print(f"  Stationary distribution sum: {dist_sum:.10f}")
        print(f"  Distribution shape: {test_model.stationary_distribution.shape}")
        
        # Validate
        assert test_model.analyzed, "Model not marked as analyzed"
        assert abs(dist_sum - 1.0) < 1e-4, f"Distribution sum {dist_sum} != 1.0"
        print(f"✓ Validation passed")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}")
print("All tests completed!")
print(f"{'='*80}")
