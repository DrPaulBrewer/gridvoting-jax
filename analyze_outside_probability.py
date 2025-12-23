"""Analyze the probability distribution outside the g=80 box in g=100 solution.

Based on BJM paper Figure 4, the stationary distribution decays roughly as exp(-kr)
where r is the L2 distance from origin. We expect small but non-zero probability
outside the g=80 box.
"""

import sys
sys.path.insert(0, '/home/paul/gridvoting-jax/gridvoting-jax/src')

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import gridvoting_jax as gv
import jax.numpy as jnp
import numpy as np

print("=" * 80)
print("Analyzing Probability Distribution Outside g=80 Box")
print("=" * 80)

# Load the g=100 solution (if we have it cached, otherwise solve)
print("\n1. Creating and solving g=100 model...")
model_100 = gv.bjm_spatial_triangle(g=100, zi=False)
model_100.analyze(solver="lazy_grid_upscaling", max_iterations=3000)
print(f"   ✓ Solved (N={model_100.model.number_of_feasible_alternatives})")
print(f"   ✓ Total probability: {jnp.sum(model_100.stationary_distribution):.10f}")

# Get grid coordinates
grid_100 = model_100.grid
x = grid_100.x
y = grid_100.y

# Calculate L2 distance from origin for each point
r = jnp.sqrt(x**2 + y**2)

print("\n2. Grid statistics:")
print(f"   ✓ x range: [{x.min():.1f}, {x.max():.1f}]")
print(f"   ✓ y range: [{y.min():.1f}, {y.max():.1f}]")
print(f"   ✓ r (distance) range: [{r.min():.1f}, {r.max():.1f}]")

# Define g=80 box
x0, x1 = -80.0, 80.0
y0, y1 = -80.0, 80.0
box_mask = grid_100.within_box(x0=x0, x1=x1, y0=y0, y1=y1)

# Split into inside and outside
prob_inside = model_100.stationary_distribution[box_mask]
prob_outside = model_100.stationary_distribution[~box_mask]

total_inside = jnp.sum(prob_inside)
total_outside = jnp.sum(prob_outside)

print("\n3. Probability distribution:")
print(f"   ✓ Points inside g=80 box:  {jnp.sum(box_mask)}")
print(f"   ✓ Points outside g=80 box: {jnp.sum(~box_mask)}")
print(f"   ✓ Probability inside:  {total_inside:.15f}")
print(f"   ✓ Probability outside: {total_outside:.15f}")
print(f"   ✓ Outside/Total ratio: {total_outside:.10e} ({total_outside*100:.6f}%)")

# Analyze the outside region
if total_outside > 0:
    print("\n4. Outside region analysis:")
    r_outside = r[~box_mask]
    prob_outside_vals = prob_outside
    
    # Sort by distance
    sorted_indices = jnp.argsort(r_outside)
    r_sorted = r_outside[sorted_indices]
    prob_sorted = prob_outside_vals[sorted_indices]
    
    print(f"   ✓ Min distance outside: {r_sorted[0]:.2f}")
    print(f"   ✓ Max distance outside: {r_sorted[-1]:.2f}")
    print(f"   ✓ Max probability outside: {jnp.max(prob_outside_vals):.10e}")
    print(f"   ✓ Min probability outside: {jnp.min(prob_outside_vals):.10e}")
    
    # Show top 10 points by probability
    top_indices = jnp.argsort(prob_outside_vals)[::-1][:10]
    print(f"\n   Top 10 points outside g=80 box by probability:")
    for i, idx in enumerate(top_indices):
        actual_idx = jnp.where(~box_mask)[0][idx]
        print(f"   {i+1}. r={r[actual_idx]:.2f}, "
              f"(x={x[actual_idx]:.1f}, y={y[actual_idx]:.1f}), "
              f"prob={prob_outside_vals[idx]:.10e}")
else:
    print("\n4. ⚠️  Probability outside is exactly zero!")
    print("   This is likely a solver artifact (numerical precision or convergence issue)")

# Analyze decay pattern
print("\n5. Analyzing exponential decay pattern:")
print("   Based on BJM Figure 4, expect: prob ~ exp(-k*r)")

# Look at probability vs distance for points inside g=80 box
r_inside = r[box_mask]
prob_inside_vals = prob_inside

# Filter to non-zero probabilities
nonzero_mask = prob_inside_vals > 1e-15
r_nonzero = r_inside[nonzero_mask]
prob_nonzero = prob_inside_vals[nonzero_mask]

if len(r_nonzero) > 0:
    # Take log of probabilities
    log_prob = jnp.log(prob_nonzero)
    
    # Simple linear fit: log(prob) = a - k*r
    # Using points with r > 50 to estimate decay rate
    far_mask = r_nonzero > 50
    if jnp.sum(far_mask) > 10:
        r_far = r_nonzero[far_mask]
        log_prob_far = log_prob[far_mask]
        
        # Estimate decay rate k
        k_estimate = -jnp.mean((log_prob_far[1:] - log_prob_far[:-1]) / 
                               (r_far[1:] - r_far[:-1]))
        
        print(f"   ✓ Estimated decay rate k ≈ {k_estimate:.6f}")
        
        # Predict probability at r=100 (edge of g=100 box)
        # Using a point at r≈80 as reference
        ref_mask = (r_nonzero > 75) & (r_nonzero < 85)
        if jnp.sum(ref_mask) > 0:
            r_ref = jnp.mean(r_nonzero[ref_mask])
            prob_ref = jnp.mean(prob_nonzero[ref_mask])
            
            # Predict at r=100
            r_predict = 100.0
            prob_predict = prob_ref * jnp.exp(-k_estimate * (r_predict - r_ref))
            
            print(f"   ✓ Reference: r={r_ref:.1f}, prob={prob_ref:.10e}")
            print(f"   ✓ Predicted at r=100: prob ≈ {prob_predict:.10e}")
            
            # Estimate total probability outside g=80 box
            # Rough estimate: integrate exp(-kr) over annulus
            # This is approximate!
            r_min_outside = 80.0 * jnp.sqrt(2)  # Corner of box
            r_max_outside = 100.0 * jnp.sqrt(2)  # Corner of g=100
            
            print(f"\n6. Expected probability outside g=80 box:")
            print(f"   ✓ Based on exponential decay model")
            print(f"   ✓ Outer corners: r ∈ [{r_min_outside:.1f}, {r_max_outside:.1f}]")
            print(f"   ✓ Expected order of magnitude: ~1e-6 to 1e-4")
            print(f"   ✓ Actual: {total_outside:.10e}")
            
            if total_outside == 0:
                print(f"\n   ⚠️  WARNING: Actual is exactly zero!")
                print(f"   This suggests numerical underflow or solver artifact")
            elif total_outside < 1e-10:
                print(f"\n   ✓ Actual is very small but non-zero (good!)")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
