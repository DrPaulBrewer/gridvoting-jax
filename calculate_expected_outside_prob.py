"""Calculate expected probability outside g=80 box using exponential decay model.

Uses actual OSF g=80 stationary distribution data to fit exp(-kr) decay model,
then predicts probability in the g=100 region outside g=80 box.
"""

import sys
sys.path.insert(0, '/home/paul/gridvoting-jax/gridvoting-jax/src')

import gridvoting_jax as gv
from gridvoting_jax.benchmarks.osf_comparison import load_osf_distribution
import jax.numpy as jnp
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

print("=" * 80)
print("Exponential Decay Analysis from OSF g=80 Data")
print("=" * 80)

# Load OSF reference data
print("\n1. Loading OSF reference data for g=80...")
ref_data = load_osf_distribution(g=80, zi=False)
if ref_data is None:
    print("   ❌ Could not load OSF data")
    sys.exit(1)

print(f"   ✓ Loaded {len(ref_data)} points")

# Extract probabilities from log10prob
ref_statd = 10 ** ref_data['log10prob'].values
ref_statd = np.array(ref_statd)

# Create g=80 model to get grid coordinates
print("\n2. Creating g=80 model to get grid coordinates...")
model_80 = gv.bjm_spatial_triangle(g=80, zi=False)
grid_80 = model_80.grid
x = np.array(grid_80.x)
y = np.array(grid_80.y)

# Calculate L2 distance from origin
r = np.sqrt(x**2 + y**2)

print(f"   ✓ Grid points: {len(x)}")
print(f"   ✓ Distance range: [{r.min():.1f}, {r.max():.1f}]")

# Filter to non-zero probabilities for fitting
nonzero_mask = ref_statd > 1e-15
r_nonzero = r[nonzero_mask]
prob_nonzero = ref_statd[nonzero_mask]
log_prob = np.log(prob_nonzero)

print(f"   ✓ Non-zero probabilities: {len(prob_nonzero)}")

# Fit exponential decay model: prob = A * exp(-k * r)
# Taking log: log(prob) = log(A) - k * r
print("\n3. Fitting exponential decay model: prob = A * exp(-k * r)")

def exponential_model(r, A, k):
    return A * np.exp(-k * r)

# Fit using points with r > 20 to avoid center region complexity
fit_mask = r_nonzero > 20
r_fit = r_nonzero[fit_mask]
prob_fit = prob_nonzero[fit_mask]

# Perform curve fit
popt, pcov = curve_fit(exponential_model, r_fit, prob_fit, p0=[1e-4, 0.01])
A_fit, k_fit = popt

print(f"   ✓ Fitted parameters:")
print(f"     A (amplitude) = {A_fit:.6e}")
print(f"     k (decay rate) = {k_fit:.6f}")

# Calculate R² to assess fit quality
prob_predicted = exponential_model(r_fit, A_fit, k_fit)
ss_res = np.sum((prob_fit - prob_predicted)**2)
ss_tot = np.sum((prob_fit - np.mean(prob_fit))**2)
r_squared = 1 - (ss_res / ss_tot)
print(f"     R² = {r_squared:.6f}")

# Show predictions at key distances
print(f"\n4. Model predictions at key distances:")
for r_test in [50, 60, 70, 80, 90, 100, 110, 120]:
    prob_pred = exponential_model(r_test, A_fit, k_fit)
    print(f"   r={r_test:3d}: prob ≈ {prob_pred:.6e}")

# Calculate expected probability outside g=80 box in g=100 grid
print("\n5. Calculating expected probability outside g=80 box...")

# Create g=100 grid to get coordinates outside g=80 box
model_100 = gv.bjm_spatial_triangle(g=100, zi=False)
grid_100 = model_100.grid
x_100 = np.array(grid_100.x)
y_100 = np.array(grid_100.y)
r_100 = np.sqrt(x_100**2 + y_100**2)

# Identify points outside g=80 box
box_mask = grid_100.within_box(x0=-80, x1=80, y0=-80, y1=80)
outside_mask = ~np.array(box_mask)

x_outside = x_100[outside_mask]
y_outside = y_100[outside_mask]
r_outside = r_100[outside_mask]

print(f"   ✓ Points outside g=80 box: {len(r_outside)}")
print(f"   ✓ Distance range outside: [{r_outside.min():.1f}, {r_outside.max():.1f}]")

# Predict probability at each outside point
prob_outside_predicted = exponential_model(r_outside, A_fit, k_fit)
total_prob_outside_predicted = np.sum(prob_outside_predicted)

print(f"\n6. Expected probability outside g=80 box:")
print(f"   ✓ Total predicted: {total_prob_outside_predicted:.6e}")
print(f"   ✓ As percentage: {total_prob_outside_predicted * 100:.6f}%")
print(f"   ✓ Max individual point: {np.max(prob_outside_predicted):.6e}")
print(f"   ✓ Min individual point: {np.min(prob_outside_predicted):.6e}")

# Show distribution by distance bins
print(f"\n7. Distribution by distance bins:")
bins = [80, 90, 100, 110, 120, 150]
for i in range(len(bins)-1):
    bin_mask = (r_outside >= bins[i]) & (r_outside < bins[i+1])
    if np.sum(bin_mask) > 0:
        bin_prob = np.sum(prob_outside_predicted[bin_mask])
        bin_count = np.sum(bin_mask)
        print(f"   r ∈ [{bins[i]:3d}, {bins[i+1]:3d}): {bin_count:5d} points, "
              f"prob ≈ {bin_prob:.6e} ({bin_prob/total_prob_outside_predicted*100:.1f}%)")

# Compare with actual g=100 solution if available
print(f"\n8. Comparison with actual g=100 solution:")
print(f"   (Run test_osf_g80_g100_cpu.py first to get actual values)")
print(f"   Expected (from model): {total_prob_outside_predicted:.6e}")
print(f"   Reported in test: 0.000000 (likely precision issue)")
print(f"\n   If actual is ~{total_prob_outside_predicted:.2e}, the model is validated!")
print(f"   If actual is exactly 0, it's a solver artifact or numerical underflow")

# Create visualization
print(f"\n9. Creating visualization...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Decay curve with data and fit
ax1.scatter(r_nonzero, prob_nonzero, alpha=0.3, s=1, label='OSF g=80 data')
r_plot = np.linspace(0, 120, 1000)
prob_plot = exponential_model(r_plot, A_fit, k_fit)
ax1.plot(r_plot, prob_plot, 'r-', linewidth=2, label=f'Fit: A·exp(-kr), k={k_fit:.4f}')
ax1.axvline(80, color='g', linestyle='--', label='g=80 boundary')
ax1.axvline(100, color='b', linestyle='--', label='g=100 boundary')
ax1.set_xlabel('Distance from origin (r)')
ax1.set_ylabel('Probability')
ax1.set_yscale('log')
ax1.set_title('Exponential Decay Model Fit')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right plot: Predicted probability outside g=80 box
ax2.scatter(r_outside, prob_outside_predicted, alpha=0.5, s=2, c=prob_outside_predicted, 
            cmap='viridis', norm=plt.matplotlib.colors.LogNorm())
ax2.set_xlabel('Distance from origin (r)')
ax2.set_ylabel('Predicted probability')
ax2.set_yscale('log')
ax2.set_title(f'Predicted Probability Outside g=80 Box\nTotal: {total_prob_outside_predicted:.2e}')
ax2.grid(True, alpha=0.3)
plt.colorbar(ax2.collections[0], ax=ax2, label='Probability')

plt.tight_layout()
plt.savefig('exponential_decay_analysis.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: exponential_decay_analysis.png")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\n📊 Key Result: Expected probability outside g=80 box = {total_prob_outside_predicted:.6e}")
print(f"   This is the value we should see in a properly converged g=100 solution")
print("=" * 80)
