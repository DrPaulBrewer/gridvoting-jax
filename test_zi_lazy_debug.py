"""Debug script to find ZI lazy bug by directly testing the functions."""

import jax.numpy as jnp
from gridvoting_jax.core.winner_determination import compute_winner_matrix_jit
from gridvoting_jax.core.zimi_succession_logic import finalize_transition_matrix

# Create simple test case: 3 voters, 5 alternatives
# Voter utilities (V=3, N=5)
utilities = jnp.array([
    [1.0, 2.0, 3.0, 4.0, 5.0],  # Voter 0 prefers 4 > 3 > 2 > 1 > 0
    [5.0, 4.0, 3.0, 2.0, 1.0],  # Voter 1 prefers 0 > 1 > 2 > 3 > 4
    [3.0, 3.0, 5.0, 1.0, 1.0],  # Voter 2 prefers 2 > 0=1 > 3=4
])

majority = 2  # Need 2 votes to win
N = 5

# Test for status quo = 0
status_quo_indices = jnp.array([0])

print("Testing ZI logic for status quo = 0")
print("=" * 60)

# Step 1: Compute winner matrix
cV = compute_winner_matrix_jit(utilities, majority, status_quo_indices)
print(f"\nWinner matrix cV (shape {cV.shape}):")
print(cV)
print(f"Row sum: {cV.sum(axis=1)}")

# Step 2: Finalize to transition matrix (ZI mode)
cP_zi = finalize_transition_matrix(cV, zi=True, nfa=N, status_quo_indices=status_quo_indices)
print(f"\nZI Transition matrix cP (shape {cP_zi.shape}):")
print(cP_zi)
print(f"Row sum: {cP_zi.sum(axis=1)}")
print(f"Diagonal value cP[0, 0]: {cP_zi[0, 0]}")

# Check if uniform
is_uniform = jnp.allclose(cP_zi[0], 1.0/N)
print(f"\nIs uniform (all 1/{N}={1.0/N:.3f})? {is_uniform}")

# Expected ZI logic:
# - Winners get probability 1/N
# - Losers get probability 0
# - Status quo gets (N - num_winners) / N
num_winners = int(cV[0].sum())
expected_sq_prob = (N - num_winners) / N
print(f"\nExpected:")
print(f"  Number of winners: {num_winners}")
print(f"  Status quo probability: {expected_sq_prob:.3f}")
print(f"  Winner probability: {1.0/N:.3f}")

# Step 3: Test MI mode for comparison
cP_mi = finalize_transition_matrix(cV, zi=False, nfa=N, status_quo_indices=status_quo_indices)
print(f"\nMI Transition matrix cP (shape {cP_mi.shape}):")
print(cP_mi)
print(f"Row sum: {cP_mi.sum(axis=1)}")
