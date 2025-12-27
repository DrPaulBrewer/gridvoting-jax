"""Direct comparison of old vs new ZI finalization logic."""

import jax.numpy as jnp
from gridvoting_jax.core.zimi_succession_logic import finalize_transition_matrix_zi_jit

# Simulate the old logic
def old_zi_logic(cV_batch, N, status_quo_indices):
    """Old ZI logic from operators.py"""
    batch_size = cV_batch.shape[0]
    
    # Count winning alternatives for each status quo
    row_sums = cV_batch.sum(axis=1)  # (B,)
    
    # Start with cV_batch (winners get 1, losers get 0)
    cP = cV_batch.astype(jnp.float32)
    
    # Add diagonal: status quo gets (N - row_sum)
    diag_values = N - row_sums  # (B,)
    cP = cP.at[jnp.arange(batch_size), status_quo_indices].add(diag_values.astype(jnp.float32))
    
    # Divide everything by N
    cP = cP / N
    
    return cP

# Test case: 3 states, status quo = [0, 1, 2]
N = 3
status_quo_indices = jnp.array([0, 1, 2])

# Winner matrix: each row shows which alternatives beat that status quo
cV_batch = jnp.array([
    [0, 0, 1],  # SQ=0: alternative 2 beats it
    [1, 0, 0],  # SQ=1: alternative 0 beats it  
    [0, 1, 0],  # SQ=2: alternative 1 beats it
], dtype=jnp.int32)

print("Input:")
print(f"cV_batch:\n{cV_batch}")
print(f"N = {N}")
print(f"status_quo_indices = {status_quo_indices}")

# Old logic
cP_old = old_zi_logic(cV_batch, N, status_quo_indices)
print(f"\nOld ZI logic result:")
print(cP_old)
print(f"Row sums: {cP_old.sum(axis=1)}")

# New logic
cP_new = finalize_transition_matrix_zi_jit(cV_batch, N, status_quo_indices, eligibility_mask=None)
print(f"\nNew ZI logic result:")
print(cP_new)
print(f"Row sums: {cP_new.sum(axis=1)}")

# Compare
diff = jnp.abs(cP_old - cP_new).sum()
print(f"\nL1 difference: {diff:.10e}")

if diff > 1e-6:
    print("⚠️  MISMATCH!")
    print(f"Old:\n{cP_old}")
    print(f"New:\n{cP_new}")
else:
    print("✓ Results match!")
