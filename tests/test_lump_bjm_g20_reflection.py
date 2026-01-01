import jax.numpy as jnp
import gridvoting_jax as gv
import pytest

def test_lump_bjm_g20_reflection():
    """
    Test lumping accuracy on BJM spatial triangle with reflection symmetry.
    1. Solve full model (g=20)
    2. Solve lumped model using reflect_x symmetry
    3. Verify L1 norm difference is small (< 1e-4)
    """
    # 1. Create BJM spatial triangle model (g=20, Minimal Intelligence mode)
    # This model has exact reflection symmetry around x=0
    model = gv.bjm_spatial_triangle(g=20, zi=False)
    n_original = model.grid.len
    
    # 2. Solve original chain using full matrix inversion
    model.analyze(solver="full_matrix_inversion")
    pi_original = model.stationary_distribution
    
    # 3. Generate partition using reflection around x=0
    # This uses the optimized fast-path implemented in v0.24.0
    partition = model.get_spatial_symmetry_partition(['reflect_x'])
    n_lumped = int(partition.max()) + 1
    
    # Verify reduction (should be nearly half)
    assert n_lumped < n_original
    assert n_lumped > n_original // 2
    
    # 4. Create and solve lumped chain
    mc = model.MarkovChain
    lumped_mc = gv.lump(mc, partition)
    lumped_mc.find_unique_stationary_distribution(solver="full_matrix_inversion")
    pi_lumped = lumped_mc.stationary_distribution
    
    # 5. Unlump the solution back to original space
    pi_unlumped = gv.unlump(pi_lumped, partition)
    
    # 6. Compare original vs unlumped distributions
    diff_l1 = float(jnp.sum(jnp.abs(pi_original - pi_unlumped)))
    
    # Verify sum is 1.0
    assert jnp.allclose(jnp.sum(pi_unlumped), 1.0, atol=1e-6)
    
    # Threshold check (requested 10^-4 or 10^-5)
    # Since this is an exact symmetry, expectation is very high accuracy
    print(f"L1 norm difference: {diff_l1:.2e}")
    assert diff_l1 < 1e-6
