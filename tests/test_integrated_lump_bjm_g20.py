import jax.numpy as jnp
import gridvoting_jax as gv
import pytest


def _test_bjm_g20_reflection_lumpability(model):
    """Common test logic for BJM g=20 reflection symmetry lumpability.
    
    Tests that:
    1. Solving full model
    2. Lumping using reflection symmetry
    3. Unlumping the solution
    4. Verifying it matches the original solution
    5. Doing steps 2-4 with the partitions= parameter
    
    Args:
        model: BMJ g=20 model (either lazy or dense)
    """
    n_original = model.grid.len
    
    # Solve original chain using full matrix inversion
    model.analyze(solver="full_matrix_inversion")
    pi_original = model.stationary_distribution
    
    # Generate partition using reflection around x=0
    partition = model.get_spatial_symmetry_partition(['reflect_x'])
    n_lumped = int(partition.max()) + 1
    
    # Verify reduction (should be nearly half)
    assert n_lumped < n_original
    assert n_lumped > n_original // 2
    
    # Create and solve lumped chain
    mc = model.MarkovChain
    lumped_mc = gv.lump(mc, partition)
    lumped_mc.solve(solver="full_matrix_inversion")
    pi_lumped = lumped_mc.stationary_distribution
    
    # Unlump the solution back to original space
    pi_unlumped = gv.unlump(pi_lumped, partition)
    
    # Compare original vs unlumped distributions
    diff_l1 = float(jnp.linalg.norm(pi_original - pi_unlumped, ord=1))
    
    # Verify sum is 1.0
    assert jnp.allclose(jnp.sum(pi_unlumped), 1.0, atol=1e-6)
    
    # Threshold check - exact symmetry should have very high accuracy
    print(f"L1 norm difference: {diff_l1:.2e}")
    assert diff_l1 < 1e-5
    
    # Do steps 2-4 with partitions= parameter
    mc.solve(solver="full_matrix_inversion", partitions=partition)
    new_pi_unlumped = mc.stationary_distribution
    diff_l1 = float(jnp.linalg.norm(pi_original - new_pi_unlumped, ord=1))
    assert jnp.allclose(jnp.sum(new_pi_unlumped), 1.0, atol=1e-6)
    print(f"L1 norm difference: {diff_l1:.2e}")
    assert diff_l1 < 1e-5


def test_bjm_g20_reflection_lumpability_lazy(bmj_g20_mi):
    """Test reflection symmetry with lazy transition matrix."""
    _test_bjm_g20_reflection_lumpability(bmj_g20_mi)


def test_bjm_g20_reflection_lumpability_dense(bmj_g20_mi):
    """Test reflection symmetry with dense transition matrix."""
    # create a deep copy
    from copy import deepcopy
    from gridvoting_jax.stochastic.markov import MarkovChain
    bmj_g20_mi_dense = deepcopy(bmj_g20_mi)
    P = bmj_g20_mi_dense.model.MarkovChain.dense_P()
    bmj_g20_mi_dense.model.MarkovChain = MarkovChain(P=P)
    _test_bjm_g20_reflection_lumpability(bmj_g20_mi_dense)
