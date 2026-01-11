import pytest
import jax.numpy as jnp
import chex
import random
def test_topcycle():
  import gridvoting_jax as gv
  from itertools import permutations
  TOLERANCE = 1e-6
  solvers = ["full_matrix_inversion", "power_method", "bifurcated_power_method"]
  perms = list(permutations(range(6)))
  n_perms = len(perms)
  assert n_perms == 720
  for perm in random.sample(perms, 10):
    aperm = jnp.array(perm)
    u = jnp.array([
      [1000,900,800,20,10,1],
      [800,1000,900,1,20,10],
      [900,800,1000,10,1,20]
    ])[:,aperm]
    correct_stationary_distribution = jnp.array([1/3,1/3,1/3,0.,0.,0.])[aperm]
    for solver in solvers:
      vm = gv.VotingModel(utility_functions=u,number_of_feasible_alternatives=6,number_of_voters=3,majority=2,zi=False)
      vm.analyze(solver=solver, time_per_digit=0.25)
      chex.assert_trees_all_close(
        vm.stationary_distribution,
        correct_stationary_distribution,
        atol=1e-4
      )
      # Check that lower cycle probabilities are effectively zero
      zero_mask = correct_stationary_distribution==0.0
      lower_cycle_sum = vm.stationary_distribution[zero_mask].sum()
      if lower_cycle_sum > TOLERANCE:
        raise RuntimeError(f"lower cycle still significant: solver={solver}, sum={lower_cycle_sum} (tolerance={TOLERANCE})")
