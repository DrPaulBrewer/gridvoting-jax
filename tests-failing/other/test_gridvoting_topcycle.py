import pytest
import chex
def test_gridvoting_topcycle():
  import gridvoting_jax as gv
  from itertools import permutations
  import numpy as np
  # Tolerance for numerical precision (handles NumPy 2.0 differences)
  TOLERANCE = 1e-7
  for perm in permutations(np.arange(6)):
    aperm = np.array(perm)
    u = np.array([
      [1000,900,800,20,10,1],
      [800,1000,900,1,20,10],
      [900,800,1000,10,1,20]
    ])[:,aperm]
    correct_stationary_distribution = np.array([1/3,1/3,1/3,0.,0.,0.])[aperm]
    vm = gv.VotingModel(utility_functions=u,number_of_feasible_alternatives=6,number_of_voters=3,majority=2,zi=False)
    vm.analyze()
    chex.assert_trees_all_close(
      vm.stationary_distribution,
      correct_stationary_distribution,
      atol=1e-6,
      rtol=0
    )
    # Check that lower cycle probabilities are effectively zero (with tolerance for NumPy 2.0)
    zero_mask = correct_stationary_distribution==0.0
    lower_cycle_sum = vm.stationary_distribution[zero_mask].sum()
    if lower_cycle_sum > TOLERANCE:
      raise RuntimeError(f"lower cycle still positive: sum={lower_cycle_sum} (tolerance={TOLERANCE})")
