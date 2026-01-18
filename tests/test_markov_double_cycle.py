import pytest
import jax.numpy as jnp
import gridvoting_jax as gv
from gridvoting_jax.core.constants import DTYPE_FLOAT

@pytest.fixture
def double_cycle_mc():
  double_cycle_P = jnp.array([
    [1/2,1/2,0,0,0,0],
    [0,1/2,1/2,0,0,0],
    [1/2,0,1/2,0,0,0],
    [0,0,0,1/2,1/2,0],
    [0,0,0,0,1/2,1/2],
    [0,0,0,1/2,0,1/2]
  ], dtype=DTYPE_FLOAT)
  mc = gv.MarkovChain(P=double_cycle_P)
  return mc
  
  
def test_gridvoting_doublecycle_full_matrix_inversion_fails(double_cycle_mc):
# This will fail because full matrix inversion is not possible for this matrix
  with pytest.raises(RuntimeError) as e_info:
    double_cycle_mc.solve(solver="full_matrix_inversion")

      
def test_gridvoting_doublecycle_power_method(double_cycle_mc):
# This will succeed because power method is not sensitive to the double cycle
  result = double_cycle_mc.solve(solver="power_method")
  l1_vs_uniform = jnp.linalg.norm(result - jnp.ones(6)/6.0, ord=1)
  assert l1_vs_uniform < 1e-6

def test_gridvoting_doublecycle_bifurcated_power_method(double_cycle_mc):
# this will fail because the two initial guesses will converge toward different distributions
  with pytest.raises(RuntimeError) as e_info:
    double_cycle_mc.solve(solver="bifurcated_power_method") 
  
