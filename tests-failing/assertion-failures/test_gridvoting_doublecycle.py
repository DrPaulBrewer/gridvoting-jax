import pytest


@pytest.fixture
def double_cycle_mc():
  import gridvoting_jax as gv
  import jax.numpy as jnp
  double_cycle_P = jnp.array([
    [1/2,1/2,0,0,0,0],
    [0,1/2,1/2,0,0,0],
    [1/2,0,1/2,0,0,0],
    [0,0,0,1/2,1/2,0],
    [0,0,0,0,1/2,1/2],
    [0,0,0,1/2,0,1/2]
  ])
  mc = gv.MarkovChain(P=double_cycle_P)
  return mc
  
  
def test_gridvoting_doublecycle_algebra(double_cycle_mc):
  with pytest.raises(RuntimeError) as e_info:
    double_cycle_mc.solve_for_unit_eigenvector()
      
