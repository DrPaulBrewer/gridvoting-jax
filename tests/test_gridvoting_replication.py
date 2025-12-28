import pytest
import chex
import jax.numpy as jnp

pytestmark = pytest.mark.benchmarks

# attempt to replicate grid boundary probability and entropy (H) from 
# Brewer, Juybari, Moberly (2023), J. Econ Interact Coord, Tab.4-5
# https://link.springer.com/article/10.1007/s11403-023-00387-8/tables/4
# https://link.springer.com/article/10.1007/s11403-023-00387-8/tables/5
# grid size 20 only active for testing
# grid size 40 is commented out because of low RAM on github actions but can be tested manually by removing '#'

@pytest.mark.parametrize("params,correct", [
    ({'g':20,'zi':False}, {'p_boundary': 0.024, 'p_voter_ideal_point_triangle': 0.458, 'entropy': 10.32, 'mean': [0,-0.1452]}),
    ({'g':20,'zi':True},  {'p_boundary': 0.0086,'p_voter_ideal_point_triangle': 0.68, 'entropy':  9.68, 'mean': [0,-0.2937]}),
#   ({'g':40,'zi':False}, {'p_boundary': 0.000254, 'p_voter_ideal_point_triangle':0.396, 'entropy': 10.92, 'mean': [0,-0.3373]}),
#   ({'g':40,'zi':True},  {'p_boundary': 2.55e-05, 'p_voter_ideal_point_triangle':0.675, 'entropy': 9.82, 'mean': [0,-0.3428]})
])
def test_replicate_spatial_voting_analysis(params, correct):
    import gridvoting_jax as gv
    import numpy as np
    g = params['g']
    zi = params['zi']
    vm = gv.models.examples.bjm_spatial.bjm_spatial_triangle(g=g, zi=zi)
    assert len(vm.grid.x) == vm.number_of_feasible_alternatives
    assert len(vm.grid.y) == vm.number_of_feasible_alternatives
    vm.analyze()
    p_boundary = vm.stationary_distribution[vm.grid.boundary].sum()
    assert p_boundary == pytest.approx(correct['p_boundary'], rel=0.05)
    triangle_of_voter_ideal_points = vm.grid.within_triangle(points=vm.voter_ideal_points)
    p_voter_ideal_point_triangle = vm.stationary_distribution[triangle_of_voter_ideal_points].sum()
    assert p_voter_ideal_point_triangle == pytest.approx(correct['p_voter_ideal_point_triangle'], rel=0.05)
    diagnostic_metrics = vm.MarkovChain.diagnostic_metrics()
    assert diagnostic_metrics['||F||'] == vm.number_of_feasible_alternatives
    assert diagnostic_metrics['(𝝨𝝿)-1'] == pytest.approx(0.0,abs=5e-5)
    assert diagnostic_metrics['||𝝿P-𝝿||_L1_norm'] < 5e-5
    summary = vm.summarize_in_context(grid=vm.grid)
    assert summary['entropy_bits'] == pytest.approx(correct['entropy'],abs=0.01)
    chex.assert_trees_all_close(
        summary['point_mean'],
        jnp.array(correct['mean']),
        atol=1e-3,
        rtol=0
    )
    chex.assert_trees_all_equal(summary['prob_max_points'], jnp.array([[0,-1]]))


@pytest.mark.parametrize("params,correct",[
    ({'g':20,'zi':False,'voters':[[0,0],[1,0],[2,0],[3,0],[4,0]]}, {'core_points':[[2,0]]}), 
    ({'g':20,'zi':True, 'voters':[[0,0],[0,1],[0,2],[0,3],[0,4]]}, {'core_points':[[0,2]]}),
    ({'g':20,'zi':False,'voters':[[-2,-2],[-1,-1],[0,0],[1,1],[2,2]]}, {'core_points':[[0,0]]}),
    ({'g':20,'zi':True,'voters':[[-10,-10],[-10,10],[10,-10],[10,10],[0,0]]}, {'core_points':[[0,0]]})
])
def test_replicate_core_Plott_theorem_example(params,correct):
    import gridvoting_jax as gv
    import numpy as np
    g = params['g']
    zi = params['zi']
    grid = gv.Grid(x0=-g,x1=g,y0=-g,y1=g)
    u = grid.spatial_utilities(
        voter_ideal_points=params['voters'],
        metric='sqeuclidean'
    )
    vm = gv.VotingModel(
        utility_functions=u,
        majority=3,
        number_of_voters=5,
        number_of_feasible_alternatives=grid.len,
        zi=zi
    )
    vm.analyze()
    summary = vm.summarize_in_context(grid=grid)
    chex.assert_trees_all_equal(summary['core_points'], jnp.array(correct['core_points']))

        
