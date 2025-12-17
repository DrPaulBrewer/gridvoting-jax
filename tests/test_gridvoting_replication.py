import pytest

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
    np = gv.np
    g = params['g']
    zi = params['zi']
    majority = 2
    grid = gv.Grid(x0=-g,x1=g,y0=-g,y1=g)
    number_of_alternatives = (2*g+1)*(2*g+1)
    assert len(grid.x) == number_of_alternatives
    assert len(grid.y) == number_of_alternatives
    voter_ideal_points = [
        [-15,-9],
        [0,17],
        [15,-9]
    ]
    number_of_voters = 3
    u = grid.spatial_utilities(
        voter_ideal_points=voter_ideal_points,
        metric='sqeuclidean'
    )
    assert u.shape == (number_of_voters, number_of_alternatives)
    vm = gv.VotingModel(
        utility_functions=u,
        majority=majority,
        zi=zi,
        number_of_voters=number_of_voters,
        number_of_feasible_alternatives=number_of_alternatives
    )
    vm.analyze()
    stat_dist = vm.stationary_distribution
    p_boundary = stat_dist[grid.boundary].sum()
    assert p_boundary == pytest.approx(correct['p_boundary'], rel=0.05)
    triangle_of_voter_ideal_points = grid.within_triangle(points=voter_ideal_points)
    p_voter_ideal_point_triangle = stat_dist[triangle_of_voter_ideal_points].sum()
    assert p_voter_ideal_point_triangle == pytest.approx(correct['p_voter_ideal_point_triangle'], rel=0.05)
    diagnostic_metrics = vm.MarkovChain.diagnostic_metrics()
    assert diagnostic_metrics['||F||'] == number_of_alternatives
    assert diagnostic_metrics['(𝝨𝝿)-1'] == pytest.approx(0.0,abs=5e-5)
    assert diagnostic_metrics['||𝝿P-𝝿||_L1_norm'] < 5e-5
    summary = vm.summarize_in_context(grid=grid)
    assert summary['entropy_bits'] == pytest.approx(correct['entropy'],abs=0.01)
    np.testing.assert_array_almost_equal(
        summary['point_mean'],
        np.array(correct['mean']),
        decimal=3
    )
    np.testing.assert_array_equal(summary['prob_max_points'],[[0,-1]])


@pytest.mark.parametrize("params,correct",[
    ({'g':20,'zi':False,'voters':[[0,0],[1,0],[2,0],[3,0],[4,0]]}, {'core_points':[[2,0]]}), 
    ({'g':20,'zi':True, 'voters':[[0,0],[0,1],[0,2],[0,3],[0,4]]}, {'core_points':[[0,2]]}),
    ({'g':20,'zi':False,'voters':[[-2,-2],[-1,-1],[0,0],[1,1],[2,2]]}, {'core_points':[[0,0]]}),
    ({'g':20,'zi':True,'voters':[[-10,-10],[-10,10],[10,-10],[10,10],[0,0]]}, {'core_points':[[0,0]]})
])
def test_replicate_core_Plott_theorem_example(params,correct):
    import gridvoting_jax as gv
    np = gv.np
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
    np.testing.assert_array_equal(summary['core_points'],np.array(correct['core_points']))

        
