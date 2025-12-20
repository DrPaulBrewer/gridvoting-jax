import pytest
import chex

def test_grid_init():
    import gridvoting_jax
    import jax.numpy as jnp
    import numpy as np
    grid = gridvoting_jax.Grid(x0=-5,x1=5,y0=-7,y1=7)
    assert grid.x0 == -5
    assert grid.x1 == 5
    assert grid.xstep == 1
    assert grid.y0 == -7
    assert grid.y1 == 7
    assert grid.gshape == (15,11)
    assert grid.extent == (-5,5,-7,7)
    assert grid.len == 165
    correct_grid_x = jnp.array([
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5]
       ])
    correct_grid_y = jnp.array(
       [
       [ 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7],
       [ 6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6],
       [ 5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5],
       [ 4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4],
       [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3],
       [ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
       [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
       [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
       [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
       [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4],
       [-5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5],
       [-6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6],
       [-7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7]
       ]
    )
    assert grid.x.shape == (165,)
    assert grid.y.shape == (165,)
    assert grid.boundary[0]
    assert grid.boundary[-1]
    assert grid.boundary.shape == (165,)
    correct_boundary = jnp.array([
    [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
    [True, False, False, False, False, False, False, False, False, False, True],
    [True, False, False, False, False, False, False, False, False, False, True],
    [True, False, False, False, False, False, False, False, False, False, True],
    [True, False, False, False, False,  False, False, False, False, False, True],
    [True, False, False, False, False,  False, False, False, False, False, True],
    [True, False, False, False, False,  False, False, False, False, False, True],
    [True, False, False, False, False, False, False, False, False, False, True],
    [True, False, False, False, False, False, False, False, False, False, True],
    [True, False, False, False, False, False, False, False, False, False, True],
    [True, False, False, False, False, False, False, False, False, False, True],
    [True, False, False, False, False, False, False, False, False, False, True],
    [True, False, False, False, False, False, False, False, False, False, True],
    [True, False, False, False, False, False, False, False, False, False, True],
    [True,   True,  True,  True, True,  True,  True,  True,  True,  True,  True]
    ])
    assert jnp.array_equal(grid.boundary.reshape(grid.gshape), correct_boundary)
    assert jnp.array_equal(grid.x.reshape(grid.gshape), correct_grid_x)
    assert jnp.array_equal(grid.y.reshape(grid.gshape), correct_grid_y)
 

def test_grid_as_xy_vectors():
    import gridvoting_jax as gv
    import jax.numpy as jnp
    import numpy as np
    grid = gv.Grid(x0=-3,x1=3,y0=-5,y1=5)
    correct_vectors = jnp.array([
       [-3,  5],
       [-2,  5],
       [-1,  5],
       [ 0,  5],
       [ 1,  5],
       [ 2,  5],
       [ 3,  5],
       [-3,  4],
       [-2,  4],
       [-1,  4],
       [ 0,  4],
       [ 1,  4],
       [ 2,  4],
       [ 3,  4],
       [-3,  3],
       [-2,  3],
       [-1,  3],
       [ 0,  3],
       [ 1,  3],
       [ 2,  3],
       [ 3,  3],
       [-3,  2],
       [-2,  2],
       [-1,  2],
       [ 0,  2],
       [ 1,  2],
       [ 2,  2],
       [ 3,  2],
       [-3,  1],
       [-2,  1],
       [-1,  1],
       [ 0,  1],
       [ 1,  1],
       [ 2,  1],
       [ 3,  1],
       [-3,  0],
       [-2,  0],
       [-1,  0],
       [ 0,  0],
       [ 1,  0],
       [ 2,  0],
       [ 3,  0],
       [-3, -1],
       [-2, -1],
       [-1, -1],
       [ 0, -1],
       [ 1, -1],
       [ 2, -1],
       [ 3, -1],
       [-3, -2],
       [-2, -2],
       [-1, -2],
       [ 0, -2],
       [ 1, -2],
       [ 2, -2],
       [ 3, -2],
       [-3, -3],
       [-2, -3],
       [-1, -3],
       [ 0, -3],
       [ 1, -3],
       [ 2, -3],
       [ 3, -3],
       [-3, -4],
       [-2, -4],
       [-1, -4],
       [ 0, -4],
       [ 1, -4],
       [ 2, -4],
       [ 3, -4],
       [-3, -5],
       [-2, -5],
       [-1, -5],
       [ 0, -5],
       [ 1, -5],
       [ 2, -5],
       [ 3, -5]
    ])
    assert jnp.array_equal(
        grid.points,
        correct_vectors
    )

@pytest.mark.parametrize("x0,x1,xstep,y0,y1,ystep,correct",[
    (None,None,None,None,None,None,(10,6)),
    (0,0,None,0,0,None,(1,1)),
    (-5,5,None,-20,20,2,(21,11)),
    (1,4,None,None,None,None,(10,4))
])
def test_grid_shape(x0,x1,xstep,y0,y1,ystep,correct):
    import gridvoting_jax as gv
    grid = gv.Grid(x0=0,x1=5,y0=0,y1=9)
    assert grid.shape(x0=x0,x1=x1,xstep=xstep,y0=y0,y1=y1,ystep=ystep) == correct

def test_grid_embedding():
    import gridvoting_jax as gv
    import jax.numpy as jnp
    import numpy as np
    grid = gv.Grid(x0=-5,x1=5,y0=-7,y1=7)
    triangle = (grid.x>=0) & (grid.y>=0) & ((grid.x+grid.y)<=4)
    correct_triangle = jnp.array([
    [False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False,  True, False, False, False, False, False],
    [False, False, False, False, False,  True,  True, False, False, False, False],
    [False, False, False, False, False,  True,  True,  True, False, False, False],
    [False, False, False, False, False,  True,  True,  True,  True, False, False],
    [False, False, False, False, False,  True,  True,  True,  True,  True, False],
    [False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False, False]
    ])
    assert jnp.array_equal(triangle.reshape(grid.gshape),correct_triangle)
    assert 15 == triangle.sum()
    triangle_points_xy = grid.points[triangle]
    correct_triangle_points_xy = jnp.array([
        [0,4],
        [0,3],
        [1,3],
        [0,2],
        [1,2],
        [2,2],
        [0,1],
        [1,1],
        [2,1],
        [3,1],
        [0,0],
        [1,0],
        [2,0],
        [3,0],
        [4,0]
    ])
    assert jnp.array_equal(
        triangle_points_xy,correct_triangle_points_xy)
    emfunc = grid.embedding(valid=triangle)
    triangle_x = grid.x[triangle]
    assert jnp.array_equal(
        triangle_x,
        jnp.array([0,0,1,0,1,2,0,1,2,3,0,1,2,3,4])
    )
    correct_embedding_result = jnp.array([
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 2., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 2., 3., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 2., 3., 4., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    ])
    assert jnp.array_equal(
        emfunc(triangle_x, fill=0.0).reshape(grid.gshape),
        correct_embedding_result
    )

def test_grid_within_box():
    import gridvoting_jax as gv
    import jax.numpy as jnp
    import numpy as np
    grid = gv.Grid(x0=0,y0=0,x1=100,y1=100)
    box = grid.within_box(x0=20,y0=20,x1=30,y1=40)
    assert box.shape == (grid.len,)
    assert box.sum() == 21*11
    assert grid.x[box].shape == (21*11,)
    equiv_grid = gv.Grid(x0=20,y0=20,x1=30,y1=40)
    assert jnp.array_equal(grid.x[box],equiv_grid.x)
    assert jnp.array_equal(grid.y[box],equiv_grid.y)

def test_grid_within_disk():
    import gridvoting_jax as gv
    import jax.numpy as jnp
    import numpy as np
    grid = gv.Grid(x0=0,y0=0,x1=100,y1=100)
    disk = grid.within_disk(x0=30,y0=40,r=50)
    assert grid.x[disk][0] == 30
    assert grid.y[disk][0] == 90
    assert grid.y[disk][-1] == 0
    correct_disk = ((grid.x-30)*(grid.x-30)+(grid.y-40)*(grid.y-40)<=(50*50))
    assert jnp.array_equal(disk,correct_disk)


def test_grid_within_triangle_many_right_triangles():
    import gridvoting_jax as gv
    import jax.numpy as jnp
    import numpy as np
    grid = gv.Grid(x0=0,y0=0,x1=100,y1=100)
    for cx in range(0,80,5):
        for cy in range(0,80,5):
            ax = cx
            by = cy
            for ay in range(cy+1,100,7):
                for bx in range(cx+1,100,7):
                    points = np.array([
                        [ax,ay],
                        [bx,by],
                        [cx,cy]
                    ])
                    slope = (0.0+by-ay)/(0.0+bx-ax)
                    assert slope<0.0
                    # add slight bias to account for floating-point roundoff error
                    correct_triangle = (grid.x>=cx) & (grid.y>=cy) & (grid.y<=(1e-5+ay+slope*(grid.x-ax)))
                    calc_triangle_A = grid.within_triangle(points=points)
                    altpoints = jnp.array([
                        [cx,cy],
                        [ax,ay],
                        [bx,by]
                    ])
                    calc_triangle_B = grid.within_triangle(points=altpoints)
                    unusual = (calc_triangle_A != correct_triangle) | (calc_triangle_B != correct_triangle)
                    if unusual.sum()>0:
                        disagree = grid.points[unusual]
                        raise Exception("input={points},disagreement={disagree},count={count}".format(points=points,disagree=disagree,count=unusual.sum()))
                    
def test_grid_spatial_utility():
    # this also tests gridvoting github issue #10
    import gridvoting_jax as gv
    import numpy as np
    grid = gv.Grid(x0=-5,x1=5,y0=-7,y1=7)
    assert grid.gshape == (15,11)
    u = grid.spatial_utilities(voter_ideal_points=[[0,1]]).reshape(grid.gshape)
# this reshaped output is correct, the dependence on the squared distance from the ideal point is clear
    correct_u = np.array([
           [-61., -52., -45., -40., -37., -36., -37., -40., -45., -52., -61.],
           [-50., -41., -34., -29., -26., -25., -26., -29., -34., -41., -50.],
           [-41., -32., -25., -20., -17., -16., -17., -20., -25., -32., -41.],
           [-34., -25., -18., -13., -10.,  -9., -10., -13., -18., -25., -34.],
           [-29., -20., -13.,  -8.,  -5.,  -4.,  -5.,  -8., -13., -20., -29.],
           [-26., -17., -10.,  -5.,  -2.,  -1.,  -2.,  -5., -10., -17., -26.],
           [-25., -16.,  -9.,  -4.,  -1.,  -0.,  -1.,  -4.,  -9., -16., -25.],
           [-26., -17., -10.,  -5.,  -2.,  -1.,  -2.,  -5., -10., -17., -26.],
           [-29., -20., -13.,  -8.,  -5.,  -4.,  -5.,  -8., -13., -20., -29.],
           [-34., -25., -18., -13., -10.,  -9., -10., -13., -18., -25., -34.],
           [-41., -32., -25., -20., -17., -16., -17., -20., -25., -32., -41.],
           [-50., -41., -34., -29., -26., -25., -26., -29., -34., -41., -50.],
           [-61., -52., -45., -40., -37., -36., -37., -40., -45., -52., -61.],
           [-74., -65., -58., -53., -50., -49., -50., -53., -58., -65., -74.],
           [-89., -80., -73., -68., -65., -64., -65., -68., -73., -80., -89.]])
    chex.assert_trees_all_equal(u, correct_u)


