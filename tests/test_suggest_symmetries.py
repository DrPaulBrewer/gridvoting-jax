import pytest
import jax.numpy as jnp
import numpy as np

pytestmark = pytest.mark.lumping
import gridvoting_jax as gv

def test_suggest_symmetries_empty():
    assert gv.suggest_symmetries([]) == []

def test_suggest_symmetries_reflection_x():
    # Points symmetric around x=0
    points = [(-1.0, 1.0), (1.0, 1.0), (-2.0, 2.0), (2.0, 2.0)]
    syms = gv.suggest_symmetries(points)
    assert 'reflect_x' in syms

def test_suggest_symmetries_reflection_y():
    # Points symmetric around y=0
    points = [(1.0, -1.0), (1.0, 1.0), (2.0, -2.0), (2.0, 2.0)]
    syms = gv.suggest_symmetries(points)
    assert 'reflect_y' in syms

def test_suggest_symmetries_reflection_xy():
    # Points symmetric around y=x
    points = [(1.0, 2.0), (2.0, 1.0), (3.0, 3.0)]
    syms = gv.suggest_symmetries(points)
    assert 'swap_xy' in syms

def test_suggest_symmetries_offset_reflection():
    # Points symmetric around x=1
    # (-0, y) <-> (2, y) -> 1 is mid
    points = [(0.0, 0.0), (2.0, 0.0)]
    syms = gv.suggest_symmetries(points)
    
    # Might return reflect_x=1
    found = False
    for s in syms:
        if isinstance(s, str) and s.startswith('reflect_x=1'):
            found = True
            break
    assert found

def test_suggest_symmetries_square_rotation():
    # Square: (1,1), (-1,1), (-1,-1), (1,-1)
    points = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    syms = gv.suggest_symmetries(points)
    
    # Should have rotation by 90 (4-fold)
    # ('rotate', 0.0, 0.0, 90.0)
    found_90 = False
    for s in syms:
        if isinstance(s, tuple) and s[0] == 'rotate':
            if abs(s[3] - 90.0) < 1e-5:
                found_90 = True
    assert found_90

def test_bjm_spatial_triangle_symmetry():
    # BJM points: (-15, -9), (0, 17), (15, -9)
    model = gv.bjm_spatial_triangle(g=20)
    points = model.voter_ideal_points
    syms = gv.suggest_symmetries(points)
    
    assert 'reflect_x' in syms
    assert 'reflect_y' not in syms
