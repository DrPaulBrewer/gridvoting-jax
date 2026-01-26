import pytest
import gridvoting_jax as gv
import jax.numpy as jnp

def test_polargrid_as_rings():
    pg = gv.geometry.PolarGrid(radius=10, thetastep=30)
    c, M = pg.as_rings(z=jnp.ones(pg.len))
    assert c == 1.0
    assert jnp.allclose(M, jnp.ones((pg.n_rvals-1, pg.n_thetavals)))
    c, M = pg.as_rings(z=jnp.zeros(pg.len))
    assert c == 0.0
    assert jnp.allclose(M, jnp.zeros((pg.n_rvals-1, pg.n_thetavals)))
    c, M = pg.as_rings(z=pg.r)
    assert c == 0.0
    assert jnp.allclose(M, jnp.outer(pg.rvals[1:], jnp.ones(pg.n_thetavals)))
    c, M = pg.as_rings(z=pg.theta_deg)
    assert c==0.0
    assert jnp.allclose(M, jnp.outer(jnp.ones(pg.n_rvals-1), pg.thetavals))


def test_polargrid_ring_sums():
    pg = gv.geometry.PolarGrid(radius=10, thetastep=30)
    assert jnp.allclose(pg.ring_sums(z=jnp.ones(pg.len)), jnp.concat((jnp.array([1.0]), pg.n_thetavals * jnp.ones(pg.n_rvals-1))))
    assert jnp.allclose(pg.ring_sums(z=jnp.zeros(pg.len)), jnp.zeros(pg.n_rvals))
    assert jnp.allclose(pg.ring_sums(z=pg.r), pg.rvals*pg.n_thetavals)
    assert jnp.allclose(pg.ring_sums(z=pg.points[:,0]), jnp.zeros(pg.n_rvals), atol=20.0*jnp.finfo(jnp.float32).eps)
    assert jnp.allclose(pg.ring_sums(z=pg.points[:,1]), jnp.zeros(pg.n_rvals), atol=20.0*jnp.finfo(jnp.float32).eps)

def test_polargrid_theta_sums():
    pg = gv.geometry.PolarGrid(radius=10, thetastep=30)
    pg.unit_vectors = pg.points[pg.r==1.0,:]
    assert jnp.allclose(pg.theta_sums(z=jnp.ones(pg.len)), 10.0*jnp.ones(pg.n_thetavals))
    assert jnp.allclose(pg.theta_sums(z=jnp.zeros(pg.len)), jnp.zeros(pg.n_thetavals))
    assert jnp.allclose(pg.theta_sums(z=pg.r), 55.0*jnp.ones(pg.n_thetavals))
    assert jnp.allclose(pg.theta_sums(z=pg.points[:,0]), 55.0*pg.unit_vectors[:,0])
    assert jnp.allclose(pg.theta_sums(z=pg.points[:,1]), 55.0*pg.unit_vectors[:,1])
