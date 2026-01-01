"""Implementation of float64 precision test, meant to be run in a subprocess."""
import jax.numpy as jnp
import gridvoting_jax as gv

def test_enable_float64_impl():
    # Enable float64
    gv.enable_float64()
    
    # Test precision with sum that should equal 1.0
    vec = jnp.full(101, 1/101)
    total = jnp.sum(vec)
    diff = abs(total - 1.0)
    
    # With float64, difference should be very small (< 1e-10)
    assert diff < 1e-10, f"Float64 precision not enabled: diff={diff}"
    assert vec.dtype == jnp.float64, f"Expected float64, got {vec.dtype}"

if __name__ == "__main__":
    test_enable_float64_impl()
    print("Float64 implementation test passed!")
