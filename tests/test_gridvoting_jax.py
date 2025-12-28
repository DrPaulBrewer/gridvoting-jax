import pytest
import jax.numpy as jnp

pytestmark = pytest.mark.essential

def test_module():
    import gridvoting_jax as gv
    import jax
    
    # Test device detection
    devices = jax.devices()
    assert len(devices) > 0
    
    # Test that device_type is set
    assert gv.device_type in ['cpu', 'gpu', 'tpu']
    
    # Test use_accelerator flag
    assert isinstance(gv.use_accelerator, bool)
    
    print(f"JAX device: {gv.device_type}")
    print(f"Accelerator: {gv.use_accelerator}")
    print(f"Devices: {devices}")
