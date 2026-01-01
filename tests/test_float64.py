"""Test float64 precision support using subprocess for isolation."""
import pytest
import subprocess
import os
import sys

# Path to the implementation file
IMPL_FILE = "tests/float64_impl.py"

def test_enable_float64():
    """Run float64 precision test in a subprocess to avoid affecting global JAX state."""
    env = os.environ.copy()
    
    # Ensure PYTHONPATH includes src
    if "PYTHONPATH" not in env:
        env["PYTHONPATH"] = "src"
    else:
        env["PYTHONPATH"] = f"src:{env['PYTHONPATH']}"
    
    # Disable pre-allocation and use platform allocator for the subprocess
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        
    cmd = [sys.executable, IMPL_FILE]
    
    # Run in subprocess
    result = subprocess.run(
        cmd, 
        env=env, 
        capture_output=True, 
        text=True
    )
    
    # Fail this wrapper test if the subprocess failed
    assert result.returncode == 0, f"Float64 test failed:\n{result.stdout}\n{result.stderr}"
