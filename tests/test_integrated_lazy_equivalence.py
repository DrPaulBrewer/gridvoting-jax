import pytest
import subprocess
import os
import sys


# Define path to the implementation file
IMPL_FILE = "tests/lazy_equivalence_impl.py"

def run_precision_test(enable_float64: bool):
    """Run the implementation tests in a subprocess with specific precision.
    
    This runs in a separate process to ensure complete isolation between
    float32 and float64 test runs, preventing any JAX state contamination.
    The tests are marked to run sequentially to avoid parallel execution.
    """
    env = os.environ.copy()
    env["GV_ENABLE_FLOAT64"] = "1" if enable_float64 else "0"
    
    # Ensure PYTHONPATH includes src
    if "PYTHONPATH" not in env:
        env["PYTHONPATH"] = "src"
    else:
        env["PYTHONPATH"] = f"src:{env['PYTHONPATH']}"
        
    cmd = [sys.executable, "-m", "pytest", IMPL_FILE, "-v"]
    
    # Run pytest in subprocess
    result = subprocess.run(
        cmd, 
        env=env, 
        capture_output=True, 
        text=True
    )
    
    # Fail this wrapper test if the subprocess failed
    if result.returncode != 0:
        precision = "Float64" if enable_float64 else "Float32"
        pytest.fail(f"{precision} tests failed:\n{result.stdout}\n{result.stderr}")

def test_lazy_equivalence_float32():
    """Run lazy equivalence tests in Float32 (default) mode.
    
    Runs in isolated subprocess. Sequential execution is guaranteed by pytest configuration.
    """
    run_precision_test(enable_float64=False)

def test_lazy_equivalence_float64():
    """Run lazy equivalence tests in Float64 mode.
    
    Runs in isolated subprocess. Sequential execution is guaranteed by pytest configuration.
    """
    run_precision_test(enable_float64=True)

