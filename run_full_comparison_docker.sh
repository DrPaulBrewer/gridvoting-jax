#!/bin/bash
# Run full OSF comparison in Docker (Float32 and Float64)

set -e

echo "Starting Docker container for Full Benchmark..."
docker run --rm -v "$(pwd):/workspace" -w /workspace ubuntu:24.04 /bin/bash -c "
    set -e
    apt-get update -qq
    apt-get install -y -qq python3 python3-pip python3-venv > /dev/null 2>&1
    
    python3 -m venv /venv
    source /venv/bin/activate
    
    pip install --quiet build
    
    # Pre-install dependencies
    pip install --quiet pandas requests
    
    # Build and install package
    python3 -m build
    pip install --quiet dist/*.whl
    
    echo '=================================================='
    echo 'Running Benchmark (Float32)'
    echo '=================================================='
    python3 -m gridvoting_jax.benchmarks.osf_comparison
    
    echo ''
    echo '=================================================='
    echo 'Running Benchmark (Float64)'
    echo '=================================================='
    export GV_ENABLE_FLOAT64=1
    python3 -m gridvoting_jax.benchmarks.osf_comparison
"
