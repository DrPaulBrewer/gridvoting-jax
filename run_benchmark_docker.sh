#!/bin/bash
# Run benchmark in Docker

set -e

echo "Starting Docker container for Benchmark..."
docker run --rm -v "$(pwd):/workspace" -w /workspace ubuntu:24.04 /bin/bash -c "
    set -e
    apt-get update -qq
    apt-get install -y -qq python3 python3-pip python3-venv > /dev/null 2>&1
    
    python3 -m venv /venv
    source /venv/bin/activate
    
    pip install --quiet build
    
    # Pre-install dependencies for benchmark
    pip install --quiet pandas requests
    
    # Build and install package
    python3 -m build
    pip install --quiet dist/*.whl
    
    echo 'Running Benchmark...'
    python3 benchmark_solvers_vs_osf.py
"
