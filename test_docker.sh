#!/bin/bash
# Docker test script for gridvoting-jax with NumPy 2.0+

set -e

echo "========================================="
echo "Testing gridvoting-jax with NumPy 2.0+"
echo "========================================="

# Create and run Docker container
# Source mounted read-only to prevent artifact pollution
echo "Starting Docker container..."
docker run --rm -v "$(pwd):/source:ro" ubuntu:24.04 /bin/bash -c "
    set -e
    echo 'Copying source to container workspace...'
    cp -r /source /workspace
    cd /workspace
    
    echo 'Installing Python and pip...'
    apt-get update -qq
    apt-get install -y -qq python3 python3-pip python3-venv > /dev/null 2>&1
    
    echo 'Creating virtual environment...'
    python3 -m venv /venv
    source /venv/bin/activate
    
    echo 'Installing build tools...'
    pip install --quiet build
    
    echo 'Building package...'
    python3 -m build
    
    echo 'Installing package...'
    pip install --quiet dist/*.whl
    pip install --quiet pytest
    
    echo 'Checking NumPy version...'
    python3 -c 'import numpy; print(f\"NumPy version: {numpy.__version__}\")'
    
    echo 'Testing import...'
    python3 -c 'import gridvoting_jax as gv; print(f\"gridvoting-jax version: {gv.__version__}\"); print(f\"Device: {gv.device_type}\")'
    
    echo 'Running tests...'
    pytest tests/ -v --tb=short
    
    echo 'Testing GV_FORCE_CPU mode...'
    GV_FORCE_CPU=1 python3 -c 'import gridvoting_jax as gv; assert gv.device_type == \"cpu\", \"GV_FORCE_CPU mode failed\"; print(\"✓ GV_FORCE_CPU mode works\")'
    
    echo '========================================='
    echo '✓ All tests passed with NumPy 2.0+!'
    echo '========================================='
"
