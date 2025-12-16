#!/bin/bash
# Docker test script for gridvoting-jax with NumPy 2.0+

set -e

echo "========================================="
echo "Testing gridvoting-jax with NumPy 2.0+"
echo "========================================="

# Build the package
echo "Building package..."
python3 -m build

# Get the wheel file
WHEEL_FILE=$(ls dist/*.whl | head -1)
echo "Built: $WHEEL_FILE"

# Create and run Docker container
echo "Starting Docker container..."
docker run --rm -v "$(pwd)/dist:/dist" -v "$(pwd)/tests:/tests" ubuntu:24.04 /bin/bash -c "
    set -e
    echo 'Installing Python and pip...'
    apt-get update -qq
    apt-get install -y -qq python3 python3-pip python3-venv > /dev/null 2>&1
    
    echo 'Creating virtual environment...'
    python3 -m venv /venv
    source /venv/bin/activate
    
    echo 'Installing package...'
    pip install --quiet /dist/*.whl
    pip install --quiet pytest
    
    echo 'Checking NumPy version...'
    python3 -c 'import numpy; print(f\"NumPy version: {numpy.__version__}\")'
    
    echo 'Testing import...'
    python3 -c 'import gridvoting_jax as gv; print(f\"gridvoting-jax version: {gv.__version__}\"); print(f\"Device: {gv.device_type}\")'
    
    echo 'Running tests...'
    pytest /tests/ -v --tb=short
    
    echo 'Testing NO_GPU mode...'
    NO_GPU=1 python3 -c 'import gridvoting_jax as gv; assert gv.device_type == \"cpu\", \"NO_GPU mode failed\"; print(\"✓ NO_GPU mode works\")'
    
    echo '========================================='
    echo '✓ All tests passed with NumPy 2.0+!'
    echo '========================================='
"
