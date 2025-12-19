#!/bin/bash
# Script to run OSF benchmarks in Docker with various configurations.
# Automatically detects GPU and runs tests for both Float32 and Float64 precisions.

set -e

# Configuration
IMAGE_CPU="ubuntu:24.04"
IMAGE_GPU="nvidia/cuda:12.3.1-base-ubuntu22.04"
CACHE_DIR="/tmp/gridvoting_osf_cache"
OSF_EXTRAS="deps" # If we had an [osf] extra, we'd use it. For now we install manually.

echo "========================================="
echo "OSF Benchmark Docker Verification"
echo "========================================="

# Create cache directory if it doesn't exist
mkdir -p "$CACHE_DIR"
echo "Data cache: $CACHE_DIR"

# 1. Detect GPU
HAS_GPU=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        echo "GPU Detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
        
        # Check if Docker supports GPUs
        echo "Checking Docker GPU support..."
        if docker run --rm --gpus all "$IMAGE_GPU" nvidia-smi &> /dev/null; then
            echo "✓ Docker GPU support confirmed."
            HAS_GPU=true
        else
            echo "⚠ WARNING: GPU detected on host, but Docker GPU support is missing."
            echo "  Error: 'could not select device driver' or similar."
            echo "  To fix, you likely need to install the NVIDIA Container Toolkit:"
            echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
            echo "  Skipping GPU tests."
            HAS_GPU=false
        fi
    else
        echo "nvidia-smi found but failed to run. Assuming CPU only."
    fi
else
    echo "nvidia-smi not found. Assuming CPU only."
fi

# Define tests: "Name|DockerFlags|EnvVars|PythonCommand"
TESTS=(
    "CPU_Float32|--rm|GV_FORCE_CPU=1|run_comparison_report()"
    "CPU_Float64|--rm|GV_FORCE_CPU=1|gv.enable_float64(); run_comparison_report()"
)

if [ "$HAS_GPU" = true ]; then
    TESTS+=(
        "GPU_Float32|--gpus all --rm| |run_comparison_report()"
        "GPU_Float64|--gpus all --rm| |gv.enable_float64(); run_comparison_report()"
    )
fi

echo "Scheduled tests:"
for test_entry in "${TESTS[@]}"; do
    IFS='|' read -r name flags env cmd <<< "$test_entry"
    echo "  - $name"
done
echo "========================================="

# 2. Run Tests
for test_entry in "${TESTS[@]}"; do
    IFS='|' read -r name flags env cmd <<< "$test_entry"
    
    echo ""
    echo "Running Test: $name"
    echo "-----------------------------------------"
    
    # Select Image and Install Command based on mode
    if [[ "$name" == *"GPU"* ]]; then
        IMAGE="$IMAGE_GPU"
        # For GPU, we need to install JAX with CUDA support
        SETUP_CMD="
            echo 'Setting up GPU environment (installing python, jax[cuda12]...)' && \
            apt-get update -qq > /dev/null && \
            apt-get install -y -qq apt-utils python3 python3-pip python3-venv git > /dev/null && \
            python3 -m venv /venv && \
            source /venv/bin/activate && \
            pip install --quiet --upgrade pip > /dev/null && \
            pip install --quiet build > /dev/null && \
            python3 -m build > /dev/null && \
            pip install --quiet dist/*.whl > /dev/null && \
            pip install --quiet pandas requests > /dev/null && \
            pip install --quiet 'jax[cuda12]' > /dev/null && \
            python3 -c 'import jax; print(\"JAX Device:\", jax.devices()[0])'
        "
    else
        IMAGE="$IMAGE_CPU"
        SETUP_CMD="
            echo 'Setting up CPU environment...' && \
            apt-get update -qq > /dev/null && \
            apt-get install -y -qq apt-utils python3 python3-pip python3-venv > /dev/null && \
            python3 -m venv /venv && \
            source /venv/bin/activate && \
            pip install --quiet --upgrade pip > /dev/null && \
            pip install --quiet build > /dev/null && \
            python3 -m build > /dev/null && \
            pip install --quiet dist/*.whl > /dev/null && \
            pip install --quiet pandas requests > /dev/null
        "
    fi
    
    # clean dist before starting to avoid conflicts
    rm -rf dist/*
    
    # Run Docker
    # We mount the current directory to /workspace and the cache dir to /tmp/gridvoting_osf_cache
    docker run $flags \
        -v "$(pwd):/workspace" \
        -v "$CACHE_DIR:$CACHE_DIR" \
        -w /workspace \
        -e PYTHONUNBUFFERED=1 \
        $IMAGE \
        /bin/bash -c "
            set -e
            $SETUP_CMD
            
            echo '========================================='
            echo 'Environment Ready. Starting Python Benchmark...'
            echo 'Command: $cmd'
            echo '========================================='
            
            # Use -u for unbuffered output
            $env /venv/bin/python3 -u -c 'import sys; print(\"Python Started\"); import gridvoting_jax as gv; from gridvoting_jax.benchmarks import run_comparison_report; print(\"Modules Imported\"); $cmd'
        "
        
    if [ $? -eq 0 ]; then
        echo "✓ $name Passed"
    else
        echo "✗ $name Failed"
        exit 1
    fi
done

echo ""
echo "========================================="
echo "All tests completed successfully!"
echo "========================================="
