#!/bin/bash
# Script to run OSF benchmarks using pre-built Docker images.
# Replaces test_docker_osf.sh with reduced overhead by skipping environment setup.

set -e

# Configuration
REPO="ghcr.io/drpaulbrewer/gridvoting-jax"
IMAGE_CPU="${REPO}-cpu:latest"
IMAGE_GPU="${REPO}-all:latest"

echo "========================================="
echo "OSF Benchmark Docker Verification (Fast)"
echo "========================================="

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
            echo "  Skipping GPU tests."
            HAS_GPU=false
        fi
    else
        echo "nvidia-smi found but failed to run. Assuming CPU only."
    fi
else
    echo "nvidia-smi not found. Assuming CPU only."
fi

# Define tests: "Name|DockerFlags|Image|EnvVars|PythonCommand"
TESTS=(
    "CPU_Float32|--rm|$IMAGE_CPU| |run_comparison_report()"
    "CPU_Float64|--rm|$IMAGE_CPU| |gv.enable_float64(); run_comparison_report()"
)

if [ "$HAS_GPU" = true ]; then
    TESTS+=(
        "GPU_Float32|--gpus all --rm|$IMAGE_GPU| |run_comparison_report()"
        "GPU_Float64|--gpus all --rm|$IMAGE_GPU| |gv.enable_float64(); run_comparison_report()"
    )
fi

echo "Scheduled tests:"
for test_entry in "${TESTS[@]}"; do
    IFS='|' read -r name flags image env cmd <<< "$test_entry"
    echo "  - $name ($image)"
done
echo "========================================="

# 2. Run Tests
for test_entry in "${TESTS[@]}"; do
    IFS='|' read -r name flags image env cmd <<< "$test_entry"
    
    echo ""
    echo "Running Test: $name"
    echo "-----------------------------------------"
    
    # Run Docker using pre-built image
    docker run $flags \
        -e PYTHONUNBUFFERED=1 \
        $env \
        $image \
        python3 -u -c "import sys; print(\"Python Started\"); import gridvoting_jax as gv; from gridvoting_jax.benchmarks import run_comparison_report; print(\"Modules Imported\"); $cmd"
        
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
