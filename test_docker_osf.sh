#!/bin/bash
# OSF validation test script with Docker support
# Usage:
#   ./test_docker_osf.sh [--dev|--version=vX.Y.Z] [--cpu|--gpu] [--quick|--extended] [--float64]

set -e

MODE="dev"
VERSION="latest"
CUDA_TYPE="cpu"
GRID_SIZES="20 40 60 80"
PRECISION="float32"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            MODE="dev"
            shift
            ;;
        --version=*)
            MODE="release"
            VERSION="${1#*=}"
            shift
            ;;
        --cpu)
            CUDA_TYPE="cpu"
            shift
            ;;
        --gpu)
            if command -v nvidia-smi &> /dev/null; then
                CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1)
                if [[ "$CUDA_VER" == "12" ]]; then
                    CUDA_TYPE="cuda12"
                elif [[ "$CUDA_VER" == "13" ]]; then
                    CUDA_TYPE="cuda13"
                else
                    CUDA_TYPE="cuda12"
                fi
            else
                echo "Error: --gpu specified but nvidia-smi not found"
                exit 1
            fi
            shift
            ;;
        --quick)
            GRID_SIZES="20 40"
            shift
            ;;
        --extended)
            GRID_SIZES="20 40 60 80 100"
            shift
            ;;
        --float64)
            PRECISION="float64"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "========================================="
echo "OSF Validation - Docker"
echo "========================================="
echo "Mode: $MODE"
echo "CUDA: $CUDA_TYPE"
echo "Grid sizes: $GRID_SIZES"
echo "Precision: $PRECISION"
echo ""

# Determine image
REGISTRY="ghcr.io/drpaulbrewer/gridvoting-jax"

if [ "$MODE" == "dev" ]; then
    IMAGE="${REGISTRY}/dev/${CUDA_TYPE}:latest"
    echo "Pulling dev image: $IMAGE"
    docker pull "$IMAGE"
    
    DOCKER_ARGS="-v $(pwd):/workspace"
else
    IMAGE="${REGISTRY}/${CUDA_TYPE}:${VERSION}"
    echo "Pulling release image: $IMAGE"
    docker pull "$IMAGE"
    
    DOCKER_ARGS=""
fi

# Add GPU support if needed
if [ "$CUDA_TYPE" != "cpu" ]; then
    DOCKER_ARGS="$DOCKER_ARGS --gpus all"
fi

# Run tests for each grid size
for g in $GRID_SIZES; do
    echo ""
    echo "Testing g=$g..."
    docker run --rm $DOCKER_ARGS "$IMAGE" \
        python3 -m pytest tests/test_osf_validation_g80_g100.py \
        -k "g${g}" -v --tb=short
done

echo ""
echo "✅ All OSF validation tests passed!"
