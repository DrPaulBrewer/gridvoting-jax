#!/bin/bash
# OSF validation test script with Docker support
# Usage:
#   ./test_docker_osf.sh [--dev|--version=vX.Y.Z] [--cpu|--gpu] [--quick|--extended] [--float64]

set -e

MODE="dev"
VERSION="latest"
CUDA_TYPE="cpu"
PRECISION="float32"
QUICK="false"
EXTENDED="false"

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
            QUICK="true"
            shift
            ;;
        --extended)
            EXTENDED="true"
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

# Run tests
echo "Running OSF Comparison Report..."

# Construct Arguments
PY_ARGS=""
if [ "$QUICK" == "true" ]; then
    PY_ARGS="--max_g 40"
fi

if [ "$EXTENDED" == "true" ]; then
    # No max_g means run all
    PY_ARGS="" 
fi

# Construct Environment Variables
ENV_VARS="-e PYTHONUNBUFFERED=1"
if [ "$PRECISION" == "float64" ]; then
    ENV_VARS="$ENV_VARS -e GV_ENABLE_FLOAT64=1"
fi

# Run Docker
docker run --rm \
    $DOCKER_ARGS \
    $ENV_VARS \
    ${CUDA_TYPE:+$([ "$CUDA_TYPE" != "cpu" ] && echo "--gpus all")} \
    "$IMAGE" \
    python3 -m gridvoting_jax.benchmarks.osf_comparison $PY_ARGS

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ OSF validation completed successfully!"
else
    echo ""
    echo "❌ OSF validation failed!"
    exit 1
fi
