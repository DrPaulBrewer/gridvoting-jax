#!/bin/bash
# Docker test script with support for dev and versioned images from GHCR
# Usage:
#   ./test_docker.sh [--dev|--version=vX.Y.Z] [--cpu|--gpu] [--command="..."] [pytest args...]
#   
# Examples:
#   ./test_docker.sh --dev --gpu tests/
#   ./test_docker.sh --version=v0.9.1 --cpu
#   ./test_docker.sh --command="pip list"
#   ./test_docker.sh --dev  # defaults to CPU

set -e

MODE="dev"  # dev or release
VERSION="latest"
CUDA_TYPE="cpu"
PYTEST_ARGS="tests/"
COMMAND=""

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
            # Auto-detect CUDA version
            if command -v nvidia-smi &> /dev/null; then
                CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1)
                if [[ "$CUDA_VER" == "12" ]]; then
                    CUDA_TYPE="cuda12"
                elif [[ "$CUDA_VER" == "13" ]]; then
                    CUDA_TYPE="cuda13"
                else
                    echo "Warning: Unknown CUDA version $CUDA_VER, defaulting to cuda12"
                    CUDA_TYPE="cuda12"
                fi
            else
                echo "Error: --gpu specified but nvidia-smi not found"
                exit 1
            fi
            shift
            ;;
        --command)
            COMMAND="$2"
            shift 2
            ;;
        --command=*)
            COMMAND="${1#*=}"
            shift
            ;;
        *)
            PYTEST_ARGS="$@"
            break
            ;;
    esac
done

# Determine image name
REGISTRY="ghcr.io/drpaulbrewer/gridvoting-jax"

if [ "$MODE" == "dev" ]; then
    IMAGE="${REGISTRY}/dev/${CUDA_TYPE}:latest"
    echo "Using dev image: $IMAGE"
    
    # Pull latest dev image
    docker pull "$IMAGE"
    
    DOCKER_ARGS="-v $(pwd):/workspace"
else
    # Release mode
    IMAGE="${REGISTRY}/${CUDA_TYPE}:${VERSION}"
    echo "Using release image: $IMAGE"
    
    # Pull release image
    docker pull "$IMAGE"
    
    DOCKER_ARGS=""
fi

# Run container
# Note: We use arrays for command arguments to properly handle quoting
docker run --rm \
    $DOCKER_ARGS \
    ${CUDA_TYPE:+$([ "$CUDA_TYPE" != "cpu" ] && echo "--gpus all")} \
    "$IMAGE" \
    /bin/bash -c "${COMMAND:-python3 -m pytest $PYTEST_ARGS}"
