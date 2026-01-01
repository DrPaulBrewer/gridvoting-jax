#!/bin/bash
# Docker test script with support for dev and versioned images from GHCR
# Usage:
#   ./test_docker.sh [--dev|--version=vX.Y.Z] [--cpu|--gpu] [--dry-run] [--parallel|--jobs=N] [--command="..."] [pytest args...]
#   
# Examples:
#   ./test_docker.sh --dev --gpu tests/
#   ./test_docker.sh --version=v0.9.1 --cpu
#   ./test_docker.sh --command="pip list"
#   ./test_docker.sh --dev --dry-run
#   ./test_docker.sh --dev --parallel tests/test_core.py

set -e

MODE="dev"  # dev or release
VERSION="latest"
CUDA_TYPE="cpu"
# Default pytest args start empty, user args will be appended
PYTEST_ARGS_LIST=()
COMMAND=""
DRY_RUN=0
PARALLEL_ARGS=""

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
        --parallel)
            PARALLEL_ARGS="-n 2"
            shift
            ;;
        --jobs=*)
            JOBS="${1#*=}"
            PARALLEL_ARGS="-n $JOBS"
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            # Collect positional args (files, folders, other pytest flags)
            PYTEST_ARGS_LIST+=("$1")
            shift
            ;;
    esac
done

# Check for conflicts
if [[ -n "$COMMAND" && -n "$PARALLEL_ARGS" ]]; then
    echo "Warning: --command override specified; ignoring --parallel/--jobs flags."
    echo "         To use parallel execution with a custom command, include '-n <cores>' inside your command string."
fi

# Determine image name
REGISTRY="ghcr.io/drpaulbrewer/gridvoting-jax"

if [ "$MODE" == "dev" ]; then
    IMAGE="${REGISTRY}/dev/${CUDA_TYPE}:latest"
    if [ $DRY_RUN -eq 0 ]; then
        echo "Using dev image: $IMAGE"
        # Pull latest dev image
        docker pull "$IMAGE"
    fi
    DOCKER_ARGS="-v $(pwd):/workspace"
else
    # Release mode
    IMAGE="${REGISTRY}/${CUDA_TYPE}:${VERSION}"
    if [ $DRY_RUN -eq 0 ]; then
        echo "Using release image: $IMAGE"
        # Pull release image
        docker pull "$IMAGE"
    fi
    DOCKER_ARGS=""
fi

# Construct final command
if [ -n "$COMMAND" ]; then
    # User specified an exact command override
    FINAL_CMD="$COMMAND"
else
    # Default behavior: wrapped pytest
    # If no args provided, default to 'tests/' (but respecting pyproject.toml testpaths)
    # Actually, if no args, pytest defaults to testpaths in config, so we don't need to force tests/
    
    # Combine parallel args and collected positional args
    # Use array expansion for creating the string
    ARGS_STR="${PARALLEL_ARGS} ${PYTEST_ARGS_LIST[*]}"
    
    # Trim whitespace safely using bash pattern matching instead of xargs/echo
    ARGS_STR="${ARGS_STR#"${ARGS_STR%%[![:space:]]*}"}"
    ARGS_STR="${ARGS_STR%"${ARGS_STR##*[![:space:]]}"}"
    
    FINAL_CMD="python3 -m pytest $ARGS_STR"
fi

# Run container
GPU_FLAG=""
if [ "$CUDA_TYPE" != "cpu" ]; then
    GPU_FLAG="--gpus all"
fi

# Note: We use arrays for command arguments to properly handle quoting
# Using bash -c allows complex commands inside the container
FULL_DOCKER_CMD="docker run --rm $DOCKER_ARGS $GPU_FLAG -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e XLA_PYTHON_CLIENT_ALLOCATOR=platform $IMAGE /bin/bash -c \"$FINAL_CMD\""

if [ $DRY_RUN -eq 1 ]; then
    echo "Dry Run: Not executing."
    echo "Command:"
    echo "$FULL_DOCKER_CMD"
else
    eval "$FULL_DOCKER_CMD"
fi
