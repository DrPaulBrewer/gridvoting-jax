# Docker Usage Guide

## Overview

The `gridvoting-jax` project uses a multi-tier Docker image system for efficient development and testing:

- **Base Images**: Heavy dependencies (JAX, CUDA, OSF data) - built once
- **Release Images**: Versioned releases from PyPI - fast builds
- **Dev Images**: For local development with mounted source code

All images are hosted on GitHub Container Registry (GHCR).

## Image Types

### Base Images (Internal Use)
Built when dependencies change, not typically used directly:
- `ghcr.io/drpaulbrewer/gridvoting-jax/jax-base-cpu:latest`
- `ghcr.io/drpaulbrewer/gridvoting-jax/jax-base-cuda12:latest`
- `ghcr.io/drpaulbrewer/gridvoting-jax/jax-base-cuda13:latest`

### Development Images
For local development with source code mounted:
- `ghcr.io/drpaulbrewer/gridvoting-jax/gridvoting-jax-dev-cpu:latest`
- `ghcr.io/drpaulbrewer/gridvoting-jax/gridvoting-jax-dev-cuda12:latest`
- `ghcr.io/drpaulbrewer/gridvoting-jax/gridvoting-jax-dev-cuda13:latest`

### Release Images
Versioned releases from PyPI:
- `ghcr.io/drpaulbrewer/gridvoting-jax/gridvoting-jax-cpu:VERSION`
- `ghcr.io/drpaulbrewer/gridvoting-jax/gridvoting-jax-gpu-cuda12:VERSION`
- `ghcr.io/drpaulbrewer/gridvoting-jax/gridvoting-jax-gpu-cuda13:VERSION`

## Quick Start

### Local Development (Recommended)
```bash
# CPU testing with local source code
./test_docker.sh --dev --cpu tests/

# GPU testing with local source code (auto-detects CUDA version)
./test_docker.sh --dev --gpu tests/

# Run specific test
./test_docker.sh --dev --gpu tests/test_lazy_consistency.py -v
```

### Testing Specific Versions
```bash
# Test a specific release
./test_docker.sh --version=v0.9.1 --cpu

# Test latest release
./test_docker.sh --version=latest --gpu
```

### OSF Validation
```bash
# Quick validation (g=20, 40)
./test_docker_osf.sh --dev --gpu --quick

# Extended validation (g=20, 40, 60, 80, 100)
./test_docker_osf.sh --dev --gpu --extended

# With float64 precision
./test_docker_osf.sh --dev --gpu --float64
```

## Test Script Options

### `test_docker.sh`
```
Usage: ./test_docker.sh [OPTIONS] [PYTEST_ARGS]

Options:
  --dev              Use dev image with mounted source (default)
  --version=vX.Y.Z   Use specific release version
  --cpu              Use CPU-only image (default)
  --gpu              Use GPU image (auto-detects CUDA version)

Examples:
  ./test_docker.sh --dev --gpu tests/
  ./test_docker.sh --version=v0.9.1 --cpu
  ./test_docker.sh --dev tests/test_lazy_consistency.py -v
```

### `test_docker_osf.sh`
```
Usage: ./test_docker_osf.sh [OPTIONS]

Options:
  --dev              Use dev image (default)
  --version=vX.Y.Z   Use specific release
  --cpu              CPU-only (default)
  --gpu              GPU (auto-detects CUDA)
  --quick            Test g=20, 40
  --extended         Test g=20, 40, 60, 80, 100
  --float64          Use float64 precision

Examples:
  ./test_docker_osf.sh --dev --gpu --quick
  ./test_docker_osf.sh --version=v0.9.1 --cpu --extended
```

## CUDA Version Detection

The test scripts automatically detect your CUDA version:
- CUDA 12.x → Uses `cuda12` images
- CUDA 13.x → Uses `cuda13` images

Check your CUDA version:
```bash
nvidia-smi | grep "CUDA Version"
```

## Development Workflow

### 1. Make Code Changes
Edit source files locally as usual.

### 2. Test with Dev Image
```bash
# Dev image mounts your local source code
./test_docker.sh --dev --gpu tests/

# Or run interactively
docker run --rm -it -v $(pwd):/workspace --gpus all \
  ghcr.io/drpaulbrewer/gridvoting-jax/gridvoting-jax-dev-cuda12:latest \
  bash
```

### 3. Run Full Test Suite
```bash
./test_docker.sh --dev --gpu tests/ -v
```

## Building Images Locally (Advanced)

### Build Base Images
```bash
# CPU
docker build -f Dockerfiles/base/Dockerfile.jax-cpu \
  -t jax-base-cpu:local .

# CUDA 12
docker build -f Dockerfiles/base/Dockerfile.jax-cuda12 \
  -t jax-base-cuda12:local .

# CUDA 13
docker build -f Dockerfiles/base/Dockerfile.jax-cuda13 \
  -t jax-base-cuda13:local .
```

### Build Dev Images
```bash
docker build -f Dockerfiles/dev/Dockerfile.dev-cpu \
  -t gridvoting-jax-dev:local .
```

## CI/CD Pipeline

### Base Images
- **Trigger**: Manual or when `Dockerfiles/base/` changes
- **Workflow**: `.github/workflows/docker-base-images.yml`
- **Build Time**: 5-10 minutes (downloads ~2.5 GB CUDA libraries)
- **Frequency**: Rarely (only for JAX updates)

### Release Images
- **Trigger**: After PyPI publish
- **Workflow**: `.github/workflows/docker-publish.yml`
- **Build Time**: ~30 seconds
- **Tags**: `vX.Y.Z` and `latest`

### Dev Images
- **Trigger**: After base images rebuild
- **Workflow**: `.github/workflows/docker-base-images.yml`
- **Build Time**: ~10 seconds

## Troubleshooting

### GPU Not Detected
```bash
# Check nvidia-smi works
nvidia-smi

# Check Docker has GPU access
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# Install nvidia-container-toolkit if needed
# (see main README)
```

### Image Pull Fails
```bash
# Login to GHCR (if private)
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Pull manually
docker pull ghcr.io/drpaulbrewer/gridvoting-jax/gridvoting-jax-dev-cpu:latest
```

### Old Images Taking Space
```bash
# Remove old images
docker image prune -a

# Remove specific version
docker rmi ghcr.io/drpaulbrewer/gridvoting-jax/gridvoting-jax-cpu:v0.9.0
```

## Image Sizes

- **Base CPU**: ~1.5 GB
- **Base CUDA 12/13**: ~4 GB (includes CUDA libraries)
- **Dev Images**: +50 MB (dev tools)
- **Release Images**: +10 MB (gridvoting-jax package)

## Version History

All release versions are preserved on GHCR indefinitely, allowing:
- Performance comparisons across versions
- Reproducible research
- Easy rollback to previous versions

List available versions:
```bash
# Via GitHub Packages web UI
https://github.com/DrPaulBrewer/gridvoting-jax/pkgs/container/gridvoting-jax%2Fgridvoting-jax-cpu

# Or use Docker CLI
docker pull ghcr.io/drpaulbrewer/gridvoting-jax/gridvoting-jax-cpu:v0.9.1
```
