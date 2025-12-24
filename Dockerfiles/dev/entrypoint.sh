#!/bin/bash
# Entrypoint for gridvoting-jax dev images
# Validates that /workspace is mounted and contains source code

if [ ! -d "/workspace/src/gridvoting_jax" ]; then
    cat << 'EOF'
╔════════════════════════════════════════════════════════════════╗
║                  gridvoting-jax Dev Container                  ║
╚════════════════════════════════════════════════════════════════╝

ERROR: /workspace is not mounted or does not contain source code.

USAGE:
  docker run -v $(pwd):/workspace [OPTIONS] IMAGE [COMMAND]

EXAMPLES:
  # Interactive shell
  docker run -it -v $(pwd):/workspace ghcr.io/drpaulbrewer/gridvoting-jax/dev/cpu bash

  # Run tests
  docker run -v $(pwd):/workspace ghcr.io/drpaulbrewer/gridvoting-jax/dev/cpu \
    python3 -m pytest tests/

  # GPU support
  docker run --gpus all -v $(pwd):/workspace ghcr.io/drpaulbrewer/gridvoting-jax/dev/cuda12 \
    python3 -m pytest tests/

NOTE:
  - PYTHONPATH is pre-configured to /workspace/src
  - The gridvoting_jax module is immediately importable
  - No need to run 'pip install -e .'
  - To test with pip install in a pristine environment, unset PYTHONPATH:
      docker run -e PYTHONPATH="" -v $(pwd):/workspace IMAGE bash

For more information, see: docs/docker.md
EOF
    exit 1
fi

# All good, execute the command
exec "$@"
