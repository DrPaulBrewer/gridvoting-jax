# Testing Guide

This document outlines the testing procedures for `gridvoting-jax`.


## Running Tests with Docker

The primary way to run tests is via the `test_docker.sh` script, which ensures a consistent environment.

### Default Behavior
Running `./test_docker.sh` without arguments effectively runs:
```bash
./test_docker.sh --dev --cpu --command="python3 -m pytest tests/"
```
*   **Image**: Uses the `dev/cpu` image (`ghcr.io/drpaulbrewer/gridvoting-jax/dev/cpu:latest`).
*   **Mode**: CPU execution.
*   **Tests**: Runs **ALL** tests in the `tests/` directory (including slow ones).

### Interactive Mode
Running `./test_docker.sh --cpu --dev --it --command=ipython` will run an ipython3 interactive environment

Options:
   *  `--cpu`  calculate on local CPU
   *  `--gpu`  calculate on local GPU
   *  `--it`   runs docker container in interactive terminal mode
   *  `--command=` runs a specific command in the container


### Other Common Commands

**Run tests with local GPU:**
```bash
./test_docker.sh --dev --gpu
```

**Run a specific test file:**
```bash
./test_docker.sh --dev --cpu tests/test_models_examples_bjm_g20_replication.py
```

**Run a specific test function:**
```bash
./test_docker.sh --dev --cpu tests/test_models_examples_bjm_g20_replication.py::test_bjm_g20_replication
```

## Continuous Integration (GitHub Actions)

Tests are automatically run on GitHub via two workflows:

1.  **Tests (`tests.yml`)**:
    *   **Triggers**: Push and Pull Request to `main`/`master`.
    *   **Environment**: `ubuntu-latest`.
    *   **Python Versions**: 3.10, 3.11, 3.12.
    *   **Command**: `pytest tests/ -v`.

2.  **Python Testing (`python-testing.yml`)**:
    *   **Triggers**: Push and Pull Request to `main`.
    *   **OS Coverage**: `ubuntu-latest`, `macOS-latest`, `windows-latest`.
    *   **Python Versions**: 3.10, 3.11.
    *   **Command**: `pytest tests/ -sv`.

