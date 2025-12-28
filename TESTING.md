# Testing Guide

This document outlines the testing procedures for `gridvoting-jax`.

## Test Organization (Pytest Markers)

To facilitate faster development loops, the test suite is organized using pytest markers. You can run specific groups of tests using `-m <marker>`.

| Marker | Description | Typical Usage |
| :--- | :--- | :--- |
| `essential` | Core functionality tests (Grid, JAX basics, Solvers) | `pytest -m essential` |
| `lazy` | Lazy solver implementation tests | `pytest -m lazy` |
| `lumping` | Symmetry detection and lumping tests | `pytest -m lumping` |
| `benchmarks` | Performance benchmarks and replication checks | `pytest -m benchmarks` |
| `budget` | Budget voting model tests | `pytest -m budget` |
| `spatial` | Spatial voting model and shape tests | `pytest -m spatial` |
| `slow` | Long-running tests (skipped in CI by default) | `pytest -m "not slow"` |

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

### Common Commands

**Run core tests only:**
```bash
./test_docker.sh --command="pytest -m essential"
```

**Run specific feature tests:**
```bash
./test_docker.sh --command="pytest -m lazy"
```

**Run in parallel (CPU only):**
```bash
./test_docker.sh --parallel  # Uses 2 processes
./test_docker.sh --jobs=4    # Uses 4 processes
```

**Run with GPU:**
```bash
./test_docker.sh --dev --gpu
```

## Continuous Integration (GitHub Actions)

Tests are automatically run on GitHub via two workflows:

1.  **Tests (`tests.yml`)**:
    *   **Triggers**: Push and Pull Request to `main`/`master`.
    *   **Environment**: `ubuntu-latest`.
    *   **Python Versions**: 3.9, 3.10, 3.11, 3.12.
    *   **Command**: `pytest tests/ -v -m "not slow"` (Skips slow tests).
    *   **Check**: Also verifies CPU-only mode explicitly.

2.  **Python Testing (`python-testing.yml`)**:
    *   **Triggers**: Push and Pull Request to `main`.
    *   **OS Coverage**: `ubuntu-latest`, `macOS-latest`, `windows-latest`.
    *   **Python Versions**: 3.10, 3.11.
    *   **Command**: `pytest tests/ -sv -m "not slow"`.

**Note:** Both workflows skip tests marked as `@pytest.mark.slow` to ensure reasonably fast feedback. Heavy benchmarks should be run locally or via specific manual triggers.
