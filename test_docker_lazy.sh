#!/bin/bash
# Test lazy solvers in Docker environment
set -e

echo "========================================="
echo "Docker Lazy Solver Test Suite"
echo "========================================="

docker build -f Dockerfiles/Dockerfile.cpu -t dev/cpu:lazy-test .

echo ""
echo "1. Lazy consistency tests..."
docker run --rm dev/cpu:lazy-test \
    python3 -m pytest tests/test_lazy_consistency.py -v

echo ""
echo "2. Lazy auto-selection tests..."
docker run --rm dev/cpu:lazy-test \
    python3 -m pytest tests/test_lazy_auto_selection.py -v

echo ""
echo "3. OSF validation (g=80, CPU)..."
docker run --rm dev/cpu:lazy-test \
    python3 -m pytest tests/test_osf_validation_g80_g100.py::TestPowerMethodCPU::test_g80_power_method_cpu -v

echo ""
echo "✅ All lazy solver tests passed!"
