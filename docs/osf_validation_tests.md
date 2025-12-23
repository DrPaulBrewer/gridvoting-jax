# OSF Validation Test Suite

## Test Organization

### Float32 Tests (Default Precision)
- **`test_g80_lazy_grid_upscaling_vs_osf`** - Quick validation of g=80
- **`test_g100_subgrid_validation`** - Full g=100 validation (marked `@slow`, `@large_grid`)

### Float64 Tests (High Precision)
- **`test_g80_float64_vs_osf`** - High-precision g=80 validation (marked `@slow`, `@large_grid`)
- **`test_g100_outside_probability_float64`** - Detailed analysis of probability outside g=80 box (marked `@slow`, `@large_grid`)

### GPU Power Method Tests
- **`test_g80_power_method_gpu`** - g=80 validation with lazy power method on GPU (marked `@slow`, `@large_grid`)
- **`test_g100_power_method_outside_probability_gpu`** - g=100 power method comparison with GMRES (marked `@slow`, `@large_grid`)

## Running Tests

### Run all OSF validation tests
```bash
pytest tests/test_osf_validation_g80_g100.py -v
```

### Run only fast tests (skip slow/large_grid)
```bash
pytest tests/test_osf_validation_g80_g100.py -v -m "not slow"
```

### Run only float32 tests
```bash
pytest tests/test_osf_validation_g80_g100.py::TestOSFValidation -v
```

### Run only float64 tests
```bash
pytest tests/test_osf_validation_g80_g100.py::TestOSFValidationFloat64 -v
```

### Run only GPU power method tests
```bash
pytest tests/test_osf_validation_g80_g100.py::TestPowerMethodGPU -v
```

### Run specific test
```bash
# Float64 outside probability test (most detailed)
pytest tests/test_osf_validation_g80_g100.py::TestOSFValidationFloat64::test_g100_outside_probability_float64 -v -s

# GPU power method comparison
pytest tests/test_osf_validation_g80_g100.py::TestPowerMethodGPU::test_g100_power_method_outside_probability_gpu -v -s
```

## Test Markers

- **`@pytest.mark.slow`** - Tests that take >1 minute
- **`@pytest.mark.large_grid`** - Tests using g≥80 grids

## Expected Results

### Float32 Tests

**g=80 validation:**
- Max absolute difference: < 1e-6
- L1 difference: < 1e-5
- Probability sum: within 1e-5 of 1.0

**g=100 validation:**
- Subgrid max absolute difference: < 1e-6
- Probability outside g=80 box: < 1e-5
- Probability sum: within 1e-5 of 1.0

### Float64 Tests

**g=80 validation:**
- Max absolute difference: < 1e-8 (better than float32)
- L1 difference: < 1e-7
- Probability sum: within 1e-10 of 1.0

**g=100 outside probability:**
- **Expected value: 7.10e-07** (from exponential decay model)
- Actual should be within factor of 10 of expected
- Distribution by distance bins shown in output
- **If exactly 0**: Test fails (indicates solver artifact)

## Key Insight: Outside Probability

The float64 test validates that the probability outside the g=80 box in the g=100 solution matches the theoretical expectation from exponential decay analysis.

**Exponential decay model** (fitted to OSF g=80 data):
```
prob(r) = A × exp(-k × r)
where:
  A = 1.67e-02 (amplitude)
  k = 0.210 (decay rate)
  r = ||(x, y)||₂ (L2 distance from origin)
```

**Expected distribution outside g=80 box:**
- r ∈ [80, 90): 73.4% of outside probability
- r ∈ [90, 100): 23.5%
- r ∈ [100, 110): 2.8%
- r ∈ [110, 120): 0.2%

## Troubleshooting

### Test shows probability outside = 0

This indicates a solver artifact, likely from:
1. Grid upscaling initial guess being too concentrated
2. Numerical underflow in float32
3. Convergence criteria stopping too early

**Solution**: Run float64 test to see if higher precision resolves it.

### Test shows probability outside >> expected

This could indicate:
1. Solver not fully converged
2. Numerical instability
3. Different convergence pattern

**Solution**: Check convergence metrics and increase max_iterations.

## CI/CD Integration

Add to `.github/workflows/tests.yml`:

```yaml
# Fast tests (run on every push)
- name: Run fast tests
  run: pytest tests/test_osf_validation_g80_g100.py -v -m "not slow"

# Slow tests (run on schedule or manual trigger)
- name: Run slow tests
  run: pytest tests/test_osf_validation_g80_g100.py -v -m "slow"
  if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
```

## References

- **Exponential decay analysis**: `calculate_expected_outside_prob.py`
- **BJM paper Figure 4**: https://link.springer.com/article/10.1007/s11403-023-00387-8/figures/4
- **OSF reference data**: Loaded via `gridvoting_jax.benchmarks.osf_comparison.load_osf_distribution()`
