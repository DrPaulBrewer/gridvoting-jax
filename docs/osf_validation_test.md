# OSF Validation Test - g=80 and g=100

## Purpose

This test validates the lazy grid upscaling solver against OSF (Open Science Framework) reference data for the BJM spatial triangle example.

## What It Tests

### Part 1: g=80 Direct Validation
- Solves g=80 BJM spatial triangle using lazy grid upscaling
- Compares results against OSF reference stationary distribution
- Validates that relative error is < 1%

### Part 2: g=100 Subgrid Validation
- Solves g=100 BJM spatial triangle using lazy grid upscaling
- Extracts the g=80 subgrid from the g=100 solution
- Compares the subgrid to the g=80 reference
- Validates that:
  - Probability mass outside the g=80 box is << 1%
  - Subgrid values are within 1% of g=80 reference

## Why This Matters

This test ensures that:
1. **Lazy solver accuracy**: The lazy grid upscaling produces accurate results
2. **Grid upscaling correctness**: Larger grids (g=100) correctly contain smaller grids (g=80)
3. **Probability conservation**: Very little probability mass "leaks" outside the expected region

## Running the Test

```bash
# With automatic CPU optimization (recommended)
python3 test_osf_g80_g100_cpu.py

# Output is saved to test_osf_g80_g100_output.txt
```

## Expected Results

### g=80 Validation
- Max relative error < 1%
- Solve time: ~30-60 seconds (CPU, 12 cores)

### g=100 Validation  
- Probability outside g=80 box: < 0.1%
- Subgrid max relative error: < 1%
- Solve time: ~60-120 seconds (CPU, 12 cores)

## CPU Optimization

The test automatically benefits from the new CPU configuration:
- 12 CPU devices exposed to JAX
- Optimal XLA flags for parallelization
- Multi-threaded BLAS operations

No manual configuration needed!
