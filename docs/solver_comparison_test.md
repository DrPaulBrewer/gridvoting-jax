# Solver Comparison Test: Lazy GMRES vs Lazy Power Method

## Purpose

This test compares the two lazy iterative solvers to ensure they produce compatible, accurate results:
- **Lazy GMRES with grid upscaling** - Uses initial guess from coarser grid
- **Lazy power method** - Iterative power method without initial guess

## What It Tests

### 1. Solver Compatibility
- Solves g=80 with both lazy GMRES and lazy power method
- Compares the two solutions directly
- Validates they agree within 1% (ideally much better)

### 2. OSF Reference Validation
- Compares both solvers against OSF reference data
- Ensures both produce accurate results independently
- Validates max relative error < 1%

### 3. Performance Comparison
- Measures solve time for each method
- Compares speedup from grid upscaling initial guess
- Expected: GMRES with grid upscaling should be faster

## Why This Matters

### Solver Compatibility
If both solvers produce the same answer, it provides:
- **Confidence in correctness** - Independent methods agreeing
- **Flexibility** - Can choose solver based on performance needs
- **Validation** - One solver can validate the other

### Grid Upscaling Benefit
The comparison shows the benefit of grid upscaling:
- **With grid upscaling**: GMRES starts with good initial guess
- **Without grid upscaling**: Power method starts from uniform distribution
- Expected speedup: 2-5x for GMRES with grid upscaling

## Running the Test

```bash
# Run solver comparison test
python3 test_solver_comparison_cpu.py
```

## Expected Results

### Solver Compatibility
- Max relative difference between solvers: < 0.1%
- Both should produce nearly identical results

### OSF Validation
- GMRES vs OSF: Max relative error < 1%
- Power method vs OSF: Max relative error < 1%

### Performance
- GMRES with grid upscaling: ~30-60s
- Power method: ~60-180s (slower due to no initial guess)
- Speedup: 2-3x for GMRES

## Key Insights

### When to Use Each Solver

**Lazy GMRES with Grid Upscaling:**
- ✅ Fastest for large grids (g>60)
- ✅ Benefits from initial guess
- ✅ Default choice for production

**Lazy Power Method:**
- ✅ Simpler algorithm
- ✅ Good for validation
- ✅ More robust to poor initial guesses
- ⚠️  Slower convergence

### Compatibility Expectations

The solvers should agree very closely because:
1. Both solve the same linear system
2. Both use the same lazy matrix operations
3. Differences should only be from:
   - Numerical precision
   - Convergence criteria
   - Iteration count differences

If solvers disagree significantly (>1%), it indicates a potential bug or convergence issue.
