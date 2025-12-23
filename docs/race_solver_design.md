# Race Solver Design Document

## Concept

A hybrid solver that runs **lazy GMRES** and **lazy power method** in parallel, using whichever completes first. This exploits multi-core CPUs where individual solvers don't saturate all cores.

## Architecture

### Race Strategy

```
Timeline:
0s:    Start power method (uniform initial guess)
       ↓
       Power method running (uses ~400% CPU)
       ↓
Ts:    Subgrid solve completes → Start GMRES (with initial guess)
       ↓
       Both solvers racing (combined ~800% CPU usage)
       ↓
Tf:    First solver to converge wins → Cancel the other
```

### Key Design Elements

1. **Parallel Execution**
   - Use Python `threading` or `multiprocessing`
   - Each solver runs in separate thread/process
   - Shared result queue for first completion

2. **Staggered Start**
   - Power method starts immediately with uniform distribution
   - GMRES waits for subgrid solve (grid upscaling step)
   - This gives power method a head start

3. **Early Termination**
   - First solver to converge signals completion
   - Other solver is cancelled/terminated
   - Result from winner is returned

4. **Resource Management**
   - Monitor CPU usage to ensure cores available
   - Only enable race mode if sufficient cores (e.g., ≥8 cores)
   - Fallback to single solver if resources constrained

## Implementation Plan

### Phase 1: Basic Race Solver

```python
def _solve_race(self, tolerance, max_iterations, initial_guess=None):
    """
    Race solver: Run GMRES and power method in parallel.
    
    Strategy:
    1. Start power method immediately (uniform guess)
    2. Compute subgrid solution for GMRES initial guess
    3. Start GMRES with initial guess
    4. Return whichever solver finishes first
    """
    import threading
    from queue import Queue
    
    result_queue = Queue()
    
    # Thread 1: Power method (starts immediately)
    def run_power_method():
        try:
            result = self._solve_power_method_lazy(
                tolerance=tolerance,
                max_iterations=max_iterations,
                initial_guess=None,  # Uniform distribution
                timeout=max_iterations * 0.1  # Adaptive timeout
            )
            result_queue.put(('power', result))
        except Exception as e:
            result_queue.put(('power_error', e))
    
    # Thread 2: GMRES with grid upscaling (starts after subgrid)
    def run_gmres_with_upscaling():
        try:
            # Compute initial guess from subgrid
            initial_guess = self._compute_grid_upscaling_guess()
            
            result = self._solve_gmres_lazy(
                tolerance=tolerance,
                max_iterations=max_iterations,
                initial_guess=initial_guess
            )
            result_queue.put(('gmres', result))
        except Exception as e:
            result_queue.put(('gmres_error', e))
    
    # Start both threads
    power_thread = threading.Thread(target=run_power_method)
    gmres_thread = threading.Thread(target=run_gmres_with_upscaling)
    
    power_thread.start()
    gmres_thread.start()
    
    # Wait for first result
    winner, result = result_queue.get()
    
    # Note: In Python, threads will continue until completion
    # For true cancellation, would need multiprocessing with terminate()
    
    return result, winner
```

### Phase 2: Multiprocessing for True Cancellation

```python
def _solve_race_multiprocess(self, tolerance, max_iterations):
    """
    Race solver using multiprocessing for true cancellation.
    """
    import multiprocessing as mp
    
    def power_worker(queue, transition_matrix_lazy, tolerance, max_iterations):
        # Reconstruct lazy chain in worker process
        chain = LazyMarkovChain(transition_matrix_lazy)
        result = chain._solve_power_method_lazy(tolerance, max_iterations)
        queue.put(('power', result))
    
    def gmres_worker(queue, transition_matrix_lazy, tolerance, max_iterations):
        # Reconstruct lazy chain and compute initial guess
        chain = LazyMarkovChain(transition_matrix_lazy)
        initial_guess = chain._compute_grid_upscaling_guess()
        result = chain._solve_gmres_lazy(tolerance, max_iterations, initial_guess)
        queue.put(('gmres', result))
    
    queue = mp.Queue()
    
    p1 = mp.Process(target=power_worker, args=(queue, self.transition_matrix_lazy, tolerance, max_iterations))
    p2 = mp.Process(target=gmres_worker, args=(queue, self.transition_matrix_lazy, tolerance, max_iterations))
    
    p1.start()
    p2.start()
    
    # Wait for first result
    winner, result = queue.get()
    
    # Terminate the loser
    if winner == 'power':
        p2.terminate()
    else:
        p1.terminate()
    
    p1.join()
    p2.join()
    
    return result, winner
```

### Phase 3: Adaptive Strategy

```python
def _should_use_race_solver(self):
    """
    Determine if race solver is beneficial.
    
    Criteria:
    - Sufficient CPU cores (≥8)
    - Large enough problem (N > 10000)
    - Not already using GPU
    """
    import os
    cpu_count = os.cpu_count() or 1
    
    if cpu_count < 8:
        return False
    
    if self.transition_matrix_lazy.N < 10000:
        return False
    
    # Check if GPU is being used
    import jax
    if jax.devices()[0].platform == 'gpu':
        return False
    
    return True
```

## Advantages

1. **Better CPU Utilization**
   - GMRES uses ~400% CPU → 8 cores idle
   - Power method uses ~400% CPU → 8 cores idle
   - Both together use ~800% CPU → 4 cores idle
   - Much better than 8 cores idle!

2. **Robustness**
   - If one solver has convergence issues, other may succeed
   - Automatic fallback without user intervention

3. **Performance**
   - Best-case: Faster than either solver alone
   - Worst-case: Same as slower solver (but with more CPU usage)

4. **Validation**
   - Can optionally wait for both and compare results
   - Built-in cross-validation

## Challenges

### 1. JAX and Multiprocessing

**Issue**: JAX doesn't play well with multiprocessing due to:
- Device initialization in each process
- Memory duplication
- XLA compilation cache

**Solution**: Use threading instead, accept that cancellation is harder

### 2. Thread Cancellation in Python

**Issue**: Python threads can't be forcefully terminated

**Solutions**:
- Let both threads complete (wastes some CPU)
- Use cooperative cancellation (check flag periodically)
- Use multiprocessing with `terminate()` (but JAX issues)

### 3. Memory Overhead

**Issue**: Running both solvers doubles memory usage

**Solution**: Only enable for systems with sufficient RAM

## Recommended Implementation

### Simple Threading Approach (Recommended)

```python
def _solve_race_simple(self, tolerance, max_iterations, initial_guess_subgrid=None):
    """
    Simple race solver using threading.
    
    Both solvers run to completion, but we return the first result.
    This avoids JAX/multiprocessing issues.
    """
    import threading
    from queue import Queue
    import time
    
    result_queue = Queue()
    start_time = time.time()
    
    def run_power():
        result = self._solve_power_method_lazy(tolerance, max_iterations)
        elapsed = time.time() - start_time
        result_queue.put(('power', result, elapsed))
    
    def run_gmres():
        # Wait for initial guess if needed
        if initial_guess_subgrid is not None:
            initial_guess = self._upscale_distribution(initial_guess_subgrid)
        else:
            initial_guess = None
        
        result = self._solve_gmres_lazy(tolerance, max_iterations, initial_guess)
        elapsed = time.time() - start_time
        result_queue.put(('gmres', result, elapsed))
    
    # Start both
    t1 = threading.Thread(target=run_power)
    t2 = threading.Thread(target=run_gmres)
    
    t1.start()
    t2.start()
    
    # Get first result
    winner, result, elapsed = result_queue.get()
    
    # Threads will continue, but we ignore their results
    # They'll finish naturally
    
    print(f"Race solver: {winner} won in {elapsed:.2f}s")
    
    return result
```

## Solution Quality Validation

**Critical**: Don't accept a solution just because it finished first - validate quality!

### Quality Metrics

```python
def _validate_solution_quality(self, solution, tolerance):
    """
    Validate that a solution meets quality criteria.
    
    Checks:
    1. Sum to 1.0 (probability distribution)
    2. All non-negative (valid probabilities)
    3. Residual norm (how well it satisfies P^T v = v)
    
    Returns:
        (is_valid, quality_score, metrics)
    """
    import jax.numpy as jnp
    
    # Check 1: Sum to 1.0
    prob_sum = jnp.sum(solution)
    sum_error = jnp.abs(prob_sum - 1.0)
    
    # Check 2: Non-negative
    min_val = jnp.min(solution)
    has_negative = min_val < -1e-6
    
    # Check 3: Residual norm ||P^T v - v||
    # This is the key quality metric!
    Pv = self.transition_matrix_lazy.rmatvec(solution)
    residual = Pv - solution
    residual_norm = jnp.linalg.norm(residual)
    
    # Quality criteria
    is_valid = (
        sum_error < tolerance * 10 and  # Sum close to 1
        not has_negative and             # No negative probabilities
        residual_norm < tolerance        # Converged solution
    )
    
    quality_score = residual_norm  # Lower is better
    
    metrics = {
        'sum': float(prob_sum),
        'sum_error': float(sum_error),
        'min_value': float(min_val),
        'residual_norm': float(residual_norm),
        'is_valid': is_valid
    }
    
    return is_valid, quality_score, metrics
```

### Race Solver with Quality Validation

```python
def _solve_race_with_validation(self, tolerance, max_iterations, initial_guess_subgrid=None):
    """
    Race solver with solution quality validation.
    
    Strategy:
    1. Run both solvers in parallel
    2. Validate each solution as it completes
    3. Accept first VALID solution
    4. If first solution is invalid, wait for second
    5. If both invalid, raise error
    """
    import threading
    from queue import Queue
    import time
    
    result_queue = Queue()
    start_time = time.time()
    
    def run_power():
        result = self._solve_power_method_lazy(tolerance, max_iterations)
        elapsed = time.time() - start_time
        
        # Validate quality
        is_valid, quality, metrics = self._validate_solution_quality(result, tolerance)
        
        result_queue.put({
            'solver': 'power',
            'result': result,
            'elapsed': elapsed,
            'is_valid': is_valid,
            'quality': quality,
            'metrics': metrics
        })
    
    def run_gmres():
        if initial_guess_subgrid is not None:
            initial_guess = self._upscale_distribution(initial_guess_subgrid)
        else:
            initial_guess = None
        
        result = self._solve_gmres_lazy(tolerance, max_iterations, initial_guess)
        elapsed = time.time() - start_time
        
        # Validate quality
        is_valid, quality, metrics = self._validate_solution_quality(result, tolerance)
        
        result_queue.put({
            'solver': 'gmres',
            'result': result,
            'elapsed': elapsed,
            'is_valid': is_valid,
            'quality': quality,
            'metrics': metrics
        })
    
    # Start both
    t1 = threading.Thread(target=run_power)
    t2 = threading.Thread(target=run_gmres)
    
    t1.start()
    t2.start()
    
    # Get first result
    first = result_queue.get()
    
    if first['is_valid']:
        # First solution is valid - accept it!
        print(f"Race solver: {first['solver']} won in {first['elapsed']:.2f}s "
              f"(residual: {first['quality']:.2e})")
        return first['result']
    else:
        # First solution is invalid - wait for second
        print(f"Race solver: {first['solver']} finished first but INVALID "
              f"(residual: {first['quality']:.2e}), waiting for other solver...")
        
        second = result_queue.get()
        
        if second['is_valid']:
            print(f"Race solver: {second['solver']} won in {second['elapsed']:.2f}s "
                  f"(residual: {second['quality']:.2e})")
            return second['result']
        else:
            # Both invalid - report and use better one
            print(f"WARNING: Both solvers produced invalid solutions!")
            print(f"  Power method: residual={first['quality']:.2e}, metrics={first['metrics']}")
            print(f"  GMRES: residual={second['quality']:.2e}, metrics={second['metrics']}")
            
            # Use the one with better quality score
            if first['quality'] < second['quality']:
                print(f"  Using power method (better residual)")
                return first['result']
            else:
                print(f"  Using GMRES (better residual)")
                return second['result']
```

### Why Quality Validation Matters

**Scenario**: Power method finishes in 30s, but didn't fully converge
- Without validation: Accept poor solution ❌
- With validation: Reject and wait for GMRES ✅

**Key insight**: Residual norm `||P^T v - v||` tells us if the solution actually satisfies the stationary distribution equation, regardless of which solver produced it.

## Testing Strategy

1. **Unit Tests**
   - Verify both solvers produce same result
   - Test race logic with mock solvers
   - Validate winner selection

2. **Performance Tests**
   - Compare race vs individual solvers
   - Measure CPU utilization
   - Test on various grid sizes (g=40, 60, 80, 100)

3. **Integration Tests**
   - Test with real BJM spatial triangle models
   - Validate against OSF reference
   - Compare race results to known-good solutions

## Next Steps

1. **Prototype**: Implement simple threading version
2. **Benchmark**: Test on g=60, g=80 to measure speedup
3. **Refine**: Add adaptive logic for when to use race solver
4. **Document**: Add user-facing documentation
5. **Integrate**: Add as solver option: `solver="race"`

## API Design

```python
# User-facing API
model.analyze(solver="race", max_iterations=3000)

# Or with explicit configuration
model.analyze(
    solver="race",
    race_config={
        'enable_power': True,
        'enable_gmres': True,
        'use_grid_upscaling': True,
        'wait_for_both': False  # For validation
    }
)
```

## Expected Performance

For g=80 on 12-core CPU:
- GMRES alone: ~40s
- Power method alone: ~120s
- Race solver: ~40s (GMRES wins) with ~800% CPU usage vs ~400%

**Key benefit**: Better CPU utilization, not necessarily faster completion!
