"""
Experimental script to measure L1 differences between dense and lazy solvers.

This script runs each solver pair and measures the actual L1 norm difference
to determine appropriate epsilon-based tolerances for test_lazy_equivalence.
"""

import jax
import jax.numpy as jnp
import gridvoting_jax as gv
from gridvoting_jax.models.examples.condorcet import condorcet_cycle

def measure_solver_difference(g, solver_dense, solver_lazy, params, model_factory=None):
    """Measure L1 difference between dense and lazy solver."""
    if model_factory is None:
        model_dense = gv.bjm_spatial_triangle(g=g, zi=False)
        model_lazy = gv.bjm_spatial_triangle(g=g, zi=False)
    else:
        model_dense = model_factory()
        model_lazy = model_factory()
    
    # Run dense
    model_dense.analyze(solver=solver_dense, **params)
    
    # Run lazy
    model_lazy.analyze_lazy(solver=solver_lazy, force_lazy=True, **params)
    
    # Measure difference
    diff = jnp.linalg.norm(
        model_dense.stationary_distribution - model_lazy.stationary_distribution,
        ord=1
    )
    
    # Get epsilon for this dtype
    dtype = model_dense.stationary_distribution.dtype
    eps = jnp.finfo(dtype).eps
    
    # Calculate factor
    factor = float(diff / eps)
    
    return {
        'diff': float(diff),
        'eps': float(eps),
        'factor': factor,
        'dtype': str(dtype)
    }

def run_experiments():
    """Run all experiments and report findings."""
    print("=" * 80)
    print("SOLVER TOLERANCE EXPERIMENTS")
    print("=" * 80)
    print()
    
    # Test configurations
    tests = [
        {
            'name': 'power_method (g=20)',
            'g': 20,
            'dense': 'power_method',
            'lazy': 'power_method',
            'params': {'max_iterations': 20, 'timeout': 60},
            'current_tol': 350.0
        },
        {
            'name': 'power_method (g=40)',
            'g': 40,
            'dense': 'power_method',
            'lazy': 'power_method',
            'params': {'max_iterations': 20, 'timeout': 60},
            'current_tol': 350.0
        },
        {
            'name': 'bifurcated_power_method (g=20)',
            'g': 20,
            'dense': 'bifurcated_power_method',
            'lazy': 'bifurcated_power_method',
            'params': {'max_iterations': 20, 'timeout': 60},
            'current_tol': 50.0
        },
        {
            'name': 'bifurcated_power_method (g=40)',
            'g': 40,
            'dense': 'bifurcated_power_method',
            'lazy': 'bifurcated_power_method',
            'params': {'max_iterations': 20, 'timeout': 60},
            'current_tol': 50.0
        },
        {
            'name': 'gmres (g=20)',
            'g': 20,
            'dense': 'gmres_matrix_inversion',
            'lazy': 'gmres',
            'params': {'max_iterations': 20},
            'current_tol': 500.0
        },
        {
            'name': 'gmres (g=40)',
            'g': 40,
            'dense': 'gmres_matrix_inversion',
            'lazy': 'gmres',
            'params': {'max_iterations': 20},
            'current_tol': 500.0
        },
    ]
    
    # Also test Condorcet cycle
    condorcet_tests = [
        {
            'name': 'power_method (condorcet)',
            'dense': 'power_method',
            'lazy': 'power_method',
            'params': {'max_iterations': 20, 'timeout': 60},
            'factory': lambda: condorcet_cycle(zi=False),
            'current_tol': 10.0
        },
        {
            'name': 'bifurcated_power_method (condorcet)',
            'dense': 'bifurcated_power_method',
            'lazy': 'bifurcated_power_method',
            'params': {'max_iterations': 20, 'timeout': 60},
            'factory': lambda: condorcet_cycle(zi=False),
            'current_tol': 10.0
        },
        {
            'name': 'gmres (condorcet)',
            'dense': 'gmres_matrix_inversion',
            'lazy': 'gmres',
            'params': {'max_iterations': 20},
            'factory': lambda: condorcet_cycle(zi=False),
            'current_tol': 10.0
        },
    ]
    
    results = []
    
    # Run BJM tests
    for test in tests:
        print(f"Testing {test['name']}...")
        result = measure_solver_difference(
            g=test['g'],
            solver_dense=test['dense'],
            solver_lazy=test['lazy'],
            params=test['params']
        )
        result['name'] = test['name']
        result['current_tol'] = test['current_tol']
        results.append(result)
        print(f"  L1 diff: {result['diff']:.2e}")
        print(f"  Factor: {result['factor']:.1f} eps (current: {test['current_tol']} eps)")
        print()
    
    # Run Condorcet tests
    for test in condorcet_tests:
        print(f"Testing {test['name']}...")
        result = measure_solver_difference(
            g=None,
            solver_dense=test['dense'],
            solver_lazy=test['lazy'],
            params=test['params'],
            model_factory=test['factory']
        )
        result['name'] = test['name']
        result['current_tol'] = test['current_tol']
        results.append(result)
        print(f"  L1 diff: {result['diff']:.2e}")
        print(f"  Factor: {result['factor']:.1f} eps (current: {test['current_tol']} eps)")
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Test':<40} {'Current':<12} {'Measured':<12} {'Recommended':<12}")
    print("-" * 80)
    
    for result in results:
        # Recommend tolerance with 20% safety margin
        recommended = int(result['factor'] * 1.2)
        # Round up to nearest 10
        recommended = ((recommended + 9) // 10) * 10
        
        status = "✓" if result['factor'] < result['current_tol'] else "⚠"
        print(f"{result['name']:<40} {result['current_tol']:<12.0f} {result['factor']:<12.1f} {recommended:<12} {status}")
    
    print()
    print("Legend:")
    print("  ✓ = Current tolerance is sufficient")
    print("  ⚠ = Current tolerance may be too tight")
    print()

if __name__ == "__main__":
    run_experiments()
