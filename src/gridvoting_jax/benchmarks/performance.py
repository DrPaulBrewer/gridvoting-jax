"""Performance benchmarking for gridvoting-jax."""

import time
import jax
from .. import Grid, VotingModel
from ..models.examples import bjm_spatial_triangle


def performance(dict=False):
    """Run performance benchmarks for gridvoting-jax.
    
    Tests algebraic solver performance across various grid sizes
    and voting models (ZI and MI modes).
    
    Args:
        dict: If True, return results as dictionary without printing.
              If False (default), print formatted output and return None.
    
    Returns:
        If dict=True, returns a dictionary with benchmark results.
        If dict=False, returns None and prints results to stdout.
    
    Usage:
        >>> import gridvoting_jax as gv
        >>> # Print results
        >>> gv.benchmarks.performance()
        >>> # Get results as dictionary
        >>> results = gv.benchmarks.performance(dict=True)
    """
    params_list = [
        {'g': 20, 'zi': False, 'label': 'g=20, zi=False'},
        {'g': 20, 'zi': True,  'label': 'g=20, zi=True'},
        {'g': 40, 'zi': False, 'label': 'g=40, zi=False'},
        {'g': 40, 'zi': True,  'label': 'g=40, zi=True'},
        {'g': 60, 'zi': False, 'label': 'g=60, zi=False'},
        {'g': 60, 'zi': True,  'label': 'g=60, zi=True'}
    ]

    results = []
    
    # Get JAX device info
    devices = jax.devices()
    default_device = devices[0] if devices else None
    device_info = f"{default_device.platform.upper()}" if default_device else "Unknown"
    
    if not dict:
        print(f"\n{'='*70}")
        print(f"JAX Performance Benchmark")
        print(f"{'='*70}")
        print(f"Device: {device_info} ({default_device})")
        print(f"JAX version: {jax.__version__}")
        print(f"{'='*70}\n")
        print(f"{'Test Case':<20} | {'Alternatives':<12} | {'Time (s)':<10} | {'Device':<10}")
        print("-" * 70)

    for params in params_list:
        g = params['g']
        zi = params['zi']
        label = params['label']
        
        # Setup (copied from test)
        # grid = Grid(x0=-g, x1=g, y0=-g, y1=g)
        # number_of_alternatives = (2*g+1)**2
        # voter_ideal_points = [[-15, -9], [0, 17], [15, -9]]
        
        # u = grid.spatial_utilities(
        #     voter_ideal_points=voter_ideal_points,
        #     metric='sqeuclidean'
        # )
        
        # vm = VotingModel(
        #    utility_functions=u,
        #    majority=2,
        #    zi=zi,
        #    number_of_voters=3,
        #    number_of_feasible_alternatives=number_of_alternatives
        # )
        
        vm = bjm_spatial_triangle(g=g, zi=zi)
        number_of_alternatives = vm.number_of_feasible_alternatives
        
        try:
            # Benchmark the algebraic solver
            start = time.time()
            vm.analyze()
            end = time.time()
            solve_time = end - start

            if not dict:
                print(f"{label:<20} | {number_of_alternatives:<12} | {solve_time:<10.4f} | {device_info:<10}")
            
            results.append({
                "test_case": label,
                "alternatives": number_of_alternatives,
                "time_seconds": solve_time,
                "device": device_info
            })

        except Exception as e:
            if not dict:
                print(f"{label:<20} | {number_of_alternatives:<12} | ERROR: {str(e)}")
            
            results.append({
                "test_case": label,
                "alternatives": number_of_alternatives,
                "time_seconds": None,
                "error": str(e),
                "device": device_info
            })

    if not dict:
        # Print summary
        print("\n\nResults Summary:")
        print("=" * 70)
        for r in results:
            print(f"Test Case: {r['test_case']}")
            if 'error' in r:
                print(f"  Alternatives: {r['alternatives']} | ERROR: {r['error']}")
            else:
                print(f"  Alternatives: {r['alternatives']} | Time: {r['time_seconds']:.4f} | Device: {r['device']}")
            print()
        return None
    else:
        return {
            "device": device_info,
            "jax_version": jax.__version__,
            "results": results
        }
