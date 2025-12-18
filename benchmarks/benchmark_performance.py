
import time
import gridvoting_jax as gv
import numpy as np
import jax

def benchmark():
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
        grid = gv.Grid(x0=-g, x1=g, y0=-g, y1=g)
        number_of_alternatives = (2*g+1)**2
        voter_ideal_points = [[-15, -9], [0, 17], [15, -9]]
        
        u = grid.spatial_utilities(
            voter_ideal_points=voter_ideal_points,
            metric='sqeuclidean'
        )
        
        vm = gv.VotingModel(
            utility_functions=u,
            majority=2,
            zi=zi,
            number_of_voters=3,
            number_of_feasible_alternatives=number_of_alternatives
        )
        
        try:
            # Benchmark the algebraic solver
            start = time.time()
            vm.analyze()
            end = time.time()
            solve_time = end - start

            print(f"{label:<20} | {number_of_alternatives:<12} | {solve_time:<10.4f} | {device_info:<10}")
            
            results.append({
                "Test Case": label,
                "Alternatives": number_of_alternatives,
                "Time (s)": f"{solve_time:.4f}",
                "Device": device_info
            })

        except Exception as e:
            print(f"{label:<20} | {number_of_alternatives:<12} | ERROR: {str(e)}")
            results.append({
                "Test Case": label,
                "Alternatives": number_of_alternatives,
                "Time (s)": "ERROR",
                "Device": device_info
            })

    # Print summary
    print("\n\nResults Summary:")
    print("=" * 70)
    for r in results:
        print(f"Test Case: {r['Test Case']}")
        print(f"  Alternatives: {r['Alternatives']} | Time: {r['Time (s)']} | Device: {r['Device']}")
        print()

if __name__ == "__main__":
    benchmark()
