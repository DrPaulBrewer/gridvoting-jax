
import os
import time
import jax.numpy as jnp
import pandas as pd
from gridvoting_jax.core import enable_float64
from gridvoting_jax.spatial import Grid
from gridvoting_jax.dynamics import VotingModel
from gridvoting_jax.benchmarks.osf_comparison import load_osf_distribution

# Enable float64 for precision comparison
enable_float64()

def benchmark_solvers_vs_osf(g=20):
    print(f"\n==================================================")
    print(f"Benchmarking Solvers vs OSF Data (g={g}, MI, Triangle 1)")
    print(f"==================================================")
    
    # 0. Load Reference Data
    print("Loading OSF data...")
    df_osf = load_osf_distribution(g=g, zi=False) # MI
    if df_osf is None:
        print("Error: Could not load OSF data. Aborting.")
        return
        
    # Extract prob from log10prob
    osf_prob = jnp.array(10 ** df_osf['log10prob'].values)
    osf_prob = osf_prob / jnp.sum(osf_prob) # Normalize
    
    # 1. Setup Grid (Triangle 1 configuration matches these bounds)
    # OSF data is usually on [-20, 20] or similar for these points
    # Let's double check alignment.
    # Triangle 1 from paper: A(-15, -9), B(0, 17), C(15, -9).
    # OSF data uses 2*g steps? 
    # g=20 -> 1681 points (41x41). 41 = 2*20 + 1.
    # g=80 -> 25921 points (161x161). 161 = 2*80 + 1.
    
    steps = 2 * g
    grid = Grid(x0=-20.0, x1=20.0, xstep=40.0/steps, y0=-20.0, y1=20.0, ystep=40.0/steps)
    N = len(grid.points)
    print(f"Grid points: {N} (OSF has {len(osf_prob)})")
    
    if N != len(osf_prob):
        print(f"Warning: Grid size mismatch! OSF: {len(osf_prob)}, Pkg: {N}")
        return

    # 2. Setup Voters
    voter_ideal_points = jnp.array([
        [-15.0, -9.0],
        [0.0, 17.0],
        [15.0, -9.0]
    ])
    
    utils = grid.spatial_utilities(voter_ideal_points=voter_ideal_points)
    
    # 3. Create Model
    model = VotingModel(
        utility_functions=utils,
        number_of_voters=3,
        number_of_feasible_alternatives=N,
        majority=2,
        zi=False
    )
    
    solvers = ["gmres_matrix_inversion", "power_method", "grid_upscaling"]
    
    results = {}
    
    for solver in solvers:
        print(f"\nRunning {solver}...")
        start_time = time.time()
        try:
            model.analyze(
                solver=solver, 
                grid=grid, 
                voter_ideal_points=voter_ideal_points,
                tolerance=1e-6,
                max_iterations=5000
            )
            duration = time.time() - start_time
            
            dist = model.stationary_distribution
            l1_diff = jnp.linalg.norm(dist - osf_prob, ord=1)
            
            print(f"  ✓ Duration: {duration:.2f}s")
            print(f"  ✓ L1 Diff vs OSF: {l1_diff:.2e}")
            results[solver] = {"status": "PASS", "l1": l1_diff, "time": duration}
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[solver] = {"status": "FAIL", "error": str(e)}

    print("\nSUMMARY")
    print("--------------------------------------------------")
    for s, res in results.items():
        if res["status"] == "PASS":
            print(f"{s:<25}: PASS (L1={res['l1']:.2e}, Time={res['time']:.2f}s)")
        else:
            print(f"{s:<25}: FAIL ({res['error']})")

if __name__ == "__main__":
    benchmark_solvers_vs_osf(g=20)
