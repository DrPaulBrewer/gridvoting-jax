"""Benchmark lazy vs dense solver implementations"""
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import gridvoting_jax as gv
import jax
import jax.numpy as jnp

def classify_accuracy(l1_norm):
    """Classify accuracy based on L1 norm"""
    if l1_norm < 1e-7:
        return "Excellent"
    elif l1_norm < 1e-6:
        return "Good"
    elif l1_norm < 1e-5:
        return "Acceptable"
    else:
        return "Poor"

def load_osf_reference(g):
    """Load OSF reference data for comparison"""
    cache_dir = Path("/tmp/gridvoting_osf_cache")
    osf_file = cache_dir / f"{g}_MI_stationary_distribution.csv"
    
    if not osf_file.exists():
        return None
    
    osf_df = pd.read_csv(osf_file)
    return 10 ** osf_df['log10prob'].values

def get_jax_memory_stats():
    """Get JAX memory statistics (GPU or CPU)"""
    device = jax.devices()[0]
    
    if device.platform == 'gpu':
        try:
            # GPU memory stats
            stats = device.memory_stats()
            return stats.get('peak_bytes_in_use', 0) / 1024 / 1024  # MB
        except:
            return None
    else:
        # CPU memory tracking via JAX
        # Note: JAX doesn't track CPU memory directly
        return None

def benchmark_solver(g, solver_name, max_iterations=3000, track_memory=True):
    """Benchmark a single solver configuration"""
    try:
        # Clear JAX cache
        jax.clear_caches()
        
        # Time execution
        start_time = time.time()
        
        model = gv.bjm_spatial_triangle(g=g, zi=False)
        model.analyze(solver=solver_name, max_iterations=max_iterations)
        
        elapsed = time.time() - start_time
        
        # Get memory stats (only if tracking enabled)
        peak_memory_mb = None
        if track_memory:
            peak_memory_mb = get_jax_memory_stats()
        
        # Calculate accuracy vs OSF
        osf_data = load_osf_reference(g)
        if osf_data is not None:
            diff = model.stationary_distribution - osf_data
            l1_norm = float(jnp.sum(jnp.abs(diff)))
            accuracy = classify_accuracy(l1_norm)
        else:
            l1_norm = None
            accuracy = "N/A"
        
        return {
            'g': g,
            'solver': solver_name,
            'time_seconds': elapsed,
            'peak_memory_mb': peak_memory_mb,
            'l1_norm': l1_norm,
            'accuracy': accuracy,
            'probability_sum': float(jnp.sum(model.stationary_distribution)),
            'N': model.model.number_of_feasible_alternatives,
            'status': 'Success'
        }
    except Exception as e:
        return {
            'g': g,
            'solver': solver_name,
            'time_seconds': None,
            'peak_memory_mb': None,
            'l1_norm': None,
            'accuracy': 'N/A',
            'probability_sum': None,
            'N': None,
            'status': f'Failed: {str(e)[:50]}'
        }

def format_results(df):
    """Format results for display with grouped comparison"""
    output = []
    output.append("\nBENCHMARK RESULTS")
    output.append("=" * 100)
    output.append(f"{'Grid':<6} {'Solver':<25} {'Time(s)':<10} {'Memory(MB)':<12} {'L1 Norm':<12} {'Accuracy':<12} {'N':<8}")
    output.append("-" * 100)
    
    for g in sorted(df['g'].unique()):
        g_data = df[df['g'] == g].sort_values('solver')
        
        for idx, row in g_data.iterrows():
            grid_label = f"g={g}" if idx == g_data.index[0] else ""
            solver_label = row['solver'].replace('_', ' ').title()
            
            if row['status'] == 'Success':
                time_str = f"{row['time_seconds']:.2f}"
                mem_str = f"{row['peak_memory_mb']:.1f}" if row['peak_memory_mb'] else "N/A"
                l1_str = f"{row['l1_norm']:.2e}" if row['l1_norm'] else "N/A"
                acc_str = row['accuracy']
                n_str = str(row['N'])
            else:
                time_str = mem_str = l1_str = "N/A"
                acc_str = row['status']
                n_str = "N/A"
            
            output.append(f"{grid_label:<6} {solver_label:<25} {time_str:<10} {mem_str:<12} {l1_str:<12} {acc_str:<12} {n_str:<8}")
        
        output.append("")  # Blank line between grid sizes
    
    return "\n".join(output)

def run_benchmarks(grid_sizes, output_file=None, track_memory=True):
    """Run full benchmark suite"""
    results = []
    
    for g in grid_sizes:
        print(f"\nBenchmarking g={g}...")
        
        # Dense solver (if memory allows)
        if g <= 60:
            print(f"  - Dense solver...")
            result = benchmark_solver(g, "dense", track_memory=track_memory)
            results.append(result)
        
        # Lazy power method
        print(f"  - Lazy power method...")
        result = benchmark_solver(g, "lazy_power_method", track_memory=track_memory)
        results.append(result)
        
        # Lazy grid upscaling (GMRES)
        print(f"  - Lazy grid upscaling...")
        result = benchmark_solver(g, "lazy_grid_upscaling", track_memory=track_memory)
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Display formatted results
    print(format_results(df))
    
    # Save to file
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--grids', nargs='+', type=int, 
                       default=[20, 40, 60],
                       help='Grid sizes to benchmark')
    parser.add_argument('--quick', action='store_true',
                       help='Quick benchmark (g=20,40)')
    parser.add_argument('--extended', action='store_true',
                       help='Extended benchmark (g=20,40,60,80,100)')
    parser.add_argument('--output', type=str,
                       default='benchmark_results.csv',
                       help='Output CSV file')
    parser.add_argument('--no-memory', action='store_true',
                       help='Disable memory tracking (faster)')
    
    args = parser.parse_args()
    
    if args.quick:
        grid_sizes = [20, 40]
    elif args.extended:
        grid_sizes = [20, 40, 60, 80, 100]
    else:
        grid_sizes = args.grids
    
    track_memory = not args.no_memory
    
    run_benchmarks(grid_sizes, args.output, track_memory=track_memory)
