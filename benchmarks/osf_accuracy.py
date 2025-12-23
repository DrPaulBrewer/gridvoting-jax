"""OSF accuracy validation for all solvers"""
import gridvoting_jax as gv
import jax.numpy as jnp
import pandas as pd
from pathlib import Path

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

def compare_with_osf(g, solver_name):
    """Compare solver result with OSF reference"""
    # Solve with specified solver
    model = gv.bjm_spatial_triangle(g=g, zi=False)
    model.analyze(solver=solver_name)
    
    # Load OSF data
    cache_dir = Path("/tmp/gridvoting_osf_cache")
    osf_file = cache_dir / f"{g}_MI_stationary_distribution.csv"
    
    if not osf_file.exists():
        return None
    
    osf_df = pd.read_csv(osf_file)
    osf_data = 10 ** osf_df['log10prob'].values
    
    # Calculate metrics
    diff = model.stationary_distribution - osf_data
    l1_norm = float(jnp.sum(jnp.abs(diff)))
    max_diff = float(jnp.max(jnp.abs(diff)))
    correlation = float(jnp.corrcoef(
        model.stationary_distribution, 
        osf_data
    )[0, 1])
    
    return {
        'g': g,
        'solver': solver_name,
        'l1_norm': l1_norm,
        'max_diff': max_diff,
        'correlation': correlation,
        'accuracy': classify_accuracy(l1_norm)
    }

def run_osf_validation(grid_sizes=[20, 40, 60, 80]):
    """Run OSF validation for all solvers"""
    results = []
    
    for g in grid_sizes:
        print(f"\nValidating g={g}...")
        
        for solver in ['dense', 'lazy_power_method', 'lazy_grid_upscaling']:
            if solver == 'dense' and g > 60:
                continue  # Skip dense for large grids
            
            print(f"  - {solver}...")
            result = compare_with_osf(g, solver)
            if result:
                results.append(result)
    
    df = pd.DataFrame(results)
    
    # Format output grouped by grid size
    print("\n" + "=" * 100)
    print("OSF ACCURACY VALIDATION")
    print("=" * 100)
    print(f"{'Grid':<6} {'Solver':<25} {'L1 Norm':<12} {'Max Diff':<12} {'Correlation':<12} {'Accuracy':<12}")
    print("-" * 100)
    
    for g in sorted(df['g'].unique()):
        g_data = df[df['g'] == g]
        for idx, row in g_data.iterrows():
            grid_label = f"g={g}" if idx == g_data.index[0] else ""
            solver_label = row['solver'].replace('_', ' ').title()
            
            print(f"{grid_label:<6} {solver_label:<25} {row['l1_norm']:<12.2e} "
                  f"{row['max_diff']:<12.2e} {row['correlation']:<12.6f} {row['accuracy']:<12}")
        print()
    
    return df

if __name__ == "__main__":
    run_osf_validation()
