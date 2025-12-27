#!/usr/bin/env python3
"""
Debug script to inspect the initial guess from grid upscaling in ZI mode.
"""

import gridvoting_jax as gv
import jax.numpy as jnp
from gridvoting_jax.models.base import VotingModel

def inspect_grid_upscaling_initial_guess(g=20, zi=True):
    """Inspect what initial guess grid upscaling creates."""
    print(f"\n{'='*60}")
    print(f"Inspecting grid upscaling initial guess: g={g}, zi={zi}")
    print(f"{'='*60}")
    
    # Create model
    grid = gv.Grid(x0=-g, x1=g, y0=-g, y1=g)
    voter_ideal_points = [[-15, -9], [0, 17], [15, -9]]
    model = gv.SpatialVotingModel(
        voter_ideal_points=voter_ideal_points,
        grid=grid,
        number_of_voters=3,
        majority=2,
        zi=zi,
        distance_measure='sqeuclidean'
    )
    
    # Manually replicate grid upscaling logic
    voter_ideal_points_arr = jnp.asarray(voter_ideal_points)
    min_xy = jnp.min(voter_ideal_points_arr, axis=0)
    max_xy = jnp.max(voter_ideal_points_arr, axis=0)
    
    x0_sub, y0_sub = min_xy[0] - grid.xstep, min_xy[1] - grid.ystep
    x1_sub, y1_sub = max_xy[0] + grid.xstep, max_xy[1] + grid.ystep
    
    box_mask = grid.within_box(x0=x0_sub, x1=x1_sub, y0=y0_sub, y1=y1_sub)
    valid_indices = jnp.nonzero(box_mask)[0]
    
    print(f"\nSubgrid info:")
    print(f"  Full grid size: {grid.len}")
    print(f"  Subgrid size: {len(valid_indices)}")
    print(f"  Subgrid bounds: x=[{x0_sub}, {x1_sub}], y=[{y0_sub}, {y1_sub}]")
    
    # Solve subproblem
    sub_utility_functions = model.utility_functions[:, valid_indices]
    sub_model = VotingModel(
        utility_functions=sub_utility_functions,
        number_of_voters=3,
        number_of_feasible_alternatives=len(valid_indices),
        majority=2,
        zi=zi
    )
    sub_model.analyze(solver="full_matrix_inversion")
    
    print(f"\nSubgrid solution:")
    print(f"  Core exists: {sub_model.core_exists}")
    print(f"  Distribution sum: {jnp.sum(sub_model.stationary_distribution):.6f}")
    print(f"  Distribution has NaN: {jnp.any(jnp.isnan(sub_model.stationary_distribution))}")
    print(f"  Distribution min/max: [{jnp.min(sub_model.stationary_distribution):.6e}, {jnp.max(sub_model.stationary_distribution):.6e}]")
    
    if not sub_model.core_exists:
        # Create upscaled initial guess
        embed_fn = grid.embedding(valid=box_mask)
        upscaled_dist = embed_fn(sub_model.stationary_distribution)
        initial_guess = upscaled_dist / jnp.sum(upscaled_dist)
        
        print(f"\nInitial guess (upscaled):")
        print(f"  Sum: {jnp.sum(initial_guess):.10f}")
        print(f"  Has NaN: {jnp.any(jnp.isnan(initial_guess))}")
        print(f"  Has Inf: {jnp.any(jnp.isinf(initial_guess))}")
        print(f"  Min/max: [{jnp.min(initial_guess):.6e}, {jnp.max(initial_guess):.6e}]")
        print(f"  Non-zero elements: {jnp.sum(initial_guess > 0)}")
        print(f"  Zero elements: {jnp.sum(initial_guess == 0)}")
        
        # Check if sum is problematic
        if jnp.sum(upscaled_dist) == 0:
            print(f"  ❌ WARNING: upscaled_dist sum is ZERO before normalization!")
        
        # Now test if this initial guess causes GMRES to fail
        print(f"\nTesting lazy GMRES with this initial guess...")
        try:
            model.model.analyze_lazy(solver="gmres", force_lazy=True, initial_guess=initial_guess)
            result = model.stationary_distribution
            print(f"  Result sum: {jnp.sum(result):.6f}")
            print(f"  Result has NaN: {jnp.any(jnp.isnan(result))}")
            if jnp.any(jnp.isnan(result)):
                print(f"  ❌ Initial guess causes NaN!")
            else:
                print(f"  ✓ Initial guess works fine")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    else:
        print(f"\n  Core exists in subgrid, no initial guess created")

if __name__ == "__main__":
    # Test ZI mode (problematic)
    inspect_grid_upscaling_initial_guess(g=20, zi=True)
    
    # Test MI mode (works fine)
    inspect_grid_upscaling_initial_guess(g=20, zi=False)
