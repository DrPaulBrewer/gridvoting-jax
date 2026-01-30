
import gridvoting_jax as gv
import jax
import jax.numpy as jnp
import time
import argparse

def run_benchmark(voters, thetastep, decimals=None):
    print(f"Benchmarking with voters={voters}, thetastep={thetastep}...")
    
    # 1. Setup Model (Lazy P)
    start_setup = time.time()
    svm = gv.models.examples.shapes.ring(g=10, r=6, voters=voters, polar=True, thetastep=thetastep, decimals=decimals)
    pg = svm.grid
    parts = pg.partition_from_rotation(angle=360//voters)
    svm.analyze() # Creates MarkovChain
    setup_time = time.time() - start_setup
    
    MC_lazy = svm.model.MarkovChain
    n_states = MC_lazy.P.shape[0]
    print(f"  State space size: {n_states}")
    print(f"  Setup time: {setup_time:.4f}s")

    # 2. Benchmark Lazy Lumping (Current Implementation)
    print("  Benchmarking Lazy Lumping...")
    # Force generic compile if needed, but we want to measure the python/lax overhead
    lump_func = gv.stochastic.markov.lump
    
    # Warmup
    _ = lump_func(MC_lazy, parts).P.block_until_ready()
    
    start_lazy = time.time()
    LM_lazy = lump_func(MC_lazy, parts)
    _ = LM_lazy.P.block_until_ready()
    lazy_time = time.time() - start_lazy
    print(f"  Lazy Lump Time: {lazy_time:.4f}s")
    
    # 3. Benchmark Dense Lumping (Baseline)
    print("  Benchmarking Dense Lumping...")
    # Materialize dense P first (exclude from timing)
    dense_P = MC_lazy.P.to_dense()
    MC_dense = gv.stochastic.markov.MarkovChain(P=dense_P)
    
    # Warmup
    _ = lump_func(MC_dense, parts).P.block_until_ready()
    
    start_dense = time.time()
    LM_dense = lump_func(MC_dense, parts)
    _ = LM_dense.P.block_until_ready()
    dense_time = time.time() - start_dense
    print(f"  Dense Lump Time: {dense_time:.4f}s")
    
    # 4. Correctness Check
    diff = jnp.max(jnp.abs(LM_lazy.P - LM_dense.P))
    print(f"  Max Difference: {diff:.2e}")
    if diff > 1e-6:
        print("  WARNING: Results differ!")
        
    return lazy_time, dense_time

if __name__ == "__main__":
    # Larger case for noticeable timing
    # voters=11, theta=2 gives roughly N=180 states? No, grid size depends on r=6, g=10.
    # The ring model has many states.
    
    run_benchmark(voters=5, thetastep=4)  # Warmup / Small
    run_benchmark(voters=9, thetastep=4) # Larger (360/9 = 40)
