
import time
import jax.numpy as jnp
import numpy as np
import gridvoting_jax as gv
from gridvoting_jax.models.examples import bjm_spatial_triangle

def benchmark_pareto_lumping():
    print(f"{'Grid':<6} | {'ZI':<5} | {'P(Pareto) Full':<14} | {'P(Pareto) Lump':<14} | {'Error':<10} | {'Time Full (s)':<13} | {'Time Lump (s)':<13}")
    print("-" * 100)

    # Loop over grid sizes and zero_interest (zi) settings
    for g in [20, 40, 60, 80]:
        for zi in [True, False]:
            try:
                # Instantiate model for this iteration
                model = bjm_spatial_triangle(g=g, zi=zi)

                # 0. Pre-requisite: Calculate Pareto Mask
                # This determines the partition. We calculate it once.
                # Accessing .Pareto triggers computation on a *copy* of the model (unanimous),
                # so it doesn't solve the main model.
                start_mask = time.time()
                pareto_mask = model.Pareto
                # Force computation if lazy
                if hasattr(pareto_mask, 'block_until_ready'):
                    pareto_mask.block_until_ready()
                # print(f"Mask calc time: {time.time() - start_mask:.4f}s") 

                # 1. Full Solution (Reference)
                # ----------------------------
                start_full = time.time()
                # Run full analysis (GMRES or best solver)
                model.analyze()
                time_full = time.time() - start_full
                
                # Get stationary distribution
                pi_full = model.stationary_distribution
                
                # Calculate reference probabilities
                p_pareto_full = float(jnp.sum(pi_full[pareto_mask]))
                # p_outside_full = float(jnp.sum(pi_full[~pareto_mask]))
                
                # 2. Approximate Lumping Solution
                # -------------------------------
                # We need the transition matrix P to create a MarkovChain for lumping.
                # Accessing model.model._get_transition_matrix() is correct for SpatialVotingModel
                P = model.model._get_transition_matrix()
                mc = gv.MarkovChain(P=P)

                # Construct partition: [Pareto Indices, Outside Indices]
                indices = jnp.arange(model.grid.len)
                pareto_indices = indices[pareto_mask].tolist()
                outside_indices = indices[~pareto_mask].tolist()
                
                if len(outside_indices) == 0:
                   # Edge case: All points are Pareto (e.g. Unanimity rule or weird geometry)
                   # Cannot lump into 2 states if one is empty. 
                   # But for BJM Triangle with Majority, this shouldn't happen.
                   # If it does, we handle it by skipping or single-state lump
                   print(f"Warning: Full Pareto set for g={g}, zi={zi}. Skipping lumping.")
                   continue

                partition = [pareto_indices, outside_indices]
                
                start_lump = time.time()
                
                # Create lumped chain (improper lumping with uniform weights)
                lumped_mc = gv.lump(mc, partition)
                
                # Solve 2x2 chain
                pi_lumped = lumped_mc.find_unique_stationary_distribution()
                
                time_lump = time.time() - start_lump
                
                # Extract approximate probabilities
                # Partition 0 is Pareto, Partition 1 is Outside
                p_pareto_lump = float(pi_lumped[0])
                p_outside_lump = float(pi_lumped[1])
                
                # Capture lumped transition matrix for output
                P_lumped_matrix = lumped_mc.P
                
            except Exception as e:
                print(f"Error in Benchmark g={g}, zi={zi}: {e}")
                # Print exception with traceback for debugging
                import traceback
                traceback.print_exc()
                continue

            # 3. Report Results
            # -----------------
            error = abs(p_pareto_full - p_pareto_lump)
            
            print(f"{g:<6} | {str(zi):<5} | {p_pareto_full:<14.6f} | {p_pareto_lump:<14.6f} | {error:<10.6f} | {time_full:<13.4f} | {time_lump:<13.4f}")
            
            # Print 2x2 Matrix
            print(f"   2x2 Lumped Matrix (g={g}, zi={zi}):")
            P_np = np.array(P_lumped_matrix)
            print(f"   [[ {P_np[0,0]:.3f}, {P_np[0,1]:.3f} ]")
            print(f"    [ {P_np[1,0]:.3f}, {P_np[1,1]:.3f} ]]")
            print("-" * 100)

if __name__ == "__main__":
    benchmark_pareto_lumping()
