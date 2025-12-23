
import jax
import jax.lax
import jax.numpy as jnp
from warnings import warn

# Import from core
from ..core import (
    TOLERANCE, 
    NEGATIVE_PROBABILITY_TOLERANCE, 
    assert_valid_transition_matrix, 
    _move_neg_prob_to_max
)

class MarkovChain:
    def __init__(self, *, P, tolerance=None):
        """initializes a MarkovChain instance by copying in the transition
        matrix P and calculating chain properties"""
        if tolerance is None:
            tolerance = TOLERANCE
        self.P = jnp.asarray(P)  # copy transition matrix to JAX array
        assert_valid_transition_matrix(P)
        diagP = jnp.diagonal(self.P)
        self.absorbing_points = jnp.equal(diagP, 1.0)
        self.unreachable_points = jnp.equal(jnp.sum(self.P, axis=0), diagP)
        self.has_unique_stationary_distribution = not jnp.any(self.absorbing_points)


    def evolve(self, x):
        """ evolve the probability vector x_t one step in the Markov Chain by returning x*P. """
        return jnp.dot(x,self.P)

    def L1_norm_of_single_step_change(self, x):
        """returns float(L1(xP-x))"""
        return float(jnp.linalg.norm(self.evolve(x) - x, ord=1))

    def solve_for_unit_eigenvector(self):
        """This is another way to potentially find the stationary distribution,
        but can suffer from numerical irregularities like negative entries.
        Assumes eigenvalue of 1.0 exists and solves for the eigenvector by
        considering a related matrix equation Q v = b, where:
        Q is P transpose minus the identity matrix I, with the first row
        replaced by all ones for the vector scaling requirement;
        v is the eigenvector of eigenvalue 1 to be found; and
        b is the first basis vector, where b[0]=1 and 0 elsewhere."""
        n = self.P.shape[0]
        Q = jnp.transpose(self.P) - jnp.eye(n)
        Q = Q.at[0].set(jnp.ones(n))  # JAX immutable update
        b = jnp.zeros(n)
        b = b.at[0].set(1.0)  # JAX immutable update        
        error_unable_msg = "unable to find unique unit eigenvector "
        try:
            unit_eigenvector = jnp.linalg.solve(Q, b)
        except Exception as err:
            warn(str(err)) # print the original exception lest it be lost for debugging purposes
            raise RuntimeError(error_unable_msg+"(solver)")
        
        if jnp.isnan(unit_eigenvector.sum()):
            raise RuntimeError(error_unable_msg+"(nan)")
        
        min_component = float(unit_eigenvector.min())
        # Use extracted constant from core for negative checks
        if ((min_component < 0.0) and (min_component > NEGATIVE_PROBABILITY_TOLERANCE)):
            unit_eigenvector = _move_neg_prob_to_max(unit_eigenvector)
            unit_eigenvector = self.evolve(unit_eigenvector)
            min_component = float(unit_eigenvector.min())
        
        if (min_component < 0.0):
            neg_msg = "(negative components: "+str(min_component)+" )"
            warn(neg_msg)
            raise RuntimeError(error_unable_msg+neg_msg)
        
        self.unit_eigenvector = unit_eigenvector
        return self.unit_eigenvector


    def find_unique_stationary_distribution(self, *, tolerance=None, solver="full_matrix_inversion", initial_guess=None, max_iterations=2000, timeout=30.0, **kwargs):
        """
        Finds the stationary distribution for a Markov Chain.
        
        Args:
            tolerance: Convergence tolerance (default: module TOLERANCE).
            solver: Strategy to use. Options:
                - "full_matrix_inversion": (Default) Direct algebraic solve (O(N^3)). Best for N < 5000.
                - "gmres_matrix_inversion": Iterative linear solver (GMRES). Low memory (O(N^2) or O(N)).
                - "power_method": Iterative power method (O(N^2)). 
                  If initial_guess is None, uses Dual-Start Entropy strategy.
                  If initial_guess is provided, uses Single-Start Refinement.
            initial_guess: Optional starting distribution for "power_method".
            max_iterations: Maximum iterations for iterative solvers.
            timeout: Maximum time in seconds for iterative solvers (default: 10.0).
        """
        if tolerance is None:
            tolerance = TOLERANCE
            
        if jnp.any(self.absorbing_points):
            self.stationary_distribution = None
            return None
            
        # Memory Check
        try:
            from ..core import get_available_memory_bytes
            available_mem = get_available_memory_bytes()
            
            if available_mem is not None:
                n = self.P.shape[0]
                # Determine element size (float32=4, float64=8)
                item_size = self.P.dtype.itemsize
                
                estimated_needed = 0
                if solver == "full_matrix_inversion":
                    # P(N^2) + Q(N^2) + Result(N^2) + Overhead
                    estimated_needed = 3 * (n**2) * item_size
                elif solver == "gmres_matrix_inversion":
                     # Matrix-vector product based (often doesn't materialize full matrix if sparse, 
                     # but here explicit P is used). 
                     # P(N^2) + Vectors(k*N)
                    estimated_needed = (n**2) * item_size + (max_iterations * n * item_size)
                
                # Safety margin (allow using up to 90% of available)
                if estimated_needed > available_mem * 0.9:
                    msg = (f"Estimated memory required ({estimated_needed / 1e9:.2f} GB) "
                           f"exceeds 90% of available memory ({available_mem / 1e9:.2f} GB) "
                           f"for solver '{solver}'.")
                    raise MemoryError(msg)
        except ImportError:
            pass # Core might not be fully initialized or circular import
        except MemoryError:
            raise # Re-raise actual memory errors
        except Exception as e:
            warn(f"Memory check failed: {e}")

        # Dispatch to solver
        if solver == "full_matrix_inversion":
            self.stationary_distribution = self._solve_full_matrix_inversion(tolerance)
        elif solver == "gmres_matrix_inversion":
            self.stationary_distribution = self._solve_gmres_matrix_inversion(tolerance, max_iterations)
        elif solver == "power_method":
            self.stationary_distribution = self._solve_power_method(tolerance, max_iterations, initial_guess, timeout)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        # Verification
        self.check_norm = self.L1_norm_of_single_step_change(self.stationary_distribution)
        if self.check_norm > tolerance:
            # If iterative solvers failed to converge tightly enough but didn't raise
            warn(f"Stationary distribution check norm {self.check_norm} exceeds tolerance {tolerance}")
            
        return self.stationary_distribution

    def _solve_full_matrix_inversion(self, tolerance):
        """Original algebraic solver using direct dense matrix inversion / linear solve."""
        return self.solve_for_unit_eigenvector()

    def _solve_gmres_matrix_inversion(self, tolerance, max_iterations, initial_guess=None):
        """
        Find stationary distribution using GMRES iterative solver.
        Solves (P.T - I)v = 0 subject to sum(v)=1.
        
        Equation: vP = v  =>  P.T v.T = v.T  => (P.T - I)v = 0
        Constraint: sum(v) = 1
        
        We enforce constraint by replacing the first equation (row) of the system
        with the sum constraint (all ones).
        
        Args:
            tolerance: Convergence tolerance
            max_iterations: Maximum GMRES iterations
            initial_guess: Optional initial guess for GMRES (useful for grid upscaling)
        """
        n = self.P.shape[0]
        I = jnp.eye(n)
        
        # System matrix A = P.T - I
        # We want to perform matrix-vector product A @ x without strictly materializing A if possible,
        # but for now, explicit A is fine as it fits in memory (unlike factorization).
        A = self.P.T - I
        
        # Enforce sum(v) = 1 constraint on the first row
        # This makes the system A' v = b where b = [1, 0, ... 0]
        # And the first row of A' is [1, 1, ... 1]
        A = A.at[0, :].set(1.0)
        
        b = jnp.zeros(n)
        b = b.at[0].set(1.0)
        
        # Prepare initial guess
        x0 = initial_guess if initial_guess is not None else jnp.ones(n) / n
        
        # Use JAX's GMRES
        # tol in gmres is residual tolerance, roughly related to error
        v, info = jax.scipy.sparse.linalg.gmres(
            lambda x: jnp.dot(A, x), 
            b,
            x0=x0,
            tol=tolerance, 
            maxiter=max_iterations
        )
        
        if info > 0:
            warn(f"GMRES did not converge in {max_iterations} iterations based on internal criteria.")
        
        # Enforce non-negativity and renormalization (numerical artifacts)
        v = jnp.abs(v)
        v = v / jnp.sum(v)
        
        return v

    def _solve_power_method(self, tolerance, max_iterations, initial_guess=None, timeout=30.0):
        """
        Power method for finding stationary distribution.
        
        Modes:
        1. Single-Start Refinement: Uses initial_guess.
        2. Dual-Start Entropy: Uses Max/Min entropy rows as starts.
        
        Args:
            timeout: Max execution time in seconds.
        """
        import time
        n = self.P.shape[0]
        start_time = time.time()
        
        # Adaptive batching for time checks
        check_interval = 10
        next_check = check_interval
        
        if initial_guess is not None:
            # Mode 1: Refine existing guess
            v = initial_guess
            i = 0
            while i < max_iterations:
                # Evolve until next check using JAX compiled loop
                batch_end = min(next_check, max_iterations)
                batch_size = batch_end - i
                
                # Use lax.fori_loop for compiled batched evolution
                # Pass P as part of carry to avoid closure capture
                def evolve_step(_, carry):
                    vec, P = carry
                    return (jnp.dot(vec, P), P)
                v, _ = jax.lax.fori_loop(0, batch_size, evolve_step, (v, self.P))
                i = batch_end
                
                # Check convergence and timeout
                diff = jnp.linalg.norm(self.evolve(v) - v, ord=1)
                if diff < tolerance:
                    return v
                
                if (time.time() - start_time) > timeout:
                    warn(f"Power method timed out after {timeout}s (iter {i}). Check norm: {diff}")
                    return v
                
                # Adaptive: Increase interval
                if check_interval < 1000:
                    check_interval *= 2
                next_check = i + check_interval
                
            # Final check
            diff = jnp.linalg.norm(self.evolve(v) - v, ord=1)
            warn(f"Power method (Single-Start) did not converge in {max_iterations} iterations. Final diff: {diff}")
            return v
            
        else:
            # Mode 2: Dual-Start Entropy
            # Calculate entropy of each row of P to find diverse starting points
            # Entropy of a row P[i]: H(i) = - sum(P[i,j] * log2(P[i,j]))
            # Avoid log(0) with mask or adding epsilon
            P_safe = jnp.where(self.P > 0, self.P, 1.0) # log(1)=0, so 0 contributions are 0
            row_entropy = -jnp.sum(self.P * jnp.log2(P_safe), axis=1)
            
            # Start 1: Max entropy (most uncertain transition)
            idx_max = jnp.argmax(row_entropy).item()
            v1 = jnp.zeros(n).at[idx_max].set(1.0)
            
            # Start 2: Min entropy (most deterministic transition)
            idx_min = jnp.argmin(row_entropy).item()
            v2 = jnp.zeros(n).at[idx_min].set(1.0)
            
            # Evolve both (batched with deferred checks)
            i = 0
            while i < max_iterations:
                # Stack once per batch
                V = jnp.stack([v1, v2], axis=0)  # Shape: (2, n)
                
                # Evolve batch until next check using JAX compiled loop
                batch_end = min(next_check, max_iterations)
                batch_size = batch_end - i
                
                # Use lax.fori_loop for compiled batched evolution
                # Pass P as part of carry to avoid closure capture
                def evolve_batch_step(_, carry):
                    V_state, P = carry
                    return (jnp.dot(V_state, P), P)
                V, _ = jax.lax.fori_loop(0, batch_size, evolve_batch_step, (V, self.P))
                i = batch_end
                
                # Unpack once per batch
                v1, v2 = V[0], V[1]
                
                # Check convergence
                diff = jnp.linalg.norm(v1 - v2, ord=1)
                if diff < tolerance:
                    return (v1 + v2) / 2.0
                
                # Check timeout
                if (time.time() - start_time) > timeout:
                    warn(f"Power method timed out after {timeout}s (iter {i}). Diff between starts: {diff}")
                    return (v1 + v2) / 2.0
                
                # Adaptive: Increase interval
                if check_interval < 1000:
                    check_interval *= 2
                next_check = i + check_interval
            
            # Final convergence check
            diff = jnp.linalg.norm(v1 - v2, ord=1)
            warn(f"Power method (Dual-Start) did not converge by {max_iterations}. Final diff between chains: {diff}")
            return (v1 + v2) / 2.0

    def diagnostic_metrics(self):
        """ return Markov chain approximation metrics in mathematician-friendly format """
        metrics = {
            '||F||': self.P.shape[0],
            '(𝝨𝝿)-1':  float(self.stationary_distribution.sum())-1.0, # cast to float to avoid singleton
            '||𝝿P-𝝿||_L1_norm': self.L1_norm_of_single_step_change(
                              self.stationary_distribution
                          )
        }
        return metrics
