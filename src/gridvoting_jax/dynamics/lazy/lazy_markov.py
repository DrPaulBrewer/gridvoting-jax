"""Lazy Markov chain implementation for memory-efficient stationary distribution solving.

This module provides LazyMarkovChain (always lazy) and FlexMarkovChain (auto-selecting)
for solving Markov chains without materializing the full transition matrix.
"""

import jax
import jax.numpy as jnp
from warnings import warn

from ..markov import MarkovChain
from .base import LazyTransitionMatrix
from ...core import TOLERANCE, _move_neg_prob_to_max, DTYPE_FLOAT


class LazyMarkovChain:
    """
    Markov chain with lazy transition matrix (always lazy).
    
    Supports only iterative solvers: gmres, power_method.
    Does NOT support full_matrix_inversion (requires dense P).
    """
    
    def __init__(self, *, lazy_P, tolerance=None):
        """
        Initialize with LazyTransitionMatrix.
        
        Args:
            lazy_P: LazyTransitionMatrix instance
            tolerance: Convergence tolerance (default: module TOLERANCE)
        """
        self.lazy_P = lazy_P
        self.tolerance = tolerance if tolerance is not None else TOLERANCE
        self._stationary_distribution = None
        self._analyzed = False
    
    def find_unique_stationary_distribution(self, *, solver="gmres", initial_guess=None, 
                                           tolerance=None, max_iterations=2000, timeout=30.0, **kwargs):
        """
        Find stationary distribution using lazy solvers.
        
        Args:
            solver: "gmres" (default), "power_method", or "bifurcated_power_method"
            initial_guess: Optional initial guess for iterative solver
            tolerance: Convergence tolerance (default: self.tolerance)
            max_iterations: Maximum iterations
            timeout: Timeout for power method (seconds)
            **kwargs: Additional solver parameters
        
        Returns:
            self (for chaining)
        """
        if tolerance is None:
            tolerance = self.tolerance
        
        if solver == "gmres":
            self._stationary_distribution = self._solve_gmres_lazy(
                tolerance=tolerance, 
                max_iterations=max_iterations, 
                initial_guess=initial_guess
            )
        elif solver == "power_method":
            self._stationary_distribution = self._solve_power_method_lazy(
                tolerance=tolerance, 
                max_iterations=max_iterations, 
                initial_guess=initial_guess,
                timeout=timeout
            )
        elif solver == "bifurcated_power_method":
            self._stationary_distribution = self._solve_bifurcated_power_method_lazy(
                tolerance=tolerance, 
                max_iterations=max_iterations,
                timeout=timeout
            )
        else:
            raise ValueError(f"LazyMarkovChain only supports 'gmres', 'power_method', or 'bifurcated_power_method', got '{solver}'")
        
        self._analyzed = True
        return self
    
    def _solve_gmres_lazy(self, tolerance, max_iterations, initial_guess=None):
        """
        GMRES using lazy matvec operations.
        
        Solves (P.T - I)v = 0 subject to sum(v)=1.
        
        Args:
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            initial_guess: Optional initial guess
        
        Returns:
            (N,) stationary distribution vector
        """
        n = self.lazy_P.N
        
        # Define matvec function: (P.T - I) @ x
        # Note: Don't use @jax.jit here - GMRES will JIT the whole thing
        def matvec_fn(x):
            return self.lazy_P.rmatvec(x) - x
        
        # Enforce sum(v) = 1 constraint by replacing first row
        # We need to modify the system to incorporate this constraint
        # Standard approach: replace first equation with sum(v) = 1
        
        # Create modified matvec that enforces constraint
        # Note: Don't use @jax.jit here - GMRES will JIT the whole thing
        def constrained_matvec(x):
            result = matvec_fn(x)
            # Replace first element with sum(x) - 1
            return result.at[0].set(jnp.sum(x) - 1.0)
        
        # Right-hand side: all zeros except first element = 0 (since sum(v) - 1 = 0)
        b = jnp.zeros(n)
        
        # GMRES with initial guess
        x0 = initial_guess if initial_guess is not None else jnp.ones(n) / n
        v, info = jax.scipy.sparse.linalg.gmres(
            constrained_matvec, 
            b, 
            x0=x0, 
            tol=tolerance, 
            maxiter=max_iterations,
            solve_method='batched'
        )
        
        if info > 0:
            warn(f"GMRES did not converge in {max_iterations} iterations.")
        
        # Post-process: ensure non-negativity and renormalization
        v = _move_neg_prob_to_max(v)
        v = v / jnp.sum(v)
        
        return v
    
    def _solve_power_method_lazy(self, tolerance, max_iterations, initial_guess=None, timeout=30.0):
        """
        Power method using lazy rmatvec operations with adaptive batching.
        
        Uses timing-based adaptive batching to efficiently use available time:
        1. Measures time for 10 iterations
        2. Calculates iterations possible in half remaining time
        3. Runs batch, saves intermediate result
        4. Repeats until timeout or convergence
        5. Returns best result based on check norm
        
        Args:
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            initial_guess: Optional initial guess
            timeout: Maximum execution time (seconds)
        
        Returns:
            (N,) stationary distribution vector
        """
        import time
        
        n = self.lazy_P.N
        
        # Initial guess
        if initial_guess is not None:
            x = jnp.asarray(initial_guess)
            x = x / jnp.sum(x)  # Normalize
        else:
            # Start with uniform distribution
            x = jnp.ones(n, dtype=DTYPE_FLOAT) / n
        
        start_time = time.time()
        
        # Calibration: measure time for 10 iterations
        calibration_iters = 10
        calibration_start = time.perf_counter_ns()
        
        x_calib = x
        for _ in range(calibration_iters):
            x_calib = self.lazy_P.rmatvec_batched(x_calib)
        
        calibration_time_ns = time.perf_counter_ns() - calibration_start
        time_per_iter_s = (calibration_time_ns / 1e9) / calibration_iters
        
        # Track best result
        best_x = x
        best_check_norm = float('inf')
        
        # Adaptive batching loop
        total_iterations = 0
        converged = False
        
        while total_iterations < max_iterations:
            # Calculate time remaining
            elapsed = time.time() - start_time
            time_remaining = timeout - elapsed
            
            if time_remaining <= 0:
                warn(f"Power method timeout after {timeout} seconds at iteration {total_iterations}")
                break
            
            # Calculate iterations for half remaining time
            iters_for_half_time = int((time_remaining / 2.0) / time_per_iter_s)
            iters_for_half_time = max(1, min(iters_for_half_time, max_iterations - total_iterations))
            
            # Run batch of iterations
            x_batch = x
            for i in range(iters_for_half_time):
                x_new = self.lazy_P.rmatvec_batched(x_batch)
                
                # Check convergence every 10 iterations (not every iteration for efficiency)
                if i % 10 == 0 and i > 0:
                    # Normalize for convergence check
                    x_new_norm = x_new / jnp.sum(x_new)
                    delta = jnp.sum(jnp.abs(x_new_norm - x_batch / jnp.sum(x_batch)))
                    
                    if delta < tolerance:
                        converged = True
                        x = x_new_norm
                        total_iterations += i + 1
                        break
                
                x_batch = x_new
            
            if converged:
                break
            
            # Normalize after batch
            x = x_batch / jnp.sum(x_batch)
            total_iterations += iters_for_half_time
            
            
            # Calculate check norm: ||P^T @ x - x||_1
            # This measures convergence to stationary distribution
            # Do NOT normalize Px - we want to measure evolution effect, not renormalization effect
            Px = self.lazy_P.rmatvec_batched(x)
            check_norm = float(jnp.sum(jnp.abs(Px - x)))
            
            # Update best result if this is better
            if check_norm < best_check_norm:
                best_check_norm = check_norm
                best_x = x
        
        if not converged:
            # Use best result based on check norm
            x = best_x
            warn(f"Power method did not converge in {total_iterations} iterations (best check norm={best_check_norm:.2e})")
        
        return x
    
    def _solve_bifurcated_power_method_lazy(self, tolerance, max_iterations, timeout=30.0):
        """
        Bifurcated (dual-start) power method using lazy rmatvec operations.
        
        Starts from two different initial guesses based on row entropy and
        evolves both until they converge to each other. More robust for detecting
        issues but more expensive than single-path power method.
        
        This matches the dense bifurcated_power_method behavior.
        
        Args:
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            timeout: Maximum execution time (seconds)
        
        Returns:
            (N,) stationary distribution vector (average of two converged paths)
        """
        import time
        
        n = self.lazy_P.N
        start_time = time.time()
        
        # Calculate entropy of each row to find diverse starting points
        # We need to materialize one row at a time to compute entropy
        # This is expensive but only done once at startup
        row_entropies = []
        for i in range(n):
            # Get row i of P
            row_i = self.lazy_P.compute_rows(jnp.array([i]))[0]
            # Compute entropy: H = -sum(p * log2(p)) for p > 0
            p_safe = jnp.where(row_i > 0, row_i, 1.0)
            entropy = -jnp.sum(row_i * jnp.log2(p_safe))
            row_entropies.append(float(entropy))
        
        row_entropies = jnp.array(row_entropies)
        
        # Start 1: Max entropy (most uncertain transition)
        idx_max = int(jnp.argmax(row_entropies))
        v1 = jnp.zeros(n, dtype=DTYPE_FLOAT).at[idx_max].set(1.0)
        
        # Start 2: Min entropy (most deterministic transition)
        idx_min = int(jnp.argmin(row_entropies))
        v2 = jnp.zeros(n, dtype=DTYPE_FLOAT).at[idx_min].set(1.0)
        
        # Adaptive batching
        check_interval = 10
        next_check = check_interval
        
        # Evolve both paths
        i = 0
        while i < max_iterations:
            # Evolve both vectors in batch
            batch_end = min(next_check, max_iterations)
            batch_size = batch_end - i
            
            # Evolve both paths for batch_size iterations
            for _ in range(batch_size):
                v1 = self.lazy_P.rmatvec_batched(v1)
                v2 = self.lazy_P.rmatvec_batched(v2)
            
            i = batch_end
            
            # Normalize
            v1 = v1 / jnp.sum(v1)
            v2 = v2 / jnp.sum(v2)
            
            # Check convergence (paths converge to each other)
            diff = float(jnp.linalg.norm(v1 - v2, ord=1))
            if diff < tolerance:
                return (v1 + v2) / 2.0
            
            # Check timeout
            if (time.time() - start_time) > timeout:
                warn(f"Bifurcated power method (lazy) timed out after {timeout}s (iter {i}). Diff between paths: {diff}")
                return (v1 + v2) / 2.0
            
            # Adaptive: Increase interval
            if check_interval < 1000:
                check_interval *= 2
            next_check = i + check_interval
        
        # Final convergence check
        diff = float(jnp.linalg.norm(v1 - v2, ord=1))
        warn(f"Bifurcated power method (lazy) did not converge in {max_iterations} iterations. Final diff between paths: {diff}")
        return (v1 + v2) / 2.0
    
    @property
    def stationary_distribution(self):
        """Get stationary distribution (must call find_unique_stationary_distribution first)."""
        if not self._analyzed:
            raise ValueError("Must call find_unique_stationary_distribution() first")
        return self._stationary_distribution
    
    @property
    def analyzed(self):
        """Check if stationary distribution has been computed."""
        return self._analyzed


class FlexMarkovChain:
    """
    Flexible Markov chain that auto-selects dense or lazy based on memory.
    
    Note: This class is created from VotingModel layer, not directly instantiated.
    Use FlexMarkovChain.from_voting_model() factory method.
    """
    
    @classmethod
    def from_voting_model(cls, model, force_lazy=False, force_dense=False, tolerance=None):
        """
        Create FlexMarkovChain from VotingModel (factory method).
        
        This maintains separation of concerns: VotingModel layer creates the chain,
        MarkovChain layer doesn't need to know about voting logic.
        
        Args:
            model: VotingModel instance
            force_lazy: Force lazy construction
            force_dense: Force dense construction
            tolerance: Convergence tolerance
        
        Returns:
            FlexMarkovChain instance with auto-selected backend
        """
        from ...core import get_available_memory_bytes
        from .utils import should_use_lazy
        
        N = model.number_of_feasible_alternatives
        dtype = model.utility_functions.dtype
        available_mem = get_available_memory_bytes()
        
        # Decision logic
        use_lazy = force_lazy or (
            not force_dense and 
            available_mem is not None and 
            should_use_lazy(N, dtype, available_mem, threshold=0.75)
        )
        
        instance = cls()
        instance.is_lazy = use_lazy
        
        if use_lazy:
            # Create lazy chain
            lazy_P = LazyTransitionMatrix(
                utility_functions=model.utility_functions,
                majority=model.majority,
                zi=model.zi,
                number_of_feasible_alternatives=N
            )
            instance.backend = LazyMarkovChain(lazy_P=lazy_P, tolerance=tolerance)
        else:
            # Create dense chain (requires materializing P)
            P = model._get_transition_matrix()
            instance.backend = MarkovChain(P=P, tolerance=tolerance)
        
        return instance
    
    def find_unique_stationary_distribution(self, *, solver="auto", **kwargs):
        """
        Delegate to backend (dense or lazy).
        
        If solver="auto", choose gmres for lazy, full_matrix_inversion for dense.
        
        Args:
            solver: Solver strategy ("auto", "gmres", "power_method", etc.)
            **kwargs: Passed to backend solver
        
        Returns:
            self (for chaining)
        """
        if solver == "auto":
            solver = "gmres" if self.is_lazy else "full_matrix_inversion"
        
        self.backend.find_unique_stationary_distribution(solver=solver, **kwargs)
        return self
    
    @property
    def stationary_distribution(self):
        """Get stationary distribution from backend."""
        return self.backend.stationary_distribution
    
    @property
    def analyzed(self):
        """Check if stationary distribution has been computed."""
        return self.backend.analyzed
    
    @property
    def MarkovChain(self):
        """Get the underlying MarkovChain (for compatibility)."""
        return self.backend
