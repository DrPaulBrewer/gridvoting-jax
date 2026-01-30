import jax
import jax.lax
import jax.numpy as jnp
from warnings import warn

# Import constants from core
from ..core import (
    constants,
    get_available_memory_bytes,
)

# Import utilities and lazy classes from local stochastic modules
from .lazy_stochastic import LazyStochasticMatrix
from .lazy_q import LazyQMatrix
from .utils import (
    _move_neg_prob_to_max,
    normalize_if_needed,
    entropy_in_bits,
    matrix_is_dense
)


def _correct_minor_negative_probabilities(x):
    """Correct minor negative probabilities in a probability vector by adjusting the
    maximum components down and zeroing out the negative components. This becomes necessary
    when the probability vector is computed from matrix algorithms that do not guarantee
    non-negative entries.
    
    Args:
        x: A probability vector.
    Returns:
        A probability vector with minor negative probabilities corrected.
    Raises:
        RuntimeError: If the probability vector contains significantly negative probabilities.
    """
    min_component = x.min().item()
    # Use extracted constant from core for negative checks
    if ((min_component < 0.0) and (min_component > constants.NEGATIVE_PROBABILITY_TOLERANCE)):
        x = _move_neg_prob_to_max(x)
        min_component = x.min().item()
    
    if (min_component < 0.0):
        neg_msg = "(negative components in probability vector: "+str(min_component)+" )"
        raise RuntimeError(neg_msg)

    return x

def dense_matrix_inversion(*, Q=None):
    """Uses a dense matrix inversion to find the stationary distribution,
    but can suffer from numerical irregularities like small negative entries.
    Assumes eigenvalue of 1.0 exists and solves for the eigenvector by
    considering a related matrix equation Q v = b, where:
    Q is P transpose minus the identity matrix I, with the first row
    replaced by all ones for the vector scaling requirement;
    v is the eigenvector of eigenvalue 1 to be found; and
    b is the first basis vector, where b[0]=1 and 0 elsewhere.
    
    Args:
        Q: Q matrix (P transpose minus the identity matrix I, with the first row replaced by all ones)
    Returns:
        Stationary distribution
    Raises:
        ValueError: If Q matrix is not provided
        ValueError: If provided Q matrix is instead a lazy P matrix (LazyStochasticMatrix)
    """
    if (Q is None):
        raise ValueError("Q matrix must be provided")
    if (isinstance(Q, LazyStochasticMatrix)):
        raise ValueError("Q matrix must be either dense or type LazyQMatrix, got LazyStochasticMatrix")
    if (isinstance(Q, LazyQMatrix)):
        Q = Q.to_dense()
    n = Q.shape[0]
    error_unable_msg = "unable to find unique unit eigenvector "
    try:
        unit_eigenvector = jnp.linalg.solve(
            Q,
            jnp.zeros(n, dtype=constants.DTYPE_FLOAT).at[0].set(1.0)
        )
    except Exception as err:
        warn(str(err)) # print the original exception lest it be lost for debugging purposes
        raise RuntimeError(error_unable_msg+"(dense_solve)")

    if jnp.isnan(unit_eigenvector.sum()):
        raise RuntimeError(error_unable_msg+"(nan)")
    
    unit_eigenvector = _correct_minor_negative_probabilities(unit_eigenvector)
    unit_eigenvector = normalize_if_needed(unit_eigenvector)

    return unit_eigenvector


def iterate_gmres(*, P=None, Q=None, iterations, initial_guess):
    """
    GMRES iteration to find the stationary distribution. Calls JAX's GMRES implementation,
    jax.scipy.sparse.linalg.gmres, to solve Q v = b, where:
    Q is P transpose minus the identity matrix I, with the first row replaced by all ones for the vector scaling requirement;
    v is the eigenvector of eigenvalue 1 to be found; and
    b is the first basis vector, where b[0]=1 and 0 elsewhere.

    Unlike dense_matrix_inversion, this method does not require a dense Q matrix,
    but instead can handle LazyQMatrix objects and thereby uses less memory.

    Args:
        P: Transition matrix (Ignored, included for API consistency 
                    when called by MarkovChain::control_iteration)
        Q: Q matrix (P transpose minus the identity matrix I, with the first row replaced by all ones)
        iterations: number of iterations
        initial_guess: Optional initial distribution (if None, uses uniform)
    Returns:
        Improved guess at the stationary distribution (shape (n))
    Raises:
        ValueError: If Q matrix is not provided
        ValueError: If provided Q matrix is instead a lazy P matrix (LazyStochasticMatrix)
    """
    if Q is None:
        raise ValueError("Q matrix must be provided")
    if initial_guess is None:
        initial_guess = jnp.ones(Q.shape[0], dtype=constants.DTYPE_FLOAT)/Q.shape[0]
    # Use JAX's GMRES
    # tol in gmres is residual tolerance, roughly related to error
    v, info = jax.scipy.sparse.linalg.gmres(
        Q, 
        jnp.zeros(Q.shape[0], dtype=constants.DTYPE_FLOAT).at[0].set(1.0),
        x0=initial_guess,
        tol=constants.TOLERANCE, 
        restart=50,
        maxiter=iterations,
        solve_method='incremental'
    )    
    # Enforce non-negativity and normalization (numerical artifacts)
    v = _move_neg_prob_to_max(v)
    v = normalize_if_needed(v)
    return v

def iterate_power_method(*, P=None, Q=None, iterations, initial_guess):
    """
    Single-path power method with uniform initial guess.
    
    This is the standard power method implementation that matches lazy power method behavior.
    Starts from uniform distribution and iterates until convergence.
    
    Args:
        P: Transition matrix
        Q: Ignored (included for API consistency when called by MarkovChain::control_iteration)
        iterations: number of iterations
        initial_guess: Optional initial distribution (if None, uses uniform)
    
    Returns:
        Improved guess at the stationary distribution (shape (2,n))
    """
    if P is None:
        raise ValueError("P matrix must be provided")
    if initial_guess is None:
        initial_guess = jnp.ones(P.shape[0], dtype=constants.DTYPE_FLOAT)/(0.0+P.shape[0])
    def evolve_step(_, vec):
        return normalize_if_needed(vec @ P)
    v = jax.lax.fori_loop(0, iterations, evolve_step, initial_guess)
    return v

def entropy_based_guess_pair(*, P):
    """
    Returns a pair of initial guesses based on the entropy of each row of P.
    v1 is the row with maximum entropy, v2 is the row with minimum entropy.
    """
    n = P.shape[0]
    row_entropies = entropy_in_bits(P)
    max_entropy_idx = jnp.argmax(row_entropies).item()
    min_entropy_idx = jnp.argmin(row_entropies).item()
    v1 = jnp.zeros(n).at[max_entropy_idx].set(1.0)
    v2 = jnp.zeros(n).at[min_entropy_idx].set(1.0)
    return jnp.stack([v1, v2], axis=0)

def geometry_based_guess_pair(*, n, atom_idx):
    """
    Returns a pair of initial guesses based on the geometry of the matrix.
    v1 is uniform, v2 is an atomic distribution at the selected state
    
    Args:
        n: Size of the state space
        atom_idx: Index for the atomic distribution in v2
    """
    if atom_idx is None:
        raise ValueError("atom_idx must be provided")
    v1 = jnp.ones(n, dtype=constants.DTYPE_FLOAT)/n
    v2 = jnp.zeros(n, dtype=constants.DTYPE_FLOAT)
    return jnp.stack([v1, v2.at[atom_idx].set(1.0)], axis=0) # shape (2,n)

def iterate_bifurcated_power_method(*, P=None, Q=None, iterations, initial_guess):
    """
    Bifurcated (dual-start) power method
    
    Starts from two different initial guesses and evolves both until they
    converge to each other. More robust for detecting issues but more expensive
    than single-path power method.
    
    Args:
        P: Transition matrix
        Q: Ignored
        iterations: number of iterations
        initial_guess: Optional initial distribution (if None, uses geometry-based guess pair)
    
    Returns:
        Improved guess at the stationary distribution (shape (2,n))
    """
    if P is None:
        raise ValueError("P matrix must be provided")
    n = P.shape[0]
    if initial_guess is None:
        initial_guess = geometry_based_guess_pair(n=n, atom_idx=n//2)
    return iterate_power_method(P=P, Q=None, iterations=iterations, initial_guess=initial_guess)

iterative_solvers = dict(
    power_method=iterate_power_method,
    bifurcated_power_method=iterate_bifurcated_power_method,
    gmres_matrix_inversion=iterate_gmres
)   

class MarkovChain:
    """Represents a Markov chain with a transition matrix P."""
    def __init__(self, *, P):
        """initializes a MarkovChain instance with transition matrix P"""
        self.P = P
        self.N = P.shape[0]

    def calculate_chain_properties(self):
        """calculates and sets properties of the Markov chain:
            self.absorbing_points: boolean vector indicating which states are absorbing
            self.has_unique_stationary_distribution: boolean indicating whether the chain has a unique stationary distribution

        Args:
            None

        Returns:
            self: the MarkovChain instance with calculated properties set:
                self.absorbing_points: boolean vector indicating which states are absorbing
                self.has_unique_stationary_distribution: boolean indicating whether the chain has a unique stationary distribution
                     (assuming full connectivity, which is not checked)
        """
        diagP = self.P.diagonal()
        self.absorbing_points = jnp.equal(diagP, 1.0)
        self.has_unique_stationary_distribution = not jnp.any(self.absorbing_points)
        return self

    def dense_P(self):
        """Materialize the transition matrix if it is a lazy matrix.

        Args:
            None

        Returns:
            P: the transition matrix as a dense array
        """
        if matrix_is_dense(self.P):
            return self.P
        else:
            return self.P.to_dense()

    def L1_step_norm(self, x):
        """Computes the L1 norm of advancing the Markov chain by one step, starting from x
        
        Args:
            x: the initial distribution

        Returns:
            ||xP - x||L1: 
            the L1 norm of advancing the Markov chain by one step, starting from x
        """
        return jnp.linalg.norm((x @ self.P ) - x, ord=1, axis=-1)

    def control_iteration(self, *, solver=iterate_power_method, time_per_digit=None, initial_guess=None):
        """
        Controls the iteration of a solver by monitoring the L1 step norm and stopping 
        when the norm fails to achieve exponential decay or converges below TOLERANCE.

        Args:
            solver: The solver function to use. Defaults to iterate_power_method.
                   Expected signature: solver(P, Q, iterations, initial_guess)
            time_per_digit: Time budget per factor of 10 decrease in L1_step_norm. Defaults to 1.0.
            initial_guess: Optional initial distribution. If None, solver will use its default.

        Returns:
            tuple: (stationary_distribution, convergence_history)
                - stationary_distribution: Final distribution vector
                - convergence_history: List of dicts with keys: 'elapsed_time', 
                                       'total_iterations', 'current_norm', 'tolerance', 
                                       'batch_norm_goal'

        Stopping criteria:
            - current_norm < TOLERANCE: Successfully converged
            - current_norm > batch_norm_goal: Failed to achieve expected exponential decay
              where batch_norm_goal = previous_norm * pow(0.1, batch_elapsed / time_per_digit)
        """
        import time
        if time_per_digit is None:
            raise ValueError("time_per_digit must be specified")
        # test if solver is callable, not simply a string
        if not callable(solver):
            raise ValueError("solver must be callable")
        if initial_guess is not None:
            initial_guess = jnp.array(initial_guess, dtype=constants.DTYPE_FLOAT)
        
        # Compute and cache Q matrix (dense) for solvers that need it
        # if memory is a concern, self.P will be lazy and Q will be made dense from lazy evaluation
        Q = None
        P = self.P  # Default: use self.P as-is
        
        if (solver is iterate_gmres):
            Q = LazyQMatrix(self.P)

        # Initialize
        current_guess = initial_guess
        total_iterations = 0
        convergence_history = []
        # For initial entry, batch_norm_goal is 1.01*current_norm (no time elapsed yet)
        # if current guess is None, use a uniform distribution to start the covergence_history
        # the individual solvers can choose their own default initial guess but we cannot access it here
        if current_guess is None:
            current_norm = float(self.L1_step_norm(jnp.ones(self.N)/self.N).max().block_until_ready())
        else:
            current_norm = float(self.L1_step_norm(current_guess).max().block_until_ready())
        convergence_history.append({
            'elapsed_time': 0.0, 
            'total_iterations': 0, 
            'batch_time': float('nan'),
            'batch_iterations': 0,
            'current_norm': float(current_norm),
            'tolerance': float(constants.TOLERANCE),
            'batch_norm_goal': float(current_norm * 1.01) # batch_norm_goal for first entry
        })

        # For second entry, run 1 iteration not on the clock (because JAX compilation startup costs)
        previous_norm = current_norm
        current_guess = solver(P=P, Q=Q, iterations=1, initial_guess=current_guess)
        total_iterations += 1
        current_norm = float(self.L1_step_norm(current_guess).max().block_until_ready())

        convergence_history.append({
            'elapsed_time': 1e-6, 
            'total_iterations': 1, 
            'batch_time': float('nan'),
            'batch_iterations': 1,
            'current_norm': current_norm,
            'tolerance': float(constants.TOLERANCE),
            'batch_norm_goal': float(previous_norm) 
        })

        # now start the clock and run the first batch

        start_time = time.time()
        batch_size = 20 if self.N < 10000 else 10
        previous_norm = current_norm
        current_guess = solver(P=P, Q=Q, iterations=batch_size, initial_guess=current_guess)
        total_iterations += batch_size
        current_norm = float(self.L1_step_norm(current_guess).max().block_until_ready())
        elapsed_time = time.time() - start_time
        # For initial entry, batch_norm_goal is just previous_norm (no time elapsed yet)
        convergence_history.append({
            'elapsed_time': float(elapsed_time), 
            'total_iterations': float(total_iterations), 
            'batch_time': float(elapsed_time),
            'batch_iterations': float(batch_size),
            'current_norm': float(current_norm),
            'tolerance': float(constants.TOLERANCE),
            'batch_norm_goal': float(previous_norm)
        })

        batch_elapsed = elapsed_time
        
        # Main iteration loop
        while True:
            # Adjust batch size to target ~1 sec per batch (always adjust)
            target_batch_time = 1.0
            batch_size_cap = min(500, 10*batch_size)
            batch_size = max(1, int(batch_size * target_batch_time / batch_elapsed))
            batch_size = min(batch_size_cap, batch_size)

            batch_start_time = time.time()

            # Run solver for one batch
            current_guess = solver(P=P, Q=Q, iterations=batch_size, initial_guess=current_guess)
            total_iterations += batch_size
            
            # Check convergence
            current_norm = float(self.L1_step_norm(current_guess).max().block_until_ready())
            batch_elapsed = time.time() - batch_start_time
            elapsed_time = time.time() - start_time
            
            # Calculate batch_norm_goal using exponential decay
            batch_norm_goal = previous_norm * pow(0.1, batch_elapsed / time_per_digit)
            
            convergence_history.append({
                'elapsed_time': float(elapsed_time),
                'total_iterations': float(total_iterations),
                'batch_time': float(batch_elapsed),
                'batch_iterations': float(batch_size),
                'current_norm': float(current_norm),
                'tolerance': float(constants.TOLERANCE),
                'batch_norm_goal': float(batch_norm_goal)
            })
            
            # Stopping criteria
            if current_norm > batch_norm_goal or current_norm < constants.TOLERANCE:
                break
                        
            previous_norm = current_norm
        
        return (current_guess, convergence_history)

    def solve(self, *, 
                                           solver="full_matrix_inversion", 
                                           initial_guess=None,
                                           partitions=None, 
                                           time_per_digit=1.0):
        """
        Finds the stationary distribution for a Markov Chain.

        In the context of voting models, this distribution is also likely to be unique, because
        the Markov Chain will usually have only one connected component. 

        Note:  This function does not check for multiple connected components.  
        If the Markov Chain contains multiple connected components, some solvers may
        fail due to numerical issues associated with multiple stationary distributions. 
        For example, the "full_matrix_inversion" solver will fail with a singular matrix. 
        
        An alternative is to perform a test for connected components but this is not implemented. 
        See tests/test_markov_double_cycle.py for an example of solver behavior with a Markov Chain
        that has two distinct connected components.
        
        Args:
            solver: Strategy to use. Options:
                - "full_matrix_inversion": (Default) Direct algebraic solve (O(N^3)). Best for N < 5000.
                - "gmres_matrix_inversion": Iterative linear solver (GMRES). Lower memory (O(N^2)).
                - "power_method": Single-path power method with uniform initial guess (O(N^2)).
                  Matches lazy power method behavior.
                - "bifurcated_power_method": Dual-start entropy-based power method (O(N^2)).
                  More robust but more expensive than power_method.
            initial_guess: Optional starting distribution for "power_method"
            partitions: Optional partitions for Markov Chain lumping/un-lumping
            time_per_digit: Time in seconds to wait for each digit of precision in the L1 step norm.
        """
        if not hasattr(self,'absorbing_points'):
            self.calculate_chain_properties()
        if hasattr(self,'stationary_distribution'):
            del self.stationary_distribution
        if jnp.any(self.absorbing_points):
            self.stationary_distribution = None
            return None
        if partitions is not None:
            if initial_guess is not None:
                raise ValueError("initial_guess is not supported for partitioned Markov Chains.")
            MC_lumped = lump(self, partitions)
            MC_lumped.solve(solver=solver, time_per_digit=time_per_digit)
            if MC_lumped.convergence_history is not None:
                self.convergence_history = MC_lumped.convergence_history
            if MC_lumped.stationary_distribution is None:
                self.stationary_distribution = None
                return None
            self.stationary_distribution = unlump(MC_lumped.stationary_distribution, partitions)
            return self.stationary_distribution            
        # Memory Check
        try:
            available_mem = get_available_memory_bytes()
            
            if available_mem is not None:
                n = self.P.shape[0]
                # Determine element size (float32=4, float64=8)
                item_size = jnp.dtype(constants.DTYPE_FLOAT).itemsize                
                estimated_needed = 0
                if solver == "full_matrix_inversion":
                    # Q(N^2) + Result(N^2) all float
                    estimated_needed = 2 * (n**2) * item_size
                elif solver in iterative_solvers:
                     # Matrix-vector product based 
                     # ~ 10 Vectors(k*N) float
                    estimated_needed = (10 * n * item_size)
                else:
                    estimated_needed = (n**2) * item_size
                
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
            self.convergence_history = None
            self.stationary_distribution = dense_matrix_inversion(Q=LazyQMatrix(self.P).to_dense())
        else:
            # iterative solvers
            if solver in iterative_solvers:
                self.stationary_distribution, self.convergence_history = self.control_iteration(
                    solver=iterative_solvers[solver],
                    initial_guess=initial_guess,
                    time_per_digit=time_per_digit
                    )
            else:
                raise ValueError(f"Unknown solver: {solver}")

        # handle 2D result from bifurcated power method (shape (2,n) needs averaging)
        # Note: for 2-state chains with other solvers, shape is (n,) which is already correct
        if solver == "bifurcated_power_method" and self.stationary_distribution.ndim == 2:
            # require stationary distributions to be similar within BAD_STATIONARY_TOLERANCE
            if jnp.linalg.norm(self.stationary_distribution[0] - self.stationary_distribution[1], ord=1) > constants.BAD_STATIONARY_TOLERANCE:
                raise RuntimeError("Markov chain convergence failure with solver='bifurcated_power_method': stationary distributions are not similar")
            self.stationary_distribution = self.stationary_distribution.mean(axis=0)

        check_sum = self.stationary_distribution.sum().block_until_ready()
        if jnp.abs(check_sum-1.0) > (2.0*constants.EPSILON*self.N):
            raise RuntimeError(f"Markov chain check sum=1 failure with solver={solver}: sum={check_sum}")

        final_check_norm = self.L1_step_norm(self.stationary_distribution).block_until_ready()
        if final_check_norm > constants.BAD_STATIONARY_TOLERANCE:
            raise RuntimeError(f"Markov chain convergence failure with solver={solver}")
 
        return self.stationary_distribution

    def diagnostic_metrics(self):
        """ return Markov chain approximation metrics in mathematician-friendly format

        args:
            None

        returns:
            dict: Dictionary with the following keys:
            `||F||`: Number of states in the Markov chain
            `(𝝨𝝿)-1`: The difference between the sum of the stationary distribution and 1
            `||𝝿P-𝝿||_L1_norm`: The L1 norm of the difference between the stationary distribution and the product of the stationary distribution and the transition matrix

        """
        metrics = {
            '||F||': self.P.shape[0],
            '(𝝨𝝿)-1':  float(self.stationary_distribution.sum())-1.0, # cast to float to avoid singleton
            '||𝝿P-𝝿||_L1_norm': self.L1_step_norm(self.stationary_distribution)
        }
        return metrics


# ============================================================================
# Markov Chain Lumping Functions
# ============================================================================

def _validate_inverse_indices(inverse_indices: jnp.ndarray, n_states: int) -> None:
    """
    Validate inverse indices is a proper partition representation.
    
    Checks (in order, fails on first violation):
    1. Correct length (matches n_states)
    2. Valid indices (all values >= 0)
    3. No gaps (all groups 0..k-1 are used)
    
    Args:
        inverse_indices: Array mapping each state to its group (0 to k-1)
        n_states: Expected number of states
    
    Raises:
        ValueError: On first violation with descriptive error message
    """
    # Check 1: Correct length
    if len(inverse_indices) != n_states:
        raise ValueError(
            f"Inverse indices length {len(inverse_indices)} != n_states {n_states}"
        )
    
    # Check 2: Valid indices (0 to k-1 for some k)
    min_idx = int(inverse_indices.min())
    max_idx = int(inverse_indices.max())
    if min_idx < 0:
        raise ValueError(f"Invalid negative index: {min_idx}")
    
    # Check 3: No gaps (all groups 0..k-1 must be used)
    k = max_idx + 1
    unique_groups = jnp.unique(inverse_indices)
    if len(unique_groups) != k:
        raise ValueError(
            f"Partition has gaps: expected {k} groups, found {len(unique_groups)}"
        )



def _compute_lumped_transition_matrix(P: jnp.ndarray, inverse_indices: jnp.ndarray) -> jnp.ndarray:
    """
    Compute lumped transition matrix using fully vectorized operations.
    
    Uses JAX's segment_sum for efficient aggregation without Python loops.
    
    P'[i,j] = (1/|Si|) * sum_{s in Si, t in Sj} P[s,t]
    
    Args:
        P: Original transition matrix (n×n)
        inverse_indices: Array mapping each state to its group (0 to k-1)
    
    Returns:
        jnp.ndarray: Lumped transition matrix (k×k)
    
    Performance:
        Fully vectorized O(n²) using segment_sum (no Python loops)
    """
    n = P.shape[0]
    k = int(inverse_indices.max()) + 1
    
    # Compute group sizes
    group_sizes = jnp.bincount(inverse_indices, length=k)
    
    # Fully vectorized: sum rows by source aggregate
    P_lumped = jax.ops.segment_sum(P, inverse_indices, num_segments=k)  # (k×n)
    
    # Sum columns by destination aggregate
    P_lumped = jax.ops.segment_sum(P_lumped.T, inverse_indices, num_segments=k).T  # (k×k)
    
    # Divide by group sizes to get average (uniform weighting)
    # This already produces properly normalized rows when input P is stochastic
    P_lumped = P_lumped / group_sizes[:, jnp.newaxis]
    
    return P_lumped
    
def _compute_lumped_transition_matrix_lazy(P, inverse_indices: jnp.ndarray) -> jnp.ndarray:
    """
    Compute lumped transition matrix using lazy matrix operations.
    
    Args:
        P: LazyStochasticMatrix or LazyWeightedStochasticMatrix object
        inverse_indices: Array mapping each state to its group (0 to k-1)
    
    Returns:
        jnp.ndarray: Lumped transition matrix (k×k)
    """
    n = P.shape[0]
    k = int(inverse_indices.max()) + 1
    
    # Compute group sizes
    group_sizes = jnp.bincount(inverse_indices, length=k)
    
    # 1. Diagonal contributions (status quo)
    # Sum status_quo values for each group: these remain in the group (diagonal of lumped)
    # Shape: (k,)
    diag_sums = jax.ops.segment_sum(P.status_quo_values, inverse_indices, num_segments=k)
    P_lumped = jnp.diag(diag_sums)
    
    # 2. Off-diagonal contributions
    # We need to sum P[i,j] for all i in group u, j in group v.
    
    # Handle different Lazy types
    if hasattr(P, 'challenger_values'):
        # LazyStochasticMatrix: P[i,j] = mask[i,j] * challenger_values[i] (for i!=j)
        # We need sum_{i \in u} challenger_values[i] * (sum_{j \in v} mask[i,j])
        
        # Count connections from each row i to each group v
        # mask_to_groups = jax.ops.segment_sum(P.mask.astype(float).T, inverse_indices, num_segments=k).T
        # Optimization: Use int32 for summation to save bandwidth/compute, then cast to float for multiplication
        # Note: Tested int8 but found no performance gain, so sticking with int32 for safety against overflow.
        mask_to_groups = jax.ops.segment_sum(P.mask.astype(jnp.int32).T, inverse_indices, num_segments=k).T
        
        # Weight by challenger values for each row
        # (n, k) * (n, 1) -> (n, k)
        weighted_to_groups = mask_to_groups * P.challenger_values[:, None]
        
    elif hasattr(P, 'row_normalized_weights'):
        # LazyWeightedStochasticMatrix: 
        # P[i,j] = mask[i,j] * challenger_scale[i] * row_normalized_weights[i,j]
        
        # Combine mask and weights
        effective_mask = P.mask * P.row_normalized_weights
        
        # Aggregate columns by group
        mask_to_groups = jax.ops.segment_sum(effective_mask.T, inverse_indices, num_segments=k).T
        
        # Weight by challenger scale
        weighted_to_groups = mask_to_groups * P.challenger_scale[:, None]
        
    else:
        # Fallback for unknown type (shouldn't happen with strict typing but good for safety)
        # Materialize row by row (slow, original implementation logic)
        warn("Unknown lazy matrix type in lumping, falling back to slow loop.")
        return jax.lax.fori_loop(
            0, n, 
            lambda i, carry: carry.at[inverse_indices[i]].add(P[i] @ jax.nn.one_hot(inverse_indices, k)), 
            jnp.zeros((k, k))
        ) / group_sizes[:, None]

    # Sum rows by source group
    # (n, k) -> (k, k)
    off_diag_sums = jax.ops.segment_sum(weighted_to_groups, inverse_indices, num_segments=k)
    
    # Add off-diagonal contributions to the lumped matrix
    # Note: P.mask has False on diagonal, so these are strictly off-diagonal transitions in the original space.
    # In lumped space, they might be self-loops (u->u) or cross-group (u->v).
    P_lumped = P_lumped + off_diag_sums
    
    # Divide by source group sizes to get average probability
    P_lumped = P_lumped / group_sizes[:, jnp.newaxis]
    
    # Renormalize rows to handle numerical errors
    P_lumped = normalize_if_needed(P_lumped)
    
    return P_lumped
    

def lump(MC: MarkovChain, inverse_indices: jnp.ndarray) -> MarkovChain:
    """
    Create a lumped (aggregated) Markov chain by combining states.
    
    States within each aggregate are assumed to have equal probability.
    The partition must be a proper partition (covering all states exactly once),
    but need not preserve the Markov property. Invalid lumpings that violate
    strong lumpability conditions are permitted but will not yield accurate
    stationary distributions when unlumped.
    
    Args:
        MC: Original MarkovChain instance
        inverse_indices: Inverse indices mapping states to their aggregate groups
    
    Returns:
        MarkovChain: New chain with k states where k = len(inverse_indices)
    
    Raises:
        ValueError: If inverse_indices is invalid (missing states, duplicates, 
                    empty groups, etc.)
    
    References:
        Kemeny, J. G., & Snell, J. L. (1976). Finite Markov Chains. 
        Springer-Verlag. (Chapter on lumpability)
    
    Examples:
        >>> # Reflection symmetry: (x,y) -> (y,x)
        >>> inverse_indices = jnp.array([0, 1, 0, 1])  # States 0,2 in group 0; 1,3 in group 1
        >>> lumped = lump(mc, inverse_indices)
        
        >>> # Swap states
        >>> inverse_indices = jnp.array([1, 0])  # Swaps states 0 and 1
        >>> lumped = lump(mc, inverse_indices)
    
    Notes:
        - inverse_indices must include all states exactly once
        - Each group in partition must be non-empty
        - States within each aggregate are weighted equally
        - Lumping may not preserve the Markov property (strong lumpability)
    """
    n_states = MC.P.shape[0]
    
    # Validate inverse indices (strict checking, fails on first violation)
    _validate_inverse_indices(inverse_indices, n_states)
    
    # Compute lumped transition matrix
    if matrix_is_dense(MC.P):
        P_lumped = _compute_lumped_transition_matrix(MC.P, inverse_indices)
    else:
        P_lumped = _compute_lumped_transition_matrix_lazy(MC.P, inverse_indices)

    # Create new MarkovChain instance
    return MarkovChain(P=P_lumped)


def unlump(lumped_distribution: jnp.ndarray, inverse_indices: jnp.ndarray) -> jnp.ndarray:
    """
    Map a probability distribution from lumped space back to original space.
    
    Distributes probability uniformly within each aggregate state.
    
    Args:
        lumped_distribution: Probability distribution over k aggregate states
        inverse_indices: Same inverse indices used to create the lumped chain
    
    Returns:
        jnp.ndarray: Probability distribution over n original states
    
    Example:
        >>> inverse_indices = jnp.array([0, 0, 1, 1, 1])  # 2 states in group 0, 3 in group 1
        >>> lumped_pi = jnp.array([0.4, 0.6])
        >>> pi = unlump(lumped_pi, inverse_indices)
        >>> # pi = [0.2, 0.2, 0.2, 0.2, 0.2]  (uniform within aggregates)
    
    Notes:
        - If the original lumping violated strong lumpability, the unlumped
          distribution will not match the original chain's stationary distribution
    """
    k = int(inverse_indices.max()) + 1
    n_states = len(inverse_indices)
    
    # Validate input
    if lumped_distribution.shape[0] != k:
        raise ValueError(
            f"Distribution size {lumped_distribution.shape[0]} doesn't match "
            f"number of groups {k}"
        )
    
    # Compute group sizes
    group_sizes = jnp.bincount(inverse_indices, length=k)
    
    # Distribute probability uniformly within each aggregate
    # For each state, get its group's probability divided by group size
    prob_per_state = lumped_distribution[inverse_indices] / group_sizes[inverse_indices]
    
    return prob_per_state


def is_lumpable(MC: MarkovChain, inverse_indices: jnp.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Test whether a partition preserves the Markov property (strong lumpability).
    
    A partition is strongly lumpable if for each aggregate state i and j,
    all states k within aggregate i have the same total transition probability
    to aggregate j:
        Σ_{l∈Lⱼ} p_{kl} = constant for all k∈Lᵢ
    
    Args:
        MC: MarkovChain instance
        inverse_indices: Inverse indices representing the partition
        tolerance: Numerical tolerance for equality check (default: 1e-6)
    
    Returns:
        bool: True if partition is strongly lumpable, False otherwise
    
    Examples:
        >>> # Test if partition preserves Markov property
        >>> P = jnp.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.1, 0.1, 0.8]])
        >>> mc = MarkovChain(P=P)
        >>> is_lumpable(mc, jnp.array([0, 0, 1]))  # True
        >>> is_lumpable(mc, jnp.array([0, 1, 0]))  # False
    
    Notes:
        - Uses vectorized matrix operations for efficiency (compatible with lazy matrices)
        - Complexity: O(nk + n²k) where n=states, k=aggregates
        - Creates indicator matrix and uses matrix-vector products to compute transition sums
    """
    _validate_inverse_indices(inverse_indices, MC.P.shape[0])
    
    n = MC.P.shape[0]
    k = int(inverse_indices.max()) + 1
    
    # Create indicator matrix using vectorized operations
    # indicator[i, s] = True if state s is in group i, else False
    indicator = jax.vmap(lambda i: inverse_indices == i)(jnp.arange(k))
    
    # For each pair of groups (i, j), compute transition probabilities
    # P_to_j[s] = sum of P[s, t] for all t in group j
    # This is equivalent to: P @ indicator[j].T
    # Shape: (n,) for each j
    
    for i in range(k):
        # If group is empty, skip (shouldn't happen with validation)
        if indicator[i].sum() == 0:
            continue
        
        for j in range(k):
            # Compute transition probability from each state to group j
            # indicator[j] is a vector with True for states in group j
            # P @ indicator[j] gives the sum of transitions to group j for each state
            probs_to_j = MC.P @ indicator[j]
            
            # Extract probabilities for states in group i using indicator[i] as mask
            probs_in_group_i = probs_to_j[indicator[i]]
            
            # Check if all states in group i have the same probability to group j
            if not jnp.allclose(probs_in_group_i, probs_in_group_i[0], atol=tolerance):
                return False
    
    return True


def partition_from_permutation_symmetry(
    n_states: int,
    state_labels: list[tuple],
    permutation_group: list[tuple]
) -> jnp.ndarray:
    """
    CAUTION: This AI-generated function is relatively untested and requires further review.


    Generate partition from permutation symmetries.
    
    Groups states that are equivalent under permutations of state labels.
    Useful for voter interchangeability in voting models.
    
    Args:
        n_states: Total number of states
        state_labels: List of tuples labeling each state
                     Example: [(0,1,2), (0,2,1), (1,0,2), ...] for 3-voter model
        permutation_group: List of permutations in cycle notation
                          Example: [((0,1),), ((1,2),), ((0,1,2),)] for S3
                          Empty tuple () represents identity
    
    Returns:
        jnp.ndarray: Inverse indices array grouping symmetric states
    
    Examples:
        >>> # 3-voter model with full S3 symmetry (all voters interchangeable)
        >>> state_labels = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
        >>> # S3 generators: (0,1) swap and (0,1,2) rotation
        >>> s3_group = [((0,1),), ((0,1,2),)]
        >>> partition = partition_from_permutation_symmetry(6, state_labels, s3_group)
        >>> # Result: jnp.array([0, 0, 0, 0, 0, 0]) - all states in group 0
        
        >>> # Z2 symmetry: swap voters 0 and 1
        >>> z2_group = [((0,1),)]
        >>> partition = partition_from_permutation_symmetry(6, state_labels, z2_group)
        >>> # Result: jnp.array([0, 0, 1, 1, 2, 2]) - pairs of swapped states
    
    Notes:
        - Permutations use cycle notation: ((0,1),) swaps 0↔1
        - ((0,1,2),) means 0→1→2→0
        - Multiple cycles: ((0,1), (2,3)) swaps 0↔1 and 2↔3
        - Identity is represented by empty tuple ()
        - Function generates closure of permutation group
    """
    # Build equivalence classes using union-find
    parent = list(range(n_states))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Helper: Apply a single cycle to a tuple
    def apply_cycle(label: tuple, cycle: tuple) -> tuple:
        if len(cycle) == 0:
            return label
        # Create mapping: cycle[i] -> cycle[i+1]
        mapping = {}
        for i in range(len(cycle)):
            next_i = (i + 1) % len(cycle)
            mapping[cycle[i]] = cycle[next_i]
        # Apply mapping to label
        return tuple(mapping.get(x, x) for x in label)
    
    # Helper: Apply a permutation (list of cycles) to a label
    def apply_permutation(label: tuple, perm: tuple) -> tuple:
        result = label
        for cycle in perm:
            result = apply_cycle(result, cycle)
        return result
    
    # For each permutation in the group
    for perm in permutation_group:
        # Apply permutation to each state
        for i in range(n_states):
            original_label = state_labels[i]
            permuted_label = apply_permutation(original_label, perm)
            
            # Find state with permuted label
            for j in range(n_states):
                if state_labels[j] == permuted_label:
                    union(i, j)
                    break
    
    # Build inverse indices from equivalence classes
    inverse_indices = jnp.zeros(n_states, dtype=jnp.int32)
    group_mapping = {}
    group_id = 0
    
    for i in range(n_states):
        root = find(i)
        if root not in group_mapping:
            group_mapping[root] = group_id
            group_id += 1
        inverse_indices = inverse_indices.at[i].set(group_mapping[root])
    
    return inverse_indices

def list_partition_to_inverse(partition: list[list[int]], n_states: int) -> jnp.ndarray:
    """
    Convert partition from list[list[int]] format to inverse indices format.
    
    This helper function is provided for migrating existing code that uses
    the old partition format.
    
    Args:
        partition: Partition as list of lists, where each inner list contains
                  state indices belonging to the same group
        n_states: Total number of states
    
    Returns:
        jnp.ndarray: Inverse indices array where inverse_indices[i] gives
                    the group ID for state i
    
    Example:
        >>> partition = [[0, 2], [1, 3]]
        >>> inverse = list_partition_to_inverse(partition, 4)
        >>> # inverse = jnp.array([0, 1, 0, 1])
        >>> # States 0 and 2 are in group 0, states 1 and 3 are in group 1
    """
    inverse = jnp.zeros(n_states, dtype=jnp.int32)
    for i, group in enumerate(partition):
        for s in group:
            inverse = inverse.at[s].set(i)
    return inverse

def _experimental_partition_from_entropy(P, decimals=3):
    """
    Generate a partition from the entropy of a transition matrix.
    
    Groups states with similar outgoing transition entropies.
    
    Args:
        P: Transition matrix
        decimals: Number of decimal places to round entropy values to create discriminant
    
    Returns:
        jnp.ndarray: Inverse indices array grouping states with similar transition entropies

    Note:  Because this partition is unlikely to be strongly Markovian, it is not recommended for
    use in most applications.  It is provided for curiosity and testing purposes and is marked with
    an underscore to indicate that it is not a primary partitioning method.

    Warning: This is an experimental feature. It may be removed in a future release.
    """
    H = entropy_in_bits(P) if matrix_is_dense(P) else P.row_entropies()
    H_rounded = jnp.round(H, decimals=decimals)
    return jnp.unique(H_rounded, return_inverse=True)[1]
