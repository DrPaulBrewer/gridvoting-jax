
import jax
import jax.lax
import jax.numpy as jnp
from warnings import warn
from collections import Counter

# Import from core
from ..core import (
   LazyLeftGVMatrix,
    LazyRightGVMatrix,
    TOLERANCE, 
    NEGATIVE_PROBABILITY_TOLERANCE, 
    DTYPE_FLOAT,
    _move_neg_prob_to_max,
    normalize_if_needed,
    entropy_in_bits,
    matrix_is_dense,
)

def _Q_matrix(P, dense=False):
    n = P.shape[0]
    if (dense) or (matrix_is_dense(P)):
        Q = P.T - jnp.eye(n)
        Q = Q.at[0].set(jnp.ones(n, dtype=DTYPE_FLOAT)) # first row of Q is all ones
    else:
        def Q_get_col(i):
            P_T_minus_I = P.get_row(i).at[i].add(-1.0)
            Q_col_i = P_T_minus_I.at[0].set(1.0)
            return Q_col_i
        Q = LazyRightGVMatrix(n=n, get_col = Q_get_col)
    
    return Q

def _correct_minor_negative_probabilities(x):
    min_component = float(x.min())
    # Use extracted constant from core for negative checks
    if ((min_component < 0.0) and (min_component > NEGATIVE_PROBABILITY_TOLERANCE)):
        x = _move_neg_prob_to_max(x)
        min_component = float(x.min())
    
    if (min_component < 0.0):
        neg_msg = "(negative components: "+str(min_component)+" )"
        raise RuntimeError(neg_msg)

    return x

def dense_matrix_inversion(*, Q=None):
    """This is another way to potentially find the stationary distribution,
    but can suffer from numerical irregularities like negative entries.
    Assumes eigenvalue of 1.0 exists and solves for the eigenvector by
    considering a related matrix equation Q v = b, where:
    Q is P transpose minus the identity matrix I, with the first row
    replaced by all ones for the vector scaling requirement;
    v is the eigenvector of eigenvalue 1 to be found; and
    b is the first basis vector, where b[0]=1 and 0 elsewhere."""
    if (Q is None):
        raise ValueError("Q matrix must be provided")    
    n = Q.shape[0]
    error_unable_msg = "unable to find unique unit eigenvector "
    try:
        unit_eigenvector = jnp.linalg.solve(
            # if Q is lazy, construct a dense Q matrix
            Q if matrix_is_dense(Q) else Q.to_dense(),
            jnp.zeros(n, dtype=DTYPE_FLOAT).at[0].set(1.0)
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
        Note:
            GMRES always uses dense Q matrix because JAX's GMRES implementation
            uses automatic differentiation internally, which is not compatible
            with our lazy matrix implementation.
    """
    if Q is None:
        raise ValueError("Q matrix must be provided")
    if initial_guess is None:
        initial_guess = jnp.ones(Q.shape[0], dtype=DTYPE_FLOAT)/Q.shape[0]
    # Use JAX's GMRES
    # tol in gmres is residual tolerance, roughly related to error
    v, info = jax.scipy.sparse.linalg.gmres(
        Q if matrix_is_dense(Q) else Q.to_dense(), 
        jnp.zeros(Q.shape[0], dtype=DTYPE_FLOAT).at[0].set(1.0),
        x0=initial_guess,
        tol=TOLERANCE, 
        maxiter=iterations,
        solve_method='incremental'
    )    
    # Enforce non-negativity and normalization (numerical artifacts)
    # GMRES can have larger deviations, so always normalize
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
        Q: Ignored
        iterations: number of iterations
        initial_guess: Optional initial distribution (if None, uses uniform)
    
    Returns:
        Stationary distribution vector
    """
    if P is None:
        raise ValueError("P matrix must be provided")
    if initial_guess is None:
        initial_guess = jnp.ones(P.shape[0], dtype=DTYPE_FLOAT)/P.shape[0]
    
    if matrix_is_dense(P):
        # Use lax.fori_loop for compiled batched evolution
        def evolve_step(_, carry):
            vec, P = carry
            return (normalize_if_needed(vec @ P), P)
        v, _ = jax.lax.fori_loop(0, iterations, evolve_step, (initial_guess, P))
    else:
        # manual loop for lazy matrix 
        v = initial_guess
        for _ in range(iterations):
            v = normalize_if_needed(v @ P)         
    return v

def entropy_based_guess_pair(*, P):
    """
    Returns a pair of initial guesses based on the entropy of each row.
    """
    n = P.shape[0]
    row_entropies = jax.vmap(entropy_in_bits)(P)
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
    v1 = jnp.ones(n, dtype=DTYPE_FLOAT)/n
    v2 = jnp.zeros(n, dtype=DTYPE_FLOAT)
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
        Stationary distribution vector (average of two converged paths)
    """
    if P is None:
        raise ValueError("P matrix must be provided")
    n = P.shape[0]
    if initial_guess is None:
        initial_guess = geometry_based_guess_pair(n=n, atom_idx=n//2)
    if matrix_is_dense(P):    
        def evolve_batch_step(_, V_state):
            V_new = V_state @ P
            V_new = jax.vmap(normalize_if_needed)(V_new)
            return V_new
        V = jax.lax.fori_loop(0, iterations, evolve_batch_step, initial_guess)
    else:
        V = initial_guess
        for _ in range(iterations):
            V = normalize_if_needed(V @ P)

    return V

iterative_solvers = dict(
    power_method=iterate_power_method,
    bifurcated_power_method=iterate_bifurcated_power_method,
    gmres_matrix_inversion=iterate_gmres
)   

class MarkovChain:
    def __init__(self, *, P):
        """initializes a MarkovChain instance by copying in the transition
        matrix P and calculating chain properties"""
        self.P = P

    def calculate_chain_properties(self):
        diagP = self.P.diagonal()
        self.absorbing_points = jnp.equal(diagP, 1.0)
        self.has_unique_stationary_distribution = not jnp.any(self.absorbing_points)
        return self

    def dense_P(self):
        """Materialize the transition matrix if it is a lazy matrix."""
        if (hasattr(self.P, 'to_dense') and callable(self.P.to_dense)):
            return self.P.to_dense()
        else:
            return self.P

    def force_dense(self):
        self.P = self.dense_P()
        if not(hasattr(self, 'has_unique_stationary_distribution')):
            self.calculate_chain_properties()
        return self

    def L1_step_norm(self, x):
        return jnp.linalg.norm((x @ self.P ) - x, ord=1, axis=-1)

    def control_iteration(self, *, solver=iterate_power_method, time_per_digit=20.0, initial_guess=None, force_dense=False):
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
                - convergence_history: List of [elapsed_time, total_iterations, current_norm, 
                                               tolerance, batch_norm_goal] entries

        Stopping criteria:
            - current_norm < TOLERANCE: Successfully converged
            - current_norm > batch_norm_goal: Failed to achieve expected exponential decay
              where batch_norm_goal = previous_norm * pow(0.1, batch_elapsed / time_per_digit)
        """
        import time
        
        # Initialize
        current_guess = initial_guess
        batch_size = 5
        total_iterations = 0
        convergence_history = []
        start_time = time.time()
        
        # Compute and cache Q matrix (dense) for solvers that need it
        # if memory is a concern, self.P will be lazy and Q will be made dense from lazy evaluation
        Q = None
        P = self.P  # Default: use self.P as-is
        
        if (solver is iterate_gmres):
            if matrix_is_dense(self.P):
                Q = _Q_matrix(self.P)
            else:
                Q = _Q_matrix(self.P).to_dense()
        elif force_dense and not matrix_is_dense(self.P):
            # For non-GMRES solvers, optionally force dense
            P = self.P.to_dense()

        # Get initial L1 norm
        if current_guess is None:
            current_guess = solver(P=P, Q=Q, iterations=batch_size, initial_guess=None)
            total_iterations += batch_size
        
        previous_norm = self.L1_step_norm(current_guess).max()
        elapsed_time = time.time() - start_time
        # For initial entry, batch_norm_goal is just previous_norm (no time elapsed yet)
        convergence_history.append([
            float(elapsed_time), 
            int(total_iterations), 
            float(previous_norm),
            float(TOLERANCE),
            float(previous_norm)  # batch_norm_goal for first entry
        ])
        
        # Main iteration loop
        while True:
            batch_start_time = time.time()
            
            # Run solver for one batch
            current_guess = solver(P=P, Q=Q, iterations=batch_size, initial_guess=current_guess)
            total_iterations += batch_size
            
            # Check convergence
            current_norm = self.L1_step_norm(current_guess).max()
            batch_elapsed = time.time() - batch_start_time
            elapsed_time = time.time() - start_time
            
            # Calculate batch_norm_goal using exponential decay
            batch_norm_goal = previous_norm * pow(0.1, batch_elapsed / time_per_digit)
            
            convergence_history.append([
                float(elapsed_time),
                int(total_iterations),
                float(current_norm),
                float(TOLERANCE),
                float(batch_norm_goal)
            ])
            
            # Stopping criteria
            if current_norm > batch_norm_goal or current_norm < TOLERANCE:
                break
            
            # Adjust batch size to target ~1 sec per batch (always adjust)
            target_batch_time = 1.0
            batch_size = max(1, int(batch_size * target_batch_time / batch_elapsed))
            batch_size = min(500, batch_size)
            
            previous_norm = current_norm
        
        return (current_guess, convergence_history)

    def find_unique_stationary_distribution(self, *, solver="dense_matrix_inversion", initial_guess=None, force_dense=False):
        """
        Finds the stationary distribution for a Markov Chain.
        
        Args:
            solver: Strategy to use. Options:
                - "dense_matrix_inversion": (Default) Direct algebraic solve (O(N^3)). Best for N < 5000.
                - "gmres_matrix_inversion": Iterative linear solver (GMRES). Lower memory (O(N^2)).
                - "power_method": Single-path power method with uniform initial guess (O(N^2)).
                  Matches lazy power method behavior.
                - "bifurcated_power_method": Dual-start entropy-based power method (O(N^2)).
                  More robust but more expensive than power_method.
            initial_guess: Optional starting distribution for "power_method".
            force_dense: Whether to force dense matrix representation for power method solvers.
        """
            
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
                item_size = jnp.dtype(DTYPE_FLOAT).itemsize
                
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
            if matrix_is_dense(self.P):
                Q = _Q_matrix(self.P)
            else:
                Q = _Q_matrix(self.P).to_dense()
            self.convergence_history = None
            self.stationary_distribution = dense_matrix_inversion(Q=Q)
        else:
            # iterative solvers
            if solver in iterative_solvers:
                self.stationary_distribution, self.convergence_history = self.control_iteration(solver=iterative_solvers[solver], initial_guess=initial_guess, force_dense=force_dense)
            else:
                raise ValueError(f"Unknown solver: {solver}")

        # handle 2-state case from bifurcated power method
        if (self.stationary_distribution.shape[0]==2):
            self.stationary_distribution = self.stationary_distribution.sum(axis=0)/2.0
 
        return self.stationary_distribution

    def diagnostic_metrics(self):
        """ return Markov chain approximation metrics in mathematician-friendly format """
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
    P_lumped = P_lumped / group_sizes[:, jnp.newaxis]
    
    # Renormalize rows
    row_sums = jnp.sum(P_lumped, axis=1, keepdims=True)
    P_lumped = P_lumped / row_sums
    
    return P_lumped
    
def _compute_lumped_transition_matrix_lazy(P: LazyLeftGVMatrix, inverse_indices: jnp.ndarray) -> jnp.ndarray:
    n = P.shape[0]
    k = int(inverse_indices.max()) + 1
    
    # Compute group sizes
    group_sizes = jnp.bincount(inverse_indices, length=k)
    
    # Initialize result
    P_lumped = jnp.zeros((k, k), dtype=P.dtype)
    
    # For each row in original matrix
    def process_row(i, P_lumped_carry):
        row_i = P[i]  # Get row i
        src_group = inverse_indices[i]  # Which group does state i belong to?
        
        # For each destination group j, sum transitions to states in that group
        def add_to_group(j, carry):
            dest_mask = (inverse_indices == j)  # States in group j
            transition_sum = jnp.sum(row_i[dest_mask])  # Sum P[i, t] for t in Sj
            return carry.at[src_group, j].add(transition_sum)
        
        return jax.lax.fori_loop(0, k, add_to_group, P_lumped_carry)
    
    # Process all rows
    P_lumped = jax.lax.fori_loop(0, n, process_row, P_lumped)
    
    # Divide by source group sizes (average over states in source group)
    P_lumped = P_lumped / group_sizes[:, jnp.newaxis]
    
    # Renormalize rows
    row_sums = jnp.sum(P_lumped, axis=1, keepdims=True)
    P_lumped = P_lumped / row_sums
    
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
        partition: List of lists, where each inner list contains indices 
                   of states to be combined into a single aggregate state.
                   Example: [[0,1], [2,3], [4,5]] combines states 0&1 into 
                   aggregate state 0, states 2&3 into aggregate state 1, etc.
    
    Returns:
        MarkovChain: New chain with k states where k = len(partition)
    
    Raises:
        ValueError: If partition is invalid (missing states, duplicates, 
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
        - Partition must include all states exactly once
        - Each group in partition must be non-empty
        - States within each aggregate are weighted equally
        - Lumping may not preserve the Markov property (strong lumpability)
    """
    n_states = MC.P.shape[0]
    
    # Validate inverse indices (strict checking, fails on first violation)
    _validate_inverse_indices(inverse_indices, n_states)
    
    # Compute lumped transition matrix
    if (type(MC.P) is jnp.ndarray) or (n_states<=60):
        P_lumped = _compute_lumped_transition_matrix(MC.dense_P(), inverse_indices)
    else:
        P_lumped = _compute_lumped_transition_matrix_lazy(MC.P, inverse_indices)

    
    # Create new MarkovChain instance
    # Preserve tolerance if available
    tolerance = getattr(MC, 'tolerance', None)
    return MarkovChain(P=P_lumped, tolerance=tolerance)


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
        - This is a dense matrix operation: O(n²k) where n=states, k=aggregates
        - For large chains, this may be expensive
    """
    _validate_inverse_indices(inverse_indices, MC.P.shape[0])
    
    k = int(inverse_indices.max()) + 1
    
    # For each pair of aggregates (i, j)
    for i in range(k):
        for j in range(k):
            # Get states in each group
            group_i_mask = (inverse_indices == i)
            group_j_mask = (inverse_indices == j)
            
            # Compute total transition probability from each state in group_i to group_j
            probs_to_j = jnp.sum(MC.P[:, group_j_mask], axis=1)[group_i_mask]
            
            # Check if all states in group_i have same transition probability to group_j
            if not jnp.allclose(probs_to_j, probs_to_j[0], atol=tolerance):
                return False
    
    return True


def partition_from_permutation_symmetry(
    n_states: int,
    state_labels: list[tuple],
    permutation_group: list[tuple]
) -> jnp.ndarray:
    """
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
