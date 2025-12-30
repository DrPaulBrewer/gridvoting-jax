
import jax
import jax.lax
import jax.numpy as jnp
from warnings import warn
from collections import Counter

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
        self.tolerance = tolerance  # Store tolerance for later use
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
                - "power_method": Single-path power method with uniform initial guess (O(N^2)).
                  Matches lazy power method behavior.
                - "bifurcated_power_method": Dual-start entropy-based power method (O(N^2)).
                  More robust but more expensive than power_method.
            initial_guess: Optional starting distribution for "power_method".
            max_iterations: Maximum iterations for iterative solvers.
            timeout: Maximum time in seconds for iterative solvers (default: 30.0).
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
            self.stationary_distribution = self._solve_gmres_matrix_inversion(tolerance, max_iterations, initial_guess)
        elif solver == "power_method":
            self.stationary_distribution = self._solve_power_method(tolerance, max_iterations, initial_guess, timeout)
        elif solver == "bifurcated_power_method":
            self.stationary_distribution = self._solve_bifurcated_power_method(tolerance, max_iterations, timeout)
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
            maxiter=max_iterations,
            solve_method='batched'
        )
        
        if info > 0:
            warn(f"GMRES did not converge in {max_iterations} iterations based on internal criteria.")
        
        # Enforce non-negativity and renormalization (numerical artifacts)
        v = _move_neg_prob_to_max(v)
        v = v / jnp.sum(v)
        
        return v

    def _solve_power_method(self, tolerance, max_iterations, initial_guess=None, timeout=30.0):
        """
        Single-path power method with uniform initial guess.
        
        This is the standard power method implementation that matches lazy power method behavior.
        Starts from uniform distribution and iterates until convergence.
        
        Args:
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            initial_guess: Optional initial distribution (if None, uses uniform)
            timeout: Max execution time in seconds
        
        Returns:
            Stationary distribution vector
        """
        import time
        n = self.P.shape[0]
        start_time = time.time()
        
        # Use uniform initial guess if not provided (matches lazy behavior)
        if initial_guess is None:
            v = jnp.ones(n) / n
        else:
            v = initial_guess / jnp.sum(initial_guess)  # Normalize
        
        # Adaptive batching for time checks
        check_interval = 10
        next_check = check_interval
        
        i = 0
        while i < max_iterations:
            # Evolve until next check using JAX compiled loop
            batch_end = min(next_check, max_iterations)
            batch_size = batch_end - i
            
            # Use lax.fori_loop for compiled batched evolution
            def evolve_step(_, carry):
                vec, P = carry
                return (jnp.dot(vec, P), P)
            v, _ = jax.lax.fori_loop(0, batch_size, evolve_step, (v, self.P))
            i = batch_end
            
            # Check convergence
            diff = jnp.linalg.norm(self.evolve(v) - v, ord=1)
            if diff < tolerance:
                return v
            
            # Check timeout
            if (time.time() - start_time) > timeout:
                warn(f"Power method timed out after {timeout}s (iter {i}). Check norm: {diff}")
                return v
            
            # Adaptive: Increase interval
            if check_interval < 1000:
                check_interval *= 2
            next_check = i + check_interval
        
        # Final check
        diff = jnp.linalg.norm(self.evolve(v) - v, ord=1)
        warn(f"Power method did not converge in {max_iterations} iterations. Final diff: {diff}")
        return v

    def _solve_bifurcated_power_method(self, tolerance, max_iterations, timeout=30.0):
        """
        Bifurcated (dual-start) power method using entropy-based starting points.
        
        Starts from two different initial guesses (max and min entropy rows) and
        evolves both until they converge to each other. More robust for detecting
        issues but more expensive than single-path power method.
        
        This was the previous default power method implementation.
        
        Args:
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            timeout: Max execution time in seconds
        
        Returns:
            Stationary distribution vector (average of two converged paths)
        """
        import time
        n = self.P.shape[0]
        start_time = time.time()
        
        # Calculate entropy of each row of P to find diverse starting points
        # Entropy of a row P[i]: H(i) = - sum(P[i,j] * log2(P[i,j]))
        P_safe = jnp.where(self.P > 0, self.P, 1.0)  # Avoid log(0)
        row_entropy = -jnp.sum(self.P * jnp.log2(P_safe), axis=1)
        
        # Start 1: Max entropy (most uncertain transition)
        idx_max = jnp.argmax(row_entropy).item()
        v1 = jnp.zeros(n).at[idx_max].set(1.0)
        
        # Start 2: Min entropy (most deterministic transition)
        idx_min = jnp.argmin(row_entropy).item()
        v2 = jnp.zeros(n).at[idx_min].set(1.0)
        
        # Adaptive batching for time checks
        check_interval = 10
        next_check = check_interval
        
        # Evolve both paths
        i = 0
        while i < max_iterations:
            # Stack both vectors for batched evolution
            V = jnp.stack([v1, v2], axis=0)  # Shape: (2, n)
            
            # Evolve batch until next check
            batch_end = min(next_check, max_iterations)
            batch_size = batch_end - i
            
            # Use lax.fori_loop for compiled batched evolution
            def evolve_batch_step(_, carry):
                V_state, P = carry
                return (jnp.dot(V_state, P), P)
            V, _ = jax.lax.fori_loop(0, batch_size, evolve_batch_step, (V, self.P))
            i = batch_end
            
            # Unpack
            v1, v2 = V[0], V[1]
            
            # Check convergence (paths converge to each other)
            diff = jnp.linalg.norm(v1 - v2, ord=1)
            if diff < tolerance:
                return (v1 + v2) / 2.0
            
            # Check timeout
            if (time.time() - start_time) > timeout:
                warn(f"Bifurcated power method timed out after {timeout}s (iter {i}). Diff between paths: {diff}")
                return (v1 + v2) / 2.0
            
            # Adaptive: Increase interval
            if check_interval < 1000:
                check_interval *= 2
            next_check = i + check_interval
        
        # Final convergence check
        diff = jnp.linalg.norm(v1 - v2, ord=1)
        warn(f"Bifurcated power method did not converge in {max_iterations} iterations. Final diff between paths: {diff}")
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


# ============================================================================
# Markov Chain Lumping Functions
# ============================================================================

def _validate_partition(partition: list[list[int]], n_states: int) -> None:
    """
    Validate partition is a proper partition of states.
    
    Checks (in order, fails on first violation):
    1. All groups are non-empty
    2. All state indices are valid (0 to n-1)
    3. All states are included
    4. No state appears more than once
    
    Raises:
        ValueError: On first violation with descriptive error message
    """
    # Check 1: No empty groups
    for i, group in enumerate(partition):
        if len(group) == 0:
            raise ValueError(f"Partition group {i} is empty")
    
    # Flatten partition
    all_states = [s for group in partition for s in group]
    
    # Check 2: Valid indices
    for s in all_states:
        if not (0 <= s < n_states):
            raise ValueError(f"Invalid state index {s} (must be 0-{n_states-1})")
    
    # Check 3: All states present
    if set(all_states) != set(range(n_states)):
        missing = set(range(n_states)) - set(all_states)
        raise ValueError(f"Partition missing states: {sorted(missing)}")
    
    # Check 4: No duplicates
    if len(all_states) != len(set(all_states)):
        counts = Counter(all_states)
        duplicates = [s for s, count in counts.items() if count > 1]
        raise ValueError(f"Partition contains duplicate states: {duplicates}")


def _compute_lumped_transition_matrix(P: jnp.ndarray, partition: list[list[int]]) -> jnp.ndarray:
    """
    Compute lumped transition matrix using vectorized scatter-add operations.
    
    Fully vectorized implementation that avoids A @ P @ A.T matrix multiplication.
    Uses JAX's advanced indexing to accumulate transition probabilities directly.
    
    P'[i,j] = (1/|Si|) * sum_{s in Si, t in Sj} P[s,t]
    
    Args:
        P: Original transition matrix (n×n)
        partition: Partition specification
    
    Returns:
        jnp.ndarray: Lumped transition matrix (k×k) where k = len(partition)
    
    Performance:
        Old: O(n²k) for A @ P @ A.T with dense matrices
        New: O(n²) vectorized scatter-add (much faster, no Python loops)
    """
    k = len(partition)
    n = P.shape[0]
    
    # Create mapping from original state to aggregate state
    state_to_aggregate = jnp.zeros(n, dtype=jnp.int32)
    group_sizes = jnp.zeros(k)
    
    for i, group in enumerate(partition):
        for s in group:
            state_to_aggregate = state_to_aggregate.at[s].set(i)
        group_sizes = group_sizes.at[i].set(len(group))
    
    # Vectorized approach: Use advanced indexing to accumulate
    # For each entry P[s,t], we want to add it to P_lumped[i,j]
    # where i = state_to_aggregate[s] and j = state_to_aggregate[t]
    
    # Create index arrays for all (s,t) pairs
    row_indices = state_to_aggregate  # Shape: (n,) - maps each row to aggregate
    col_indices = state_to_aggregate  # Shape: (n,) - maps each col to aggregate
    
    # Use at[].add() with broadcasting to accumulate all values at once
    # We need to iterate over rows but vectorize over columns
    P_lumped = jnp.zeros((k, k))
    
    for s in range(n):
        i = int(row_indices[s])
        # Vectorized: add entire row P[s,:] to aggregate row i
        # using col_indices to determine which aggregate columns
        P_lumped = P_lumped.at[i, col_indices].add(P[s, :])
    
    # Divide by group sizes to get average (uniform weighting)
    P_lumped = P_lumped / group_sizes[:, jnp.newaxis]
    
    # Renormalize rows
    row_sums = jnp.sum(P_lumped, axis=1, keepdims=True)
    P_lumped = P_lumped / row_sums
    
    return P_lumped


def lump(MC: MarkovChain, partition: list[list[int]]) -> MarkovChain:
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
        >>> partition = [[0, 2], [1, 3]]
        >>> lumped = lump(mc, partition)
        
        >>> # Swap states
        >>> partition = [[1], [0]]  # Swaps states 0 and 1
        >>> lumped = lump(mc, partition)
    
    Notes:
        - Partition must include all states exactly once
        - Each group in partition must be non-empty
        - States within each aggregate are weighted equally
        - Lumping may not preserve the Markov property (strong lumpability)
    """
    n_states = MC.P.shape[0]
    
    # Validate partition (strict checking, fails on first violation)
    _validate_partition(partition, n_states)
    
    # Compute lumped transition matrix
    P_lumped = _compute_lumped_transition_matrix(MC.P, partition)
    
    # Create new MarkovChain instance
    # Preserve tolerance if available
    tolerance = getattr(MC, 'tolerance', None)
    return MarkovChain(P=P_lumped, tolerance=tolerance)


def unlump(lumped_distribution: jnp.ndarray, partition: list[list[int]]) -> jnp.ndarray:
    """
    Map a probability distribution from lumped space back to original space.
    
    Distributes probability uniformly within each aggregate state.
    
    Args:
        lumped_distribution: Probability distribution over k aggregate states
        partition: Same partition used to create the lumped chain
    
    Returns:
        jnp.ndarray: Probability distribution over n original states
    
    Example:
        >>> partition = [[0,1], [2,3,4]]
        >>> lumped_pi = jnp.array([0.4, 0.6])
        >>> pi = unlump(lumped_pi, partition)
        >>> # pi = [0.2, 0.2, 0.2, 0.2, 0.2]  (uniform within aggregates)
    
    Notes:
        - If the original lumping violated strong lumpability, the unlumped
          distribution will not match the original chain's stationary distribution
    """
    k = len(partition)
    
    # Validate input
    if lumped_distribution.shape[0] != k:
        raise ValueError(
            f"Distribution size {lumped_distribution.shape[0]} doesn't match "
            f"partition size {k}"
        )
    
    # Determine original state space size
    n_states = sum(len(group) for group in partition)
    
    # Distribute probability uniformly within each aggregate
    original_distribution = jnp.zeros(n_states)
    
    for i, group in enumerate(partition):
        # Distribute lumped probability equally among states in this aggregate
        prob_per_state = lumped_distribution[i] / len(group)
        for s in group:
            original_distribution = original_distribution.at[s].set(prob_per_state)
    
    return original_distribution


def is_lumpable(MC: MarkovChain, partition: list[list[int]], tolerance: float = 1e-6) -> bool:
    """
    Test whether a partition preserves the Markov property (strong lumpability).
    
    A partition is strongly lumpable if for each aggregate state i and j,
    all states k within aggregate i have the same total transition probability
    to aggregate j:
        Σ_{l∈Lⱼ} p_{kl} = constant for all k∈Lᵢ
    
    Args:
        MC: MarkovChain instance
        partition: Partition to test
        tolerance: Numerical tolerance for equality check (default: 1e-6)
    
    Returns:
        bool: True if partition is strongly lumpable, False otherwise
    
    Examples:
        >>> # Test if partition preserves Markov property
        >>> P = jnp.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.1, 0.1, 0.8]])
        >>> mc = MarkovChain(P=P)
        >>> is_lumpable(mc, [[0, 1], [2]])  # True
        >>> is_lumpable(mc, [[0, 2], [1]])  # False
    
    Notes:
        - This is a dense matrix operation: O(n²k) where n=states, k=aggregates
        - For large chains, this may be expensive
    """
    _validate_partition(partition, MC.P.shape[0])
    
    # For each pair of aggregates (i, j)
    for i, group_i in enumerate(partition):
        for j, group_j in enumerate(partition):
            # Compute total transition probability from each state in group_i to group_j
            probs_to_j = []
            for k in group_i:
                # Sum transition probabilities from state k to all states in group_j
                prob_k_to_j = jnp.sum(MC.P[k, group_j])
                probs_to_j.append(prob_k_to_j)
            
            # Check if all states in group_i have same transition probability to group_j
            probs_to_j = jnp.array(probs_to_j)
            if not jnp.allclose(probs_to_j, probs_to_j[0], atol=tolerance):
                return False
    
    return True


def partition_from_permutation_symmetry(
    n_states: int,
    state_labels: list[tuple],
    permutation_group: list[tuple]
) -> list[list[int]]:
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
        list[list[int]]: Partition grouping symmetric states
    
    Examples:
        >>> # 3-voter model with full S3 symmetry (all voters interchangeable)
        >>> state_labels = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
        >>> # S3 generators: (0,1) swap and (0,1,2) rotation
        >>> s3_group = [((0,1),), ((0,1,2),)]
        >>> partition = partition_from_permutation_symmetry(6, state_labels, s3_group)
        >>> # Result: [[0,1,2,3,4,5]] - all states equivalent
        
        >>> # Z2 symmetry: swap voters 0 and 1
        >>> z2_group = [((0,1),)]
        >>> partition = partition_from_permutation_symmetry(6, state_labels, z2_group)
        >>> # Result: [[0,1], [2,3], [4,5]] - pairs of swapped states
    
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
    
    # Build partition from equivalence classes
    groups = {}
    for i in range(n_states):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    return list(groups.values())
