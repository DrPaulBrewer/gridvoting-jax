import jax
import jax.numpy as jnp
import copy
from warnings import warn

# Import from core and dynamics
from ..core import (
    assert_valid_transition_matrix, 
    assert_zero_diagonal_matrix
)
from ..core.winner_determination import compute_winner_matrix_jit
from ..dynamics import MarkovChain
from ..dynamics.lazy import FlexMarkovChain


class VotingModel:
    def __init__(
        self,
        *,
        utility_functions,
        number_of_voters,
        number_of_feasible_alternatives,
        majority,
        zi
    ):
        """initializes a VotingModel with utility_functions for each voter,
        the number_of_voters,
        the number_of_feasible_alternatives,
        the majority size, and whether to use zi fully random agenda or
        intelligent challengers random over winning set+status quo"""
        assert utility_functions.shape == (
            number_of_voters,
            number_of_feasible_alternatives,
        )
        self.utility_functions = utility_functions
        self.number_of_voters = number_of_voters
        self.number_of_feasible_alternatives = number_of_feasible_alternatives
        self.majority = majority
        self.zi = zi
        self.analyzed = False
        self._pareto_core = None

    def unanimize(self):
        """
        Returns a shallow copy of the model with majority set to unanimity.
        
        The new model requires all voters to agree to move from the status quo.
        Used for identifying Pareto optimal sets.
        """
        # Create shallow copy
        new_model = copy.copy(self)
        
        # Set new parameters
        new_model.majority = new_model.number_of_voters
        
        # Reset analysis state
        new_model.analyzed = False
        new_model.MarkovChain = None
        new_model.stationary_distribution = None
        new_model.core_points = None
        new_model.core_exists = None
        new_model._pareto_core = None
        
        return new_model

    @property
    def Pareto(self):
        """
        Returns the Pareto Optimal set (Core under unanimity).
        
        Returns:
            JAX boolean array indicating points in the Pareto set.
        """
        if self._pareto_core is not None:
            return self._pareto_core
            
        # Create unanimized model
        unanimous_model = self.unanimize()
        
        # Analyze to find core
        # Use full matrix inversion as default for robustness on small-medium grids
        unanimous_model.analyze(solver="full_matrix_inversion")
        
        # Cache and return core points
        self._pareto_core = unanimous_model.core_points
        return self._pareto_core

    def E_𝝿(self,z):
        """returns mean, i.e., expected value of z under the stationary distribution"""
        return jnp.dot(self.stationary_distribution,z)

    def analyze(self, *, solver="full_matrix_inversion", **kwargs):
        """
        Analyzes the voting model to find the stationary distribution.
        
        Args:
            solver: Strategy to use. 
                - "full_matrix_inversion" (Default)
                - "gmres_matrix_inversion"
                - "power_method"
            **kwargs: Passed to find_unique_stationary_distribution (e.g. tolerance, max_iterations).
        """
        # Main Analysis
        self.MarkovChain = MarkovChain(P=self._get_transition_matrix())
        self.core_points = self.MarkovChain.absorbing_points
        self.core_exists = jnp.any(self.core_points)
        if not self.core_exists:
            self.stationary_distribution = self.MarkovChain.find_unique_stationary_distribution(
                solver=solver, 
                **kwargs
            )
        self.analyzed = True

    def analyze_lazy(self, *, solver="auto", force_lazy=False, force_dense=False, **kwargs):
        """
        Analyzes the voting model using lazy matrix construction.
        
        This method uses FlexMarkovChain which auto-selects dense/lazy based on memory,
        or can be forced to use lazy construction for large grids.
        
        Args:
            solver: Strategy to use.
                - "auto" (Default) - Auto-select gmres for lazy, full_matrix_inversion for dense
                - "gmres" - Use GMRES solver
                - "power_method" - Use power method solver
            force_lazy: Force lazy construction (useful for large grids)
            force_dense: Force dense construction
            **kwargs: Passed to find_unique_stationary_distribution (e.g. tolerance, max_iterations).
        
        Example:
            >>> model = gv.bjm_spatial_triangle(g=80, zi=False)
            >>> model.analyze_lazy(force_lazy=True)  # Avoids GPU OOM on large grids
        """
        # Create FlexMarkovChain (auto-selects dense/lazy based on memory)
        flex_mc = FlexMarkovChain.from_voting_model(
            self,
            force_lazy=force_lazy,
            force_dense=force_dense
        )
        
        # Analyze
        flex_mc.find_unique_stationary_distribution(solver=solver, **kwargs)
        
        # Store results
        self.MarkovChain = flex_mc.backend
        
        # Handle core points (LazyMarkovChain doesn't compute absorbing points)
        if hasattr(self.MarkovChain, 'absorbing_points'):
            self.core_points = self.MarkovChain.absorbing_points
        else:
            # For lazy chains, we don't compute absorbing points (would require dense P)
            self.core_points = jnp.zeros(self.number_of_feasible_alternatives, dtype=bool)
        
        self.core_exists = jnp.any(self.core_points)
        
        if not self.core_exists:
            self.stationary_distribution = flex_mc.stationary_distribution
        
        self.analyzed = True

    def what_beats(self, *, index):
        """returns array of size number_of_feasible_alternatives
        with value 1 where alternative beats current index by some majority"""
        assert self.analyzed
        points = (self.MarkovChain.P[index, :] > 0)
        points = points.at[index].set(0)
        return points

    def what_is_beaten_by(self, *, index):
        """returns array of size number_of_feasible_alternatives
        with value 1 where current index beats alternative by some majority"""
        assert self.analyzed
        points = (self.MarkovChain.P[:, index] > 0)
        points = points.at[index].set(0)
        return points
        
    def summarize_in_context(self,*,grid,valid=None):
        """calculate summary statistics for stationary distribution using grid's coordinates and optional subset valid"""
        # missing valid defaults to all True array for grid
        valid = jnp.full((grid.len,), True) if valid is None else valid
        # check valid array shape 
        assert valid.shape == (grid.len,)
        # get X and Y coordinates for valid grid points
        validX = grid.x[valid]
        validY = grid.y[valid]
        valid_points = grid.points[valid]
        if self.core_exists:
            return {
                'core_exists': self.core_exists,
                'core_points': valid_points[self.core_points]
            }
        # core does not exist, so evaulate mean, cov, min, max of stationary distribution
        # first check that the number of valid points matches the dimensionality of the stationary distribution
        assert (valid.sum(),) == self.stationary_distribution.shape
        point_mean = self.E_𝝿(valid_points) 
        cov = jnp.cov(valid_points, rowvar=False, ddof=0, aweights=self.stationary_distribution)
        (prob_min,prob_min_points,prob_max,prob_max_points) = \
            grid.extremes(self.stationary_distribution,valid=valid)
        _nonzero_statd = self.stationary_distribution[self.stationary_distribution>0]
        entropy_bits = -_nonzero_statd.dot(jnp.log2(_nonzero_statd))
        return {
            'core_exists': self.core_exists,
            'point_mean': point_mean,
            'point_cov': cov,
            'prob_min': prob_min,
            'prob_min_points': prob_min_points,
            'prob_max': prob_max,
            'prob_max_points': prob_max_points,
            'entropy_bits': entropy_bits 
        }

    def plots(
        self,
        *,
        grid,
        voter_ideal_points,
        diagnostics=False,
        log=True,
        embedding=lambda z, fill: z,
        zoomborder=0,
        dpi=72,
        figsize=(10, 10),
        fprefix=None,
        title_core="Core (absorbing) points",
        title_sad="L1 norm of difference in two rows of P^power",
        title_diff1="L1 norm of change in corner row",
        title_diff2="L1 norm of change in center row",
        title_sum1minus1="Corner row sum minus 1.0",
        title_sum2minus1="Center row sum minus 1.0",
        title_unreachable_points="Dominated (unreachable) points",
        title_stationary_distribution_no_grid="Stationary Distribution",
        title_stationary_distribution="Stationary Distribution",
        title_stationary_distribution_zoom="Stationary Distribution (zoom)"
    ):
        import matplotlib.pyplot as plt
        import numpy as np
        
        def _fn(name):
            return None if fprefix is None else fprefix + name

        def _save(fname):
            if fprefix is not None:
                plt.savefig(fprefix + fname)

        if self.core_exists:
            grid.plot(
                embedding(self.core_points.astype("int32"), fill=np.nan),
                log=log,
                points=voter_ideal_points,
                zoom=True,
                title=title_core,
                dpi=dpi,
                figsize=figsize,
                fname=_fn("core.png"),
            )
            return None  # when core exists abort as additional plots undefined
        z = self.stationary_distribution
        if grid is None:
            plt.figure(figsize=figsize)
            plt.plot(z)
            plt.title(title_stationary_distribution_no_grid)
            _save("stationary_distribution_no_grid.png")
        else:
            grid.plot(
                embedding(z, fill=np.nan),
                log=log,
                points=voter_ideal_points,
                title=title_stationary_distribution,
                figsize=figsize,
                dpi=dpi,
                fname=_fn("stationary_distribution.png"),
            )
            if voter_ideal_points is not None:
                grid.plot(
                    embedding(z, fill=np.nan),
                    log=log,
                    points=voter_ideal_points,
                    zoom=True,
                    border=zoomborder,
                    title=title_stationary_distribution_zoom,
                    figsize=figsize,
                    dpi=dpi,
                    fname=_fn("stationary_distribution_zoom.png"),
                )

    def _get_transition_matrix(self, batch_size=128):
        """
        Computes the transition matrix P.
        Dispatches to vectorized or batched implementation based on problem size.
        """
        nfa = self.number_of_feasible_alternatives
        
        # Heuristic: For small grids, full vectorization is faster and creates smaller graphs.
        # For large grids (g>=30 -> N>=3700), the O(V*N^2) intermediate tensor risks OOM.
        # g=30 -> N=3721. N^2 ~ 14M. V=3 -> 42M floats/ints. Safe.
        # g=60 -> N=14641. N^2 ~ 214M. V=3 -> 642M floats. ~2.5GB. Risk.
        # Threshold: N > 5000 uses batching.
        if nfa > 5000:
            return self._get_transition_matrix_batched(batch_size=batch_size)
        else:
            return self._get_transition_matrix_vectorized()

    def _get_transition_matrix_vectorized(self):
        """Original fully vectorized implementation. O(V * N^2) memory."""
        utility_functions = self.utility_functions
        majority = self.majority
        zi = self.zi
        nfa = self.number_of_feasible_alternatives
        cU = jnp.asarray(utility_functions)
        
        # Create indices for all alternatives as status quo
        status_quo_indices = jnp.arange(nfa)
        
        # Use core JIT function for winner determination
        # cV[i, j] = 1 if j beats i (where i is status quo)
        cV = compute_winner_matrix_jit(utility_functions, majority, status_quo_indices)
        
        return self._finalize_transition_matrix(cV)

    def _get_transition_matrix_batched(self, batch_size=128):
        """Batched implementation to save memory. O(V * N * batch_size) memory."""
        nfa = self.number_of_feasible_alternatives
        
        # Calculate padding needed
        remainder = nfa % batch_size
        pad_len = (batch_size - remainder) if remainder > 0 else 0
        total_len = nfa + pad_len
        num_batches = total_len // batch_size
        
        # Create indices [0, 1, ... nfa-1, 0, 0 ...] (padding with 0 is fine, we slice later)
        indices = jnp.arange(total_len)
        # We need to mask out the padded indices so they don't affect computation if meaningful
        # (Though here we just slice the result, so exact values for padding don't matter as long as valid)
        indices = jnp.where(indices < nfa, indices, 0)
        
        # Reshape to (num_batches, batch_size)
        batched_indices = indices.reshape((num_batches, batch_size))
        
        # Define the function to map over batches
        # We need cU in closure
        cU = jnp.asarray(self.utility_functions) # (V, N)
        majority = self.majority
        
        def process_batch(batch_idx):
            # batch_idx shape: (batch_size,) containing SQ indices
            
            # batch_idx shape: (batch_size,) containing SQ indices
            
            # Use core JIT function for winner determination
            # Returns (batch_size, N) matrix
            cV_batch = compute_winner_matrix_jit(cU, majority, batch_idx)
            
            return cV_batch

        # Map over the batches
        # Result shape: (num_batches, batch_size, N)
        # We use jax.lax.map for efficiency (sequential compilation, parallel execution potential)
        batched_cV = jax.lax.map(process_batch, batched_indices)
        
        # Collapse batch dimension -> (total_len, N)
        cV_padded = batched_cV.reshape((total_len, nfa))
        
        # Slice off padding -> (nfa, nfa)
        cV = cV_padded[:nfa, :]
        
        # For strict correctness, ensure diagonal is 0 (though logic should ensure it naturally)
        # logic: CH vs SQ. if CH==SQ, prefs is False (not greater). votes=0. 0 < majority (usually).
        # So cV diagonal is 0.
        
        return self._finalize_transition_matrix(cV)

    def _finalize_transition_matrix(self, cV):
        """Convert winner matrix to transition matrix using ZI/MI succession logic."""
        from ..core.zimi_succession_logic import finalize_transition_matrix

        # By convention in cV matrix, the diagonal from state j -> same state j should have 0 votes
        assert_zero_diagonal_matrix(cV)
        
        # Create status_quo_indices for full matrix (all states)
        status_quo_indices = jnp.arange(self.number_of_feasible_alternatives)
        
        # Use shared ZI/MI succession logic
        cP = finalize_transition_matrix(
            cV,
            self.zi,
            status_quo_indices,
            eligibility_mask=None  # Future: self._get_eligibility_mask()
        )
        
        assert_valid_transition_matrix(cP)
        return cP
