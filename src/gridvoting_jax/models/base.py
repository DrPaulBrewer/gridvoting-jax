import jax
import jax.numpy as jnp
import copy
from warnings import warn

# Import from core and dynamics
from ..core import (
    LazyLeftGVMatrix
)
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
        unanimous_model.analyze(solver=None)
        
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
        self.MarkovChain = MarkovChain(P=self.transition_matrix())
        self.MarkovChain.calculate_chain_properties()
        self.core_points = self.MarkovChain.absorbing_points
        self.core_exists = jnp.any(self.core_points)
        if not self.core_exists and solver is not None:
            self.stationary_distribution = self.MarkovChain.find_unique_stationary_distribution(
                solver=solver, 
                **kwargs
            )
        self.analyzed = True


    def what_beats(self, *, i:int):
        """Returns boolean array of size number_of_feasible_alternatives
        with value True where alternative beats current state i by some majority.
        
        Args:
            i: Index of the alternative to compare against
        
        Returns:
            Boolean array where True indicates alternative beats i
        """
        cU = self.utility_functions
        N = self.number_of_feasible_alternatives
        
        # Get utilities for alternative i (status quo)
        # U_i shape: (V,)

        U_i = cU[:, i]
        
        # Generate preferences: does each voter prefer j over i?
        # cU shape: (V, N)
        # U_i shape: (V,) -> broadcast to (V, 1)
        # Result: (V, N) where [v, j] = "does voter v prefer j over i?"
        prefs = jnp.greater(cU, U_i[:, jnp.newaxis])
        
        # Sum votes for each alternative -> (N,)
        votes = prefs.sum(axis=0)
        
        # Determine winners: alternative j beats i if votes[j] >= majority
        beats_i = jnp.greater_equal(votes, self.majority)
        
        # Set diagonal to False (alternative doesn't beat itself)
        beats_i = beats_i.at[i].set(False)
        
        return beats_i
    
    def what_is_beaten_by(self, *, i:int):
        """Returns array of size number_of_feasible_alternatives
        with value 1 where current state i beats alternative by some majority.
        
        This is the converse of what_beats: instead of finding what beats i,
        we find what i beats.
        
        Args:
            i: Index of the alternative doing the beating
            
        Returns:
            Boolean array where True indicates i beats that alternative
        """
        cU = self.utility_functions
        N = self.number_of_feasible_alternatives
        
        # Get utilities for alternative i (the challenger)
        # U_i shape: (V,)
        U_i = cU[:, i]
        
        # Generate preferences: does each voter prefer i over j?
        # cU shape: (V, N)
        # U_i shape: (V,) -> broadcast to (V, 1)
        # Result: (V, N) where [v, j] = "does voter v prefer i over j?"
        prefs = jnp.greater(U_i[:, jnp.newaxis], cU)
        
        # Sum votes for each comparison -> (N,)
        votes = prefs.sum(axis=0)
        
        # Determine which alternatives i beats: i beats j if votes[j] >= majority
        i_beats = jnp.greater_equal(votes, self.majority)
        
        # Set diagonal to False (alternative doesn't beat itself)
        i_beats = i_beats.at[i].set(False)
        
        return i_beats

    def transition_matrix_row(self, i:int):
        """Returns row i of transition matrix"""
        
        winner_mask = self.what_beats(i=i)
        number_of_winners = winner_mask.sum()
        number_of_losers = self.number_of_feasible_alternatives - number_of_winners
        status_quo_value = jnp.where(
            self.zi,
            (0.0+number_of_losers)/(0.0+self.number_of_feasible_alternatives),
            (1.0/(1.0+number_of_winners))
        )
        challenger_value = jnp.where(
            self.zi,
            1.0/(0.0+self.number_of_feasible_alternatives),
            (1.0/(1.0+number_of_winners))
        )
        row = jnp.zeros(self.number_of_feasible_alternatives, dtype=DTYPE_FLOAT)
        row = row.at[i].set(status_quo_value)
        row = row.at[winner_mask].set(challenger_value)
        return row

    def transition_matrix(self):
        """Returns the a transition matrix for the model's Markov Chain as a LazyLeftGVMatrix"""
        return core.LazyLeftGVMatrix(n=self.number_of_feasible_alternatives, get_row=self.transition_matrix_row)

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

