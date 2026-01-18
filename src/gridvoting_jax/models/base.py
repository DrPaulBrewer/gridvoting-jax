import jax
import jax.numpy as jnp
import copy
from warnings import warn
from ..core import constants

# Import from stochastic
from ..stochastic import (
    LazyStochasticMatrix,
    LazyWeightedStochasticMatrix,
    MarkovChain
)


class VotingModel:
    def __init__(
        self,
        *,
        utility_functions,
        number_of_voters,
        number_of_feasible_alternatives,
        weights=None,
        majority,
        zi
    ):
        """initializes a VotingModel with utility_functions for each voter,
        the number_of_voters,
        the number_of_feasible_alternatives,
        the weights for each alternative,
        the majority size, and whether to use zi fully random agenda or
        intelligent challengers random over winning set+status quo"""
        assert utility_functions.shape == (
            number_of_voters,
            number_of_feasible_alternatives,
        )
        self.utility_functions = utility_functions
        self.number_of_voters = number_of_voters
        self.number_of_feasible_alternatives = number_of_feasible_alternatives
        self.weights = weights
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
        Returns a boolean mask for the grid indicating the Pareto Optimal set (Core under unanimity).  
 
        The Pareto set is the set of alternatives for which no other alternative is 
        universally preferred by all voters (under unanimity, not majority). Equivalently,
        an alternative is Pareto optimal when a change would lower some voter's utility. 
        
        Returns:
            JAX boolean array indicating points in the Pareto set.

        Note: This function is cached, so it will only be computed once.
        
        Uses: For the grid coordinates of Pareto Optimal alternatives, use grid.points[voting_model.Pareto]

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
        """
        Returns the mean, i.e., expected value of z under the stationary distribution.
        
        Args:
            z: Array of values for each alternative
        
        Returns:
            Mean of z under the voting model's stationary distribution
        """
        return jnp.dot(self.stationary_distribution,z)

    def analyze(self, *, solver="full_matrix_inversion", **kwargs):
        """
        Analyzes the voting model to find the stationary distribution.
        
        Args:
            solver: Strategy to use. 
                - "full_matrix_inversion" (Default)
                - "gmres_matrix_inversion"
                - "power_method"
            **kwargs: Passed to solve (e.g. tolerance, max_iterations).
        """
        # Main Analysis
        self.MarkovChain = MarkovChain(P=self.transition_matrix())
        self.MarkovChain.calculate_chain_properties()
        self.core_points = self.MarkovChain.absorbing_points
        self.core_exists = jnp.any(self.core_points)
        if not self.core_exists and solver is not None:
            self.stationary_distribution = self.MarkovChain.solve(
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

    def stochastic_matrix_parameters(self):
        ### 
        # Calculates the parameters needed by class LazyStochasticMatrix
        #
        # Args:
        #     None
        #
        # Returns:
        #     a dict:
        #  mask: a boolean mask of size (number_of_feasible_alternatives, number_of_feasible_alternatives)
        #     mask[i,j] is true is alternative j wins a vote over alternative i with majority self.majority
        #     
        #  status_quo_values: an array of size (number_of_feasible_alternatives,)
        #     status_quo_values[i] is the probability of an alternative i, that is the status quo,
        #                          winning the vote in the ZI or MI challenger process
        ###
        winner_mask = jnp.vectorize(lambda i:self.what_beats(i=i))(jnp.arange(self.number_of_feasible_alternatives))
        number_of_winning_alternatives = winner_mask.sum(axis=1)
        if self.zi:
            number_of_losing_alternatives = (self.number_of_feasible_alternatives - number_of_winning_alternatives).astype(constants.DTYPE_FLOAT)
            status_quo_values = number_of_losing_alternatives/(0.0+self.number_of_feasible_alternatives)
        else:
            status_quo_values = 1.0/(1.0+number_of_winning_alternatives)
        return {
            'mask': winner_mask,
            'status_quo_values': status_quo_values        
            }

    def weighted_stochastic_matrix_parameters(self):
        ###
        # Calculates the parameters needed by class LazyWeightedStochasticMatrix
        #
        # Args:
        #     None
        #
        # Returns:
        #     a dict:
        #  mask: a boolean mask of size (number_of_feasible_alternatives, number_of_feasible_alternatives)
        #     mask[i,j] is true is alternative j wins a vote over alternative i with majority self.majority
        #     
        #  status_quo_values: an array of size (number_of_feasible_alternatives,)
        #     status_quo_values[i] is the probability of an alternative i, that is the status quo,
        #                          winning the vote in the ZI or MI challenger process
        #
        #  weights: an array of size (number_of_feasible_alternatives,)
        #     weights[i] is the weight of alternative i
        ###
        winner_mask = jnp.vectorize(lambda i:self.what_beats(i=i))(jnp.arange(self.number_of_feasible_alternatives))
        weight_of_winning_alternatives = (winner_mask @ self.weights).astype(constants.DTYPE_FLOAT)
        total_weight = self.weights.sum().astype(constants.DTYPE_FLOAT)
        weight_of_losing_alternatives = total_weight - weight_of_winning_alternatives
        if self.zi:
            status_quo_values = (weight_of_losing_alternatives/total_weight).astype(constants.DTYPE_FLOAT)    
        else:
            status_quo_values = (self.weights/(self.weights+weight_of_winning_alternatives)).astype(constants.DTYPE_FLOAT)
        return {
            'mask': winner_mask,
            'status_quo_values': status_quo_values,
            'weights': self.weights
            }

    def transition_matrix(self):
        """Returns the transition matrix for the model's Markov Chain as a LazyStochasticMatrix
        
        Args:
            None
        
        Returns:
            an instance of LazyStochasticMatrix
        """
        if self.weights is None:
            return LazyStochasticMatrix(**self.stochastic_matrix_parameters())
        else:
            return LazyWeightedStochasticMatrix(**self.weighted_stochastic_matrix_parameters())

    def summarize_in_context(self,*,grid,valid=None):
        """
        calculate summary statistics for stationary distribution using grid's coordinates and optional subset valid
        
        Args:
            grid: a Grid instance
            valid (optional): an array of booleans of size grid.len, indicating which grid points are valid
                defaults to all True array for grid
        
        Returns:
            a dict containing summary statistics for the stationary distribution

            if core exists:
                'core_exists': True
                'core_points': array of valid points in core
            else:
                'core_exists': False
                'point_mean': the weighted mean point coordinates
                'point_cov': the covariance of the point coordinates
                'prob_min': the minimum probability
                'prob_min_points': (n_min,2) array of [x,y] points with minimum probability
                'prob_max': the maximum probability
                'prob_max_points': (n_max,2) array of [x,y] points with maximum probability
                'entropy_bits': the Shannon entropy of the distribution in bits

        """
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
        """
        Creates a set of plots for the Voting Model.

        This function was copied from the original gridvoting repository, and has not
        been updated.  It is provided for reference and future development and probably
        does not work as intended.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        def _fn(name):
            """
            Return the filename with the prefix if it is not None.

            Args:
                name: the filename
            """
            return None if fprefix is None else fprefix + name

        def _save(fname):
            """
            Save the current figure to a file.

            Args:
                fname: the filename to save the figure to
            """
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

