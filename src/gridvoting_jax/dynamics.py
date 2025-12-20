
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from warnings import warn

# Import from core and spatial
from .core import (
    TOLERANCE, 
    NEGATIVE_PROBABILITY_TOLERANCE, 
    assert_valid_transition_matrix, 
    assert_zero_diagonal_int_matrix, 
    _move_neg_prob_to_max
)
from .spatial import Grid

class MarkovChain:
    def __init__(self, *, P, computeNow=True, tolerance=None):
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
        if computeNow and self.has_unique_stationary_distribution:
            self.find_unique_stationary_distribution(tolerance=tolerance)

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


    def find_unique_stationary_distribution(self, *, tolerance=None, **kwargs):
        """finds the stationary distribution for a Markov Chain using algebraic method"""
        if tolerance is None:
            tolerance = TOLERANCE
        if jnp.any(self.absorbing_points):
            self.stationary_distribution = None
            return None
        self.stationary_distribution = self.solve_for_unit_eigenvector()
        self.check_norm = self.L1_norm_of_single_step_change(self.stationary_distribution)
        if self.check_norm > tolerance:
            raise RuntimeError(f"Stationary distribution check norm {self.check_norm} exceeds tolerance {tolerance}")
        return self.stationary_distribution

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

    def E_𝝿(self,z):
        """returns mean, i.e., expected value of z under the stationary distribution"""
        return jnp.dot(self.stationary_distribution,z)

    def analyze(self):
        self.MarkovChain = MarkovChain(P=self._get_transition_matrix())
        self.core_points = self.MarkovChain.absorbing_points
        self.core_exists = jnp.any(self.core_points)
        if not self.core_exists:
            self.stationary_distribution = self.MarkovChain.stationary_distribution
        self.analyzed = True

    def what_beats(self, *, index):
        """returns array of size number_of_feasible_alternatives
        with value 1 where alternative beats current index by some majority"""
        assert self.analyzed
        points = (self.MarkovChain.P[index, :] > 0).astype("int32")
        points = points.at[index].set(0)
        return points

    def what_is_beaten_by(self, *, index):
        """returns array of size number_of_feasible_alternatives
        with value 1 where current index beats alternative by some majority"""
        assert self.analyzed
        points = (self.MarkovChain.P[:, index] > 0).astype("int32")
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
        
        # Vectorized computation: compare all alternatives at once
        # cU shape: (n_voters, nfa)
        # cU[:, :, jnp.newaxis] shape: (n_voters, nfa, 1) to broadcast vs challengers (rows)
        # cU[:, jnp.newaxis, :] shape: (n_voters, 1, nfa) to broadcast vs status quo (cols) 
        # Note: Previous implementation comment had axes swapped in explanation but logic was correct for outcome.
        # Let's align with the standard logic:
        # P[i, j] is prob of moving i -> j.
        # i is Status Quo (SQ), j is Challenger (CH).
        # We need votes for CH against SQ.
        # Utility for SQ: cU[:, i] (column i)
        # Utility for CH: cU[:, j] (column j)
        # pref = u(CH) > u(SQ)
        
        # In the original code:
        # preferences = jnp.greater(cU[:, jnp.newaxis, :], cU[:, :, jnp.newaxis])
        # LHS: cU[:, 1, N] -> varying last dim is COLUMNS (CH)
        # RHS: cU[:, N, 1] -> varying middle dim is ROWS (SQ)
        # Result: (V, SQ, CH).  [v, i, j] is "does v prefer j over i?"
        # Correct.
        
        preferences = jnp.greater(cU[:, jnp.newaxis, :], cU[:, :, jnp.newaxis])
        
        # Sum votes across voters: shape (nfa, nfa) -> (SQ, CH)
        total_votes = preferences.astype("int32").sum(axis=0)
        
        # Determine winners: 1 if challenger gets majority, 0 otherwise
        # cV[i, j] = 1 if j beats i
        cV = jnp.greater_equal(total_votes, majority).astype("int32")
        
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
            
            # Get utilities for Status Quo alternatives in this batch
            # U_sq shape: (V, batch_size)
            U_sq = cU[:, batch_idx]
            
            # Generate preferences: CH (all N) vs SQ (batch)
            # LHS: cU -> (V, N) -> reshape to (V, 1, N) for broadcasting
            # RHS: U_sq -> (V, B) -> reshape to (V, B, 1)
            # Result: (V, B, N)
            
            # "Does voter prefer CH (last dim) over SQ (middle dim)?"
            prefs = jnp.greater(cU[:, jnp.newaxis, :], U_sq[:, :, jnp.newaxis])
            
            # Sum votes -> (B, N)
            votes = prefs.astype("int32").sum(axis=0)
            
            # Threshold -> (B, N)
            cV_batch = jnp.greater_equal(votes, majority).astype("int32")
            
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
        """Shared logic to convert winner matrix cV to transition matrix cP"""
        nfa = self.number_of_feasible_alternatives
        zi = self.zi
        
        assert_zero_diagonal_int_matrix(cV)
        cV_sum_of_row = cV.sum(axis=1)  # number of winning alternatives for each SQ
        
        # set up the ZI and MI transition matrices
        if zi:
            # ZI: Uniform random over ALL alternatives.
            # If ch beats sq: move to ch (prob 1/N)
            # If ch loses to sq: stay at sq
            # Plus picked sq itself: stay at sq
            # So prob(move i->j) = 1/N if j beats i
            # prob(stay i) = (1/N) * (count(j that lose to i) + 1)
            #              = (1/N) * ((N - count(win) - 1) + 1)
            #              = (N - row_sum)/N
            # logic in code: cV + diag(N - row_sum) / N
            cP = jnp.divide(
                jnp.add(cV, jnp.diag(jnp.subtract(nfa, cV_sum_of_row))), 
                nfa
            )
        else:
            # MI: Uniform random over Winning Set(i) U {i}
            # Size of set = row_sum + 1
            # Prob(move i->j) = 1/(row_sum+1) if j beats i
            # Prob(stay i) = 1/(row_sum+1)
            # logic in code: (cV + I) / (1 + row_sum)
            cP = jnp.divide(
                jnp.add(cV, jnp.eye(nfa)), 
                (1 + cV_sum_of_row)[:, jnp.newaxis]
            )
        
        assert_valid_transition_matrix(cP)
        return cP


class CondorcetCycle(VotingModel):
    def __init__(self, *, zi):
        # docs suggest to call superclass directly
        # instead of using super()
        # https://docs.python.org/3/tutorial/classes.html#inheritance
        VotingModel.__init__(
            self,
            zi=zi,
            number_of_voters=3,
            majority=2,
            number_of_feasible_alternatives=3,
            utility_functions=jnp.array(
                [
                    [3, 2, 1],  # first agent prefers A>B>C
                    [1, 3, 2],  # second agent prefers B>C>A
                    [2, 1, 3],  # third agents prefers C>A>B
                ]
            ),
        )
