"""Budget Voting Model for gridvoting-jax.

This module implements budget allocation voting where 3 voters divide a fixed budget.
The feasible set forms a triangular simplex.
"""

import jax
import jax.numpy as jnp
import numpy as np
from .base import VotingModel
from ..spatial import Grid


class BudgetVotingModel:
    """
    Budget allocation voting model.
    
    Models division of a fixed budget among 3 voters. The feasible set of allocations
    forms a triangular simplex defined by: x + y <= budget, x >= 0, y >= 0.
    
    Attributes:
        budget: Total budget to allocate
        zi: Zero Intelligence mode (True) or Minimal Intelligence mode (False)
        grid: Square grid containing the triangle
        valid: Boolean mask for valid (triangle) points
        number_of_alternatives: Number of feasible allocations
        u1, u2, u3: Utility for each voter at each alternative
        U: Stacked utility matrix (3 x number_of_alternatives)
        GiniSS: Gini-like inequality index for each alternative
        model: Underlying VotingModel instance
    """
    
    def __init__(self, budget=100, zi=False):
        """
        Initialize budget voting model.
        
        Args:
            budget: Total budget to divide (default 100)
            zi: Zero Intelligence mode (default False for MI)
        """
        self.budget = budget
        self.zi = zi
        
        # Create square grid
        self.grid = Grid(x0=0, x1=budget, y0=0, y1=budget)
        
        # Triangle constraint: x + y <= budget
        self.valid = (self.grid.x + self.grid.y) <= budget
        self.number_of_alternatives = int(self.valid.sum())
        
        # Verify: (budget+1)*(budget+2)//2
        expected = (budget + 1) * (budget + 2) // 2
        assert self.number_of_alternatives == expected, \
            f"Expected {expected} alternatives, got {self.number_of_alternatives}"
        
        # Utility functions
        # Voter 1 gets x, Voter 2 gets y, Voter 3 gets budget-x-y
        self.u1 = self.grid.x[self.valid]
        self.u2 = self.grid.y[self.valid]
        self.u3 = budget - self.u1 - self.u2
        self.U = jnp.array([self.u1, self.u2, self.u3])
        
        # GiniSS inequality index
        # Scaled to [0,1] where 0=equality, 1=one voter gets all
        self.GiniSS = 1.5 * ((jnp.abs(self.u1 - self.u2) + 
                              jnp.abs(self.u1 - self.u3) + 
                              jnp.abs(self.u2 - self.u3)) / (3 * budget))
        
        # Embedding function for plotting (use fill=np.nan for matplotlib)
        self.embed_triangle = self.grid.embedding(valid=self.valid)
        
        # Create underlying VotingModel
        self.model = VotingModel(
            utility_functions=self.U,
            number_of_voters=3,
            majority=2,
            zi=zi,
            number_of_feasible_alternatives=self.number_of_alternatives
        )
    
    def analyze(self, **kwargs):
        """
        Analyze the budget voting model.
        
        Args:
            **kwargs: Passed to VotingModel.analyze() (e.g., solver, tolerance)
        
        Returns:
            Result from VotingModel.analyze()
        """
        return self.model.analyze(**kwargs)
    
    @property
    def stationary_distribution(self):
        """Stationary distribution over alternatives."""
        return self.model.stationary_distribution
    
    @property
    def MarkovChain(self):
        """Underlying MarkovChain instance."""
        return self.model.MarkovChain
    
    @property
    def core_exists(self):
        """Whether a core exists."""
        return self.model.core_exists

    @property
    def Pareto(self):
        """Pareto optimal set (delegate to model)."""
        return self.model.Pareto
    
    @property
    def core_points(self):
        """Core points (if core exists)."""
        return self.model.core_points
    
    def plot_stationary_distribution(self, **kwargs):
        """
        Plot stationary distribution on triangle.
        
        Args:
            **kwargs: Passed to Grid.plot()
        
        Returns:
            Matplotlib figure
        """
        # Use fill=np.nan for plotting (matplotlib omits NaN values)
        embedded = self.embed_triangle(self.stationary_distribution, fill=np.nan)
        return self.grid.plot(embedded, **kwargs)
    
    def voter_utility_distribution(self, voter_index):
        """
        Calculate probability distribution of a voter's utility under stationary distribution.
        
        Args:
            voter_index: 0, 1, or 2 for voter 1, 2, or 3
        
        Returns:
            Tuple of (utility_values, probabilities) where:
            - utility_values: JAX array [0, 1, 2, ..., budget]
            - probabilities[i]: Probability that voter receives utility i
        
        Note:
            For voter 1: utility = x (u1)
            For voter 2: utility = y (u2)  
            For voter 3: utility = budget - x - y (u3)
        """
        utility_values = jnp.arange(self.budget + 1)
        
        # Get utility for this voter at each alternative
        voter_utilities = self.U[voter_index]  # Shape: (number_of_alternatives,)
        
        # JIT-compiled helper for probability accumulation
        @jax.jit
        def compute_prob_for_utility(u):
            mask = (voter_utilities == u)
            return jnp.sum(jnp.where(mask, self.stationary_distribution, 0.0))
        
        # Vectorized computation
        probabilities = jax.vmap(compute_prob_for_utility)(utility_values)
        
        return utility_values, probabilities
    
    def giniss_distribution(self, granularity=0.10):
        """
        Calculate probability distribution of GiniSS inequality index.
        
        Args:
            granularity: Bin width (default 0.10 for 11 bins)
        
        Returns:
            Tuple of (gini_values, gini_probabilities) where:
            - gini_values: Array of bin edges [0.0, granularity, 2*granularity, ..., 1.0]
            - gini_probabilities[i]: Probability that GiniSS is in bin i
        
        Note:
            GiniSS ranges from 0 (perfect equality) to 1 (one voter gets all).
            Bin i covers [i*granularity, (i+1)*granularity).
            Final bin (n_bins-1) contains ONLY gini_val==1.0 (triangle vertices).
        """
        n_bins = int(jnp.round(1.0 / granularity)) + 1
        granularity_scalar = 1.0/granularity
        gini_values = jnp.linspace(0.0, 1.0, n_bins)
        gini_probabilities = jnp.zeros(n_bins)
        
        # JIT-compiled helper for bin assignment
        @jax.jit
        def assign_to_bin(gini_val):
            # Special case: gini_val == 1.0 goes to final bin
            # This avoids floating point issues with ceil/floor
            is_max = (gini_val >= (1.0 - jnp.finfo(gini_val.dtype).eps))  # Tolerance for float comparison
            bin_idx = jnp.where(
                is_max,
                n_bins - 1,  # Final bin for gini_val == 1.0
                jnp.floor(gini_val * granularity_scalar).astype(int)
            )
            # Clamp to valid range
            return jnp.clip(bin_idx, 0, n_bins - 1)
        
        # Vectorized bin assignment
        bin_indices = jax.vmap(assign_to_bin)(self.GiniSS)
        
        # Accumulate probabilities per bin
        for i in range(len(self.GiniSS)):
            bin_idx = int(bin_indices[i])
            gini_probabilities = gini_probabilities.at[bin_idx].add(
                self.stationary_distribution[i]
            )
        
        return gini_values, gini_probabilities
    
    def get_permutation_symmetry_partition(self, permutation_group=None):
        """
        Generate partition from voter permutation symmetries.
        
        Useful for lumping when voters are interchangeable. For 3-voter budget model,
        the default is full S3 symmetry (all voters interchangeable).
        
        Args:
            permutation_group: List of permutations in cycle notation.
                              Default: S3 generators [((0,1),), ((0,1,2),)]
                              Example for Z2: [((0,1),)] swaps voters 0 and 1
        
        Returns:
            list[list[int]]: Partition grouping symmetric alternatives
        
        Examples:
            >>> # Full S3 symmetry (default)
            >>> partition = model.get_permutation_symmetry_partition()
            
            >>> # Z2 symmetry: swap voters 0 and 1
            >>> partition = model.get_permutation_symmetry_partition([((0,1),)])
        
        Notes:
            - State labels are (x, y) coordinates representing allocations
            - Permutation (0,1) swaps voter 1 ↔ voter 2
            - Permutation (0,1,2) rotates voter 1 → 2 → 3 → 1
            - Default S3 means all (x,y,budget-x-y) permutations are equivalent
        """
        from ..dynamics import partition_from_permutation_symmetry
        
        # Default: Full S3 symmetry (all voters interchangeable)
        if permutation_group is None:
            # S3 generators: (0,1) swap and (0,1,2) 3-cycle
            permutation_group = [((0,1),), ((0,1,2),)]
        
        # Build state labels from grid coordinates
        # Each state is labeled by (x, y) which determines (u1, u2, u3)
        state_labels = []
        for i in range(self.number_of_alternatives):
            x = int(self.u1[i])
            y = int(self.u2[i])
            # Label by utilities (u1, u2, u3)
            state_labels.append((x, y, self.budget - x - y))
        
        return partition_from_permutation_symmetry(
            self.number_of_alternatives,
            state_labels,
            permutation_group
        )
