
import jax.numpy as jnp
from warnings import warn

from .base import VotingModel
from ..spatial import Grid


class SpatialVotingModel:
    """
    Voting model with spatial geometry.
    
    Builds VotingModel from ideal points, distance measure, and Grid.
    Handles grid_upscaling solver and spatial visualization.
    """
    
    def __init__(
        self,
        *,
        voter_ideal_points,
        grid,
        number_of_voters,
        majority,
        zi,
        distance_measure="sqeuclidean"
    ):
        """
        Args:
            voter_ideal_points: Array of shape (number_of_voters, 2)
            grid: Grid instance
            number_of_voters: int
            majority: int
            zi: bool
            distance_measure: "sqeuclidean", "euclidean", or custom callable
        """
        self.voter_ideal_points = jnp.asarray(voter_ideal_points)
        self.grid = grid
        self.number_of_voters = number_of_voters
        self.majority = majority
        self.zi = zi
        self.distance_measure = distance_measure
        
        # Compute utility functions using grid.spatial_utilities()
        self.utility_functions = self.grid.spatial_utilities(
            voter_ideal_points=self.voter_ideal_points,
            metric=self.distance_measure
        )
        
        # Create underlying VotingModel
        self.model = VotingModel(
            utility_functions=self.utility_functions,
            number_of_voters=number_of_voters,
            number_of_feasible_alternatives=grid.len,
            majority=majority,
            zi=zi
        )
    
    def analyze(self, *, solver="full_matrix_inversion", **kwargs):
        """
        Analyze with spatial-aware solvers.
        
        Supports all base solvers plus:
        - grid_upscaling: Solve on subgrid then refine (uses dense for final solve)
        - lazy_grid_upscaling: Solve on subgrid then refine (uses lazy for final solve)
        """
        if solver == "grid_upscaling":
            return self._analyze_grid_upscaling(**kwargs)
        elif solver == "lazy_grid_upscaling":
            return self._analyze_lazy_grid_upscaling(**kwargs)
        else:
            return self.model.analyze(solver=solver, **kwargs)
    
    def analyze_lazy(self, *, solver="auto", force_lazy=False, force_dense=False, **kwargs):
        """
        Analyze using lazy matrix construction (delegates to underlying VotingModel).
        
        Args:
            solver: "auto", "gmres", or "power_method"
            force_lazy: Force lazy construction (useful for large grids)
            force_dense: Force dense construction
            **kwargs: Passed to find_unique_stationary_distribution
        
        Example:
            >>> model = gv.bjm_spatial_triangle(g=80, zi=False)
            >>> model.analyze_lazy(force_lazy=True)  # Avoids GPU OOM
        """
        return self.model.analyze_lazy(solver=solver, force_lazy=force_lazy, force_dense=force_dense, **kwargs)
    
    def _analyze_grid_upscaling(self, **kwargs):
        """Grid upscaling implementation (moved from VotingModel.analyze)."""
        # 1. Define Subgrid (Bounding Box of Ideal Points + 1 unit border)
        voter_ideal_points = jnp.asarray(self.voter_ideal_points)
        min_xy = jnp.min(voter_ideal_points, axis=0)
        max_xy = jnp.max(voter_ideal_points, axis=0)
        
        # Add 1 unit border
        x0_sub, y0_sub = min_xy[0] - self.grid.xstep, min_xy[1] - self.grid.ystep
        x1_sub, y1_sub = max_xy[0] + self.grid.xstep, max_xy[1] + self.grid.ystep
        
        # Mask for subgrid
        box_mask = self.grid.within_box(x0=x0_sub, x1=x1_sub, y0=y0_sub, y1=y1_sub)
        valid_indices = jnp.nonzero(box_mask)[0] # Indices in full grid
        
        if len(valid_indices) == 0:
            raise ValueError("Subgrid is empty. Check ideal points and grid bounds.")
            
        # 2. Solve Sub-problem
        # Create sub-model with utilities sliced for valid indices
        sub_utility_functions = self.utility_functions[:, valid_indices]
        
        sub_model = VotingModel(
            utility_functions=sub_utility_functions,
            number_of_voters=self.number_of_voters,
            number_of_feasible_alternatives=len(valid_indices),
            majority=self.majority,
            zi=self.zi
        )
        # Recursively solve submodel using robust default (full_matrix)
        sub_model.analyze(solver="full_matrix_inversion", **kwargs)
        
        initial_guess = None
        if not sub_model.core_exists:
            # 3. Upscale & Refine
            # embedding maps sub-distribution (size M) to full grid (size N) with 0-padding
            embed_fn = self.grid.embedding(valid=box_mask)
            upscaled_dist = embed_fn(sub_model.stationary_distribution)
            
            # Normalize to create initial guess for GMRES
            initial_guess = upscaled_dist / jnp.sum(upscaled_dist)
        else:
            # If core exists in subgrid, fallback to standard solver
            warn("Core found in subgrid_upscaling. Falling back to standard solver.")
        
        # Solve on full grid with GMRES using upscaled solution as initial guess
        # This should converge much faster than power_method or starting from uniform
        return self.model.analyze(solver="gmres_matrix_inversion", initial_guess=initial_guess, **kwargs)
    
    def _analyze_lazy_grid_upscaling(self, **kwargs):
        """Grid upscaling with lazy solver for large grids (avoids OOM)."""
        # 1. Define Subgrid (Bounding Box of Ideal Points + 1 unit border)
        voter_ideal_points = jnp.asarray(self.voter_ideal_points)
        min_xy = jnp.min(voter_ideal_points, axis=0)
        max_xy = jnp.max(voter_ideal_points, axis=0)
        
        # Add 1 unit border
        x0_sub, y0_sub = min_xy[0] - self.grid.xstep, min_xy[1] - self.grid.ystep
        x1_sub, y1_sub = max_xy[0] + self.grid.xstep, max_xy[1] + self.grid.ystep
        
        # Mask for subgrid
        box_mask = self.grid.within_box(x0=x0_sub, x1=x1_sub, y0=y0_sub, y1=y1_sub)
        valid_indices = jnp.nonzero(box_mask)[0]
        
        if len(valid_indices) == 0:
            raise ValueError("Subgrid is empty. Check ideal points and grid bounds.")
        
        # 2. Solve Sub-problem
        sub_utility_functions = self.utility_functions[:, valid_indices]
        
        sub_model = VotingModel(
            utility_functions=sub_utility_functions,
            number_of_voters=self.number_of_voters,
            number_of_feasible_alternatives=len(valid_indices),
            majority=self.majority,
            zi=self.zi
        )
        # Solve submodel (dense is fine for subgrid)
        sub_model.analyze(solver="full_matrix_inversion", **kwargs)
        
        initial_guess = None
        if not sub_model.core_exists:
            # 3. Upscale & Refine
            embed_fn = self.grid.embedding(valid=box_mask)
            upscaled_dist = embed_fn(sub_model.stationary_distribution)
            initial_guess = upscaled_dist / jnp.sum(upscaled_dist)
        else:
            warn("Core found in subgrid_upscaling. Falling back to lazy solver.")
        
        # Solve on full grid with LAZY GMRES (avoids OOM for large grids)
        # Use upscaled solution as initial guess
        return self.model.analyze_lazy(solver="gmres", force_lazy=True, initial_guess=initial_guess, **kwargs)
    
    # Delegate properties to underlying model
    @property
    def stationary_distribution(self):
        return self.model.stationary_distribution
    
    @property
    def MarkovChain(self):
        return self.model.MarkovChain
    
    @property
    def analyzed(self):
        return self.model.analyzed
    
    @property
    def core_points(self):
        return self.model.core_points
    
    @property
    def core_exists(self):
        return self.model.core_exists

    @property
    def Pareto(self):
        """Delegate to model.Pareto."""
        return self.model.Pareto
    
    def summarize_in_context(self, grid=None, **kwargs):
        """Delegate to model, using self.grid if not provided."""
        if grid is None:
            grid = self.grid
        return self.model.summarize_in_context(grid=grid, **kwargs)
    
    def what_beats(self, **kwargs):
        """Delegate to model."""
        return self.model.what_beats(**kwargs)
    
    def what_is_beaten_by(self, **kwargs):
        """Delegate to model."""
        return self.model.what_is_beaten_by(**kwargs)
    
    def E_𝝿(self, z):
        """Delegate to model."""
        return self.model.E_𝝿(z)
    
    # Spatial-specific methods
    def plot_stationary_distribution(self, **kwargs):
        """Visualize distribution on grid using grid.plot()."""
        return self.grid.plot(self.stationary_distribution, **kwargs)
    
    def plots(self, **kwargs):
        """Delegate to model with grid and voter_ideal_points."""
        return self.model.plots(
            grid=self.grid,
            voter_ideal_points=self.voter_ideal_points,
            **kwargs
        )
    
    def get_spatial_symmetry_partition(self, symmetries, tolerance=1e-6):
        """
        Generate partition from spatial symmetries.
        
        Convenience method that delegates to grid.partition_from_symmetry().
        
        Args:
            symmetries: List of symmetry specifications (see Grid.partition_from_symmetry)
            tolerance: Distance tolerance for matching points (default: 1e-6)
        
        Returns:
            list[list[int]]: Partition grouping symmetric grid points
        
        Examples:
            >>> # Reflection around y-axis
            >>> partition = model.get_spatial_symmetry_partition(['reflect_x'])
            
            >>> # 120° rotation for BJM spatial triangle
            >>> partition = model.get_spatial_symmetry_partition(
            ...     [('rotate', 0, 0, 120)], tolerance=0.5
            ... )
        
        Notes:
            - This is a convenience wrapper around grid.partition_from_symmetry()
            - See Grid.partition_from_symmetry() for full documentation
        """
        return self.grid.partition_from_symmetry(symmetries, tolerance=tolerance)
