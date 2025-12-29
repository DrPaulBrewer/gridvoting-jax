
import jax.numpy as jnp
from jax.experimental import sparse
from warnings import warn

from .base import VotingModel
from ..spatial import Grid


def create_outline_interpolation_matrix(fine_grid, coarse_grid):
    """
    Create sparse interpolation matrix for outline-based solvers.
    
    Uses pattern-based approach with row/column parity to eliminate coordinate lookups.
    For grids where coarse has 2x spacing of fine (same boundaries), this creates a
    sparse BCOO matrix that maps coarse grid probabilities to fine grid via interpolation.
    
    Args:
        fine_grid: Grid instance with finer spacing
        coarse_grid: Grid instance with 2x spacing of fine_grid
    
    Returns:
        jax.experimental.sparse.BCOO: Sparse interpolation matrix of shape (n_fine, n_coarse)
    
    Pattern:
        - (even_row, even_col): Direct copy from coarse[row//2, col//2]
        - (even_row, odd_col): Average of left-right neighbors
        - (odd_row, even_col): Average of up-down neighbors
        - (odd_row, odd_col): Average of 4 diagonal neighbors
    
    Example:
        >>> fine_grid = Grid(x0=0, x1=10, xstep=1, y0=0, y1=10, ystep=1)
        >>> coarse_grid = Grid(x0=0, x1=10, xstep=2, y0=0, y1=10, ystep=2)
        >>> C = create_outline_interpolation_matrix(fine_grid, coarse_grid)
        >>> # Use C @ coarse_dist to interpolate to fine grid
    """
    # Get grid shapes
    n_rows_fine, n_cols_fine = fine_grid.shape()
    n_rows_coarse, n_cols_coarse = coarse_grid.shape()
    
    # Build sparse matrix data as coordinate lists
    rows = []
    cols = []
    data = []
    
    for row_f in range(n_rows_fine):
        for col_f in range(n_cols_fine):
            # Fine grid 1D index
            idx_f = row_f * n_cols_fine + col_f
            
            # Determine parity
            row_even = (row_f % 2 == 0)
            col_even = (col_f % 2 == 0)
            
            if row_even and col_even:
                # Direct copy from coarse grid
                row_c = row_f // 2
                col_c = col_f // 2
                idx_c = row_c * n_cols_coarse + col_c
                rows.append(idx_f)
                cols.append(idx_c)
                data.append(1.0)
                
            elif row_even and not col_even:
                # Left-right interpolation
                row_c = row_f // 2
                col_c_left = col_f // 2
                col_c_right = col_c_left + 1
                
                # Collect valid neighbors
                neighbors = []
                if col_c_left < n_cols_coarse:
                    neighbors.append(row_c * n_cols_coarse + col_c_left)
                if col_c_right < n_cols_coarse:
                    neighbors.append(row_c * n_cols_coarse + col_c_right)
                
                # Average neighbors
                weight = 1.0 / len(neighbors)
                for idx_c in neighbors:
                    rows.append(idx_f)
                    cols.append(idx_c)
                    data.append(weight)
                    
            elif not row_even and col_even:
                # Up-down interpolation
                col_c = col_f // 2
                row_c_up = row_f // 2
                row_c_down = row_c_up + 1
                
                # Collect valid neighbors
                neighbors = []
                if row_c_up < n_rows_coarse:
                    neighbors.append(row_c_up * n_cols_coarse + col_c)
                if row_c_down < n_rows_coarse:
                    neighbors.append(row_c_down * n_cols_coarse + col_c)
                
                # Average neighbors
                weight = 1.0 / len(neighbors)
                for idx_c in neighbors:
                    rows.append(idx_f)
                    cols.append(idx_c)
                    data.append(weight)
                    
            else:  # not row_even and not col_even
                # 4-neighbor interpolation
                row_c_up = row_f // 2
                row_c_down = row_c_up + 1
                col_c_left = col_f // 2
                col_c_right = col_c_left + 1
                
                # Collect valid neighbors
                neighbors = []
                if row_c_up < n_rows_coarse and col_c_left < n_cols_coarse:
                    neighbors.append(row_c_up * n_cols_coarse + col_c_left)
                if row_c_up < n_rows_coarse and col_c_right < n_cols_coarse:
                    neighbors.append(row_c_up * n_cols_coarse + col_c_right)
                if row_c_down < n_rows_coarse and col_c_left < n_cols_coarse:
                    neighbors.append(row_c_down * n_cols_coarse + col_c_left)
                if row_c_down < n_rows_coarse and col_c_right < n_cols_coarse:
                    neighbors.append(row_c_down * n_cols_coarse + col_c_right)
                
                # Average neighbors
                weight = 1.0 / len(neighbors)
                for idx_c in neighbors:
                    rows.append(idx_f)
                    cols.append(idx_c)
                    data.append(weight)
    
    # Convert to sparse BCOO matrix
    indices = jnp.column_stack([jnp.array(rows), jnp.array(cols)])
    values = jnp.array(data)
    
    return sparse.BCOO((values, indices), shape=(fine_grid.len, coarse_grid.len))



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
        - grid_upscaling: Solve on subgrid then refine with dense GMRES
        - grid_upscaling_lazy_gmres: Solve on subgrid then refine with lazy GMRES
        - grid_upscaling_lazy_power: Solve on subgrid then refine with lazy power method
        - outline_and_fill: Solve on coarsened grid (2x spacing) and interpolate
        - outline_and_power: Solve on coarsened grid then refine with power_method
        - outline_and_gmres: Solve on coarsened grid then refine with gmres
        """
        if solver == "grid_upscaling":
            return self._analyze_grid_upscaling(**kwargs)
        elif solver == "grid_upscaling_lazy_gmres":
            return self._analyze_lazy_grid_upscaling(step2_solver="gmres", **kwargs)
        elif solver == "grid_upscaling_lazy_power":
            return self._analyze_lazy_grid_upscaling(step2_solver="power_method", **kwargs)
        elif solver == "outline_and_fill":
            return self._analyze_outline_and_fill(**kwargs)
        elif solver == "outline_and_power":
            return self._analyze_outline_and_power(**kwargs)
        elif solver == "outline_and_gmres":
            return self._analyze_outline_and_gmres(**kwargs)
        # Backward compatibility
        elif solver == "lazy_grid_upscaling":
            warn("'lazy_grid_upscaling' is deprecated, use 'grid_upscaling_lazy_gmres'", DeprecationWarning)
            return self._analyze_lazy_grid_upscaling(step2_solver="gmres", **kwargs)
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
    
    def _analyze_subgrid(self, border_units=1, **kwargs):
        """
        Solve on subgrid (bounding box of voter ideal points + border).
        
        Args:
            border_units: Number of grid steps to add as border (default: 5)
            **kwargs: Passed to subgrid solver
        
        Returns:
            tuple: (sub_model, box_mask, initial_guess)
                - sub_model: Solved VotingModel on subgrid
                - box_mask: Boolean mask for subgrid in full grid
                - initial_guess: Upscaled distribution for full grid (or None if core exists)
        """
        # 1. Define Subgrid (Bounding Box of Ideal Points + border)
        voter_ideal_points = jnp.asarray(self.voter_ideal_points)
        min_xy = jnp.min(voter_ideal_points, axis=0)
        max_xy = jnp.max(voter_ideal_points, axis=0)
        
        # Add border
        x0_sub = min_xy[0] - border_units * self.grid.xstep
        y0_sub = min_xy[1] - border_units * self.grid.ystep
        x1_sub = max_xy[0] + border_units * self.grid.xstep
        y1_sub = max_xy[1] + border_units * self.grid.ystep

        # Check that subgrid fits inside main grid
        # Note: This will fail for small grids (e.g. g=20) where the 5-unit border
        # extends beyond the grid bounds. This is expected behavior - grid upscaling
        # is designed for larger grids where voters don't occupy the entire grid.
        if x0_sub < self.grid.x0 or y0_sub < self.grid.y0 or x1_sub > self.grid.x1 or y1_sub > self.grid.y1:
            raise ValueError("Subgrid extends beyond main grid bounds.")

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
        
        # 3. Create upscaled initial guess
        initial_guess = None
        if not sub_model.core_exists:
            # test sub_model stationary distribution
            assert jnp.isnan(sub_model.stationary_distribution).any() == False, "Submodel stationary distribution contains NaN"
            assert jnp.isinf(sub_model.stationary_distribution).any() == False, "Submodel stationary distribution contains Inf"
            assert jnp.all(sub_model.stationary_distribution>=0), "Submodel stationary distribution contains negative values"

            # Upscale & prepare initial guess
            # Place 0.99 of probability mass on subgrid and 0.01 distributed evenly on non-subgrid points
            N = self.grid.len
            num_subgrid = len(valid_indices)
            num_nonsubgrid = N - num_subgrid
            
            subgrid_mass = 0.99
            nonsubgrid_mass = 0.01
            
            # Scale subgrid distribution to 99% of total mass
            scaled_subgrid_dist = sub_model.stationary_distribution * subgrid_mass
            
            # Calculate fill value for non-subgrid points (1% distributed evenly)
            fill_value = nonsubgrid_mass / num_nonsubgrid if num_nonsubgrid > 0 else 0.0
            
            # Embed with fill - no renormalization needed
            embed_fn = self.grid.embedding(valid=box_mask)
            initial_guess = embed_fn(scaled_subgrid_dist, fill=fill_value)
            
            # Validate
            assert jnp.isclose(initial_guess.sum(), 1.0, atol=1e-4), \
                f"Initial guess sum {initial_guess.sum()} != 1.0"
        else:
            raise AssertionError(
                "Core found in subgrid_upscaling. Grid upscaling is not supported for this case."
            )
        
        return sub_model, box_mask, initial_guess
    
    def _analyze_grid_upscaling(self, **kwargs):
        """Grid upscaling implementation (moved from VotingModel.analyze)."""
        # Solve on subgrid and get upscaled initial guess
        sub_model, box_mask, initial_guess = self._analyze_subgrid(border_units=1, **kwargs)
        
        # Solve on full grid with GMRES using upscaled solution as initial guess
        # This should converge much faster than power_method or starting from uniform
        return self.model.analyze(solver="gmres_matrix_inversion", initial_guess=initial_guess, **kwargs)
    
    def _analyze_lazy_grid_upscaling(self, *, step2_solver="gmres", **kwargs):
        """
        Grid upscaling with lazy solver for large grids (avoids OOM).
        
        Args:
            step2_solver: Solver for refinement step - "gmres" or "power_method"
            **kwargs: Passed to solver
        """
        # Solve on subgrid and get upscaled initial guess
        sub_model, box_mask, initial_guess = self._analyze_subgrid(border_units=1, **kwargs)
        
        # Solve on full grid with lazy solver
        # Use upscaled solution as initial guess
        return self.model.analyze_lazy(solver=step2_solver, force_lazy=True, initial_guess=initial_guess, **kwargs)
    
    def _create_coarsened_model(self):
        """
        Create a coarsened SpatialVotingModel with 2x grid spacing.
        
        Returns:
            SpatialVotingModel: Coarsened model with same boundaries and voter ideal points
        """
        # Create coarsened grid with 2x spacing
        coarse_grid = Grid(
            x0=self.grid.x0,
            x1=self.grid.x1,
            xstep=2 * self.grid.xstep,
            y0=self.grid.y0,
            y1=self.grid.y1,
            ystep=2 * self.grid.ystep
        )
        
        # Create coarsened model with same voter ideal points
        coarse_model = SpatialVotingModel(
            voter_ideal_points=self.voter_ideal_points,
            grid=coarse_grid,
            number_of_voters=self.number_of_voters,
            majority=self.majority,
            zi=self.zi,
            distance_measure=self.distance_measure
        )
        
        return coarse_model
    
    def _solve_and_interpolate_outline(self, interpolation_matrix=None, **kwargs):
        """
        Solve on coarsened grid and interpolate to original grid.
        
        Args:
            interpolation_matrix: Optional pre-computed interpolation matrix
            **kwargs: Passed to coarse solver (coarse_solver, tolerance, max_iterations, etc.)
        
        Returns:
            VotingModel: self.model with stationary distribution set
        """
        # Task 1: Create coarsened model
        coarse_model = self._create_coarsened_model()
        
        # Task 2: Validate grid alignment
        assert coarse_model.grid.x0 == self.grid.x0, "Grid x0 mismatch"
        assert coarse_model.grid.x1 == self.grid.x1, "Grid x1 mismatch"
        assert coarse_model.grid.y0 == self.grid.y0, "Grid y0 mismatch"
        assert coarse_model.grid.y1 == self.grid.y1, "Grid y1 mismatch"
        
        # Task 3: Create or use interpolation matrix
        if interpolation_matrix is None:
            interpolation_matrix = create_outline_interpolation_matrix(
                self.grid, 
                coarse_model.grid
            )
        
        # Task 4: Solve coarsened model
        coarse_solver = kwargs.pop('coarse_solver', 'full_matrix_inversion')
        coarse_model.analyze(solver=coarse_solver, **kwargs)
        
        # Task 5: Interpolate using matrix multiplication
        result = interpolation_matrix @ coarse_model.stationary_distribution
        
        # Task 6: Normalize
        result = result / result.sum()
        
        # Set on underlying model
        self.model.stationary_distribution = result
        self.model.analyzed = True
        
        return self.model
    
    def _analyze_outline_and_fill(self, **kwargs):
        """
        Outline-based solver: Returns raw interpolated solution.
        
        Solves on coarsened grid (2x spacing) and interpolates to original grid.
        No refinement step.
        """
        return self._solve_and_interpolate_outline(**kwargs)
    
    def _analyze_outline_and_power(self, **kwargs):
        """
        Outline-based solver: Refines with power_method.
        
        Solves on coarsened grid, interpolates, then refines using power_method
        with the interpolated solution as initial guess.
        """
        # Get interpolated solution as initial guess
        self._solve_and_interpolate_outline(**kwargs)
        initial_guess = self.model.stationary_distribution
        
        # Refine with power_method
        return self.model.analyze(solver="power_method", initial_guess=initial_guess, **kwargs)
    
    def _analyze_outline_and_gmres(self, **kwargs):
        """
        Outline-based solver: Refines with gmres.
        
        Solves on coarsened grid, interpolates, then refines using gmres_matrix_inversion
        with the interpolated solution as initial guess.
        """
        # Get interpolated solution as initial guess
        self._solve_and_interpolate_outline(**kwargs)
        initial_guess = self.model.stationary_distribution
        
        # Refine with gmres
        return self.model.analyze(solver="gmres_matrix_inversion", initial_guess=initial_guess, **kwargs)
    

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
    def number_of_feasible_alternatives(self):
        return self.model.number_of_feasible_alternatives

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
