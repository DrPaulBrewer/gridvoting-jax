
import jax.numpy as jnp
from jax.experimental import sparse
from warnings import warn

from .base import VotingModel
from ..geometry import Grid
from ..stochastic import normalize_if_needed


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
        - outline_and_fill: Solve on coarsened grid (2x spacing) and interpolate
        - outline_and_power: Solve on coarsened grid then refine with power_method
        - outline_and_gmres: Solve on coarsened grid then refine with gmres
        """
        if solver == "outline_and_fill":
            return self._analyze_outline_and_fill(**kwargs)
        elif solver == "outline_and_power":
            return self._analyze_outline_and_power(**kwargs)
        elif solver == "outline_and_gmres":
            return self._analyze_outline_and_gmres(**kwargs)
        else:
            return self.model.analyze(solver=solver, **kwargs)
    
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
        result = normalize_if_needed(result).block_until_ready()
        
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
            jnp.ndarray: Inverse indices array grouping symmetric grid points
        
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
