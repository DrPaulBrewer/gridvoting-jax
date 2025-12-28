
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy.sparse


# Import from core
from .core import TOLERANCE, GEOMETRY_EPSILON, DTYPE_FLOAT, PLOT_LOG_BIAS

# Wait, distance functions were in __init__.py. I should move them here or core?
# Plan said: spatial.py contains dist_sqeuclidean, dist_manhattan, _is_in_triangle_single

@jax.jit
def dist_sqeuclidean(XA, XB):
    """JAX-based squared Euclidean pairwise distance calculation.
    
    Args:
        XA: array of shape (m, n)
        XB: array of shape (p, n)
    
    Returns:
        Distance matrix of shape (m, p)
    """
    XA = jnp.asarray(XA)
    XB = jnp.asarray(XB)
    # Squared Euclidean: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    XA_sq = jnp.sum(XA**2, axis=1, keepdims=True)
    XB_sq = jnp.sum(XB**2, axis=1, keepdims=True)
    return XA_sq + XB_sq.T - 2 * jnp.dot(XA, XB.T)

@jax.jit
def dist_manhattan(XA, XB):
    """JAX-based Manhattan pairwise distance calculation.
    
    Args:
        XA: array of shape (m, n)
        XB: array of shape (p, n)
    
    Returns:
        Distance matrix of shape (m, p)
    """
    XA = jnp.asarray(XA)
    XB = jnp.asarray(XB)
    # Manhattan distance: sum(|a-b|)
    return jnp.sum(jnp.abs(XA[:, None, :] - XB[None, :, :]), axis=2)

@jax.jit
def _is_in_triangle_single(p, a, b, c):
    """
    Returns True if point p is in triangle (a, b, c).
    Robust for arbitrary vertex winding (CW or CCW).
    
    Args:
        p: Point as [x, y]
        a, b, c: Triangle vertices as [x, y]
    
    Returns:
        Boolean indicating if p is inside triangle

    See also:  computational geometry, half-plane test;
    Stack Overflow answer to https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
       https://stackoverflow.com/a/2049593/103081 
       by https://stackoverflow.com/users/233522/kornel-kisielewicz
    """
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    s1 = cross(p, a, b)
    s2 = cross(p, b, c)
    s3 = cross(p, c, a)

    # Use centralized epsilon from core
    eps = GEOMETRY_EPSILON
    has_neg = (s1 < -eps) | (s2 < -eps) | (s3 < -eps)
    has_pos = (s1 > eps) | (s2 > eps) | (s3 > eps)
    
    return ~(has_neg & has_pos)


class Grid:
    def __init__(self, *, x0, x1, xstep=1, y0, y1, ystep=1):
        """initializes 2D grid with x0<=x<=x1 and y0<=y<=y1;
        Creates a 1D JAX array of grid coordinates in self.x and self.y"""
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.xstep = xstep
        self.ystep = ystep
        xvals = jnp.arange(x0, x1 + xstep, xstep)
        yvals = jnp.arange(y1, y0 - ystep, -ystep)
        xgrid, ygrid = jnp.meshgrid(xvals, yvals)
        self.x = jnp.ravel(xgrid)
        self.y = jnp.ravel(ygrid)
        self.points = jnp.column_stack((self.x,self.y))
        # extent should match extent=(x0,x1,y0,y1) for compatibility with matplotlib.pyplot.contour
        # see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html
        self.extent = (self.x0, self.x1, self.y0, self.y1)
        self.gshape = self.shape()
        self.boundary = ((self.x==x0) | (self.x==x1) | (self.y==y0) | (self.y==y1))
        self.len = self.gshape[0] * self.gshape[1]

    def shape(self, *, x0=None, x1=None, xstep=None, y0=None, y1=None, ystep=None):
        """returns a tuple(number_of_rows,number_of_cols) for the natural shape of the current grid, or a subset"""
        x0 = self.x0 if x0 is None else x0
        x1 = self.x1 if x1 is None else x1
        y0 = self.y0 if y0 is None else y0
        y1 = self.y1 if y1 is None else y1
        xstep = self.xstep if xstep is None else xstep
        ystep = self.ystep if ystep is None else ystep
        if x1 < x0:
            raise ValueError
        if y1 < y0:
            raise ValueError
        if xstep <= 0:
            raise ValueError
        if ystep <= 0:
            raise ValueError
        number_of_rows = 1 + int((y1 - y0) / ystep)
        number_of_cols = 1 + int((x1 - x0) / xstep)
        return (number_of_rows, number_of_cols)

    def within_box(self, *, x0=None, x1=None, y0=None, y1=None):
        """returns a 1D numpy boolean array, suitable as an index mask, for testing whether a grid point is also in the defined box"""
        x0 = self.x0 if x0 is None else x0
        x1 = self.x1 if x1 is None else x1
        y0 = self.y0 if y0 is None else y0
        y1 = self.y1 if y1 is None else y1
        return (self.x >= x0) & (self.x <= x1) & (self.y >= y0) & (self.y <= y1)

    def within_disk(self, *, x0, y0, r, metric="euclidean", **kwargs):
        """returns 1D JAX boolean array, suitable as an index mask, for testing whether a grid point is also in the defined disk"""
        center = jnp.array([[x0, y0]])
        
        if metric == "euclidean":
            # For Euclidean distance, use squared Euclidean and compare r^2
            distances_sq = dist_sqeuclidean(center, self.points)
            mask = (distances_sq <= r**2).flatten()
        elif metric == "manhattan":
            distances = dist_manhattan(center, self.points)
            mask = (distances <= r).flatten()
        else:
            raise ValueError(f"Unsupported metric: {metric}. Use 'euclidean' or 'manhattan'.")
        
        return mask
    
    def within_triangle(self, *, points):
        """returns 1D JAX boolean array, suitable as an index mask, for testing whether a grid point is also in the defined triangle"""
        points = jnp.asarray(points)
        a, b, c = points[0], points[1], points[2]
        
        # Vectorized cross-product triangle containment test
        # Use vmap to apply the single-point test to all grid points
        mask = jax.vmap(
            lambda p: _is_in_triangle_single(p, a, b, c)
        )(self.points)
        
        return mask

    def index(self, *, x, y, tolerance=1e-9):
        """
        Returns the unique 1D array index for grid point (x,y).
        
        Uses direct computation for O(1) lookup instead of linear search.
        For regular grid: index = row * n_cols + col
        where row = (y1 - y) / ystep, col = (x - x0) / xstep
        
        Args:
            x: x-coordinate
            y: y-coordinate
            tolerance: tolerance for coordinate matching (default: 1e-9)
        
        Returns:
            int: Grid index, or raises ValueError if point not on grid
        """
        # Compute row and column indices
        col = round((x - self.x0) / self.xstep)
        row = round((self.y1 - y) / self.ystep)
        
        # Check if within bounds
        n_rows, n_cols = self.gshape
        if not (0 <= row < n_rows and 0 <= col < n_cols):
            raise ValueError(f"Point ({x}, {y}) is outside grid bounds")
        
        # Compute index
        idx = row * n_cols + col
        
        # Verify the point matches (within tolerance)
        if abs(self.x[idx] - x) > tolerance or abs(self.y[idx] - y) > tolerance:
            raise ValueError(f"Point ({x}, {y}) does not match grid point at computed index")
        
        return int(idx)

    def embedding(self, *, valid):
        """
        returns an embedding function efunc(z,fill=0.0) from 1D arrays z of size sum(valid)
        to arrays of size self.len

        valid is a jnp.array of type boolean, of size self.len

        fill is the value for indices outside the embedding. The default
        is zero (0.0).  Setting fill=jnp.nan can be useful for
        plotting purposes as matplotlib will omit jnp.nan values from various
        kinds of plots.
        """

        correct_z_len = valid.sum()

        def efunc(z, fill=0.0):
            v = jnp.full(self.len, fill)
            return v.at[valid].set(z)

        return efunc

    def extremes(self, z, *, valid=None):
        # if valid is None return unrestricted min,points_min,max,points_max
        # if valid is a boolean array, return constrained min,points_min,max,points_max
        # note that min/max is always calculated over all of z, it is the points that must be restricted
        # because valid indicates that z came from a subset of the points
        min_z = float(z.min())
        # Use GEOMETRY_EPSILON from core for consistency with strict tolerance checks
        min_z_mask = jnp.abs(z-min_z) < GEOMETRY_EPSILON
        max_z = float(z.max())
        max_z_mask = jnp.abs(z-max_z) < GEOMETRY_EPSILON
        if valid is None:
           return (min_z,self.points[min_z_mask],max_z,self.points[max_z_mask]) 
        return (min_z,self.points[valid][min_z_mask],max_z,self.points[valid][max_z_mask])

    def spatial_utilities(
        self, *, voter_ideal_points, metric="sqeuclidean", scale=-1
    ):
        """returns utility function values for each voter at each grid point"""
        voter_ideal_points = jnp.asarray(voter_ideal_points)
        
        if metric == "sqeuclidean":
            distances = dist_sqeuclidean(voter_ideal_points, self.points)
        elif metric == "manhattan":
            distances = dist_manhattan(voter_ideal_points, self.points)
        else:
            raise ValueError(f"Unsupported metric: {metric}. Use 'sqeuclidean' or 'manhattan'.")
        
        return scale * distances

    def plot(
        self,
        z,
        *,
        title=None,
        cmap=cm.gray_r,
        alpha=0.6,
        alpha_points=0.3,
        log=True,
        points=None,
        zoom=False,
        border=1,
        logbias=PLOT_LOG_BIAS, # Use constant from core
        figsize=(10, 10),
        dpi=72,
        fname=None
    ):
        """plots values z defined on the grid;
        optionally plots additional 2D points
         and zooms to fit the bounding box of the points"""
        # Convert JAX arrays to NumPy for matplotlib compatibility
        z = np.array(z)
        grid_x = np.array(self.x)
        grid_y = np.array(self.y)
        
        plt.figure(figsize=figsize, dpi=dpi)
        plt.rcParams["font.size"] = "24"
        fmt = "%1.2f" if log else "%.2e"
        if zoom:
            points = np.asarray(points)
            [min_x, min_y] = np.min(points, axis=0) - border
            [max_x, max_y] = np.max(points, axis=0) + border
            box = {"x0": min_x, "x1": max_x, "y0": min_y, "y1": max_y}
            inZoom = np.array(self.within_box(**box))
            zshape = self.shape(**box)
            extent = (min_x, max_x, min_y, max_y)
            zraw = np.copy(z[inZoom]).reshape(zshape)
            x = np.copy(grid_x[inZoom]).reshape(zshape)
            y = np.copy(grid_y[inZoom]).reshape(zshape)
        else:
            zshape = self.gshape
            extent = self.extent
            zraw = z.reshape(zshape)
            x = grid_x.reshape(zshape)
            y = grid_y.reshape(zshape)
        zplot = np.log10(logbias + zraw) if log else zraw
        contours = plt.contour(x, y, zplot, extent=extent, cmap=cmap)
        plt.clabel(contours, inline=True, fontsize=12, fmt=fmt)
        plt.imshow(zplot, extent=extent, cmap=cmap, alpha=alpha)
        if points is not None:
            plt.scatter(points[:, 0], points[:, 1], alpha=alpha_points, color="black")
        if title is not None:
            plt.title(title)
        if fname is None:
            plt.show()
        else:
            plt.savefig(fname)

    def partition_from_symmetry(
        self,
        symmetries: list,
        tolerance: float = 1e-6
    ) -> list[list[int]]:
        """
        Generate partition from spatial symmetries.
        
        Builds partition by grouping grid points that are equivalent under
        the specified spatial symmetries. Does not verify symmetry in the
        transition matrix - assumes user-specified symmetries are correct.
        
        Args:
            symmetries: List of symmetry specifications:
                - 'reflect_x' or 'reflect_x=0': Reflection around x=0
                - 'reflect_x=c': Reflection around x=c
                - 'reflect_y' or 'reflect_y=0': Reflection around y=0
                - 'reflect_y=c': Reflection around y=c
                - 'reflect_xy': Reflection around line y=x
                - 'swap_xy': Swap x and y coordinates (equivalent to reflect_xy)
                - ('rotate', center_x, center_y, degrees): Rotation around (cx, cy)
                  Example: ('rotate', 0, 0, 120) for 120° rotation around origin
            tolerance: Distance tolerance for matching rotated points (default: 1e-6)
                       Useful for approximate symmetries like 120° rotation on grid
        
        Returns:
            list[list[int]]: Partition grouping symmetric points
        
        Examples:
            >>> # Reflection symmetry around y-axis
            >>> partition = grid.partition_from_symmetry(['reflect_x'])
            
            >>> # (x,y) <-> (y,x) symmetry
            >>> partition = grid.partition_from_symmetry(['swap_xy'])
            
            >>> # 120° rotation (BJM spatial triangle example)
            >>> # Grid points near 120° rotations are grouped
            >>> partition = grid.partition_from_symmetry(
            ...     [('rotate', 0, 0, 120)], tolerance=0.5
            ... )
        
        Notes:
            - Symmetries are applied iteratively to build equivalence classes
            - Does not validate that the Markov chain respects these symmetries
            - Rotation tolerance allows approximate symmetries
            - User is responsible for ensuring symmetries are appropriate
            - Optimized for regular grids using direct index computation
        """
        n_states = self.len
        
        # We will build a list of edges (u, v) representing symmetric equivalence
        # source_indices = []
        # target_indices = []
        
        # Use JAX/NumPy for vectorized coordinate transformation
        # x, y are standard numpy arrays (or JAX arrays)
        X = self.x
        Y = self.y
        
        # Accumulate edges in efficient list of arrays
        edges_src = []
        edges_dst = []
        
        # Always include self-loops to ensure every node is in the graph
        # (though connected_components handles isolated nodes, explicitly adding identity is safe)
        edges_src.append(jnp.arange(n_states))
        edges_dst.append(jnp.arange(n_states))

        for sym in symmetries:
            # 1. Vectorized Transformation
            # ----------------------------------------------------------------
            if isinstance(sym, str):
                if sym == 'swap_xy' or sym == 'reflect_xy':
                    # Swap: (x, y) -> (y, x)
                    X_new, Y_new = Y, X
                
                elif sym.startswith('reflect_x'):
                    # Reflect x around c: x' = 2c - x
                    c = float(sym.split('=')[1]) if '=' in sym else 0.0
                    X_new = 2 * c - X
                    Y_new = Y
                
                elif sym.startswith('reflect_y'):
                    # Reflect y around c: y' = 2c - y
                    c = float(sym.split('=')[1]) if '=' in sym else 0.0
                    X_new = X
                    Y_new = 2 * c - Y
                else:
                    raise ValueError(f"Unknown symmetry string: {sym}")
            
            elif isinstance(sym, tuple) and sym[0] == 'rotate':
                # Rotate around (cx, cy) by degrees
                _, cx, cy, degrees = sym
                theta = np.radians(degrees)
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                
                # Apply rotation matrix
                dx = X - cx
                dy = Y - cy
                X_new = cx + (dx * cos_t - dy * sin_t)
                Y_new = cy + (dx * sin_t + dy * cos_t)
            else:
                 raise ValueError(f"Unknown symmetry spec: {sym}")

            # 2. Vectorized Index Lookup (Regular Grid)
            # ----------------------------------------------------------------
            # Expected indices (nearest integer grid point)
            # col = (x - x0) / xstep
            # row = (y1 - y) / ystep  (note y1 is top)
            
            # Use jnp.rint (round to nearest integer)
            col_new = jnp.rint((X_new - self.x0) / self.xstep).astype(jnp.int32)
            row_new = jnp.rint((self.y1 - Y_new) / self.ystep).astype(jnp.int32)
            
            # 3. Filtering
            # ----------------------------------------------------------------
            n_rows, n_cols = self.gshape
            
            # Check bounds
            mask_bounds = (col_new >= 0) & (col_new < n_cols) & \
                          (row_new >= 0) & (row_new < n_rows)
            
            # Calculate 1D index for potentially valid points
            # We must be careful not to access invalid indices, so we apply mask immediately
            # But JAX supports careful masking.
            # Let's compute hypothetical indices, then filter.
            idx_new = row_new * n_cols + col_new
            
            # Check coordinate match (tolerance)
            # Only check where bounds are valid to avoid OOB indexing
            # For OOB, we set error distance to infinity or just mask them out first
            
            # Strategy: Filter indices first
            valid_indices = jnp.where(mask_bounds)[0]
            target_indices = idx_new[valid_indices]
            
            # Check distance on valid candidates
            # X_target = self.x[target_indices]
            # Y_target = self.y[target_indices]
            dist_x = jnp.abs(self.x[target_indices] - X_new[valid_indices])
            dist_y = jnp.abs(self.y[target_indices] - Y_new[valid_indices])
            
            mask_match = (dist_x <= tolerance) & (dist_y <= tolerance)
            
            # Final valid edges
            # Source: valid_indices[mask_match]
            # Target: target_indices[mask_match]
            
            final_src = valid_indices[mask_match]
            final_dst = target_indices[mask_match]
            
            edges_src.append(final_src)
            edges_dst.append(final_dst)
            
        # 4. Graph Construction & Partitioning (CPU/SciPy)
        # ----------------------------------------------------------------
        # Concatenate all edges
        all_src = jnp.concatenate(edges_src)
        all_dst = jnp.concatenate(edges_dst)
        
        # Convert to numpy for SciPy
        all_src_np = np.array(all_src)
        all_dst_np = np.array(all_dst)
        
        # Build Sparse Matrix (Adjacency)
        # We need a symmetric graph for connected components, but csgraph handles directed/undirected
        # connection_type='weak' treats directed edges as undirected for components
        # Weights don't matter, just connectivity. Use 1s.
        data = np.ones(len(all_src_np), dtype=bool)
        
        adj = scipy.sparse.coo_matrix(
            (data, (all_src_np, all_dst_np)),
            shape=(n_states, n_states)
        )
        
        # Find Connected Components
        # connection='weak' means if u->v or v->u, they are connected.
        # This is correct because symmetry implies equivalence A~B.
        n_components, labels = scipy.sparse.csgraph.connected_components(
            adj, 
            directed=True, 
            connection='weak',
            return_labels=True
        )
        
        # 5. Group into Partition List
        # ----------------------------------------------------------------
        # labels is array of shape (n_states,) with component ID
        # We need list of lists.
        
        # Sort by label to group
        order = np.argsort(labels)
        sorted_labels = labels[order]
        sorted_indices = order
        
        # Find split points where label changes
        # diff gives non-zero where value changes
        # np.where returns indices
        split_indices = np.where(np.diff(sorted_labels))[0] + 1
        
        # Split sorted_indices into groups
        # np.split returns a list of arrays
        groups_arrays = np.split(sorted_indices, split_indices)
        
        # Convert to list of lists (Python ints)
        partition = [arr.tolist() for arr in groups_arrays]
        
        return partition
