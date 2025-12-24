
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Import from core
from .core import TOLERANCE, GEOMETRY_EPSILON, PLOT_LOG_BIAS

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

    def index(self, *, x, y):
        """returns the unique 1D array index for grid point (x,y)"""
        isSelectedPoint = (self.x == x) & (self.y == y)
        indexes = jnp.flatnonzero(isSelectedPoint)
        return int(indexes[0])

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
        """
        n_states = self.len
        
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
        
        # Apply each symmetry
        for sym in symmetries:
            if isinstance(sym, str):
                # String-based symmetries
                if sym == 'swap_xy' or sym == 'reflect_xy':
                    # Swap x and y coordinates
                    for i in range(n_states):
                        x, y = self.x[i], self.y[i]
                        # Find point with (y, x)
                        for j in range(n_states):
                            if abs(self.x[j] - y) < tolerance and abs(self.y[j] - x) < tolerance:
                                union(i, j)
                                break
                
                elif sym.startswith('reflect_x'):
                    # Reflection around vertical line x=c
                    if '=' in sym:
                        c = float(sym.split('=')[1])
                    else:
                        c = 0.0
                    for i in range(n_states):
                        x, y = self.x[i], self.y[i]
                        # Reflected point: (2c - x, y)
                        x_reflected = 2 * c - x
                        for j in range(n_states):
                            if abs(self.x[j] - x_reflected) < tolerance and abs(self.y[j] - y) < tolerance:
                                union(i, j)
                                break
                
                elif sym.startswith('reflect_y'):
                    # Reflection around horizontal line y=c
                    if '=' in sym:
                        c = float(sym.split('=')[1])
                    else:
                        c = 0.0
                    for i in range(n_states):
                        x, y = self.x[i], self.y[i]
                        # Reflected point: (x, 2c - y)
                        y_reflected = 2 * c - y
                        for j in range(n_states):
                            if abs(self.x[j] - x) < tolerance and abs(self.y[j] - y_reflected) < tolerance:
                                union(i, j)
                                break
            
            elif isinstance(sym, tuple) and sym[0] == 'rotate':
                # Rotation symmetry
                _, cx, cy, degrees = sym
                theta = np.radians(degrees)
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                
                for i in range(n_states):
                    x, y = self.x[i], self.y[i]
                    # Translate to origin
                    x_rel, y_rel = x - cx, y - cy
                    # Rotate
                    x_rot = x_rel * cos_theta - y_rel * sin_theta
                    y_rot = x_rel * sin_theta + y_rel * cos_theta
                    # Translate back
                    x_new = x_rot + cx
                    y_new = y_rot + cy
                    
                    # Find closest point within tolerance
                    for j in range(n_states):
                        dist = np.sqrt((self.x[j] - x_new)**2 + (self.y[j] - y_new)**2)
                        if dist < tolerance:
                            union(i, j)
                            break
        
        # Build partition from equivalence classes
        groups = {}
        for i in range(n_states):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        # Convert to list of lists
        partition = list(groups.values())
        
        return partition
