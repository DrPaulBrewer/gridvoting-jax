__version__ = "0.0.1"

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from warnings import warn
import jax
import jax.numpy as jnp

# Default tolerance for float32 calculations
TOLERANCE = 5e-5

# Device detection with NO_GPU override
use_accelerator = False
device_type = 'cpu'

if os.environ.get('NO_GPU', '0') != '1':
    # Check for available accelerators (TPU > GPU > CPU)
    devices = jax.devices()
    if devices:
        default_device = devices[0]
        device_type = default_device.platform
        if device_type in ['gpu', 'tpu']:
            use_accelerator = True
            warn(f"JAX using {device_type.upper()}: {default_device}")
        else:
            warn("JAX using CPU (no GPU/TPU detected)")
else:
    warn("NO_GPU=1: JAX forced to CPU-only mode")

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

    # Small epsilon for numerical tolerance on edges/vertices
    eps = 1e-10
    has_neg = (s1 < -eps) | (s2 < -eps) | (s3 < -eps)
    has_pos = (s1 > eps) | (s2 > eps) | (s3 > eps)
    
    return ~(has_neg & has_pos)

@jax.jit
def _move_neg_prob_to_max(pvector):
    """Fix negative probability components by moving mass to maximum values.
    
    Redistributes the total mass from negative components equally among
    all indices that share the maximum value (within TOLERANCE).
    
    Args:
        pvector: JAX array that may contain small negative values
        
    Returns:
        fixed_pvector: JAX array with negative values zeroed and mass 
                      redistributed equally to all maximum-value indices
    """
    # Identify negative components and calculate mass to redistribute
    # Use jnp.where to avoid boolean indexing which is incompatible with JIT
    to_zero = pvector < 0.0
    mass_destroyed = jnp.where(to_zero, pvector, 0.0).sum()
    
    # Zero out negative components
    fixed_pvector = jnp.where(to_zero, 0.0, pvector)
    
    # Find ALL indices with maximum value (within TOLERANCE for float32)
    max_val = fixed_pvector.max()
    is_max = jnp.abs(fixed_pvector - max_val) < TOLERANCE
    num_max_indices = is_max.sum()
    
    # Distribute mass equally among all maximum indices
    mass_per_index = mass_destroyed / num_max_indices
    fixed_pvector = jnp.where(is_max, fixed_pvector + mass_per_index, fixed_pvector)
    
    return fixed_pvector



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
        # missing valid defaults to all True array for grid
        valid = jnp.full((self.len,), True) if valid is None else valid
        min_z = z.min()
        min_z_mask = jnp.abs(z-min_z)<1e-10  # Strict tolerance for exact min/max
        max_z = z.max()
        max_z_mask = jnp.abs(z-max_z)<1e-10  # Strict tolerance for exact min/max
        return (min_z,self.points[valid][min_z_mask],max_z,self.points[valid][max_z_mask])
        
    def spatial_utilities(
        self, *, voter_ideal_points, metric="sqeuclidean", scale=-1, **kwargs
    ):
        """returns utility function values for each voter at each grid point"""
        voter_ideal_points = np.asarray(voter_ideal_points)
        
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
        logbias=1e-100,
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


def assert_valid_transition_matrix(P, *, decimal=6):
    """asserts that jax or numpy array is square and that each row sums to 1.0
    with default tolerance of 6 decimal places (appropriate for float32)"""
    rows, cols = P.shape
    assert rows == cols
    # Convert to numpy for testing
    P_np = np.array(P)
    np.testing.assert_array_almost_equal(
        P_np.sum(axis=1), 
        np.ones(shape=(rows)), 
        decimal
    )


def assert_zero_diagonal_int_matrix(M):
    """asserts that jax or numpy array is square and the diagonal is 0.0"""
    rows, cols = M.shape
    assert rows == cols
    M_np = np.array(M)
    np.testing.assert_array_equal(
        np.diag(M_np), 
        np.zeros(shape=(rows), dtype=int)
    )

class MarkovChainCPUGPU:
    def __init__(self, *, P, computeNow=True, tolerance=None):
        """initializes a MarkovChainCPUGPU instance by copying in the transition
        matrix P and calculating chain properties"""
        if tolerance is None:
            tolerance = TOLERANCE
        self.P = jnp.asarray(P)  # copy transition matrix to JAX array
        assert_valid_transition_matrix(P)
        diagP = jnp.diagonal(self.P)
        self.absorbing_points = jnp.equal(diagP, 1.0)
        self.unreachable_points = jnp.equal(jnp.sum(self.P, axis=0), diagP)
        self.has_unique_stationary_distibution = not jnp.any(self.absorbing_points)
        if computeNow and self.has_unique_stationary_distibution:
            self.find_unique_stationary_distribution(tolerance=tolerance)

    def L1_norm_of_single_step_change(self, x):
        """returns float(L1(xP-x))"""
        return float(jnp.linalg.norm(jnp.dot(x, self.P) - x, ord=1))

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
        Q = jnp.transpose(self.P).astype(jnp.float32) - jnp.eye(n, dtype=jnp.float32)
        Q = Q.at[0].set(jnp.ones(n, dtype=jnp.float32))  # JAX immutable update
        b = jnp.zeros(n, dtype=jnp.float32)
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
        # Increased threshold for NumPy 2.0 compatibility (was -1e-7)
        if ((min_component<0.0) and (min_component>-2e-7)):
            unit_eigenvector = _move_neg_prob_to_max(unit_eigenvector)
            unit_eigenvector = jnp.dot(unit_eigenvector, self.P)
            min_component = float(unit_eigenvector.min())
        
        if (min_component<0.0):
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
            '(𝝨𝝿)-1':  float(self.stationary_distribution.sum())-1.0, # cast to float to avoid cupy array singleton
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
        return np.dot(self.stationary_distribution,z)

    def analyze(self):
        self.MarkovChain = MarkovChainCPUGPU(P=self._get_transition_matrix())
        self.core_points = np.array(self.MarkovChain.absorbing_points)
        self.core_exists = np.any(self.core_points)
        if not self.core_exists:
            self.stationary_distribution = np.array(
                self.MarkovChain.stationary_distribution
            )
        self.analyzed = True

    def what_beats(self, *, index):
        """returns array of size number_of_feasible_alternatives
        with value 1 where alternative beats current index by some majority"""
        assert self.analyzed
        points = np.array(self.MarkovChain.P[index, :] > 0).astype("int32")
        points[index] = 0
        return points

    def what_is_beaten_by(self, *, index):
        """returns array of size number_of_feasible_alternatives
        with value 1 where current index beats alternative by some majority"""
        assert self.analyzed
        points = np.array(self.MarkovChain.P[:, index] > 0).astype("int32")
        points[index] = 0
        return points
        
    def summarize_in_context(self,*,grid,valid=None):
        """calculate summary statistics for stationary distribution using grid's coordinates and optional subset valid"""
        # missing valid defaults to all True array for grid
        valid = np.full((grid.len,), True) if valid is None else valid
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
        cov = np.cov(valid_points,rowvar=False,ddof=0,aweights=self.stationary_distribution)
        (prob_min,prob_min_points,prob_max,prob_max_points) = \
            grid.extremes(self.stationary_distribution,valid=valid)
        _nonzero_statd = self.stationary_distribution[self.stationary_distribution>0]
        entropy_bits = -_nonzero_statd.dot(np.log2(_nonzero_statd))
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
        title_core="Core (aborbing) points",
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

    def _get_transition_matrix(self):
        utility_functions = self.utility_functions
        majority = self.majority
        zi = self.zi
        nfa = self.number_of_feasible_alternatives
        cU = jnp.asarray(utility_functions)
        
        # Vectorized computation: compare all alternatives at once
        # cU shape: (n_voters, nfa)
        # cU[:, :, jnp.newaxis] shape: (n_voters, nfa, 1)
        # cU[:, jnp.newaxis, :] shape: (n_voters, 1, nfa)
        # Result shape: (n_voters, nfa, nfa) where [v, sq, ch] = voter v prefers challenger ch over status quo sq
        preferences = jnp.greater(cU[:, jnp.newaxis, :], cU[:, :, jnp.newaxis])
        
        # Sum votes across voters: shape (nfa, nfa) where [sq, ch] = votes for ch when sq is status quo
        total_votes = preferences.astype("int32").sum(axis=0)
        
        # Determine winners: 1 if challenger gets majority, 0 otherwise
        cV = jnp.greater_equal(total_votes, majority).astype("int32")
        
        assert_zero_diagonal_int_matrix(cV)
        cV_sum_of_row = cV.sum(axis=1)  # sum up all col for each row
        
        # set up the ZI and MI transition matrices
        if zi:
            cP = jnp.divide(
                jnp.add(cV, jnp.diag(jnp.subtract(nfa, cV_sum_of_row))), 
                nfa
            ).astype(jnp.float32)
        else:
            cP = jnp.divide(
                jnp.add(cV, jnp.eye(nfa)), 
                (1 + cV_sum_of_row)[:, jnp.newaxis]
            ).astype(jnp.float32)
        
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
            utility_functions=np.array(
                [
                    [3, 2, 1],  # first agent prefers A>B>C
                    [1, 3, 2],  # second agent prefers B>C>A
                    [2, 1, 3],  # third agents prefers C>A>B
                ]
            ),
        )

# Import benchmarks submodule
from . import benchmarks
