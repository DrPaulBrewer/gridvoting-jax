# gridvoting-jax

**A JAX-powered derivative of the original [gridvoting](https://github.com/drpaulbrewer/gridvoting) project**

[![PyPI version](https://badge.fury.io/py/gridvoting-jax.svg)](https://badge.fury.io/py/gridvoting-jax)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This library provides GPU/TPU/CPU-accelerated spatial voting simulations using Google's JAX framework with float32 precision.

## Origin and Development

This project is derived from the original `gridvoting` module, which was developed for the research publication:

> Brewer, P., Juybari, J. & Moberly, R.  
> A comparison of zero- and minimal-intelligence agendas in majority-rule voting models.  
> J Econ Interact Coord (2023). https://doi.org/10.1007/s11403-023-00387-8

**Migration to JAX**: The computational backend was refactored from NumPy/CuPy to JAX using Google's Antigravity AI assistant. This migration provides:
- ✨ Unified CPU/GPU/TPU support through JAX
- 🚀 Improved performance through JIT compilation  
- 💾 Float32 precision for efficiency
- 🔗 Better compatibility with modern ML/AI workflows

**Original Project**: https://github.com/drpaulbrewer/gridvoting

---

## Quick Start

```python
import gridvoting_jax as gv

# Create a grid
grid = gv.Grid(x0=-20, x1=20, y0=-20, y1=20)

# Define voter ideal points
voter_ideal_points = [[-15, -9], [0, 17], [15, -9]]

# Generate utility functions
utilities = grid.spatial_utilities(voter_ideal_points=voter_ideal_points)

# Create and analyze voting model
vm = gv.VotingModel(
    utility_functions=utilities,
    majority=2,
    zi=False,  # Minimal Intelligence agenda
    number_of_voters=3,
    number_of_feasible_alternatives=grid.len
)

vm.analyze()

# View results
print(f"Device: {gv.device_type}")  # Shows 'gpu', 'tpu', or 'cpu'
print(f"Stationary distribution: {vm.stationary_distribution[:5]}...")
```

---

## Installation

### Google Colab (Recommended)
All dependencies are pre-installed! Just run:
```python
!pip install gridvoting-jax
```

### Local Installation
```bash
pip install gridvoting-jax
```

**GPU Support**: JAX automatically detects and uses NVIDIA GPUs (CUDA) when available.

**TPU Support**: JAX automatically detects TPUs on Google Cloud.

**CPU-Only Mode**: Set environment variable `NO_GPU=1` to force CPU-only execution:
```bash
NO_GPU=1 python your_script.py
```

---

## Requirements

- Python 3.9+
- numpy >= 2.0.0
- matplotlib >= 3.8.0
- jax >= 0.4.20

**Google Colab**: All dependencies are pre-installed (numpy 2.0.2, matplotlib 3.10, jax 0.7).

**Note**: pandas and scipy are NOT required. gridvoting-jax uses only JAX for numerical operations.

---

## Performance

gridvoting-jax uses JAX's JIT compilation for high performance:

- **First run**: ~1-2s (includes JIT compilation)
- **Subsequent runs**: ~0.03-0.05s (comparable to CuPy)
- **Vectorized operations**: All computations run on GPU/TPU when available

**Benchmark** (g=20, 1681 alternatives, Nvidia 1080Ti):
- Analysis time: 0.033s (after JIT compilation)
- Test suite: 23 tests in ~80s (including slow benchmark test)
- Speedup: 10-30x faster than CPU-only

---

## Differences from Original gridvoting

This JAX version differs from the original in several ways:

| Feature | Original gridvoting | gridvoting-jax |
|---------|-------------------|----------------|
| **Backend** | NumPy/CuPy | JAX |
| **Precision** | Float64 | Float32 |
| **Solver** | Power + Algebraic | Algebraic only |
| **Tolerance** | 1e-10 | 5e-5 |
| **Device Detection** | GPU/CPU | TPU/GPU/CPU |
| **Import** | `import gridvoting` | `import gridvoting_jax` |

**Numerical Accuracy**: Float32 provides ~7 decimal digits of precision, which is sufficient for spatial voting simulations. Tolerance of 5e-5 ensures robust convergence on grids up to 60x60.

---

## Random Sequential Voting Simulations

This follows [section 2 of our research paper](https://link.springer.com/article/10.1007/s11403-023-00387-8#Sec4).

A simulation consists of:
- A sequence of times: `t=0,1,2,3,...`
- A finite feasible set of alternatives **F**
- A set of voters who have preferences over the alternatives and vote truthfully
- A rule for voting and selecting challengers
- A mapping of the set of alternatives **F** into a 2D grid

The active or status quo alternative at time t is called `f[t]`.

At each t, there is a majority-rule vote between alternative `f[t]` and a challenger alternative `c[t]`. The winner of that vote becomes the next status quo `f[t+1]`.

**Randomness** enters through two possible rules for choosing the challenger `c[t]`:
- **Zero Intelligence (ZI)** (`zi=True`): `c[t]` is chosen uniformly at random from **F**
- **Minimal Intelligence (MI)** (`zi=False`): `c[t]` is chosen uniformly from the status quo `f[t]` and the possible winning alternatives given `f[t]`

---

## API Documentation

### class Grid

#### Constructor

```python
gridvoting_jax.Grid(x0, x1, xstep=1, y0, y1, ystep=1)
```

Constructs a 2D grid in x and y dimensions.

**Parameters:**
- `x0`: leftmost grid x-coordinate
- `x1`: rightmost grid x-coordinate  
- `xstep=1`: optional, grid spacing in x dimension
- `y0`: lowest grid y-coordinate
- `y1`: highest grid y-coordinate
- `ystep=1`: optional, grid spacing in y dimension

**Example:**
```python
import gridvoting_jax as gv
grid = gv.Grid(x0=-5, x1=5, y0=-7, y1=7)
```

**Instance Properties:**
- `grid.x0, grid.x1, grid.xstep, grid.y0, grid.y1, grid.ystep` - constructor parameters
- `grid.points` - 2D numpy array of grid points in typewriter order `[[x0,y1],[x0+1,y1],...,[x1,y0]]`
- `grid.x` - 1D numpy array of x-coordinates in typewriter order
- `grid.y` - 1D numpy array of y-coordinates in typewriter order
- `grid.gshape` - natural shape `(number_of_rows, number_of_cols)`
- `grid.extent` - tuple `(x0, x1, y0, y1)` for matplotlib
- `grid.len` - number of points on the grid
- `grid.boundary` - 1D boolean array indicating boundary points

#### Methods

**`grid.spatial_utilities(voter_ideal_points, metric='sqeuclidean', scale=-1)`**

Returns utility function values for each voter at each grid point as a function of distance from an ideal point.

- `voter_ideal_points`: array of 2D coordinates `[[xv1,yv1],[xv2,yv2],...]`
- `metric`: distance metric (default `'sqeuclidean'`). See [scipy.spatial.distance.cdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)

**`grid.within_box(x0=None, x1=None, y0=None, y1=None)`**

Returns 1D boolean array for testing whether grid points are in the defined box.

**`grid.within_disk(x0, y0, r, metric='euclidean')`**

Returns 1D boolean array for testing whether grid points are in the defined disk.

**`grid.within_triangle(points)`**

Returns 1D boolean array for testing whether grid points are in the defined triangle.
- `points`: shape `(3,2)` array of triangle vertices

**`grid.embedding(valid)`**

Returns an embedding function `efunc(z, fill=0.0)` that maps 1D arrays of size `valid.sum()` to arrays of size `grid.len`.

- `valid`: boolean array of length `grid.len` selecting valid grid points
- `fill`: value for invalid indices (default 0.0, use `np.nan` for plotting)

**`grid.plot(z, title=None, log=True, points=None, zoom=False, ...)`**

Creates a contour plot of values z defined on the grid.

---

### class VotingModel

#### Constructor

```python
gridvoting_jax.VotingModel(
    utility_functions,
    number_of_voters,
    number_of_feasible_alternatives,
    majority,
    zi
)
```

**Parameters:**
- `utility_functions`: 2D array of shape `(number_of_voters, number_of_feasible_alternatives)`
- `number_of_voters`: integer
- `number_of_feasible_alternatives`: integer
- `majority`: integer, number of votes needed to win
- `zi`: boolean, True for Zero Intelligence, False for Minimal Intelligence

#### Methods

**`analyze()`**

Computes the transition matrix and stationary distribution.

**`what_beats(index)`**

Returns array indicating which alternatives beat the alternative at `index`.

**`what_is_beaten_by(index)`**

Returns array indicating which alternatives are beaten by the alternative at `index`.

**`summarize_in_context(grid, valid=None)`**

Calculate summary statistics for stationary distribution using grid coordinates.

**`plots(grid, voter_ideal_points, ...)`**

Creates visualization plots of the stationary distribution.

---

### class MarkovChainCPUGPU

#### Constructor

```python
gridvoting_jax.MarkovChainCPUGPU(P, computeNow=True, tolerance=5e-5)
```

**Parameters:**
- `P`: valid transition matrix (square JAX/numpy array whose rows sum to 1.0)
- `computeNow=True`: immediately compute Markov Chain properties
- `tolerance=5e-5`: tolerance for checking convergence (appropriate for float32)

#### Methods

**`solve_for_unit_eigenvector()`**

Finds the stationary distribution by solving for the unit eigenvector.

**`find_unique_stationary_distribution(tolerance=5e-5)`**

Finds the unique stationary distribution using the algebraic method.

**`diagnostic_metrics()`**

Returns dictionary of diagnostic metrics for the Markov chain.

---

## Benchmarks

Run performance benchmarks to test solver speed across different grid sizes:

```python
import gridvoting_jax as gv

# Print formatted benchmark results
gv.benchmarks.performance()

# Get results as dictionary for programmatic use
results = gv.benchmarks.performance(dict=True)
print(f"Device: {results['device']}")
print(f"JAX version: {results['jax_version']}")
for test in results['results']:
    print(f"{test['test_case']}: {test['time_seconds']:.4f}s")
```

**Benchmark Test Cases**:
- Grid sizes: g=20, g=40, g=60
- Voting modes: ZI (Zero Intelligence) and MI (Minimal Intelligence)
- 6 test cases total

---

## Testing

### Run Tests

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run all tests (23 tests, ~80s)
pytest tests/

# Skip slow tests (22 tests, ~15s)
pytest tests/ -m "not slow"

# Run only slow tests (benchmark test)
pytest tests/ -m slow

# Run with coverage
pytest tests/ --cov=gridvoting_jax -m "not slow"
```

**Test Markers**:
- `@pytest.mark.slow`: Long-running tests (benchmarks)
- Use `-m "not slow"` to skip slow tests during development

### Google Colab

```python
!pip install gridvoting-jax
!pytest /usr/local/lib/python3.*/dist-packages/gridvoting_jax/
```

---

## License

The software is provided under the standard [MIT License](./LICENSE.md).

You are welcome to try the software, read it, copy it, adapt it to your needs, and redistribute your adaptations. If you change the software, be sure to change the module name so that others know it is not the original. See the LICENSE file for more details.

---

## Disclaimers

The software is provided in the hope that it may be useful to others, but it is not a full-featured turnkey system for conducting arbitrary voting simulations. Additional coding is required to define a specific simulation.

Automated tests exist and run on GitHub Actions. However, this cannot guarantee that the software is free of bugs or defects or that it will run on your computer without adjustments.

The [MIT License](./LICENSE.md) includes this disclaimer:

> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Research Data

Code specific to the spatial voting and budget voting portions of our research publication -- as well as output data -- is deposited at: [OSF Dataset for A comparison of zero and minimal Intelligence agendas in majority rule voting models](https://osf.io/k2phe/) and is freely available.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Citation

If you use this software in your research, please cite the original paper:

```bibtex
@article{brewer2023comparison,
  title={A comparison of zero-and minimal-intelligence agendas in majority-rule voting models},
  author={Brewer, Paul and Juybari, Jeremy and Moberly, Raymond},
  journal={Journal of Economic Interaction and Coordination},
  year={2023},
  publisher={Springer},
  doi={10.1007/s11403-023-00387-8}
}
```
