# gridvoting-jax

**A JAX-powered derivative of the original [gridvoting](https://github.com/drpaulbrewer/gridvoting) project**

[![PyPI version](https://img.shields.io/pypi/v/gridvoting-jax.svg)](https://pypi.org/project/gridvoting-jax/)
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

### Spatial Voting Example

```python
import gridvoting_jax as gv

# Use a pre-built example or create your own
# Triangle 1 from Brewer, Juybari & Moberly (2023)
# Voter ideal points: [[-15, -9], [0, 17], [15, -9]]
# https://doi.org/10.1007/s11403-023-00387-8
model = gv.bjm_spatial_triangle(g=20, zi=False)
model.analyze()

print(f"Device: {gv.device_type}")  # Shows 'gpu', 'tpu', or 'cpu'
print(f"Core exists: {model.core_exists}")
print(f"Stationary distribution: {model.stationary_distribution[:5]}...")
```

### Budget Voting Example (New in v0.9.0)

```python
import gridvoting_jax as gv

# Create budget voting model (divide $100 among 3 voters)
model = gv.BudgetVotingModel(budget=100, zi=False)
model.analyze()

print(f"Alternatives: {model.number_of_alternatives}")  # 5151
print(f"GiniSS inequality: {model.GiniSS[:5]}...")

# Get voter utility distributions
utility_values, probabilities = model.voter_utility_distribution(voter_index=0)

# Get GiniSS inequality distribution
gini_values, gini_probs = model.giniss_distribution(granularity=0.10)
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

**GPU Support**: JAX automatically detects and uses NVIDIA GPUs (CUDA) when available. An Nvidia A100 works well if you have one, but even an old 2017 gaming Nvidia 1080Ti will run some models.

**TPU Support**: JAX automatically detects TPUs on Google Cloud. TPUs we tried have been quirky with this code.

**CPU Support**: Should run with most CPUs. It will fall back to CPU mode if a GPU or TPU is not detected. RAM >=32GB is useful for some tasks.

**CPU-Only Mode**: If you have a GPU or TPU but want to force CPU-only execution, set environment variable `GV_FORCE_CPU=1`:
```bash
GV_FORCE_CPU=1 python your_script.py
```

### Docker Usage

> **⚠️ EXPERIMENTAL - Docker Images Under Development**  
> The new Docker infrastructure is still being tested and may not work correctly in all environments. Images should be finalized after a few patches (v0.10.1, v0.10.2, etc.). Please report any issues on GitHub.

The project uses a multi-tier Docker image system hosted on GitHub Container Registry (GHCR):
- **Base Images**: JAX + CUDA + OSF data (built once)
- **Release Images**: Versioned releases from PyPI (~30s builds)
- **Dev Images**: Local development with mounted source code

**Quick Start - Local Development:**
```bash
# CPU testing with your local source code
./test_docker.sh --dev --cpu tests/

# GPU testing (auto-detects CUDA 12 or 13)
./test_docker.sh --dev --gpu tests/
```

**Testing Specific Versions:**
```bash
# Test a specific release
./test_docker.sh --version=v0.9.1 --gpu

# Run OSF validation
./test_docker_osf.sh --dev --gpu --quick
```

**Detailed Documentation:**  
See [`docs/docker.md`](docs/docker.md) for comprehensive Docker usage guide including:
- Image types and GHCR paths
- Development workflow
- CI/CD pipeline
- Troubleshooting

**Building Images Locally:**
```bash
# Build base images (one-time, slow)
docker build -f Dockerfiles/base/Dockerfile.jax-cpu -t base/cpu:local .

# Build dev images (fast)
docker build -f Dockerfiles/dev/Dockerfile.dev-cpu -t dev/cpu:local .
```

**Float64 Precision**: By default, JAX uses 32-bit floats for better GPU performance. To enable 64-bit precision for higher accuracy:
```python
import gridvoting_jax as gv
gv.enable_float64()
# All subsequent JAX operations will use float64
```

---

## Requirements

- Python 3.9+
- numpy >= 2.0.0
- matplotlib >= 3.8.0
- jax >= 0.4.20
- chex >= 0.1.0

**Google Colab**: All dependencies are pre-installed (numpy 2.0.2, matplotlib 3.10, jax 0.7).

**Note**: pandas and scipy are NOT required. gridvoting-jax uses only JAX for numerical operations.

---

## Performance

*Under review*

---

## Differences from Original gridvoting

This JAX version differs from the original in several ways:

| Feature | Original gridvoting | gridvoting-jax |
|---------|---------------------|----------------|
| **Backend** | NumPy/CuPy | JAX |
| **Precision** | Float64 | Float32 (default)<br>Float64 (available) |
| **Solver** | Power + Algebraic | Algebraic only |
| **Tolerance** | 1e-10 | 5e-5 |
| **Device Detection** | GPU/CPU | TPU/GPU/CPU |
| **Import** | `import gridvoting` | `import gridvoting_jax` |

**Numerical Accuracy**: Float32 provides ~7 decimal digits of precision, which is sufficient for many spatial voting simulations. 

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

## API Documentation (v0.9.0)

The package is organized into submodules, but the public API is exposed at the top level for convenience.

```python
import gridvoting_jax as gv
```

### Core Configuration (`gv.core`)

Centralized configuration and constants.

- **`gv.enable_float64()`**: Enable 64-bit floating point precision globally for JAX
- **`gv.TOLERANCE`**: Default tolerance for floating-point comparisons (5e-5 for float32)
- **`gv.device_type`**: Current device type ('gpu', 'tpu', or 'cpu')
- **`gv.use_accelerator`**: Boolean indicating if GPU/TPU is available

### Spatial Components (`gv.spatial`)

#### `class Grid`

```python
grid = gv.Grid(x0, x1, xstep=1, y0, y1, ystep=1)
```

Constructs a 2D grid for spatial voting models.

**Properties:**
- `grid.points`: JAX array of shape `(N, 2)` containing `[x, y]` coordinates
- `grid.x`, `grid.y`: 1D JAX arrays of x and y coordinates
- `grid.boundary`: 1D boolean mask for boundary points
- `grid.len`: Total number of grid points

**Methods:**
- **`spatial_utilities(voter_ideal_points, metric='sqeuclidean')`**: Distance-based utility calculation
- **`within_box/disk/triangle(...)`**: Geometric query methods returning boolean masks
- **`extremes(z, valid=None)`**: Find min/max values and their locations
- **`embedding(valid)`**: Create embedding function for plotting subsets
- **`plot(z, ...)`**: Plot scalar fields on the grid using Matplotlib

### 6. Symmetry & Dimension Reduction
Reduce computational cost by exploiting spatial symmetries.

Known issue: Practicality.  Time currently required to calculate symmetries and lumping is typically higher than solution time
of the original model.

```python
import gridvoting_jax as gv
from gridvoting_jax.symmetry import suggest_symmetries

# 1. Detect Symmetries
# Suggest valid spatial symmetries for the mode
symmetries = suggest_symmetries(model)
print(f"Detected: {symmetries}") 

# 2. Partition Grid
# Create equivalence classes (e.g. reflection across y-axis)
partition = model.grid.partition_from_symmetry(['reflect_x'])

# 3. Lump Markov Chain
# Solve on reduced state space (e.g., 50% fewer states)
lumped_mc = gv.lump(model.MarkovChain, partition)
lumped_pi = lumped_mc.find_unique_stationary_distribution()

# 4. Unlump
# Map results back to full grid
stationary_distribution = gv.unlump(lumped_pi, partition)
```

### 7. Pareto Efficiency
Analyze the Pareto Optimal set (points where no other point is unanimously preferred).

```python
# Get boolean mask of Pareto set
pareto_mask = model.Pareto

# Visualize
model.grid.plot(pareto_mask, title="Pareto Set")

# Create Unanimous Model
unanimous_model = model.unanimize()
```
### Voting Models

#### `class VotingModel`

Geometry-agnostic base voting model.

```python
vm = gv.VotingModel(
    utility_functions,
    number_of_voters,
    number_of_feasible_alternatives,
    majority,
    zi
)
```

**Methods:**
- **`analyze(solver="full_matrix_inversion")`**: Compute stationary distribution
- **`what_beats(index)`**: Returns alternatives that beat the given index
- **`summarize_in_context(grid)`**: Calculate entropy, mean, and covariance

**Properties:**
- `stationary_distribution`: Probability distribution over alternatives
- `core_exists`: Boolean indicating if a core exists
- `core_points`: Boolean mask of core points

#### `class SpatialVotingModel`

Geometry-aware spatial voting model (delegates to `VotingModel`).

```python
model = gv.SpatialVotingModel(
    voter_ideal_points,
    grid,
    number_of_voters,
    majority,
    zi
)
```

**Additional Methods:**
- **`plot_stationary_distribution(**kwargs)`**: Visualize results on grid
- **`analyze_lazy(solver="auto", force_lazy=False, force_dense=False, **kwargs)`** *(New in v0.10.0)*: 
  - Analyze using lazy matrix construction for large grids (g=80, g=100)
  - Auto-selects dense or lazy based on memory
  - Solvers: `"auto"`, `"gmres"`, `"power_method"`
  - Example: `model.analyze_lazy(force_lazy=True)` for g=80+

#### `class BudgetVotingModel` *(New in v0.9.0)*

Budget allocation voting model for dividing a fixed budget among 3 voters.

```python
model = gv.BudgetVotingModel(budget=100, zi=False)
```

**Features:**
- Feasible set forms triangular simplex: `x + y <= budget`
- Number of alternatives: `(budget+1)*(budget+2)//2`
- Utility functions: `u1=x`, `u2=y`, `u3=budget-x-y`
- GiniSS inequality index: scaled to [0,1]
- Symmetry property: `π[x,y] ≈ π[y,x]`

**Methods:**
- **`analyze(solver="full_matrix_inversion")`**: Compute stationary distribution
- **`voter_utility_distribution(voter_index)`**: Probability distribution of voter payoffs
- **`giniss_distribution(granularity=0.10)`**: Probability distribution of GiniSS index
- **`plot_stationary_distribution(**kwargs)`**: Visualize on triangular simplex

**Properties:**
- `budget`: Total budget to allocate
- `u1, u2, u3`: Utility for each voter at each alternative
- `GiniSS`: Gini-like inequality index for each alternative
- `stationary_distribution`: Probability distribution over allocations

### Example Models *(New in v0.9.0)*

#### Plott's Theorem Examples

Demonstrate core existence conditions from Plott's median voter theorem:

> Plott, C. R. (1967). A notion of equilibrium and its possibility under majority rule. *American Economic Review*, 57(4), 787-806.

```python
# Core existence examples
model = gv.core1(g=20, zi=False)  # 5 voters on horizontal line
model = gv.core2(g=20, zi=False)  # 5 voters on vertical line  
model = gv.core3(g=20, zi=False)  # 5 voters on diagonal
model = gv.core4(g=20, zi=False)  # 4 corners + center
model = gv.ring_with_central_core(g=20, r=10, voters=7)  # Ring + center

# No-core example
model = gv.nocore_triangle(g=20, zi=False)  # Equilateral triangle (cycling)
```

#### Shapes Submodule

Random and geometric configurations:

```python
# Random triangle
model = gv.shapes.random_triangle(g=20, within=10, zi=False)

# Ring of voters (must be odd)
model = gv.shapes.ring(g=20, r=10, voters=5, round_ideal_points=True)
```

#### BJM Research Examples

Examples from published research:

> Brewer, P., Juybari, J. & Moberly, R. (2023). A comparison of zero- and minimal-intelligence agendas in majority-rule voting models. *Journal of Economic Interaction and Coordination*. https://doi.org/10.1007/s11403-023-00387-8

```python
# Spatial voting (Triangle 1 from OSF)
model = gv.bjm_spatial_triangle(g=20, zi=False)

# Budget voting
model = gv.bjm_budget_triangle(budget=100, zi=False)
```

### Markov Chain (`gv.dynamics`)

#### `class MarkovChain`

```python
mc = gv.MarkovChain(P, tolerance=5e-5)
mc.find_unique_stationary_distribution(solver="full_matrix_inversion")
```

**Dense Solvers** (construct full transition matrix):
- **`"full_matrix_inversion"`** (default): Direct matrix inversion
  - Fastest for small grids (g≤40)
  - Memory: O(N²) where N = grid.len
  - Accuracy: Excellent (limited by float32 precision)
  - Fails: g≥80 (out of memory on some GPUs)

- **`"gmres_matrix_inversion"`**: Iterative GMRES solver
  - Memory-efficient for medium grids (g=40-60)
  - Memory: O(N²) for matrix + O(N) for solver
  - Accuracy: Excellent
  - Fails: g≥80 (out of memory on some GPUs matrix construction OOM)

- **`"power_method"`**: Power iteration
  - Memory: O(N²) for matrix + O(N) for vectors
  - Slower convergence than GMRES
  - Robust for difficult cases
  - Fails: g≥80 (out of memory on some GPUs, matrix construction OOM)

- **`"bifurcated_power_method"`**: Dual-start power iteration
  - Starts from min and max utility points
  - Better convergence for some models
  - Same memory requirements as power_method

**Lazy Solvers** (matrix-free, for large grids):
- **`"power_method (lazy)"`**: Lazy power iteration
  - No matrix construction - computes P.T @ v on-the-fly
  - Memory: O(N) only
  - Works for g=80, g=100
  - Slower than dense but enables large grids
  - Use via: `model.analyze_lazy(solver="power_method")`

- **`"bifurcated_power_method (lazy)"`**: Lazy dual-start power iteration
  - Lazy version of bifurcated_power_method
  - Memory: O(N)
  - Use via: `model.analyze_lazy(solver="power_method", force_lazy=True)`

**Spatial Solvers** (SpatialVotingModel only):
- **`"grid_upscaling"`**: Solve on subgrid, refine with dense GMRES
  - Solves on bounding box of voter ideal points
  - Upscales solution to full grid
  - Refines with GMRES using upscaled solution as initial guess
  - Memory: O(N²) for full grid GMRES
  - Fails: g≥80 (GMRES OOM on some GPUs)

- **`"grid_upscaling_lazy_gmres"`**: Solve on subgrid, refine with lazy GMRES
  - Same as grid_upscaling but uses lazy GMRES for refinement
  - Memory: O(N) for lazy solver
  - Works for g=80 (MI mode)
  - May produce NaN in some ZI cases (known issue)

- **`"grid_upscaling_lazy_power"`**: Solve on subgrid, refine with lazy power method
  - Same as grid_upscaling but uses lazy power method for refinement
  - Memory: O(N)
  - Works for g=80
  - More robust than lazy GMRES for ZI mode

**Outline Solvers** *(New in v0.19.0)* (SpatialVotingModel only):
- **`"outline_and_fill"`**: Solve on coarsened grid (2x spacing), interpolate
  - Creates coarsened model with 2x grid spacing (same boundaries)
  - Solves coarsened model (4x fewer points)
  - Interpolates to original grid using sparse BCOO matrix
  - Normalizes result
  - **Fast but low accuracy** (L1 ~0.02-0.03)
  - Use case: Quick approximation or initial guess generation
  - Memory: O(N_coarse²) + O(N_fine) for interpolation

- **`"outline_and_power"`**: Outline + power_method refinement
  - Uses outline_and_fill solution as initial guess
  - Refines with power_method solver
  - **Best for large grids** - never hits OOM at g=80
  - Memory: O(N²) for dense power method
  - Accuracy: Good to Excellent (L1 ~1e-06 to 5e-04)
  - **Recommended for g=80** when GMRES fails

- **`"outline_and_gmres"`**: Outline + GMRES refinement
  - Uses outline_and_fill solution as initial guess
  - Refines with gmres_matrix_inversion
  - Excellent accuracy when memory allows
  - Memory: O(N²) for dense GMRES
  - Fails: g≥80 (GMRES OOM)
  - **Recommended for g≤60** for best accuracy

**Solver Selection Guide**:

| Grid Size | Recommended Solver | Alternative | Notes |
|-----------|-------------------|-------------|-------|
| g=20-40 | `full_matrix_inversion` | `outline_and_gmres` | Fastest, most accurate |
| g=60 | `gmres_matrix_inversion` | `outline_and_gmres` | Balance speed/memory |
| g=80 (MI) | `outline_and_power` | `grid_upscaling_lazy_power` | Only solvers that work |
| g=80 (ZI) | `outline_and_power` | `power_method` | ZI mode harder |
| g=100 | `grid_upscaling_lazy_power` | N/A | Requires lazy solver |

**Pre-computed Interpolation Matrix** *(New in v0.19.0)*:
```python
from gridvoting_jax.models.spatial import create_outline_interpolation_matrix

# Pre-compute for batch processing
C = create_outline_interpolation_matrix(model.grid, coarse_grid)

# Reuse for multiple models with same grid size
model1.analyze(solver="outline_and_fill", interpolation_matrix=C)
model2.analyze(solver="outline_and_fill", interpolation_matrix=C)
```

#### `class LazyMarkovChain` *(New in v0.10.0)*

Memory-efficient Markov chain for large grids (g=80, g=100).

```python
from gridvoting_jax.dynamics.lazy import LazyMarkovChain, LazyTransitionMatrix

lazy_P = LazyTransitionMatrix(utility_functions, majority, zi, number_of_feasible_alternatives)
mc = LazyMarkovChain(lazy_P=lazy_P)
mc.find_unique_stationary_distribution(solver="gmres")
```

**Solvers:** `"gmres"` (default), `"power_method"`

**Methods:**
- `find_unique_stationary_distribution(solver, initial_guess, tolerance, max_iterations, timeout)`
- `stationary_distribution`, `analyzed` (properties)

#### `class LazyTransitionMatrix` *(New in v0.10.0)*

**Methods:**
- `rmatvec(v)`: P.T @ v (non-batched, for GMRES)
- `rmatvec_batched(v)`: P.T @ v (batched, for power method)
- `matvec(v)`: P @ v
- `todense()`: Materialize full matrix

#### `class FlexMarkovChain` *(New in v0.10.0)*

Auto-selects dense or lazy based on memory.

```python
from gridvoting_jax.dynamics.lazy import FlexMarkovChain

mc = FlexMarkovChain.from_voting_model(model)
mc.find_unique_stationary_distribution(solver="auto")
```

### Large Grid Support *(New in v0.10.0)*

- **g=80**: Validated (L1 ~1e-08)
- **g=100**: 10,201 alternatives, uses lazy solvers + grid upscaling

```python
model = gv.bjm_spatial_triangle(g=100, zi=False)
model.analyze(solver="grid_upscaling")  # Uses lazy GMRES
```


### Datasets (`gv.datasets`)

- **`gv.datasets.fetch_osf_spatial_voting_2022_a100()`**: Downloads OSF reference dataset

---

## Benchmarks

Run performance benchmarks to test solver speed across different grid sizes:

```python
import gridvoting_jax as gv

# Print formatted benchmark results
gv.benchmarks.run_osf_comparison()

# run the full suite of benchmarks
gv.benchmarks.performance()

# download the reference dataset
gv.datasets.fetch_osf_spatial_voting_2022_a100()

# compare your computed results to the published scientific record using L1 norm
gv.benchmarks.run_osf_comparison()

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

## Replication & Verification against OSF Data

You can automatically verify the library's output against the original A100 GPU replication data deposited on OSF. This benchmark downloads the reference data and compares stationary distributions using the L1 norm.

```python
from gridvoting_jax.benchmarks.osf_comparison import run_comparison_report

# Run complete comparison report
# Automatically downloads reference data to /tmp/gridvoting_osf_cache
report = run_comparison_report()

# Or test specific configurations
# report = run_comparison_report([(20, False)])  # g=20, MI mode
```

### Google Colab Usage

In a Colab notebook, you can run the full verification suite in a single cell:

```python
!pip install gridvoting-jax

from gridvoting_jax.benchmarks.osf_comparison import run_comparison_report

# Run all 8 replication configurations (g=20, 40, 60, 80)
report = run_comparison_report()
```

This compares your computer's simulation results to the published scientific record.

---

## Testing

### Run Tests

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run all tests (43 tests in v0.9.0)
pytest tests/

# Skip slow tests
pytest tests/ -m "not slow"

# Run only slow tests (benchmark test)
pytest tests/ -m slow

# Run with coverage
pytest tests/ --cov=gridvoting_jax -m "not slow"
```

**Test Coverage (v0.9.0)**:
- Budget voting: 7 tests (symmetry, ZI/MI modes, distributions)
- Plott's theorem examples: 3 tests (core existence/absence)
- Shapes: 4 tests (random triangles, rings)
- BJM examples: 3 tests (OSF validation)
- Core functionality: 26 tests (grid, voting, solvers)

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
