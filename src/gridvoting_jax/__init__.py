"""gridvoting-jax: Spatial voting models with JAX."""

__version__ = "0.21.0"

# Core utilities
from .core import TOLERANCE, enable_float64, device_type, use_accelerator

# Core components
from .dynamics import MarkovChain, LazyMarkovChain, FlexMarkovChain, LazyTransitionMatrix, lump, unlump, is_lumpable, partition_from_permutation_symmetry
from .symmetry import suggest_symmetries
from .spatial import Grid, dist_sqeuclidean, dist_manhattan

# Models
from .models import VotingModel, SpatialVotingModel, BudgetVotingModel

# Examples
from .models.examples import (
    condorcet_cycle,
    core1, core2, core3, core4, nocore_triangle, ring_with_central_core,
    bjm_spatial_triangle, bjm_budget_triangle,
    shapes
)

# Datasets and benchmarks
from . import datasets
from . import benchmarks

# Backward compatibility
CondorcetCycle = condorcet_cycle

__all__ = [
    # Version
    '__version__',
    # Core utilities
    'TOLERANCE', 'enable_float64', 'device_type', 'use_accelerator',
    # Core components
    'MarkovChain', 'LazyMarkovChain', 'FlexMarkovChain', 'LazyTransitionMatrix',
    'lump', 'unlump', 'is_lumpable', 'partition_from_permutation_symmetry',
    'suggest_symmetries',
    'Grid', 'dist_sqeuclidean', 'dist_manhattan',
    # Models
    'VotingModel', 'SpatialVotingModel', 'BudgetVotingModel',
    # Examples
    'condorcet_cycle', 'CondorcetCycle',
    'core1', 'core2', 'core3', 'core4', 'nocore_triangle', 'ring_with_central_core',
    'bjm_spatial_triangle', 'bjm_budget_triangle',
    'shapes',
    # Modules
    'datasets', 'benchmarks',
]
