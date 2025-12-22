__version__ = "0.8.2"

# Core configuration and types
from .core import (
    TOLERANCE, 
    enable_float64,
    device_type,
    use_accelerator
)

# Spatial components (Grid, Distances)
from .spatial import (
    Grid,
    dist_sqeuclidean,
    dist_manhattan
)

# Dynamical systems / Voting
from .dynamics import MarkovChain
from .models import VotingModel, SpatialVotingModel
from .models.examples import condorcet_cycle

# Backward compatibility alias
CondorcetCycle = condorcet_cycle

# Datasets
from . import datasets

# Benchmarks
from . import benchmarks
