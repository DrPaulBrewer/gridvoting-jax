__version__ = "0.4.1"

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
from .dynamics import (
    MarkovChain,
    VotingModel,
    CondorcetCycle
)

# Datasets
from . import datasets

# Benchmarks
from . import benchmarks
