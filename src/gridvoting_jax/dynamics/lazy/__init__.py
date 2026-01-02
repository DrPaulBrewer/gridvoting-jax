"""Lazy matrix construction for memory-efficient large-scale models."""
    
from .utils import should_use_lazy, estimate_memory_for_dense_matrix
from .lazy_markov import FlexMarkovChain

__all__ = [
    'should_use_lazy', 
    'estimate_memory_for_dense_matrix',
    'FlexMarkovChain'
]

