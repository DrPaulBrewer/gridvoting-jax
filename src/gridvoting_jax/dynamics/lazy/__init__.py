"""Lazy matrix construction for memory-efficient large-scale models."""

from .base import LazyTransitionMatrix
from .lazy_markov import LazyMarkovChain, FlexMarkovChain
from .utils import should_use_lazy, estimate_memory_for_dense_matrix

__all__ = [
    'LazyTransitionMatrix', 
    'LazyMarkovChain', 
    'FlexMarkovChain',
    'should_use_lazy', 
    'estimate_memory_for_dense_matrix'
]

