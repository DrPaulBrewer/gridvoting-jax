"""Markov Chain dynamics module."""

from .markov import MarkovChain, lump, unlump, is_lumpable, partition_from_permutation_symmetry
from .lazy import LazyMarkovChain, FlexMarkovChain, LazyTransitionMatrix

__all__ = ['MarkovChain', 'LazyMarkovChain', 'FlexMarkovChain', 'LazyTransitionMatrix', 'lump', 'unlump', 'is_lumpable', 'partition_from_permutation_symmetry']
