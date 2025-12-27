"""Core voting logic modules.

This package contains shared voting-specific logic used across the codebase:
- zimi_succession_logic: ZI/MI succession rules
- winner_determination: Pairwise winner computation
"""

from .zimi_succession_logic import (
    finalize_transition_matrix,
    finalize_transition_matrix_zi_jit,
    finalize_transition_matrix_mi_jit,
)

from .winner_determination import (
    compute_winner_matrix_jit,
)

__all__ = [
    'finalize_transition_matrix',
    'finalize_transition_matrix_zi_jit',
    'finalize_transition_matrix_mi_jit',
    'compute_winner_matrix_jit',
]
