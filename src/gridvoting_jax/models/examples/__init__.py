"""Example voting models."""

from .condorcet import condorcet_cycle
from .plott_theorem import core1, core2, core3, core4, nocore_triangle, ring_with_central_core
from .bjm_spatial import bjm_spatial_triangle, BJM_TRIANGLE_VOTER_IDEAL_POINTS
from .bjm_budget import bjm_budget_triangle
from . import shapes

__all__ = [
    'condorcet_cycle',
    'core1', 'core2', 'core3', 'core4', 'nocore_triangle', 'ring_with_central_core',
    'bjm_spatial_triangle', 'BJM_TRIANGLE_VOTER_IDEAL_POINTS', 'bjm_budget_triangle',
    'shapes'
]
