"""Voting models module."""

from .base import VotingModel
from .spatial import SpatialVotingModel
from .budget import BudgetVotingModel

__all__ = ['VotingModel', 'SpatialVotingModel', 'BudgetVotingModel']
