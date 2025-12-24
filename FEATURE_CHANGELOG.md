# Feature Changelog: Pareto Efficiency

## [0.2.0-pareto] - 2025-12-24

### Added
- `VotingModel.unanimize()`: Method to create a copy of the voting model with majority rule set to unanimity (N/N).
- `VotingModel.Pareto`: Property that returns the Pareto Optimal set (the core under unanimity).
- `SpatialVotingModel.Pareto`: Exposed property delegating to the underlying model.
- `BudgetVotingModel.Pareto`: Exposed property delegating to the underlying model.
