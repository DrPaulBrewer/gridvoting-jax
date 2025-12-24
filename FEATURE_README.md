# Feature: Pareto Efficiency

This feature adds support for identifying the Pareto Optimal set of alternatives in a voting model.

## Usage

### Pareto Set
The `.Pareto` property returns a boolean mask of the grid points that are Pareto optimal. A point is Pareto optimal if no other point is strictly preferred by *all* voters (unanimity).

```python
import gridvoting_jax as gv

# Create a spatial model (e.g., triangle)
model = gv.models.examples.shapes.triangle.triangle(g=20)

# Get the Pareto set
pareto_mask = model.Pareto

# Visualize
model.grid.plot(pareto_mask)
```

### Unanimize
The `unanimize()` method returns a new model instance where the decision rule is set to unanimity.

```python
unanimous_model = model.unanimize()
unanimous_model.analyze()
print(unanimous_model.core_points) # Identical to model.Pareto
```
