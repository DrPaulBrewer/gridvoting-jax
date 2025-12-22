import gridvoting_jax as gv
import jax.numpy as jnp
import numpy as np

class BudgetVotingModel():
  def __init__(budget=100, zi):
    # create a square grid of edge size (budget+1)
    self.grid = Grid(x0=0,x1=budget,y0=0,y1=budget)
    self.number_of_voters = 3
    # this is about splitting $budget among voter 1, voter 2, and voter 3
    # many of the points on the grid exceed the budget
    # the area of validity is 0<=x<=budget, 0<=y<=budget, 0<=(budget-x-y)<=100
    self.valid = (grid.x+grid.y) <= budget
    # the area of validity is a triangle
    # create an embedding for the triangle into the square grid 
    # this embedding is for plotting purposes, so we use fill=np.nan
    self.embed_triangle_in_square = grid.embedding(valid=valid, fill=np.nan)
    # coordinate mappings for triangle-sized lists
    self.triangleX = grid.x[valid]
    self.triangleY = grid.y[valid]
    # there should be (budget+1)*(budget+2) alternatives
    expected_alternatives = (budget+1)*(budget+2)
    self.number_of_alternatives = int(valid.sum())
    assert(number_of_alternatives==expected_alternatives)
    # define utility function values for each agent
    self.u1 = grid.x[valid]  # x coordinate is voter 1's portion of the budget
    self.u2 = grid.y[valid]  # y coordinate is voter 2's portion of the budget
    self.u3 = budget-u1-u2   # voter 3 receives what is left over
    self.U = jnp.array([self.u1,self.u2,self.u3])
    # GiniSS is a Gini-like index scaled to be between 0 (equality) and 1 (one voter gets all)
    self.GiniSS = 1.5*((np.abs(u1-u2)+np.abs(u1-u3)+np.abs(u2-u3))/(3*budget))
    # voter ideal points here are only for plotting purposes
    self.voter_ideal_points = np.array([
      [budget,0],
      [0, budget],
      [0,0]
    ])
    self.voting_model =  VotingModel(
      utility_functions = self.U,
      number_of_voters = 3,
      majority=2,
      zi=zi,
      number_of_alternatives=self.number_of_alternatives
    )

# to be completed


