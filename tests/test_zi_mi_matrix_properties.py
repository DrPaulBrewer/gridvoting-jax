"""
Test ZI/MI transition matrix properties.

Validates fundamental properties of transition probability matrices (P) for
Zero Intelligence (ZI) and Minimal Intelligence (MI) modes.
"""

import pytest
import jax
import jax.numpy as jnp
import gridvoting_jax as gv
from gridvoting_jax.core import EPSILON

pytestmark = pytest.mark.essential


def _finalize_transition_matrix(vm: gv.VotingModel, cV):
    """Shared logic to convert winner matrix cV to transition matrix cP"""
    nfa = vm.number_of_feasible_alternatives
    zi = vm.zi
    cV_sum_of_row = cV.sum(axis=1)  # number of winning alternatives for each SQ
        
    # set up the ZI and MI transition matrices
    if zi:
        # ZI: Uniform random over ALL alternatives.
        # If ch beats sq: move to ch (prob 1/N)
        # If ch loses to sq: stay at sq
        # Plus picked sq itself: stay at sq
        # So prob(move i->j) = 1/N if j beats i
        # prob(stay i) = (1/N) * (count(j that lose to i) + 1)
        #              = (1/N) * ((N - count(win) - 1) + 1)
        #              = (N - row_sum)/N
        # logic in code: cV + diag(N - row_sum) / N
        cP = jnp.divide(
            jnp.add(cV, jnp.diag(jnp.subtract(nfa, cV_sum_of_row))), 
            nfa
            )
    else:
        # MI: Uniform random over Winning Set(i) U {i}
        # Size of set = row_sum + 1
        # Prob(move i->j) = 1/(row_sum+1) if j beats i
        # Prob(stay i) = 1/(row_sum+1)
        # logic in code: (cV + I) / (1 + row_sum)
        cP = jnp.divide(
            jnp.add(cV, jnp.eye(nfa)), 
            (1 + cV_sum_of_row)[:, jnp.newaxis]
            )
    return cP

def _get_transition_matrix_vectorized(vm: gv.VotingModel):
    """adapted from v0.9.1:  Original fully vectorized implementation. O(V * N^2) memory."""
    utility_functions = vm.utility_functions
    majority = vm.majority
    cU = jnp.asarray(utility_functions) 
    
    # Vectorized computation: compare all alternatives at once
    # cU shape: (n_voters, nfa)
    # cU[:, :, jnp.newaxis] shape: (n_voters, nfa, 1) to broadcast vs challengers (rows)
    # cU[:, jnp.newaxis, :] shape: (n_voters, 1, nfa) to broadcast vs status quo (cols) 
    # Note: Previous implementation comment had axes swapped in explanation but logic was correct for outcome.
    # Let's align with the standard logic:
    # P[i, j] is prob of moving i -> j.
    # i is Status Quo (SQ), j is Challenger (CH).
    # We need votes for CH against SQ.
    # Utility for SQ: cU[:, i] (column i)
    # Utility for CH: cU[:, j] (column j)
    # pref = u(CH) > u(SQ)
    
    # In the original code:
    # preferences = jnp.greater(cU[:, jnp.newaxis, :], cU[:, :, jnp.newaxis])
    # LHS: cU[:, 1, N] -> varying last dim is COLUMNS (CH)
    # RHS: cU[:, N, 1] -> varying middle dim is ROWS (SQ)
    # Result: (V, SQ, CH).  [v, i, j] is "does v prefer j over i?"
    # Correct.
    
    preferences = jnp.greater(cU[:, jnp.newaxis, :], cU[:, :, jnp.newaxis])
    
    # Sum votes across voters: shape (nfa, nfa) -> (SQ, CH)
    total_votes = preferences.astype("int32").sum(axis=0)
    
    # Determine winners: 1 if challenger gets majority, 0 otherwise
    # cV[i, j] = 1 if j beats i
    cV = jnp.greater_equal(total_votes, majority)
    
    return _finalize_transition_matrix(vm,cV)


@pytest.fixture(scope="module")
def bmj_model_mi():
    """BMJ spatial triangle model with MI (zi=False), g=20."""
    return gv.bjm_spatial_triangle(g=20, zi=False)


@pytest.fixture(scope="module")
def bmj_model_zi():
    """BMJ spatial triangle model with ZI (zi=True), g=20."""
    return gv.bjm_spatial_triangle(g=20, zi=True)


@pytest.fixture(scope="module")
def bmj_P_mi_dense(bmj_model_mi):
    """Dense MI transition matrix for BMJ triangle model.
    
    Returns tuple: (vectorized, lazy.to_dense())
    """
    vectorized = _get_transition_matrix_vectorized(bmj_model_mi.model)
    lazy_dense = bmj_model_mi.model.transition_matrix().to_dense()
    return (vectorized, lazy_dense)


@pytest.fixture(scope="module")
def bmj_P_zi_dense(bmj_model_zi):
    """Dense ZI transition matrix for BMJ triangle model.
    
    Returns tuple: (vectorized, lazy.to_dense())
    """
    vectorized = _get_transition_matrix_vectorized(bmj_model_zi.model)
    lazy_dense = bmj_model_zi.model.transition_matrix().to_dense()
    return (vectorized, lazy_dense)


@pytest.fixture(scope="module")
def bmj_P_mi_diagonal(bmj_model_mi):
    """Diagonal of MI transition matrix for BMJ triangle model."""
    return bmj_model_mi.model.transition_matrix().diagonal()


@pytest.fixture(scope="module")
def bmj_P_zi_diagonal(bmj_model_zi):
    """Diagonal of ZI transition matrix for BMJ triangle model."""
    return bmj_model_zi.model.transition_matrix().diagonal()


def test_mi_diagonal_is_positive(bmj_P_mi_diagonal):
    """Validate that MI transition matrix has positive diagonal.
    
    MI includes status quo in the selection set (winners ∪ {status quo}),
    so prob(i→i) = 1/set_size > 0.
    """
    diagonal = bmj_P_mi_diagonal
    
    assert jnp.all(diagonal > 0.0), "MI diagonal must be positive (status quo in selection set)"


def test_zi_diagonal_is_positive(bmj_P_zi_diagonal):
    """Validate that ZI transition matrix has strictly positive diagonal.
    
    ZI always has non-zero probability of proposing status quo against itself.
    """
    diagonal = bmj_P_zi_diagonal
    
    assert jnp.all(diagonal > 0.0), "ZI diagonal must be positive (allows self-transitions)"


def test_zi_diagonal_greater_than_mi(bmj_P_mi_diagonal, bmj_P_zi_diagonal):
    """Validate that ZI diagonal elements are >= MI diagonal elements.
    
    ZI spreads probability uniformly over all alternatives, while MI concentrates
    on winners, resulting in higher self-transition probability for ZI.
    """
    diagonal_mi = bmj_P_mi_diagonal
    diagonal_zi = bmj_P_zi_diagonal
    
    assert jnp.all(diagonal_zi >= diagonal_mi), "ZI diagonal must be >= MI diagonal at all positions"

def test_zi_mi_offdiagonal_relationship(bmj_P_mi_dense, bmj_P_zi_dense):
    """Validate that non-diagonal elements satisfy MI >= ZI relationship.
    
    Key properties:
    - Boolean masks for non-zero locations are identical
    - At each non-zero location: P_mi[i,j] >= P_zi[i,j]
    - MI concentrates probability on winning alternatives
    """
    # Use lazy.to_dense() version for this test
    P_mi = bmj_P_mi_dense[1]
    P_zi = bmj_P_zi_dense[1]
    
    # Create off-diagonal matrices
    P_mi_offdiag = P_mi - jnp.diag(jnp.diag(P_mi))
    P_zi_offdiag = P_zi - jnp.diag(jnp.diag(P_zi))
    
    # Find non-zero locations in MI
    nonzero_indices = jnp.where(P_mi_offdiag > 0)
    
    # Test 1: Boolean masks are identical
    mi_mask = P_mi_offdiag > 0
    zi_mask = P_zi_offdiag > 0
    assert jnp.all(mi_mask == zi_mask), "MI and ZI must have identical non-zero patterns"
    
    # Test 2: At non-zero locations, MI >= ZI
    mi_values = P_mi_offdiag[nonzero_indices]
    zi_values = P_zi_offdiag[nonzero_indices]
    assert jnp.all(mi_values >= zi_values), "MI values must be >= ZI values at all non-zero locations"
    
    # Test 3: Verify the relationship mask matches the non-zero mask
    relationship_mask = P_mi_offdiag >= P_zi_offdiag
    assert jnp.all(relationship_mask[mi_mask]), "MI >= ZI must hold at all non-zero locations"


def _lazy_check_diagonal(label, lazy_P):
    # Sample 200 diagonal positions uniformly
    n = lazy_P.shape[0]
    diag = lazy_P.diagonal()
    sample_indices = jnp.linspace(0, n-1, 200, dtype=int)
    
    for i in sample_indices:
        i = int(i)
        e_i = jnp.zeros(n)
        e_i = e_i.at[i].set(1.0)
                
        # rmatvec: e_i^T @ P gives row i  
        row_i = e_i @ lazy_P

        assert row_i[i] > 0.0, f"Lazy {label} rmatvec diagonal[{i}] must be positive"

        # matvec and rmatvec values for [i,i] should match
        diff_in_eps = round(abs(diag[i] -row_i[i])/EPSILON)
        assert diff_in_eps<= 2, f"Lazy {label} matvec and rmatvec values for [{i},{i}] should match (diff_in_eps={diff_in_eps})"



def test_lazy_mi_diagonal_is_positive(bmj_model_mi):
    """Validate lazy representation produces positive diagonal for MI.
    
    Tests both matvec and rmatvec operations by sampling diagonal positions.
    """
    _lazy_check_diagonal("MI", bmj_model_mi.model.transition_matrix())
    
    

def test_lazy_zi_diagonal_is_positive(bmj_model_zi):
    """Validate lazy representation produces positive diagonal for ZI.
    
    Tests both matvec and rmatvec operations by sampling diagonal positions.
    """
    _lazy_check_diagonal("ZI", bmj_model_zi.model.transition_matrix())


def test_lazy_matches_dense(bmj_model_mi, bmj_model_zi, bmj_P_mi_dense, bmj_P_zi_dense):
    """Validate lazy representation matches dense for both ZI and MI.
    
    Also includes direct comparison of lazy MI vs lazy ZI.
    Uses both vectorized and lazy.to_dense() representations.
    """
    # Unpack tuples: (vectorized, lazy.to_dense())
    P_mi_vectorized, P_mi_lazy_dense = bmj_P_mi_dense
    P_zi_vectorized, P_zi_lazy_dense = bmj_P_zi_dense
    
    lazy_P_mi = bmj_model_mi.model.transition_matrix()
    lazy_P_zi = bmj_model_zi.model.transition_matrix()
    
    # Test with random vector
    n = P_mi_vectorized.shape[0]
    rng = jax.random.PRNGKey(42)
    v = jax.random.normal(rng, (n,))
    
    # MI: lazy evolution matches vectorized dense
    result_dense_evolution_mi = v@P_mi_vectorized
    result_lazy_evolution_mi = v@lazy_P_mi
    assert jnp.allclose(result_dense_evolution_mi, result_lazy_evolution_mi, atol=1e-6, rtol=1e-4), \
        "Lazy MI evolution must match vectorized dense"
    
    # ZI: lazy evolution matches vectorized dense
    result_dense_evolution_zi = v@P_zi_vectorized
    result_lazy_evolution_zi = v@lazy_P_zi
    assert jnp.allclose(result_dense_evolution_zi, result_lazy_evolution_zi, atol=1e-6, rtol=1e-4), \
        "Lazy ZI evolution must match vectorized dense"
    
    # Test vectorized vs lazy.to_dense() for MI
    assert jnp.allclose(P_mi_vectorized, P_mi_lazy_dense, atol=1e-6, rtol=1e-4), \
        "Vectorized MI must match lazy.to_dense() MI"

    # Test vectorized vs lazy.to_dense() for ZI
    assert jnp.allclose(P_zi_vectorized, P_zi_lazy_dense, atol=1e-6, rtol=1e-4), \
        "Vectorized ZI must match lazy.to_dense() ZI"

def test_row_sums_stochastic(bmj_P_mi_dense, bmj_P_zi_dense):
    """Validate row sums are approximately 1.0 within floating-point error.
    
    Expected error scales with number of alternatives due to accumulation.
    Uses lazy.to_dense() from fixtures.
    """
    # Use lazy.to_dense() from fixtures (index 1 of tuple)
    P_mi_dense = bmj_P_mi_dense[1]
    P_zi_dense = bmj_P_zi_dense[1]
    
    n = P_mi_dense.shape[0]
    ones = jnp.ones(n)
    row_sums = P_mi_dense @ ones
    
    # Expected error from floating point arithmetic
    # Error ~ n * eps where we're summing n terms of ~1/n magnitude
    expected_error = n * EPSILON
    
    # All row sums should be 1.0 within tolerance
    assert jnp.allclose(row_sums, 1.0, atol=expected_error * 10), \
        f"MI row sums deviate from 1.0 beyond expected floating-point error"
    
    # Also test for ZI
    row_sums_zi = P_zi_dense @ ones
    assert jnp.allclose(row_sums_zi, 1.0, atol=expected_error * 10), \
        f"ZI row sums deviate from 1.0 beyond expected floating-point error"


def test_probability_bounds(bmj_P_mi_dense, bmj_P_zi_dense):
    """Validate all matrix elements are in [0, 1].
    
    Tests strict bounds without tolerance.
    Tests both vectorized and lazy.to_dense() representations.
    """
    # Test both representations
    for idx, label in enumerate(["vectorized", "lazy.to_dense()"]):
        P_mi = bmj_P_mi_dense[idx]
        P_zi = bmj_P_zi_dense[idx]
    
        # Test MI
        assert jnp.all(P_mi >= 0.0), f"All MI {label} elements must be >= 0"
        assert jnp.all(P_mi <= 1.0), f"All MI {label} elements must be <= 1"
        
        # Test ZI  
        assert jnp.all(P_zi >= 0.0), f"All ZI {label} elements must be >= 0"
        assert jnp.all(P_zi <= 1.0), f"All ZI {label} elements must be <= 1"


if __name__ == "__main__":
    print("Running ZI/MI matrix property tests...")
    # Create fixtures manually for standalone execution
    model_mi = gv.bjm_spatial_triangle(g=20, zi=False)
    model_zi = gv.bjm_spatial_triangle(g=20, zi=True)
    # Create tuples: (vectorized, lazy.to_dense())
    P_mi_dense = (_get_transition_matrix_vectorized(model_mi.model), 
                  model_mi.model.transition_matrix().to_dense())
    P_zi_dense = (_get_transition_matrix_vectorized(model_zi.model),
                  model_zi.model.transition_matrix().to_dense())
    P_mi_diagonal = model_mi.model.transition_matrix().diagonal()
    P_zi_diagonal = model_zi.model.transition_matrix().diagonal()
    
    test_mi_diagonal_is_positive(P_mi_diagonal)
    print("✓ Test 1: MI diagonal is positive")
    test_zi_diagonal_is_positive(P_zi_diagonal)
    print("✓ Test 2: ZI diagonal is positive")
    test_zi_diagonal_greater_than_mi(P_mi_diagonal, P_zi_diagonal)
    print("✓ Test 3: ZI diagonal >= MI diagonal")
    test_zi_mi_offdiagonal_relationship(P_mi_dense, P_zi_dense)
    print("✓ Test 4: ZI/MI off-diagonal relationship (MI >= ZI)")
    test_lazy_mi_diagonal_is_positive(model_mi)
    print("✓ Test 5: Lazy MI diagonal is positive")
    test_lazy_zi_diagonal_is_positive(model_zi)
    print("✓ Test 6: Lazy ZI diagonal is positive")
    test_lazy_matches_dense(model_mi, model_zi, P_mi_dense, P_zi_dense)
    print("✓ Test 7: Lazy matches dense (both modes)")
    test_row_sums_stochastic(P_mi_dense, P_zi_dense)
    print("✓ Test 8: Row sums are stochastic")
    test_probability_bounds(P_mi_dense, P_zi_dense)
    print("✓ Test 9: Probability bounds [0, 1]")
    print("\n✅ All ZI/MI matrix property tests passed!")
