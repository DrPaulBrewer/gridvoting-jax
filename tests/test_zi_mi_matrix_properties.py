"""
Test ZI/MI transition matrix properties.

Validates fundamental properties of transition probability matrices (P) for
Zero Intelligence (ZI) and Minimal Intelligence (MI) modes.
"""

import pytest
import jax
import jax.numpy as jnp
import gridvoting_jax as gv
from gridvoting_jax.dynamics.lazy.base import LazyTransitionMatrix

pytestmark = pytest.mark.essential


def test_mi_diagonal_is_positive():
    """Validate that MI transition matrix has positive diagonal.
    
    MI includes status quo in the selection set (winners ∪ {status quo}),
    so prob(i→i) = 1/set_size > 0.
    """
    model_mi = gv.bjm_spatial_triangle(g=20, zi=False)
    P_mi = model_mi.model._get_transition_matrix()
    diagonal = jnp.diag(P_mi)
    
    assert jnp.all(diagonal > 0.0), "MI diagonal must be positive (status quo in selection set)"


def test_zi_diagonal_is_positive():
    """Validate that ZI transition matrix has strictly positive diagonal.
    
    ZI always has non-zero probability of proposing status quo against itself.
    """
    model_zi = gv.bjm_spatial_triangle(g=20, zi=True)
    P_zi = model_zi.model._get_transition_matrix()
    diagonal = jnp.diag(P_zi)
    
    assert jnp.all(diagonal > 0.0), "ZI diagonal must be positive (allows self-transitions)"


def test_zi_diagonal_greater_than_mi():
    """Validate that ZI diagonal elements are >= MI diagonal elements.
    
    ZI spreads probability uniformly over all alternatives, while MI concentrates
    on winners, resulting in higher self-transition probability for ZI.
    """
    model_mi = gv.bjm_spatial_triangle(g=20, zi=False)
    model_zi = gv.bjm_spatial_triangle(g=20, zi=True)
    P_mi = model_mi.model._get_transition_matrix()
    P_zi = model_zi.model._get_transition_matrix()
    
    diagonal_mi = jnp.diag(P_mi)
    diagonal_zi = jnp.diag(P_zi)
    
    assert jnp.all(diagonal_zi >= diagonal_mi), "ZI diagonal must be >= MI diagonal at all positions"
def test_zi_mi_offdiagonal_relationship():
    """Validate that non-diagonal elements satisfy MI >= ZI relationship.
    
    Key properties:
    - Boolean masks for non-zero locations are identical
    - At each non-zero location: P_mi[i,j] >= P_zi[i,j]
    - MI concentrates probability on winning alternatives
    """
    model_mi = gv.bjm_spatial_triangle(g=20, zi=False)
    model_zi = gv.bjm_spatial_triangle(g=20, zi=True)
    P_mi = model_mi.model._get_transition_matrix()
    P_zi = model_zi.model._get_transition_matrix()
    
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


def test_lazy_mi_diagonal_is_positive():
    """Validate lazy representation produces positive diagonal for MI.
    
    Tests both matvec and rmatvec operations by sampling diagonal positions.
    """
    model_mi = gv.bjm_spatial_triangle(g=20, zi=False)
    P_mi_dense = model_mi.model._get_transition_matrix()
    
    # Create lazy representation
    lazy_P = LazyTransitionMatrix(
        utility_functions=model_mi.model.utility_functions,
        majority=model_mi.majority,
        zi=False,
        number_of_feasible_alternatives=model_mi.model.number_of_feasible_alternatives
    )
    
    # Sample 200 diagonal positions uniformly
    n = lazy_P.shape[0]
    sample_indices = jnp.linspace(0, n-1, 200, dtype=int)
    
    for i in sample_indices:
        i = int(i)
        e_i = jnp.zeros(n)
        e_i = e_i.at[i].set(1.0)
        
        # matvec: P @ e_i gives column i
        col_i = lazy_P.matvec(e_i)
        assert col_i[i] > 0.0, f"Lazy MI matvec diagonal[{i}] must be positive"
        
        # rmatvec: e_i^T @ P gives row i  
        row_i = lazy_P.rmatvec(e_i)
        assert row_i[i] > 0.0, f"Lazy MI rmatvec diagonal[{i}] must be positive"


def test_lazy_zi_diagonal_is_positive():
    """Validate lazy representation produces positive diagonal for ZI.
    
    Tests both matvec and rmatvec operations by sampling diagonal positions.
    """
    model_zi = gv.bjm_spatial_triangle(g=20, zi=True)
    P_zi_dense = model_zi.model._get_transition_matrix()
    
    # Create lazy representation
    lazy_P = LazyTransitionMatrix(
        utility_functions=model_zi.model.utility_functions,
        majority=model_zi.majority,
        zi=True,
        number_of_feasible_alternatives=model_zi.model.number_of_feasible_alternatives
    )
    
    # Sample 200 diagonal positions uniformly
    n = lazy_P.shape[0]
    sample_indices = jnp.linspace(0, n-1, 200, dtype=int)
    
    for i in sample_indices:
        i = int(i)
        e_i = jnp.zeros(n)
        e_i = e_i.at[i].set(1.0)
        
        # matvec: P @ e_i gives column i
        col_i = lazy_P.matvec(e_i)
        assert col_i[i] > 0.0, f"Lazy ZI matvec diagonal[{i}] must be positive"
        
        # rmatvec: e_i^T @ P gives row i  
        row_i = lazy_P.rmatvec(e_i)
        assert row_i[i] > 0.0, f"Lazy ZI rmatvec diagonal[{i}] must be positive"


def test_lazy_matches_dense():
    """Validate lazy representation matches dense for both ZI and MI.
    
    Also includes direct comparison of lazy MI vs lazy ZI.
    """
    # Test MI
    model_mi = gv.bjm_spatial_triangle(g=20, zi=False)
    P_mi_dense = model_mi.model._get_transition_matrix()
    
    lazy_P_mi = LazyTransitionMatrix(
        utility_functions=model_mi.model.utility_functions,
        majority=model_mi.majority,
        zi=False,
        number_of_feasible_alternatives=model_mi.model.number_of_feasible_alternatives
    )
    
    # Test ZI
    model_zi = gv.bjm_spatial_triangle(g=20, zi=True)
    P_zi_dense = model_zi.model._get_transition_matrix()
    
    lazy_P_zi = LazyTransitionMatrix(
        utility_functions=model_zi.model.utility_functions,
        majority=model_zi.majority,
        zi=True,
        number_of_feasible_alternatives=model_zi.model.number_of_feasible_alternatives
    )
    
    # Test with random vector
    n = P_mi_dense.shape[0]
    rng = jax.random.PRNGKey(42)
    v = jax.random.normal(rng, (n,))
    
    # MI: lazy matvec matches dense
    result_dense_mi = P_mi_dense @ v
    result_lazy_matvec_mi = lazy_P_mi.matvec(v)
    assert jnp.allclose(result_dense_mi, result_lazy_matvec_mi, atol=1e-6, rtol=1e-4), \
        "Lazy MI matvec must match dense"
    
    # MI: lazy rmatvec matches dense
    result_dense_rmatvec_mi = P_mi_dense.T @ v
    result_lazy_rmatvec_mi = lazy_P_mi.rmatvec(v)
    assert jnp.allclose(result_dense_rmatvec_mi, result_lazy_rmatvec_mi, atol=1e-6, rtol=1e-4), \
        "Lazy MI rmatvec must match dense"
    
    # ZI: lazy matvec matches dense
    result_dense_zi = P_zi_dense @ v
    result_lazy_matvec_zi = lazy_P_zi.matvec(v)
    assert jnp.allclose(result_dense_zi, result_lazy_matvec_zi, atol=1e-6, rtol=1e-4), \
        "Lazy ZI matvec must match dense"
    
    # ZI: lazy rmatvec matches dense
    result_dense_rmatvec_zi = P_zi_dense.T @ v
    result_lazy_rmatvec_zi = lazy_P_zi.rmatvec(v)
    assert jnp.allclose(result_dense_rmatvec_zi, result_lazy_rmatvec_zi, atol=1e-6, rtol=1e-4), \
        "Lazy ZI rmatvec must match dense"
    
    # Direct comparison: lazy MI vs lazy ZI (off-diagonal relationship)
    # Remove diagonal contributions
    P_mi_offdiag = P_mi_dense - jnp.diag(jnp.diag(P_mi_dense))
    P_zi_offdiag = P_zi_dense - jnp.diag(jnp.diag(P_zi_dense))
    
    # Compute off-diagonal contributions to matvec
    result_mi_offdiag = P_mi_offdiag @ v
    result_zi_offdiag = P_zi_offdiag @ v
    
    # Lazy versions (subtract diagonal contribution)
    result_lazy_mi_offdiag = result_lazy_matvec_mi - jnp.diag(P_mi_dense) * v
    result_lazy_zi_offdiag = result_lazy_matvec_zi - jnp.diag(P_zi_dense) * v
    
    # Verify lazy off-diagonal matches dense off-diagonal
    assert jnp.allclose(result_lazy_mi_offdiag, result_mi_offdiag, atol=1e-6, rtol=1e-4), \
        "Lazy MI off-diagonal must match dense"
    assert jnp.allclose(result_lazy_zi_offdiag, result_zi_offdiag, atol=1e-6, rtol=1e-4), \
        "Lazy ZI off-diagonal must match dense"


def test_row_sums_stochastic():
    """Validate row sums are approximately 1.0 within floating-point error.
    
    Expected error scales with number of alternatives due to accumulation.
    """
    model_mi = gv.bjm_spatial_triangle(g=20, zi=False)
    P_mi = model_mi.model._get_transition_matrix()
    
    n = P_mi.shape[0]
    row_sums = jnp.sum(P_mi, axis=1)
    
    # Expected error from floating point arithmetic
    # Error ~ n * eps where we're summing n terms of ~1/n magnitude
    dtype = P_mi.dtype
    eps = jnp.finfo(dtype).eps
    expected_error = n * eps
    
    # All row sums should be 1.0 within tolerance
    assert jnp.allclose(row_sums, 1.0, atol=expected_error * 10), \
        f"MI row sums deviate from 1.0 beyond expected floating-point error"
    
    # Also test for ZI
    model_zi = gv.bjm_spatial_triangle(g=20, zi=True)
    P_zi = model_zi.model._get_transition_matrix()
    row_sums_zi = jnp.sum(P_zi, axis=1)
    assert jnp.allclose(row_sums_zi, 1.0, atol=expected_error * 10), \
        f"ZI row sums deviate from 1.0 beyond expected floating-point error"


def test_probability_bounds():
    """Validate all matrix elements are in [0, 1].
    
    Tests strict bounds without tolerance.
    """
    model_mi = gv.bjm_spatial_triangle(g=20, zi=False)
    model_zi = gv.bjm_spatial_triangle(g=20, zi=True)
    P_mi = model_mi.model._get_transition_matrix()
    P_zi = model_zi.model._get_transition_matrix()
    
    # Test MI
    assert jnp.all(P_mi >= 0.0), "All MI elements must be >= 0"
    assert jnp.all(P_mi <= 1.0), "All MI elements must be <= 1"
    
    # Test ZI  
    assert jnp.all(P_zi >= 0.0), "All ZI elements must be >= 0"
    assert jnp.all(P_zi <= 1.0), "All ZI elements must be <= 1"


if __name__ == "__main__":
    print("Running ZI/MI matrix property tests...")
    test_mi_diagonal_is_positive()
    print("✓ Test 1: MI diagonal is positive")
    test_zi_diagonal_is_positive()
    test_zi_diagonal_greater_than_mi()
    print("✓ Test 3: ZI diagonal >= MI diagonal")
    print("✓ Test 2: ZI diagonal is positive")
    test_zi_mi_offdiagonal_relationship()
    print("✓ Test 4: ZI/MI off-diagonal relationship (MI >= ZI)")
    test_lazy_mi_diagonal_is_positive()
    print("✓ Test 5: Lazy MI diagonal is positive")
    test_lazy_zi_diagonal_is_positive()
    print("✓ Test 6: Lazy ZI diagonal is positive")
    test_lazy_matches_dense()
    print("✓ Test 7: Lazy matches dense (both modes)")
    test_row_sums_stochastic()
    print("✓ Test 8: Row sums are stochastic")
    test_probability_bounds()
    print("✓ Test 9: Probability bounds [0, 1]")
    print("\n✅ All ZI/MI matrix property tests passed!")
