"""Pytest configuration for gridvoting-jax tests."""

import pytest
import gridvoting_jax as gv
from tests.test_utils import get_transition_matrix_vectorized




# ============================================================================
# Session-Level Model Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def bmj_g10_mi():
    """BMJ spatial triangle, g=10, MI mode (zi=False)."""
    return gv.bjm_spatial_triangle(g=10, zi=False)


@pytest.fixture(scope="session")
def bmj_g20_mi():
    """BMJ spatial triangle, g=20, MI mode (zi=False)."""
    return gv.bjm_spatial_triangle(g=20, zi=False)


@pytest.fixture(scope="session")
def bmj_g20_zi():
    """BMJ spatial triangle, g=20, ZI mode (zi=True)."""
    return gv.bjm_spatial_triangle(g=20, zi=True)


@pytest.fixture(scope="session")
def bmj_g40_mi():
    """BMJ spatial triangle, g=40, MI mode (zi=False)."""
    return gv.bjm_spatial_triangle(g=40, zi=False)


@pytest.fixture(scope="session")
def condorcet_mi():
    """Condorcet cycle, MI mode (zi=False)."""
    return gv.condorcet_cycle(zi=False)


@pytest.fixture(scope="session")
def condorcet_zi():
    """Condorcet cycle, ZI mode (zi=True)."""
    return gv.condorcet_cycle(zi=True)


# ============================================================================
# Session-Level Derived Data Fixtures (for g=20 models)
# ============================================================================

@pytest.fixture(scope="session")
def bmj_g20_mi_P_dense(bmj_g20_mi):
    """Dense MI transition matrix for g=20 BMJ model.
    
    Returns tuple: (vectorized, lazy.to_dense())
    """
    vectorized = get_transition_matrix_vectorized(bmj_g20_mi.model)
    lazy_dense = bmj_g20_mi.model.transition_matrix().to_dense()
    return (vectorized, lazy_dense)


@pytest.fixture(scope="session")
def bmj_g20_zi_P_dense(bmj_g20_zi):
    """Dense ZI transition matrix for g=20 BMJ model.
    
    Returns tuple: (vectorized, lazy.to_dense())
    """
    vectorized = get_transition_matrix_vectorized(bmj_g20_zi.model)
    lazy_dense = bmj_g20_zi.model.transition_matrix().to_dense()
    return (vectorized, lazy_dense)


@pytest.fixture(scope="session")
def bmj_g20_mi_P_diagonal(bmj_g20_mi):
    """Diagonal of MI transition matrix for g=20 BMJ model."""
    return bmj_g20_mi.model.transition_matrix().diagonal()


@pytest.fixture(scope="session")
def bmj_g20_zi_P_diagonal(bmj_g20_zi):
    """Diagonal of ZI transition matrix for g=20 BMJ model."""
    return bmj_g20_zi.model.transition_matrix().diagonal()
