"""Test benchmarks submodule."""

import pytest
import gridvoting_jax as gv

pytestmark = pytest.mark.benchmarks


@pytest.mark.slow
def test_benchmarks_performance():
    """Test that benchmarks.performance() runs without error and returns a dict when dict=True."""
    # Run benchmark with dict=True to get results without printing
    result = gv.benchmarks.performance(dict=True)
    
    # Verify it returns a dictionary
    assert isinstance(result, dict), "performance(dict=True) should return a dictionary"
    
    # Verify the dictionary has expected top-level keys
    assert "device" in result, "Result should contain 'device' key"
    assert "jax_version" in result, "Result should contain 'jax_version' key"
    assert "results" in result, "Result should contain 'results' key"
    
    # Verify results is a list
    assert isinstance(result["results"], list), "results should be a list"
    
    # Verify we have benchmark results (should be 6 test cases)
    assert len(result["results"]) > 0, "Should have at least one benchmark result"
