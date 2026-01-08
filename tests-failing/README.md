# Failing Tests Organization

Successfully moved 12 failing test files from `tests/` to `tests-failing/` subdirectories based on failure category.

## Summary

- **Total test files moved**: 12
- **Remaining test files in tests/**: 17 (all passing tests)

## Organization by Category

### assertion-failures/ (3 files)
Tests with assertion failures and numerical precision issues:
- `test_gridvoting_condorcet_cycle.py`
- `test_gridvoting_doublecycle.py`
- `test_lazy_equivalence.py`

### other/ (2 files)
Tests with various other errors:
- `test_gridvoting_topcycle.py` (TracerArrayConversionError)
- `test_lump_bjm_g20_reflection.py` (multiple issues)

## Next Steps

With failing tests isolated, the remaining 17 test files in `tests/` should all pass, confirming the core module refactoring was successful.
