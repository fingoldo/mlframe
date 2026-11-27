"""Cluster 24 P3 regressions: the ACE selector and the functional adapters are exported from the package, and the per-window Hurst docstring states
what it returns (audit 2026-11-20, findings 24-F64 and 24-F51).

ACESelector and the four functional adapters were defined but missing from the package's public names, unlike the other selectors. The
_hurst_rs_single docstring claimed the same definition as compute_hurst_rs, but it returns the log-scaled estimate, not the raw R/S ratio.
"""

import mlframe.feature_selection as fs
from mlframe.feature_engineering import hurst

EXPORTED = ["ACESelector", "ForwardSelectSelector", "GreedyBackwardEliminationSelector", "ZeroImportancePruningSelector", "CascadeSelectSelector"]


def test_ace_and_functional_selectors_are_public_names():
    """Each selector is importable from the package and listed in __all__ (24-F64)."""
    assert EXPORTED, "the list of selectors under test must not be empty"
    for name in EXPORTED:
        assert name in fs.__all__, name
        assert hasattr(fs, name), name


def test_hurst_window_docstring_states_the_log_scaled_estimate():
    """The per-window Hurst docstring says it returns log(R/S)/log(n), not the raw R/S ratio (24-F51)."""
    doc = hurst._hurst_rs_single.__doc__ or ""
    assert "log(R/S)" in doc and "raw R/S" in doc
