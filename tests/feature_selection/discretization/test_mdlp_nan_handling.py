"""Wave 9.1 loop-iter-48 regression: ``mdlp_bin_edges`` MUST drop NaN
inputs before processing and produce identical output across njit /
python backends.

Pre-fix at ``supervised_binning.py:68-78``:
- ``np.argsort(x)`` placed NaN at the tail (numpy convention).
- The njit recursion then sliced ``x[best_idx+1:]`` and could include
  the NaN tail, poisoning subsequent ``x[best_idx] + x[best_idx+1]``
  midpoint computations.
- Result: njit backend emitted literal NaN values into the ``splits``
  list, producing ``edges = [-inf, ..., NaN, ..., +inf]`` - violating
  the "edges are sorted" invariant downstream ``np.searchsorted``
  relies on.
- Python backend handled NaN differently, so the two backends DISAGREED
  silently on NaN-bearing inputs - even though the docstring promises
  identical semantics ("backend is a perf-only knob").

Severity: medium-high. Silent backend divergence on NaN-bearing input;
downstream searchsorted indices arbitrary when edges contain NaN.

Fix at supervised_binning.py:68: drop ``~np.isfinite(x)`` rows from
both x and y at the function entry point (single source of truth).
Empty-after-filter input returns the trivial ``[-inf, +inf]`` edges.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize("seed", [0, 1, 3, 7, 42])
def test_njit_and_python_backends_agree_on_nan_input(seed):
    """The iter-48 contract: backends must agree on identical input
    even when that input contains NaN.
    """
    from mlframe.feature_selection.filters.supervised_binning import mdlp_bin_edges

    rng = np.random.default_rng(int(seed))
    n = 200
    x = rng.standard_normal(n).astype(np.float64)
    y = (x > 0).astype(np.int64)
    nan_idx = rng.choice(n, 30, replace=False)
    x[nan_idx] = np.nan
    e_njit = mdlp_bin_edges(x, y, backend="njit")
    e_py = mdlp_bin_edges(x, y, backend="python")
    np.testing.assert_array_equal(e_njit, e_py)


@pytest.mark.parametrize("backend", ["njit", "python"])
def test_no_nan_in_returned_edges(backend):
    """Returned edges must NOT contain NaN regardless of input
    NaN-density.
    """
    from mlframe.feature_selection.filters.supervised_binning import mdlp_bin_edges

    rng = np.random.default_rng(0)
    n = 200
    x = rng.standard_normal(n).astype(np.float64)
    y = (x > 0).astype(np.int64)
    x[::5] = np.nan  # 20% NaN
    edges = mdlp_bin_edges(x, y, backend=backend)
    assert not np.any(np.isnan(edges)), f"backend={backend}: edges contain NaN: {edges}"


@pytest.mark.parametrize("backend", ["njit", "python"])
def test_edges_remain_sorted(backend):
    """``edges`` must be monotonically non-decreasing - the
    ``np.searchsorted`` invariant downstream relies on this.
    """
    from mlframe.feature_selection.filters.supervised_binning import mdlp_bin_edges

    rng = np.random.default_rng(1)
    n = 300
    x = rng.standard_normal(n).astype(np.float64)
    y = (x > 0).astype(np.int64)
    x[rng.choice(n, 50, replace=False)] = np.nan
    edges = mdlp_bin_edges(x, y, backend=backend)
    assert (np.diff(edges) >= 0).all(), f"edges not sorted: {edges}"


def test_all_nan_input_returns_trivial_edges():
    """Edge case: all-NaN input must return ``[-inf, +inf]`` (no
    splits) instead of crashing or returning NaN-bearing edges.
    """
    from mlframe.feature_selection.filters.supervised_binning import mdlp_bin_edges

    x = np.full(50, np.nan)
    y = np.zeros(50, dtype=np.int64)
    edges = mdlp_bin_edges(x, y)
    np.testing.assert_array_equal(edges, np.array([-np.inf, np.inf]))


def _y_only_nan_fixture(seed=0, n=2000, nan_stride=13):
    """A strong 2-class x->y signal (continuous-dtype y, few distinct finite values so the
    quantile-rebucketing branch never fires) with NaN scattered through Y ONLY -- x stays clean."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = (x > 0.3).astype(np.float64)
    nan_mask = np.zeros(n, dtype=bool)
    nan_mask[::nan_stride] = True
    y_with_nan = y.copy()
    y_with_nan[nan_mask] = np.nan
    return x, y_with_nan, nan_mask


class TestB13NanInY:
    """mrmr_audit_2026-07-20 B-13: mdlp_bin_edges (+ its two _mdlp_validated_split.py siblings) dropped
    NaN rows from x explicitly, but never filtered NaN out of y before casting it to int64 -- a NaN-y
    row silently became a platform-defined garbage class label (typically INT64_MIN) instead of being
    dropped, polluting every entropy computation the split search relies on."""

    def test_mdlp_bin_edges_nan_y_matches_caller_predropped(self):
        """Passing NaN-bearing y directly must give the IDENTICAL result to a caller who pre-drops the
        NaN rows themselves -- proving the internal filter, not a garbage class, is what's happening."""
        from mlframe.feature_selection.filters.supervised_binning import mdlp_bin_edges

        x, y_with_nan, nan_mask = _y_only_nan_fixture()
        edges_internal = mdlp_bin_edges(x, y_with_nan)
        edges_predropped = mdlp_bin_edges(x[~nan_mask], y_with_nan[~nan_mask])
        np.testing.assert_array_equal(edges_internal, edges_predropped)

    @pytest.mark.parametrize(
        "fn_name",
        ["mdlp_bin_edges_validated", "mdlp_bin_edges_oos_validated"],
    )
    def test_validated_siblings_nan_y_matches_caller_predropped(self, fn_name):
        """Same contract for both _mdlp_validated_split.py siblings named in the audit finding."""
        from mlframe.feature_selection.filters import _mdlp_validated_split as mod

        fn = getattr(mod, fn_name)
        x, y_with_nan, nan_mask = _y_only_nan_fixture(seed=1)
        edges_internal = fn(x, y_with_nan)
        edges_predropped = fn(x[~nan_mask], y_with_nan[~nan_mask])
        np.testing.assert_array_equal(edges_internal, edges_predropped)

    def test_mdlp_bin_edges_nan_y_edges_finite(self):
        """Returned edges must never contain NaN even when y (not x) carries the NaN."""
        from mlframe.feature_selection.filters.supervised_binning import mdlp_bin_edges

        x, y_with_nan, _ = _y_only_nan_fixture(seed=2)
        edges = mdlp_bin_edges(x, y_with_nan)
        assert not np.any(np.isnan(edges)), f"edges contain NaN when only y carried NaN: {edges}"
