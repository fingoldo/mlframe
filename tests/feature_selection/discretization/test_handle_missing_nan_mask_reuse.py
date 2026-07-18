"""Regression: ``_handle_missing`` reuses a caller-supplied NaN mask instead of rescanning ``arr``.

Pre-fix, ``categorize_dataset`` computed ``_nan_mask = np.isnan(arr)`` once, then called
``_handle_missing`` (default ``nan_strategy='separate_bin'``), which independently recomputed
``np.isnan(arr).any()``, then ``np.nanmedian(arr, axis=0)`` (an internal NaN rescan), then
``filled = np.where(np.isnan(arr), col_medians, arr)`` (a THIRD full-size ``np.isnan(arr)`` scan
plus a brand-new full-size float64 output array instead of an in-place fill).

This module pins: (a) the new mask-threaded + in-place path produces byte-identical output to the
pre-fix ``np.where``-based logic on a fixed-seed NaN-containing dataset, and (b) with a precomputed
mask supplied, ``_handle_missing`` does not call ``np.isnan`` on the full-size array a second time.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlframe.feature_selection.filters.discretization import _handle_missing


def _legacy_handle_missing(arr: np.ndarray, strategy: str) -> np.ndarray:
    """Pre-fix reference implementation (verbatim logic), used as the oracle for output-equivalence."""
    if not np.isnan(arr).any():
        return arr
    if strategy == "fillna_zero":
        return np.where(np.isnan(arr), 0.0, arr)
    if strategy in ("separate_bin", "propagate"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col_medians = np.nanmedian(arr, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        return np.where(np.isnan(arr), col_medians, arr)
    if strategy == "raise":
        raise ValueError("input contains NaN values")
    raise ValueError(f"unknown missing-value strategy: {strategy!r}")


def _make_nan_array(seed: int = 42, shape=(500, 6)) -> np.ndarray:
    """Make nan array."""
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=shape).astype(np.float64)
    nan_mask = rng.random(shape) < 0.15
    arr[nan_mask] = np.nan
    return arr


@pytest.mark.parametrize("strategy", ["separate_bin", "propagate", "fillna_zero"])
def test_output_matches_legacy_where_based_fill(strategy):
    """Output matches legacy where based fill."""
    arr = _make_nan_array()
    legacy = _legacy_handle_missing(arr.copy(), strategy)
    mask = np.isnan(arr)
    new = _handle_missing(arr.copy(), strategy=strategy, nan_mask=mask)
    np.testing.assert_array_equal(legacy, new)


@pytest.mark.parametrize("strategy", ["separate_bin", "propagate", "fillna_zero"])
def test_output_matches_legacy_when_mask_not_supplied(strategy):
    """``nan_mask=None`` (default) must fall back to computing the mask internally -- byte-identical
    to the legacy path for any caller that does not have a precomputed mask on hand."""
    arr = _make_nan_array(seed=7)
    legacy = _legacy_handle_missing(arr.copy(), strategy)
    new = _handle_missing(arr.copy(), strategy=strategy)
    np.testing.assert_array_equal(legacy, new)


def _count_full_array_isnan_calls(monkeypatch, fn, arr) -> int:
    """Run ``fn()`` (a zero-arg closure) with a counting ``np.isnan`` wrapper installed; returns
    the number of calls whose argument shape matches ``arr.shape`` (i.e. a full-array rescan, as
    opposed to a tiny per-column ``col_medians`` shape)."""
    full_shape_calls = []
    orig_isnan = np.isnan

    def counting_isnan(x, *a, **kw):
        """Counting isnan."""
        full_shape_calls.append(np.shape(x))
        return orig_isnan(x, *a, **kw)

    monkeypatch.setattr(np, "isnan", counting_isnan)
    try:
        fn()
    finally:
        monkeypatch.setattr(np, "isnan", orig_isnan)
    return sum(1 for s in full_shape_calls if s == arr.shape)


def test_no_full_array_isnan_rescan_when_mask_supplied(monkeypatch):
    """With a precomputed mask, ``_handle_missing`` must call ``np.isnan`` on the FULL-size array
    STRICTLY FEWER times than the legacy path -- eliminating the two explicit full-array scans this
    fix targets (the ``.any()`` early-exit check and the final ``np.where(np.isnan(arr), ...)``
    output-selection call). One full-array call remains and is EXPECTED: ``np.nanmedian``'s own
    internal NaN-detection is unavoidable (finding 2 explicitly documents it as the un-eliminable
    "scan #2" -- only the redundant explicit scans around it are the bug)."""
    arr = _make_nan_array(seed=99)
    mask = np.isnan(arr)

    legacy_calls = _count_full_array_isnan_calls(monkeypatch, lambda: _legacy_handle_missing(arr.copy(), "separate_bin"), arr)
    new_calls = _count_full_array_isnan_calls(monkeypatch, lambda: _handle_missing(arr.copy(), strategy="separate_bin", nan_mask=mask), arr)

    assert legacy_calls == 3, f"sanity: the legacy path should make 3 full-array isnan calls (early-exit + nanmedian + final np.where); got {legacy_calls}"
    assert (
        new_calls == 1
    ), f"expected exactly 1 full-array isnan call (nanmedian's own unavoidable internal scan) when a precomputed mask is supplied; got {new_calls}"
    assert new_calls < legacy_calls, f"mask reuse must reduce full-array isnan calls: new={new_calls} vs legacy={legacy_calls}"


def test_isnan_still_called_internally_when_no_mask_supplied(monkeypatch):
    """Sanity check for the counting harness itself: without a precomputed mask, the full-array
    isnan scan DOES happen (falls back to legacy behaviour)."""
    arr = _make_nan_array(seed=100)

    full_shape_calls = []
    orig_isnan = np.isnan

    def counting_isnan(x, *a, **kw):
        """Counting isnan."""
        full_shape_calls.append(np.shape(x))
        return orig_isnan(x, *a, **kw)

    monkeypatch.setattr(np, "isnan", counting_isnan)
    try:
        _handle_missing(arr.copy(), strategy="separate_bin")
    finally:
        monkeypatch.setattr(np, "isnan", orig_isnan)

    assert any(s == arr.shape for s in full_shape_calls), "expected at least one full-array isnan scan when no mask is supplied"


def test_in_place_fill_mutates_writable_input():
    """The optimisation mutates ``arr`` in place (returns the same object) when writable."""
    arr = _make_nan_array(seed=5)
    mask = np.isnan(arr)
    out = _handle_missing(arr, strategy="separate_bin", nan_mask=mask)
    assert out is arr, "expected the in-place fill to return the same array object"
    assert not np.isnan(out).any()


def test_read_only_input_falls_back_to_allocating_fill():
    """A read-only array (e.g. a genuine zero-copy view) must not raise -- falls back to an
    allocating np.where fill instead of mutating in place."""
    arr = _make_nan_array(seed=6)
    mask = np.isnan(arr)
    arr.flags.writeable = False
    out = _handle_missing(arr, strategy="separate_bin", nan_mask=mask)
    assert out is not arr
    assert not np.isnan(out).any()
