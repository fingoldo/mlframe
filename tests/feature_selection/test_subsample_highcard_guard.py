"""Regression tests for the high-cardinality classification subsample guard + the logged (non-silent)
resolver fallback.

Root cause (wellbore-shape investigation, 2026-07): the classification stratified subsample guarantees a
``min(2, count)`` floor per class so a rare class is never wholly dropped. But when the label is
(near-)continuous / very high cardinality, those floors SUM TO ~n, so a "30k" draw returned ~n rows and
SILENTLY DEFEATED the FE screen subsample -- every downstream MI/CMI consumer then ran at FULL n (~33x cost
at n~1M), invisible in a profile. The guard falls back to a plain uniform draw of exactly ``size`` when the
floored minimum cannot fit the budget. The bare ``except Exception: return None`` fallbacks that make the
whole screen silently run at full-n now log at WARNING so such a regression is diagnosable.
"""

import logging

import numpy as np

from mlframe.feature_selection.filters._fe_subsample import (
    resolve_shared_fe_subsample_idx,
    stratified_subsample_idx,
)


def test_highcard_clf_does_not_return_full_n():
    """A (near-)continuous target passed as classification must NOT return ~n rows (the silent-full-n bug)."""
    n, size = 200_000, 30_000
    y = np.random.default_rng(0).lognormal(0.0, 1.5, n)  # ~all-unique => ~n classes
    idx = stratified_subsample_idx(np.random.default_rng(7), y, size, is_clf=True)
    assert idx.shape[0] <= size, f"high-card clf subsample must fit the budget, got {idx.shape[0]} > {size}"
    # A pre-fix run returned ~n (>= 100k); pin well below that so a regression is caught.
    assert idx.shape[0] < n // 2


def test_highcard_int_clf_does_not_return_full_n():
    n, size = 200_000, 30_000
    y = np.random.default_rng(1).integers(0, 100_000, n)  # >size distinct labels
    idx = stratified_subsample_idx(np.random.default_rng(7), y, size, is_clf=True)
    assert idx.shape[0] <= size


def test_resolve_shared_highcard_clf_subsamples():
    """End-to-end resolver: is_clf=True + high-card target must return ~size, not ~n (or None)."""
    n, size = 998_327, 30_000
    y = np.random.default_rng(2).lognormal(0.0, 1.5, n)
    idx = resolve_shared_fe_subsample_idx(y, n, size, is_clf=True, stratify_knob=True, random_seed=42)
    assert idx is not None
    assert idx.shape[0] <= size, f"resolver returned {idx.shape[0]} rows for a 'size={size}' high-card clf draw"


def test_legit_classification_still_stratifies():
    """The guard must NOT break a genuine low-cardinality classification: a rare class stays represented."""
    n, size = 100_000, 30_000
    rng = np.random.default_rng(3)
    y = rng.integers(0, 5, n)
    y[:500] = 7  # a rare 0.5% class
    idx = stratified_subsample_idx(np.random.default_rng(7), y, size, is_clf=True)
    assert idx.shape[0] <= size + 10
    assert (y[idx] == 7).any(), "rare class must survive the stratified draw"
    # all 6 classes represented
    assert len(np.unique(y[idx])) == 6


def test_regression_path_still_subsamples():
    n, size = 998_327, 30_000
    y = np.random.default_rng(4).lognormal(0.0, 1.5, n)
    idx = resolve_shared_fe_subsample_idx(y, n, size, is_clf=False, stratify_knob=None, random_seed=42)
    assert idx is not None and idx.shape[0] <= size + 10


def test_resolver_logs_on_failure_instead_of_silent_none(caplog):
    """A resolver failure must WARN (diagnosable full-n fallback), not return None silently."""
    # A non-numeric object array with an uncomparable element makes the internal np draw raise; the
    # resolver must log + fall back rather than swallow. Force stratify to exercise the stratified path.
    bad = np.array([object()] * 100, dtype=object)
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.filters.mrmr"):
        out = resolve_shared_fe_subsample_idx(bad, 100, 30, is_clf=True, stratify_knob=True, random_seed=1)
    # Either it succeeds (returns indices) or, if it fails, it must have logged a WARNING (never a silent None).
    if out is None:
        assert any("FULL n" in rec.message for rec in caplog.records), "silent None without a WARNING log"
