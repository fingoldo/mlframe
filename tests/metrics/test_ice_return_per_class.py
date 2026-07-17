"""Bit-identity regression for the per-class ICE reuse fast route.

``compute_probabilistic_multiclass_error`` returns the per-class ICE vector
(via ``return_per_class=True``) that its batched kernel already computes during
the full (N, K) call. The probabilistic report indexes that vector in its
per-class loop instead of recomputing each 1-D column. These tests pin that the
indexed per-class values EQUAL the per-class recompute exactly (maxdiff 0.0),
and that the report's stored per-class ``class_integral_error`` is unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.core import compute_probabilistic_multiclass_error


def _softmax_probs(rng, n, k):
    """Helper: Softmax probs."""
    logits = rng.normal(size=(n, k))
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return p


@pytest.mark.parametrize("k", [2, 3, 8])
def test_per_class_ice_indexed_equals_recompute(k):
    """Indexed per-class ICE == 1-D per-class recompute, maxdiff 0.0."""
    rng = np.random.default_rng(123 + k)
    n = 2000
    y_true = rng.integers(0, k, n)
    probs = _softmax_probs(rng, n, k)

    _agg, per_class = compute_probabilistic_multiclass_error(y_true=y_true, y_score=probs, nbins=10, return_per_class=True)
    assert isinstance(per_class, dict) and per_class, "fastpath should populate per-class dict"

    # Binary skips class 0; multiclass evaluates all K.
    eval_ids = [1] if k == 2 else list(range(k))
    assert sorted(per_class) == eval_ids

    maxdiff = 0.0
    for c in eval_ids:
        recompute = compute_probabilistic_multiclass_error(y_true=(y_true == c).astype(np.int8), y_score=probs[:, c], nbins=10)
        maxdiff = max(maxdiff, abs(float(recompute) - float(per_class[c])))
    assert maxdiff == 0.0, f"per-class ICE must be bit-identical, got maxdiff={maxdiff}"


def test_return_per_class_scalar_still_matches():
    """Aggregate scalar is unchanged whether or not per-class is requested."""
    rng = np.random.default_rng(7)
    n, k = 1500, 5
    y_true = rng.integers(0, k, n)
    probs = _softmax_probs(rng, n, k)

    scalar = compute_probabilistic_multiclass_error(y_true=y_true, y_score=probs, nbins=10)
    agg, _ = compute_probabilistic_multiclass_error(y_true=y_true, y_score=probs, nbins=10, return_per_class=True)
    assert agg == scalar


def test_return_per_class_empty_dict_on_legacy_path():
    """Non-multicrit/verbose path returns an empty dict (caller falls back)."""
    rng = np.random.default_rng(11)
    n, k = 800, 4
    y_true = rng.integers(0, k, n)
    probs = _softmax_probs(rng, n, k)

    _agg, per_class = compute_probabilistic_multiclass_error(y_true=y_true, y_score=probs, method="brier_score", return_per_class=True)
    assert per_class == {}


def test_report_per_class_integral_error_unchanged():
    """The report's stored per-class ICE is identical old-vs-new on K=8.

    Old behaviour == forcing the per-class recompute by disabling the fast
    route (custom_ice_metric that rejects return_per_class). New behaviour ==
    the default fast route. The per-class ``class_integral_error`` in the
    metrics dict must match exactly.
    """
    from functools import partial

    from mlframe.training.reporting._reporting_probabilistic import (
        report_probabilistic_model_perf,
    )

    rng = np.random.default_rng(2026)
    n, k = 1500, 8
    y_true = rng.integers(0, k, n)
    probs = _softmax_probs(rng, n, k)

    fast_metric = partial(compute_probabilistic_multiclass_error, nbins=10)

    def legacy_metric(*, y_true, y_score):
        # Rejects return_per_class -> report falls back to per-class recompute.
        """Legacy metric."""
        return compute_probabilistic_multiclass_error(y_true=y_true, y_score=y_score, nbins=10)

    common = dict(
        targets=y_true,
        columns=[f"f{i}" for i in range(3)],
        model_name="m",
        model=None,
        probs=probs,
        classes=list(range(k)),
        print_report=False,
        show_perf_chart=False,
        show_prob_histogram=False,
        fairness_calibration_charts=False,
        calibration_by_feature_charts=False,
        calibration_heatmap_2d_charts=False,
    )

    m_new: dict = {}
    report_probabilistic_model_perf(custom_ice_metric=fast_metric, metrics=m_new, **common)

    m_old: dict = {}
    report_probabilistic_model_perf(custom_ice_metric=legacy_metric, metrics=m_old, **common)

    new_ice = {cid: m_new[cid]["class_integral_error"] for cid in range(k) if cid in m_new}
    old_ice = {cid: m_old[cid]["class_integral_error"] for cid in range(k) if cid in m_old}
    assert set(new_ice) == set(old_ice)
    maxdiff = max(abs(new_ice[c] - old_ice[c]) for c in new_ice)
    assert maxdiff == 0.0, f"report per-class ICE must be bit-identical, maxdiff={maxdiff}"


def test_perf_sentinel_index_faster_than_recompute():
    """Indexing the precomputed vector beats K per-class recomputes.

    Floor 1.5x (measured ~1.88x at N=200k/K=8). Guards against a future change that silently reverts the report to the
    per-class recompute path (which would collapse the ratio to ~1x). The floor was 2x when the ICE kernel sorted
    internally with numba's slow argsort -- hoisting the sort to numpy's C argsort (bit-identical, the AUC walk is
    tie-invariant) made BOTH the batched and the recompute path faster, so the recompute penalty -- and thus this
    ratio -- shrank while the batched route stayed strictly faster.
    """
    import timeit

    rng = np.random.default_rng(0)
    n, k = 200_000, 8
    y_true = rng.integers(0, k, n)
    probs = _softmax_probs(rng, n, k)

    def old():
        """Old."""
        compute_probabilistic_multiclass_error(y_true=y_true, y_score=probs, nbins=10)
        return [compute_probabilistic_multiclass_error(y_true=(y_true == c).astype(np.int8), y_score=probs[:, c], nbins=10) for c in range(k)]

    def new():
        """New."""
        _, per = compute_probabilistic_multiclass_error(y_true=y_true, y_score=probs, nbins=10, return_per_class=True)
        return [per[c] for c in range(k)]

    old()
    new()  # warm numba
    to = min(timeit.repeat(old, number=2, repeat=4)) / 2
    tn = min(timeit.repeat(new, number=2, repeat=4)) / 2
    assert to / tn >= 1.5, f"fast route should be >=1.5x; got {to / tn:.2f}x (old={to * 1e3:.1f}ms new={tn * 1e3:.1f}ms)"


def test_report_non_arange_labels_fall_back_safely():
    """Non-0-indexed labels disable the fast route (correctness gate)."""
    from functools import partial

    from mlframe.training.reporting._reporting_probabilistic import (
        report_probabilistic_model_perf,
    )

    rng = np.random.default_rng(99)
    n, k = 1200, 3
    # Labels [1, 2, 3] -- kernel column index (0..2) != class label.
    y_true = rng.integers(1, k + 1, n)
    probs = _softmax_probs(rng, n, k)
    fast_metric = partial(compute_probabilistic_multiclass_error, nbins=10)

    metrics: dict = {}
    report_probabilistic_model_perf(
        targets=y_true,
        columns=["a"],
        model_name="m",
        model=None,
        probs=probs,
        classes=[1, 2, 3],
        print_report=False,
        show_perf_chart=False,
        show_prob_histogram=False,
        fairness_calibration_charts=False,
        calibration_by_feature_charts=False,
        calibration_heatmap_2d_charts=False,
        custom_ice_metric=fast_metric,
        metrics=metrics,
    )
    # Recompute the honest per-class value (targets == label) and compare.
    for i, label in enumerate([1, 2, 3]):
        honest = compute_probabilistic_multiclass_error(y_true=(y_true == label).astype(np.int8), y_score=probs[:, i], nbins=10)
        assert metrics[i]["class_integral_error"] == honest
