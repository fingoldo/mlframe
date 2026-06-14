"""iter97: parallel bucket-split descending argsort for the large-N metrics CPU path.

The metric kernels' descending argsort is tie-order-invariant (AUC uses fractional ranks; KS folds tied scores into a
single CDF jump), so any sort that orders ``y_score`` identically to ``np.argsort(y_score)[::-1]`` is admissible. The
parallel bucket variant must (a) order y_score identically on continuous, tied, and constant inputs, (b) be wired into
the dispatcher at N >= the gate, and (c) leave the full calibration report byte-identical.

FAILS on pre-fix code: ``_argsort_desc_par_bucket`` / ``_PAR_BUCKET_ARGSORT_MIN_N`` do not exist at HEAD (ImportError).
"""
import numpy as np
import pytest

from mlframe.metrics import _core_auc_brier as cab


def test_par_bucket_orders_yscore_identically_continuous():
    rng = np.random.default_rng(7)
    y = (1 / (1 + np.exp(-rng.normal(0, 1, 300_000)))).astype(np.float64)
    new = cab._argsort_desc_par_bucket(y)
    old = np.argsort(y)[::-1]
    assert np.array_equal(y[new], y[old])


def test_par_bucket_orders_yscore_identically_tied():
    rng = np.random.default_rng(8)
    y = np.round((1 / (1 + np.exp(-rng.normal(0, 1, 200_000)))).astype(np.float64), 2)
    new = cab._argsort_desc_par_bucket(y)
    old = np.argsort(y)[::-1]
    assert np.array_equal(y[new], y[old])


def test_par_bucket_constant_column_is_full_permutation():
    y = np.full(120_000, 0.5)
    new = cab._argsort_desc_par_bucket(y)
    assert len(new) == 120_000
    assert np.array_equal(np.sort(new), np.arange(120_000))


def test_dispatcher_routes_to_par_bucket_above_gate(monkeypatch):
    calls = {"n": 0}
    orig = cab._argsort_desc_par_bucket

    def spy(y):
        calls["n"] += 1
        return orig(y)

    monkeypatch.setattr(cab, "_argsort_desc_par_bucket", spy)
    # Force the GPU branch off so the large-N CPU path is the parallel bucket sort deterministically (the GPU radix is
    # the higher-priority backend when a CUDA device is visible; this test pins the CPU dispatch, not the GPU one).
    monkeypatch.setattr(cab, "_GPU_ARGSORT_MIN_N", 10 ** 12)
    rng = np.random.default_rng(9)
    big = rng.random(cab._PAR_BUCKET_ARGSORT_MIN_N + 1000).astype(np.float64)
    small = rng.random(1000).astype(np.float64)
    cab._argsort_desc_for_metrics(big)
    assert calls["n"] == 1, "large-N CPU path must route through the parallel bucket sort"
    cab._argsort_desc_for_metrics(small)
    assert calls["n"] == 1, "small-N path must stay on scalar numpy argsort"


def test_full_report_byte_identical_with_and_without_par_bucket():
    from mlframe.metrics.classification._classification_report import fast_calibration_report

    rng = np.random.default_rng(11)
    n = 300_000
    z = rng.normal(0, 1, n)
    yt = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(np.int64)
    yp = (1 / (1 + np.exp(-(z + rng.normal(0, 0.5, n))))).astype(np.float64)

    new = fast_calibration_report(yt, yp, show_plots=False)[:-2]
    saved = cab._PAR_BUCKET_ARGSORT_MIN_N
    try:
        cab._PAR_BUCKET_ARGSORT_MIN_N = 10 ** 12  # force scalar numpy path
        old = fast_calibration_report(yt, yp, show_plots=False)[:-2]
    finally:
        cab._PAR_BUCKET_ARGSORT_MIN_N = saved

    for a, b in zip(new, old):
        if a is None or b is None:
            assert a is b
        else:
            assert a == b
