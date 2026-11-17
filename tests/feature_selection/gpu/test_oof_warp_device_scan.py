"""The device scan of the out-of-fold warp family scores the columns as the host pool does and leaves the emitted column to the host service."""

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _oof_warp_fe as warp
from mlframe.feature_selection.filters._fe_frame_ops import fe_stack_rows

_STATS = ("mi_raw", "mi_warp", "gain", "se", "bulk_gain", "bulk_se")


def _frame(n: int = 8000, classes: int = 2) -> "tuple[pd.DataFrame, np.ndarray]":
    """Eight float32 columns: a sine and a bump effect on the target, noise, one column with missing and infinite values, one nominal-like, one constant, one with heavy tails."""
    rng = np.random.default_rng(3)
    xs = rng.standard_normal((n, 8)).astype(np.float32)
    score = np.sin(4.0 * xs[:, 0]) + (np.abs(xs[:, 1]) < 0.4) * 1.5 + 0.3 * rng.standard_normal(n)
    xs[rng.random(n) < 0.03, 3] = np.nan
    xs[:5, 3] = np.inf
    xs[:, 4] = rng.integers(0, 3, n)
    xs[:, 5] = 1.0
    xs[:, 6] = rng.standard_cauchy(n).astype(np.float32)
    y = np.digitize(score, np.quantile(score, np.linspace(0, 1, classes + 1)[1:-1])).astype(np.int64)
    return pd.DataFrame(xs, columns=[f"c{j}" for j in range(8)]), y


def _scan_both(X: pd.DataFrame, y: np.ndarray, **kw) -> "tuple[list, list]":
    """The host pool's and the device scan's candidates; skipped when the device scan did not run."""
    host = warp.warp_candidates(X, y, list(X.columns), device=False, **kw)
    dev = warp.warp_candidates(X, y, list(X.columns), device=True, **kw)
    if not dev or any(c["scored_on"] != "device" for c in dev):
        pytest.skip("the device warp scan is not available on this host")
    assert host and all(c["scored_on"] == "host" for c in host)
    return host, dev


def _assert_same(host: list, dev: list) -> None:
    """Same candidates in the same order with the same statistics."""
    assert [(c["col"], c["cls"]) for c in host] == [(c["col"], c["cls"]) for c in dev]
    for k in _STATS:
        np.testing.assert_allclose([c[k] for c in dev], [c[k] for c in host], rtol=1e-6, atol=1e-9, err_msg=k)


def test_device_scan_matches_the_host_pool():
    """Every usable column gets the host's statistics; the nominal-like and the constant column are skipped by both."""
    X, y = _frame()
    host, dev = _scan_both(X, y)
    _assert_same(host, dev)
    assert {c["col"] for c in dev} == {"c0", "c1", "c2", "c3", "c6", "c7"}
    assert all(c["fit"] is None for c in dev)


def test_device_scan_matches_on_a_row_sample_and_with_one_vs_rest_targets():
    """The scan rows are a sample of the frame and each class has its own warp target."""
    X, y = _frame(9000, classes=3)
    host, dev = _scan_both(X, y, scan_rows=4001, multiclass="ovr")
    assert len({c["cls"] for c in dev}) == 3
    _assert_same(host, dev)


def test_device_scored_candidate_is_fitted_by_the_host_service():
    """The fit made on request and the training column of a device-scored candidate are the host scan's, to the bit."""
    X, y = _frame()
    host, dev = _scan_both(X, y)
    for h, d in zip(host, dev):
        fit = warp.candidate_fit(d)
        for k in ("oof", "cx", "cy"):
            np.testing.assert_array_equal(fit[k], h["fit"][k])
        assert fit["fill"] == h["fit"]["fill"]
        np.testing.assert_array_equal(warp.training_column(d), warp.training_column(h))


def test_family_emits_the_same_columns_through_the_device_scan(monkeypatch):
    """``hybrid_oof_warp_fe`` appends the same warp columns, values and recipes whichever side scanned."""
    X, y = _frame()
    _scan_both(X, y)
    out = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("MLFRAME_OOF_WARP_DEVICE", flag)
        _x_aug, appended, recipes, enc = warp.hybrid_oof_warp_fe(X, y)
        out[flag] = (appended, enc, recipes)
    assert out["0"][0] == out["1"][0] and out["1"][0]
    pd.testing.assert_frame_equal(out["0"][1], out["1"][1])
    for a, b in zip(out["0"][2], out["1"][2]):
        np.testing.assert_array_equal(a.extra["cy"], b.extra["cy"])


def test_stack_rows_is_the_stack_of_the_columns():
    """Row ``j`` is column ``j`` cast to the dtype, for strided columns longer than a tile."""
    base = np.random.default_rng(0).standard_normal((2500, 5))
    got = fe_stack_rows([base[:, j] for j in range(5)], np.float32)
    assert got.flags.c_contiguous and got.dtype == np.float32
    np.testing.assert_array_equal(got, base.T.astype(np.float32))


def test_row_quantiles_are_numpys_to_the_bit():
    """The device edges equal ``np.quantile`` of each row's sample exactly, also where the position of a quantile is a whole number (29406 values: an edge that is a sample itself decides
    the bin of that sample, and a position off by one rounding moved a row to the next bin and the warp table with it)."""
    from mlframe.feature_selection.filters._oof_warp_gpu import _row_quantiles

    rng = np.random.default_rng(1)
    counts = np.array([29406, 29405, 401, 21, 30000], dtype=np.int64)
    rows = np.full((len(counts), 30000), np.inf, dtype=np.float32)
    for r, c in enumerate(counts):
        rows[r, :c] = np.sort(rng.standard_normal(c).astype(np.float32))
    for probs in (np.linspace(0.0, 1.0, 21)[1:-1], np.linspace(0.0, 1.0, 11)[1:-1], np.array([0.025, 0.975])):
        got = cp.asnumpy(_row_quantiles(cp, cp.asarray(rows), cp.asarray(counts), probs))
        want = np.stack([np.quantile(rows[r, :c], probs) for r, c in enumerate(counts)])
        np.testing.assert_array_equal(got, want)
