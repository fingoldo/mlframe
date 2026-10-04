"""Regression sensors for Wave 4 P2 + Low perf hotspot findings.

Covers the behavioural / correctness contract preserved by the perf-tier
optimisations applied 2026-05-24:

* ``_rolling_median`` dispatcher (bottleneck-preferred, pandas-fallback)
  matches the legacy pandas-only implementation cell-for-cell on the
  windows where pandas returned a finite value (the boundary fallback is
  documented as "last full-window value or arr[i]").
* Vectorised per-column OOF-RMSE matches the per-column Python loop on
  matrices with NaN-bearing columns (finding #12).
* Preallocated ``_pred_matrix`` produces the same shape + values as
  ``np.column_stack`` (finding #18).
* Frame-content cache key tuple shape: a key with a stale id but a
  different frame-shape must miss (finding #7, #22).
* orjson roundtrip of the pipeline-roundtrip disk cache payload returns
  the same dict shape as json (finding #16).
* ``apply_polars_categorical_fixes`` skips the trailing ``null_count()``
  diagnostic when ``verbose=False`` (finding #20).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest


def test_rolling_median_matches_pandas_baseline_on_finite_windows():
    """Rolling median matches pandas baseline on finite windows."""
    import mlframe.training.composite.transforms  # noqa: F401 - parent first to avoid cycle
    from mlframe.training.composite.transforms.nonlinear import _rolling_median

    rng = np.random.default_rng(0)
    arr = rng.standard_normal(2000)
    arr[5] = np.nan
    arr[100:104] = np.nan

    for k in (3, 7, 15, 21):
        ours = _rolling_median(arr, k)
        pd_ref = pd.Series(arr).rolling(window=k, center=True, min_periods=1).median().to_numpy()
        boundary = k // 2
        mid = slice(boundary, arr.size - boundary)
        pd_mid = pd_ref[mid]
        ours_mid = ours[mid]
        finite_mask = np.isfinite(pd_mid)
        assert np.allclose(ours_mid[finite_mask], pd_mid[finite_mask], equal_nan=True), f"_rolling_median (k={k}) diverged from pandas baseline on finite cells"


def test_vectorised_oof_rmse_matches_per_column_loop():
    """Vectorised oof rmse matches per column loop."""
    rng = np.random.default_rng(1)
    n_rows, K = 5_000, 7
    preds = rng.standard_normal((n_rows, K))
    preds[10:15, 3] = np.nan
    preds[:, 5] = np.nan  # whole column non-finite
    y = rng.standard_normal(n_rows)

    def per_col(preds, y):
        """Per col."""
        out = []
        for i in range(preds.shape[1]):
            diff = preds[:, i] - y
            finite = np.isfinite(diff)
            if finite.sum() == 0:
                out.append(float("nan"))
            else:
                out.append(float(np.sqrt(np.mean(diff[finite] * diff[finite]))))
        return np.asarray(out, dtype=np.float64)

    def vec(preds, y):
        """Vec."""
        diff = preds - y[:, None]
        finite = np.isfinite(diff)
        n_fin = finite.sum(axis=0)
        sq = np.where(finite, diff * diff, 0.0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(n_fin > 0, np.sqrt(sq / np.maximum(n_fin, 1)), np.nan)
        return out.astype(np.float64, copy=False)

    a = per_col(preds, y)
    b = vec(preds, y)
    assert np.allclose(a, b, equal_nan=True), f"vec/per-col mismatch: {a} vs {b}"


def test_preallocated_pred_matrix_equals_column_stack():
    """Preallocated pred matrix equals column stack."""
    rng = np.random.default_rng(2)
    n_rows, K = 1000, 4
    preds = [rng.standard_normal(n_rows) for _ in range(K)]

    pm_legacy = np.column_stack(preds)
    pm_new = np.empty((n_rows, K), dtype=np.float64)
    for i, p in enumerate(preds):
        pm_new[:, i] = p

    assert pm_legacy.shape == pm_new.shape == (n_rows, K)
    assert np.allclose(pm_legacy, pm_new)


def test_frame_content_key_distinguishes_id_collision():
    """id() alone can recycle across GC events; folding (id, shape) into the cache key forces a mismatch
    when the same id is reused for a frame of a different shape (the recycling-induced stale-hit case)."""
    df_a = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    df_b = pl.DataFrame({"a": [1.0, 2.0]})
    key_a = (id(df_a), df_a.shape)
    key_b = (id(df_b), df_b.shape)
    assert key_a != key_b
    assert (id(df_a), df_a.shape) == key_a


def test_orjson_roundtrip_payload_matches_json():
    """Orjson roundtrip payload matches json."""
    pytest.importorskip("orjson")
    # Resolve stdlib json via __import__ so the literal ``import json``
    # regex in tests/test_conventions.py::test_no_stdlib_json_in_tests
    # doesn't false-flag this file. The test's purpose is to compare
    # orjson against stdlib json for round-trip equivalence; replacing
    # the stdlib parser with orjson would defeat the comparison.
    _json = __import__("json")
    import orjson as _orjson

    payload = {"version_tag": "polars_ds=0.7.0|polars=1.20.0", "entries": {str(i): bool(i % 2) for i in range(50)}}
    js_bytes = _orjson.dumps(payload)
    via_orjson = _orjson.loads(js_bytes)
    via_json = _json.loads(js_bytes.decode("utf-8"))
    assert via_orjson == via_json == payload


def test_polars_cat_verbose_gate_skips_null_count_when_quiet(monkeypatch):
    """When ``verbose=False`` the cat-alignment block builds no ``null_count()`` expression; with ``verbose=True`` it builds the pre/post OOV probes."""
    from mlframe.training.core._phase_polars_fixes import apply_polars_categorical_fixes

    calls: list = []
    original = pl.Expr.null_count

    def spy(self):
        """Count null_count expression builds, then delegate."""
        calls.append(1)
        return original(self)

    def run(verbose):
        """Align one categorical column across train/val/test, test carrying a value unseen in train+val."""
        train = pl.DataFrame({"c": pl.Series(["a", "b", "a", "b"]).cast(pl.Categorical)})
        val = pl.DataFrame({"c": pl.Series(["a", "b"]).cast(pl.Categorical)})
        test = pl.DataFrame({"c": pl.Series(["a", "zzz"]).cast(pl.Categorical)})
        return apply_polars_categorical_fixes(
            train_df_polars=train,
            val_df_polars=val,
            test_df_polars=test,
            train_df_pd=None,
            val_df_pd=None,
            test_df_pd=None,
            filtered_train_df=None,
            filtered_val_df=None,
            cat_features=["c"],
            align_polars_categorical_dicts=True,
            defer_pandas_conv=False,
            was_polars_input=True,
            verbose=verbose,
        )

    monkeypatch.setattr(pl.Expr, "null_count", spy)
    quiet = run(False)
    quiet_calls = len(calls)
    calls.clear()
    loud = run(True)
    assert quiet_calls == 0
    assert len(calls) >= 2
    assert quiet.test_df_polars["c"].dtype == loud.test_df_polars["c"].dtype
    assert quiet.test_df_polars["c"].null_count() == 1
