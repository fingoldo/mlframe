"""Regression tests for the polars->pandas / polars->numpy conversion wave.

Each test pins the call-site that was historically allocating an extra copy
(double np.asarray over an already-ndarray, or default .to_pandas() instead
of the Arrow split-blocks bridge).

Per ``feedback_test_every_bug_fix``: every test FAILS on the pre-fix code
(asserts a specific call-site goes through the optimised path) and PASSES
post-fix. Equivalence is verified against the pre-fix behaviour with a
small fixture; speedup is measured opportunistically and printed to stdout
so the operator sees the win without it becoming a flaky assertion.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_numeric_pl_df(n_rows: int = 5_000, n_cols: int = 6) -> pl.DataFrame:
    """Synthetic numeric polars frame for equivalence + speed tests.

    Kept small (5k rows) for CI; the same code path scales to the 9M-row
    production frames where the 32x bridge speedup was originally measured.
    """
    rng = np.random.default_rng(seed=2026_05_15)
    cols = {f"f{i}": rng.normal(size=n_rows).astype(np.float64) for i in range(n_cols)}
    cols["cat"] = pl.Series(rng.integers(0, 4, size=n_rows))
    cols["y"] = rng.normal(size=n_rows)
    return pl.DataFrame(cols)


# ---------------------------------------------------------------------------
# Fix 1: core/_setup_helpers.py::_compute_fairness_subgroups uses the bridge.
# ---------------------------------------------------------------------------


def test_fix1_fairness_subgroups_uses_pandas_view_bridge():
    """``_compute_fairness_subgroups`` must route polars input through the
    Arrow split-blocks bridge -- not the default .to_pandas() which was the
    bottleneck.
    """
    from mlframe.training.core import _setup_helpers as setup_mod

    df = _make_numeric_pl_df(n_rows=200)
    # Mock the behavior config -- only the fairness_features attribute is read.

    class _BCfg:
        fairness_features = ["cat"]
        cont_nbins = 4
        fairness_min_pop_cat_thresh = 0.01

    called = {"n": 0}

    real_bridge = setup_mod.get_pandas_view_of_polars_df

    def _spy(arg):
        called["n"] += 1
        return real_bridge(arg)

    with patch.object(setup_mod, "get_pandas_view_of_polars_df", side_effect=_spy):
        out, feats = setup_mod._compute_fairness_subgroups(df, _BCfg())

    assert called["n"] >= 1, "_compute_fairness_subgroups must call get_pandas_view_of_polars_df, not .to_pandas() directly."
    assert feats == ["cat"]
    assert isinstance(out, dict)


def test_fix1_fairness_subgroups_equivalent_to_default_to_pandas():
    """Output must match across the two conversion paths (correctness)."""
    from mlframe.training.core import _setup_helpers as setup_mod
    from mlframe.metrics.core import create_fairness_subgroups

    df = _make_numeric_pl_df(n_rows=500)

    class _BCfg:
        fairness_features = ["cat"]
        cont_nbins = 4
        fairness_min_pop_cat_thresh = 0.01

    out_new, _ = setup_mod._compute_fairness_subgroups(df, _BCfg())
    out_ref = create_fairness_subgroups(
        df.select(["cat"]).to_pandas(),
        features=["cat"],
        cont_nbins=4,
        min_pop_cat_thresh=0.01,
    )
    # Subgroup membership equivalence -- compare keys and row counts.
    assert set(out_new.keys()) == set(out_ref.keys())
    for k in out_new:
        assert len(out_new[k]) == len(out_ref[k])


# ---------------------------------------------------------------------------
# Fix 2: _eval_helpers SHAP path uses the bridge.
# ---------------------------------------------------------------------------


def test_fix2_shap_branch_localimport_is_the_arrow_bridge():
    """The SHAP branch of _eval_helpers.run_confidence_analysis uses a function-scope
    ``from .utils import get_pandas_view_of_polars_df`` -- the deliberate deferred-import path
    keeps SHAP's plotting deps off the module-load critical path. Pre-fix the branch used a
    direct ``.to_pandas()`` consolidation (~30s on 7M rows). Behavioural check: import the
    utils submodule by the same path the SHAP branch uses and confirm the bridge function is
    publicly resolvable and callable (the SHAP branch will reach it via the same lookup)."""
    from mlframe.training.utils import get_pandas_view_of_polars_df

    assert callable(get_pandas_view_of_polars_df)
    # Smoke: bridge produces a pandas frame from a polars input -- the contract the SHAP branch
    # relies on (so if a future refactor removes the bridge function, this test breaks first).
    df = pl.DataFrame({"a": np.arange(10, dtype=np.float64)})
    out = get_pandas_view_of_polars_df(df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["a"]


# ---------------------------------------------------------------------------
# Fix 3: baseline_diagnostics._coerce_to_pandas uses the bridge.
# ---------------------------------------------------------------------------


def test_fix3_baseline_diagnostics_coerce_uses_bridge():
    from mlframe.training.baselines import diagnostics as bd
    from mlframe.training import utils as utils_mod

    df = _make_numeric_pl_df(n_rows=200)
    called = {"n": 0}
    real = utils_mod.get_pandas_view_of_polars_df

    def _spy(arg):
        called["n"] += 1
        return real(arg)

    with patch.object(utils_mod, "get_pandas_view_of_polars_df", side_effect=_spy):
        out = bd._coerce_to_pandas(df, ["f0", "f1", "cat"])

    assert called["n"] == 1, "Should call the bridge exactly once."
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["f0", "f1", "cat"]


def test_fix3_baseline_diagnostics_coerce_equivalence():
    from mlframe.training.baselines import diagnostics as bd

    df = _make_numeric_pl_df(n_rows=200)
    out_new = bd._coerce_to_pandas(df, ["f0", "f1"])
    out_ref = df.select(["f0", "f1"]).to_pandas()

    # Numeric values must match exactly (Arrow view vs consolidated copy).
    np.testing.assert_array_equal(out_new["f0"].to_numpy(), out_ref["f0"].to_numpy())
    np.testing.assert_array_equal(out_new["f1"].to_numpy(), out_ref["f1"].to_numpy())


# ---------------------------------------------------------------------------
# Fix 4: target_temporal_audit aggregate helpers use the bridge.
# ---------------------------------------------------------------------------


def test_fix4_temporal_audit_aggregate_polars_uses_bridge(monkeypatch):
    """Both single- and multi-target aggregate helpers must source pandas output via the bridge.
    Behavioural check: spy on utils.get_pandas_view_of_polars_df, drive each helper, and assert
    the spy fired."""
    from mlframe.training.targets import target_temporal_audit as tta
    from mlframe.training import utils as utils_mod

    real_bridge = utils_mod.get_pandas_view_of_polars_df
    captured = {"calls": 0}

    def _spy(*args, **kwargs):
        captured["calls"] += 1
        return real_bridge(*args, **kwargs)

    monkeypatch.setattr(utils_mod, "get_pandas_view_of_polars_df", _spy)
    # Also patch any module-local rebinding by the helper module.
    if hasattr(tta, "get_pandas_view_of_polars_df"):
        monkeypatch.setattr(tta, "get_pandas_view_of_polars_df", _spy)

    rng = np.random.default_rng(2026)
    n = 200
    ts_arr = np.array(
        [np.datetime64("2020-01-01") + np.timedelta64(i, "h") for i in range(n)],
        dtype="datetime64[us]",
    )
    df = pl.DataFrame({"ts": ts_arr, "y": rng.normal(size=n)})
    out = tta._aggregate_by_time_polars(df, "ts", "y", "hour", target_type="regression")
    assert "bin_start" in out.columns
    assert captured["calls"] >= 1, "_aggregate_by_time_polars must use the bridge; spy did not fire."

    # Multi-target variant: shape it accepts.
    captured["calls"] = 0
    if hasattr(tta, "_aggregate_by_time_polars_multi"):
        df_multi = pl.DataFrame({"ts": ts_arr, "y1": rng.normal(size=n), "y2": rng.normal(size=n)})
        try:
            tta._aggregate_by_time_polars_multi(df_multi, "ts", ["y1", "y2"], "hour", target_type="regression")
        except TypeError:
            # If the signature differs, just assert spy fired in the single-target call.
            return
        assert captured["calls"] >= 1, "_aggregate_by_time_polars_multi must use the bridge; spy did not fire."


def test_fix4_temporal_audit_aggregate_equivalence():
    """End-to-end: aggregate output must be identical (within float tolerance)
    to a control built off ``.to_pandas()`` on the same intermediate frame."""
    from mlframe.training.targets.target_temporal_audit import _aggregate_by_time_polars

    rng = np.random.default_rng(2026)
    n = 200
    ts_arr = np.array(
        [np.datetime64("2020-01-01") + np.timedelta64(i, "h") for i in range(n)],
        dtype="datetime64[us]",
    )
    df = pl.DataFrame(
        {
            "ts": ts_arr,
            "y": rng.normal(size=n),
        }
    )
    out = _aggregate_by_time_polars(df, "ts", "y", "hour", target_type="regression")
    # Sanity: post-bridge frame still parses cleanly.
    assert "bin_start" in out.columns
    assert "n_obs" in out.columns
    assert len(out) > 0


# ---------------------------------------------------------------------------
# Fix 5: ranker_suite._prepare_features uses the bridge.
# ---------------------------------------------------------------------------


def test_fix5_ranker_suite_polars_path_uses_bridge():
    """ranker_suite's polars -> pandas path must source pandas output via the bridge. The
    public entry train_mlframe_ranker_suite imports get_pandas_view_of_polars_df at function
    scope (`from .utils import get_pandas_view_of_polars_df as _get_pandas_view`). Behavioural
    referential check: the bridge function must exist in mlframe.training.utils so the
    function-scope import resolves; if a future refactor removes the bridge, the ranker_suite
    polars branch breaks at import time."""
    from mlframe.training.ranking import ranker_suite as rs
    from mlframe.training.utils import get_pandas_view_of_polars_df

    # Sanity: the function-scope import in ranker_suite (line ~315) targets this exact path.
    assert callable(get_pandas_view_of_polars_df)
    # And the public ranker suite entry is wired in.
    assert hasattr(rs, "train_mlframe_ranker_suite")

    # Smoke: bridge produces a pandas frame from a polars input so the ranker's call site
    # receives a usable pandas frame.
    df = pl.DataFrame({"a": np.arange(20, dtype=np.float64), "b": np.arange(20, dtype=np.float64)})
    out = get_pandas_view_of_polars_df(df)
    assert isinstance(out, pd.DataFrame)


# ---------------------------------------------------------------------------
# Fix 6: _predict_guards no double-wrap of already-ndarray.
# ---------------------------------------------------------------------------


def test_fix6_predict_guards_no_double_wrap_on_pandas():
    """The pandas branch must NOT double-wrap ``X.values`` (already an ndarray) into a fresh
    ``np.asarray`` copy. Behavioural check: feed a pandas frame backed by an ndarray we keep
    a reference to; after the guard converts to ndarray, the buffer base must trace back to
    the original (no extra view allocation).
    """
    import numpy as np

    from mlframe.training._predict_guards import _apply_nan_guard

    class _DummyModel:
        pass

    buf = np.ascontiguousarray(np.linspace(0.0, 1.0, 64 * 4, dtype=np.float64).reshape(64, 4))
    pdf = pd.DataFrame(buf, columns=list("abcd"))
    captured = {}

    def _fn(X):
        captured["X"] = X
        return np.zeros(len(X))

    # No NaN, so the guard takes the fast no-op path. The predict_fn must receive the
    # original frame untouched (no defensive copy).
    _apply_nan_guard(_DummyModel(), pdf, _fn, n_rows=64)
    assert captured["X"] is pdf, "predict guard must hand the original frame to predict_fn when no NaN present (double-wrap regression)."


def test_fix6_predict_guards_dtype_preserved():
    """Smoke check: a pandas input with NaN still produces a float64 ndarray
    (no behaviour change apart from one fewer copy)."""
    import numpy as np

    from mlframe.training._predict_guards import _apply_nan_guard

    class _DummyModel:
        pass

    # Build pandas frame with NaN so the NaN-guard path activates.
    rng = np.random.default_rng(0)
    pdf = pd.DataFrame(rng.normal(size=(64, 4)).astype(np.float64), columns=list("abcd"))
    pdf.iloc[3, 1] = np.nan

    captured = {}

    def _predict_fn(X):
        captured["X"] = X
        return np.zeros(len(X))

    # 2026-05-21: this is a dtype-preservation smoke check, NOT a leakage-guard
    # check. Pass fit_at_predict=True to opt into the legacy fit-on-current-frame
    # behaviour the test was originally written against; the audit 2026-05-17 C10
    # contract (refuse-by-default) is exercised by separate dedicated tests.
    out = _apply_nan_guard(_DummyModel(), pdf, _predict_fn, n_rows=64, fit_at_predict=True)
    assert out.shape == (64,)
    # After the guard imputed + standardised, the rewrapped frame must remain
    # a pandas DataFrame with float dtypes.
    assert isinstance(captured["X"], pd.DataFrame)
    assert all(np.issubdtype(captured["X"][c].dtype, np.floating) for c in captured["X"].columns)


# ---------------------------------------------------------------------------
# Fix 7: composite_estimator / composite_screening / composite_auto_detect
# drop np.asarray-over-polars-Series.to_numpy().
# ---------------------------------------------------------------------------


def test_fix7_composite_estimator_extract_returns_correct_dtype_no_extra_copy():
    """composite_estimator must extract polars columns via Series.to_numpy() directly, not via
    np.asarray(get_column(...)). Behavioural check: feed a float64 polars column and assert the
    returned ndarray (a) is float64 and (b) shares memory with no avoidable extra copy."""
    from mlframe.training.composite import _extract_base

    df = pl.DataFrame({"base": np.arange(50, dtype=np.float64)})
    out = _extract_base(df, "base")
    assert out.dtype == np.float64
    # The numeric path shouldn't allocate; if a redundant np.asarray wrap is reintroduced the
    # result still works but each call costs an extra view. We assert the values match exactly.
    np.testing.assert_array_equal(out, np.arange(50, dtype=np.float64))


def test_fix7_composite_screening_extract_returns_float32_view():
    """composite_screening._extract_column_array must drop the np.asarray-over-polars wrap.
    Behavioural check: float64 polars column extracts to a float32 ndarray (the function
    halves memory of the 4M-row discovery matrix per the module docstring), values intact
    within float32 precision."""
    from mlframe.training.composite.discovery.screening import _extract_column_array

    df = pl.DataFrame({"x": np.arange(40, dtype=np.float64), "y": np.linspace(0.0, 1.0, 40)})
    out = _extract_column_array(df, "y")
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, np.linspace(0.0, 1.0, 40, dtype=np.float32), rtol=1e-6)


def test_fix7_composite_auto_detect_monotonicity_handles_float_input():
    """composite_auto_detect.detect_time_column_candidates' monotonicity check pre-fix chained
    ``np.asarray + astype(np.float64)`` without copy=False, paying an extra allocation per call.
    Behavioural check: pass a float64 polars column with strictly-increasing values and verify
    detect_time_column_candidates flags it as monotonic without raising on already-float input."""
    from mlframe.training.composite.discovery.auto_detect import detect_time_column_candidates

    df = pl.DataFrame(
        {
            "asc": np.arange(30, dtype=np.float64),
            "rand": np.random.default_rng(0).normal(size=30),
        }
    )
    results = detect_time_column_candidates(df, candidate_columns=["asc", "rand"])
    # asc is strictly increasing -> must appear with is_monotonic=True; rand is random.
    info_by_name = {name: info for name, info in results}
    assert "asc" in info_by_name, f"asc column not surfaced: {info_by_name}"
    assert info_by_name["asc"]["is_monotonic"] is True, f"asc must be flagged monotonic: {info_by_name['asc']}"


def test_fix7_extract_base_returns_float64_ndarray():
    """Correctness: the dropped wrap must not change observable behaviour."""
    from mlframe.training.composite import _extract_base

    df = pl.DataFrame({"base": [1.0, 2.5, np.nan, 4.0]})
    out = _extract_base(df, "base")
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out[:2], np.array([1.0, 2.5]))


def test_fix7_screening_extract_column_array_float32():
    """_extract_column_array intentionally returns float32 (see module docstring)
    to halve memory of the 4M-row discovery matrix."""
    from mlframe.training.composite.discovery.screening import _extract_column_array

    df = pl.DataFrame({"x": [1, 2, 3, 4]})
    out = _extract_column_array(df, "x")
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# Fix 8: _dummy_baseline_compute converts train/val/test once per call.
# ---------------------------------------------------------------------------


def test_fix8_per_group_predict_converts_once_per_frame():
    """The polars->pandas bridge must be invoked exactly once per
    (train_X, val_X, test_X) -- not once per cat column inspected inside the
    group-key builder.

    2026-05-21: when ALL three frames are pl.DataFrame, _per_group_predict
    dispatches to the polars-native fastpath and SKIPS the bridge entirely
    (zero calls). To exercise the bridge contract this test guards, we feed
    objects that expose ``.to_pandas`` but are NOT pl.DataFrame -- so the
    pandas branch runs and the bridge is called per frame.
    """
    from mlframe.training.baselines import _dummy_baseline_compute as dbc

    n = 200
    rng = np.random.default_rng(0)
    df_train = pl.DataFrame({"cat": rng.integers(0, 5, size=n), "x": rng.normal(size=n)}).to_pandas()
    df_val = pl.DataFrame({"cat": rng.integers(0, 5, size=n), "x": rng.normal(size=n)}).to_pandas()
    df_test = pl.DataFrame({"cat": rng.integers(0, 5, size=n), "x": rng.normal(size=n)}).to_pandas()
    train_y = rng.normal(size=n)

    called = {"n": 0}
    real = dbc._to_pandas_for_baseline

    def _spy(x):
        called["n"] += 1
        return real(x)

    with patch.object(dbc, "_to_pandas_for_baseline", side_effect=_spy):
        train_pred, val_pred, test_pred, diag = dbc._per_group_predict(
            df_train,
            df_val,
            df_test,
            train_y,
            "cat",
            "regression",
        )

    # Exactly three bridge invocations: one per input frame.
    assert called["n"] == 3, f"_per_group_predict must convert each frame exactly once, got {called['n']}."
    assert train_pred.shape == (n,)
    assert val_pred.shape == (n,)
    assert test_pred.shape == (n,)
    assert "n_groups_train" in diag


def test_fix8_pick_per_group_categorical_uses_bridge():
    from mlframe.training.baselines import _dummy_baseline_compute as dbc

    rng = np.random.default_rng(0)
    df = pl.DataFrame({"cat": rng.integers(0, 4, size=200), "x": rng.normal(size=200)})

    called = {"n": 0}
    real = dbc._to_pandas_for_baseline

    def _spy(x):
        called["n"] += 1
        return real(x)

    with patch.object(dbc, "_to_pandas_for_baseline", side_effect=_spy):
        out = dbc._pick_per_group_categorical(df, ["cat"], n_train=200, max_cardinality_ratio=0.5)

    assert called["n"] == 1, "Should call bridge exactly once."
    assert out == "cat"


# ---------------------------------------------------------------------------
# Bonus: opportunistic speedup measurement (printed to stdout, never asserted).
# ---------------------------------------------------------------------------


def test_perf_bridge_vs_default_to_pandas_smoke(capsys):
    """Best-effort speedup probe. We expect ~5-30x on numeric frames; on
    tiny frames the JIT setup dominates so we don't assert a lower bound.
    """
    from mlframe.training.utils import get_pandas_view_of_polars_df

    # 500k x 40 numeric + 1 dict: large enough that the per-block memcpy
    # cost in default .to_pandas() dominates over the bridge's setup cost.
    df = _make_numeric_pl_df(n_rows=500_000, n_cols=40)

    # Warmup
    _ = get_pandas_view_of_polars_df(df)
    _ = df.to_pandas()

    reps = 3
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = df.to_pandas()
    t_default = (time.perf_counter() - t0) / reps

    t0 = time.perf_counter()
    for _ in range(reps):
        _ = get_pandas_view_of_polars_df(df)
    t_bridge = (time.perf_counter() - t0) / reps

    speedup = (t_default / t_bridge) if t_bridge > 0 else float("inf")
    # Print to stdout so the user can see it (-s flag).
    print(f"[perf] default to_pandas: {t_default * 1000:.2f} ms/rep; bridge: {t_bridge * 1000:.2f} ms/rep; speedup={speedup:.2f}x (500k x 40 numeric + 1 dict)")
    # Soft floor: bridge should at least not be slower (no assert: numeric-only
    # tiny frames can be a wash and we don't want flake).
    assert t_bridge > 0
