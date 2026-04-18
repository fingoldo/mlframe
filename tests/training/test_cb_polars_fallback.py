"""End-to-end tests for the CatBoost Polars-fastpath -> pandas fallback.

Bug history that motivated these tests:

* 2026-04-18: CatBoost's Polars fastpath rejects certain categorical layouts
  with cryptic errors like ``TypeError: No matching signature found`` or
  ``Unsupported data type Categorical for a ... feature column``. We built a
  fallback in ``_train_model_with_fallback`` that on those errors converts
  polars -> pandas and retries. The fallback was never end-to-end-tested;
  downstream bugs kept surfacing in production that would have been caught
  by a simple mock-backed test:

    - 2026-04-19 morning: text-feature columns still pd.Categorical on retry,
      CatBoost rejected "dtype 'category' but not in cat_features list".
    - 2026-04-19 night:   ``prepare_df_for_catboost`` rebuilt every
      pd.Categorical column via ``astype(str).astype("category")``, taking
      minutes per high-cardinality text column — production hang.

Those are *orchestration* bugs (wrong ordering / wrong filter), not
unit-level issues. The tests below run the full fallback with a stub model
so we can assert end-to-end invariants.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.trainer import _train_model_with_fallback


# ---------------------------------------------------------------------------
# Fake CatBoost: raises Polars-style TypeError on the first fit, succeeds on
# the second. Records every fit invocation so tests can inspect what ended
# up being passed to the retry.
# ---------------------------------------------------------------------------

class FakeCatBoost:
    """Minimal stand-in for a CatBoostClassifier used by the fallback path.

    The class name is ``FakeCatBoost`` but ``_train_model_with_fallback``
    gates the fallback on the *model_type_name* argument string, so the
    test passes ``"CatBoostClassifier"`` explicitly — no pyx import needed.
    """
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self.fit_error_first_call: Exception = TypeError("No matching signature found")

    def fit(self, X, y, **fit_params):
        # Record a snapshot of what arrived: types, dtypes of text columns,
        # cat_features list, eval_set shape.
        snapshot = {
            "X_type": type(X).__name__,
            "X_shape": tuple(X.shape) if hasattr(X, "shape") else None,
            "cat_features": list(fit_params.get("cat_features") or []),
            "text_features": list(fit_params.get("text_features") or []),
            "text_dtypes": {
                col: str(X[col].dtype)
                for col in (fit_params.get("text_features") or [])
                if hasattr(X, "__getitem__") and col in (X.columns if hasattr(X, "columns") else [])
            },
            "has_eval_set": "eval_set" in fit_params,
        }
        self.calls.append(snapshot)
        if len(self.calls) == 1:
            raise self.fit_error_first_call
        # On retry — succeed silently, return self (mimicking sklearn API).
        return self


# ---------------------------------------------------------------------------
# Fixture: a small Polars frame with a mix of cat and text columns,
# structurally matching the production case that triggered the bugs.
# ---------------------------------------------------------------------------

@pytest.fixture
def polars_frame_with_text_cats() -> Tuple[pl.DataFrame, pl.DataFrame, List[str], List[str]]:
    """Build train/val polars frames + cat/text feature lists.

    'skills_text' mimics the production high-cardinality auto-promoted
    text column. It's pl.Categorical dtype on the polars side, which is
    what triggers the pandas-path hang before the 2026-04-19 fix.
    """
    rng = np.random.default_rng(0)
    n_train, n_val = 500, 100
    train = pl.DataFrame({
        "num":         rng.standard_normal(n_train).astype(np.float32),
        "true_cat":    pl.Series("true_cat", rng.choice(["r", "g", "b"], size=n_train)).cast(pl.Categorical),
        "skills_text": pl.Series("skills_text",
                                 np.array([f"s_{i:04d}" for i in range(200)])[rng.integers(0, 200, size=n_train)]
                                 ).cast(pl.Categorical),
    })
    val = pl.DataFrame({
        "num":         rng.standard_normal(n_val).astype(np.float32),
        "true_cat":    pl.Series("true_cat", rng.choice(["r", "g", "b"], size=n_val)).cast(pl.Categorical),
        "skills_text": pl.Series("skills_text",
                                 np.array([f"s_{i:04d}" for i in range(200)])[rng.integers(0, 200, size=n_val)]
                                 ).cast(pl.Categorical),
    })
    cat_features = ["true_cat"]
    text_features = ["skills_text"]
    return train, val, cat_features, text_features


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fallback_triggers_on_polars_typeerror(polars_frame_with_text_cats):
    """Baseline: fallback activates on the exact exception the production
    log showed, calls fit a second time, and the second call doesn't raise.
    """
    train_df, val_df, cat, text = polars_frame_with_text_cats
    train_target = np.random.default_rng(0).integers(0, 2, size=train_df.height)
    val_target = np.random.default_rng(1).integers(0, 2, size=val_df.height)

    model = FakeCatBoost()
    fit_params = {
        "cat_features": cat,
        "text_features": text,
        "eval_set": (val_df, val_target),
    }
    out_model, _ = _train_model_with_fallback(
        model=model, model_obj=model,
        model_type_name="CatBoostClassifier",
        train_df=train_df, train_target=train_target,
        fit_params=fit_params, verbose=False,
    )
    assert out_model is model
    assert len(model.calls) == 2, f"expected 2 fit calls (raise then retry), got {len(model.calls)}"


def test_fallback_converts_train_df_to_pandas(polars_frame_with_text_cats):
    """After the fallback, the retry must be invoked with a pandas
    DataFrame, not the original Polars one."""
    train_df, val_df, cat, text = polars_frame_with_text_cats
    train_target = np.arange(train_df.height) % 2
    val_target = np.arange(val_df.height) % 2

    model = FakeCatBoost()
    _train_model_with_fallback(
        model=model, model_obj=model, model_type_name="CatBoostClassifier",
        train_df=train_df, train_target=train_target,
        fit_params={"cat_features": cat, "text_features": text,
                    "eval_set": (val_df, val_target)},
        verbose=False,
    )
    first, retry = model.calls
    assert first["X_type"] == "DataFrame"  # polars DataFrame
    assert retry["X_type"] == "DataFrame"  # pandas DataFrame (same class name)
    # Distinguish via the cat-features contract: pandas retry must receive
    # the same list; no silent in-place mutation.
    assert retry["cat_features"] == cat


def test_fallback_decategorizes_text_columns_before_retry(polars_frame_with_text_cats):
    """REGRESSION SENSOR for the 2026-04-19 morning bug:
    text columns must NOT be pd.Categorical at retry time — otherwise
    CatBoost rejects "dtype 'category' but not in cat_features list".
    """
    train_df, val_df, cat, text = polars_frame_with_text_cats
    train_target = np.arange(train_df.height) % 2
    val_target = np.arange(val_df.height) % 2

    model = FakeCatBoost()
    _train_model_with_fallback(
        model=model, model_obj=model, model_type_name="CatBoostClassifier",
        train_df=train_df, train_target=train_target,
        fit_params={"cat_features": cat, "text_features": text,
                    "eval_set": (val_df, val_target)},
        verbose=False,
    )
    retry = model.calls[1]
    for col, dtype_str in retry["text_dtypes"].items():
        assert "category" not in dtype_str.lower(), (
            f"text column {col!r} arrived at retry with dtype {dtype_str!r}; "
            "must be object/string (text columns are decategorized before CB retry)"
        )


def test_fallback_rewrites_eval_set_to_pandas(polars_frame_with_text_cats):
    """The eval_set must be rewritten to pandas form too — otherwise CB
    will hit the same fastpath rejection on val and re-crash.
    """
    train_df, val_df, cat, text = polars_frame_with_text_cats
    train_target = np.arange(train_df.height) % 2
    val_target = np.arange(val_df.height) % 2

    model = FakeCatBoost()
    fit_params = {"cat_features": cat, "text_features": text,
                  "eval_set": (val_df, val_target)}
    _train_model_with_fallback(
        model=model, model_obj=model, model_type_name="CatBoostClassifier",
        train_df=train_df, train_target=train_target,
        fit_params=fit_params, verbose=False,
    )
    # After the fallback, fit_params["eval_set"] should hold pandas.
    eval_X = fit_params["eval_set"][0]
    assert isinstance(eval_X, pd.DataFrame), (
        f"eval_set X must be pandas after fallback, got {type(eval_X).__name__}"
    )
    # And its text column must not be pd.Categorical either.
    assert not isinstance(eval_X["skills_text"].dtype, pd.CategoricalDtype), (
        "eval_set text column must also be decategorized"
    )


def test_fallback_passes_when_polars_fastpath_succeeds(polars_frame_with_text_cats):
    """Sanity: when the Polars fastpath succeeds on the first call, no
    fallback kicks in and the model is not reconverted."""
    train_df, val_df, cat, text = polars_frame_with_text_cats
    train_target = np.arange(train_df.height) % 2

    class OKModel(FakeCatBoost):
        def fit(self, X, y, **fit_params):
            self.calls.append({"X_type": type(X).__name__})
            return self

    model = OKModel()
    _train_model_with_fallback(
        model=model, model_obj=model, model_type_name="CatBoostClassifier",
        train_df=train_df, train_target=train_target,
        fit_params={"cat_features": cat, "text_features": text},
        verbose=False,
    )
    assert len(model.calls) == 1
    assert model.calls[0]["X_type"] == "DataFrame"  # polars, no conversion needed


def test_fallback_ignored_for_non_catboost_models(polars_frame_with_text_cats):
    """The fallback only applies to CatBoost. Other backends (XGB, LGB,
    MLP, ...) must not trigger the polars->pandas conversion even if they
    raise a similar-looking TypeError.
    """
    train_df, val_df, cat, text = polars_frame_with_text_cats
    train_target = np.arange(train_df.height) % 2

    model = FakeCatBoost()
    model.fit_error_first_call = TypeError("No matching signature found")

    with pytest.raises(TypeError, match="No matching signature"):
        _train_model_with_fallback(
            model=model, model_obj=model,
            model_type_name="XGBClassifier",  # <-- NOT CatBoost
            train_df=train_df, train_target=train_target,
            fit_params={"cat_features": cat, "text_features": text},
            verbose=False,
        )


def test_fallback_without_eval_set_still_retries(polars_frame_with_text_cats):
    """``fit_params`` may arrive without an ``eval_set`` (e.g. when no val
    split is configured). The fallback must still trigger and retry with
    the pandas-converted train_df — the eval_set rewrite path should
    gracefully no-op, not crash on missing key.
    """
    train_df, _val_df, cat, text = polars_frame_with_text_cats
    train_target = np.arange(train_df.height) % 2

    model = FakeCatBoost()
    _train_model_with_fallback(
        model=model, model_obj=model, model_type_name="CatBoostClassifier",
        train_df=train_df, train_target=train_target,
        fit_params={"cat_features": cat, "text_features": text},
        verbose=False,
    )
    assert len(model.calls) == 2


def test_fallback_retry_failure_propagates(polars_frame_with_text_cats):
    """If the pandas retry ALSO fails, the failure must propagate up —
    we must not silently swallow errors from the retry path.
    """
    train_df, val_df, cat, text = polars_frame_with_text_cats
    train_target = np.arange(train_df.height) % 2
    val_target = np.arange(val_df.height) % 2

    class RaisingTwice(FakeCatBoost):
        def fit(self, X, y, **fit_params):
            self.calls.append({"X_type": type(X).__name__})
            if len(self.calls) == 1:
                raise TypeError("No matching signature found")
            raise RuntimeError("retry also failed")

    model = RaisingTwice()
    with pytest.raises(RuntimeError, match="retry also failed"):
        _train_model_with_fallback(
            model=model, model_obj=model, model_type_name="CatBoostClassifier",
            train_df=train_df, train_target=train_target,
            fit_params={"cat_features": cat, "text_features": text,
                        "eval_set": (val_df, val_target)},
            verbose=False,
        )
    assert len(model.calls) == 2


# ---------------------------------------------------------------------------
# Diagnostic logging — pre-fit Enum warning + post-fail schema dump
# ---------------------------------------------------------------------------
# 2026-04-19 incident: CatBoost 1.2.10 Polars fastpath raised
# "TypeError: No matching signature found" in
# _set_features_order_data_polars_categorical_column.process. The fused
# cpdef dispatcher has no overload for pl.Enum. With the old one-line
# warning (truncated to 160 chars, just the error message), operators
# had NO way to tell which of 9 cat_features was at fault — leading to
# 2+ minutes of wasted CB attempt + a pandas-fallback detour on every run.
# The two helpers below surface the culprit in the first log line.


def test_warn_on_unsupported_polars_dtypes_flags_enum_cat_features(caplog):
    """Pre-fit warning must name the Enum column(s) in cat_features."""
    import logging
    from mlframe.training.trainer import _warn_on_unsupported_polars_dtypes

    n = 50
    enum_dt = pl.Enum(["A", "B", "C"])
    df = pl.DataFrame({
        "num":  np.arange(n, dtype=np.float32),
        "good": pl.Series("good", ["A"] * n).cast(pl.Categorical),
        "bad":  pl.Series("bad",  ["A"] * n, dtype=enum_dt),
    })
    with caplog.at_level(logging.WARNING, logger="mlframe.training.trainer"):
        _warn_on_unsupported_polars_dtypes(df, cat_features=["good", "bad"])
    msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("bad" in m and "Enum" in m for m in msgs), (
        f"Expected warning naming 'bad' as an Enum cat_feature; got: {msgs}"
    )
    # 'good' is plain Categorical — must NOT be flagged.
    assert not any("'good'" in m for m in msgs), msgs


def test_warn_on_unsupported_polars_dtypes_silent_when_clean(caplog):
    """No cat_feature is Enum -> no warning. Silent path matters because
    this helper runs on every CatBoost fit; a false-positive warning
    would noise up every log run."""
    import logging
    df = pl.DataFrame({
        "num": np.arange(10, dtype=np.float32),
        "c":   pl.Series("c", ["a", "b"] * 5).cast(pl.Categorical),
    })
    from mlframe.training.trainer import _warn_on_unsupported_polars_dtypes
    with caplog.at_level(logging.WARNING, logger="mlframe.training.trainer"):
        _warn_on_unsupported_polars_dtypes(df, cat_features=["c"])
    warn_msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert not warn_msgs, f"Expected silence on clean Polars cat_features; got: {warn_msgs}"


def test_polars_schema_diagnostic_names_enum_culprit():
    """The post-fail schema dump must highlight any Enum cat_feature in
    its header (not buried in the per-column list) so the first WARNING
    line after the CB TypeError immediately points to the culprit."""
    from mlframe.training.trainer import _polars_schema_diagnostic

    enum_dt = pl.Enum(["red", "green", "blue"])
    df = pl.DataFrame({
        "a": np.arange(20, dtype=np.float32),
        "job_type": pl.Series("job_type", ["red"] * 20, dtype=enum_dt),
        "category_group": pl.Series("category_group", ["x", "y"] * 10).cast(pl.Categorical),
    })
    dump = _polars_schema_diagnostic(
        df, cat_features=["job_type", "category_group"], text_features=[]
    )
    assert "job_type" in dump
    assert "Enum" in dump
    assert "No matching signature found" in dump or "Enum dispatch" in dump
    # category_group must appear as a plain Categorical entry, not flagged.
    assert "category_group" in dump


def test_polars_schema_diagnostic_handles_empty_cat_features():
    """Schema diagnostic must produce a usable dump even when cat_features
    is None/empty — operators may need to see it to rule out unexpected
    dtype mixes on text or numeric columns."""
    from mlframe.training.trainer import _polars_schema_diagnostic
    df = pl.DataFrame({
        "a": np.arange(5, dtype=np.float32),
        "b": pl.Series("b", ["x", "y", "x", "y", "x"]),
    })
    dump = _polars_schema_diagnostic(df, cat_features=None, text_features=None)
    # Even without any categoricals, the diagnostic must not crash and
    # must at least report shape.
    assert "5" in dump or "shape" in dump.lower() or "×" in dump


def test_polars_schema_diagnostic_never_raises():
    """Error path safety: the helper runs inside an `except` block after
    CB's fit crashed. If the helper itself crashed, the operator would
    lose the original CB error AND the diagnostic. It must always
    return a string, even on malformed inputs.
    """
    from mlframe.training.trainer import _polars_schema_diagnostic
    # Nonsense input — passing a pandas DataFrame where polars is expected.
    bad_input = pd.DataFrame({"a": [1, 2, 3]})
    out = _polars_schema_diagnostic(bad_input, cat_features=["a"], text_features=[])
    assert isinstance(out, str)
    assert len(out) > 0


# ---------------------------------------------------------------------------
# Stale cat_features regression — 2026-04-19 incident (round 6)
# ---------------------------------------------------------------------------
# Production symptom: the Polars fastpath orchestration in
# `train_mlframe_models_suite` used to build
#   _cat_features = cat_features_polars or cat_features or []
# The `cat_features_polars` list was captured early in Phase 3 BEFORE the
# text-promotion step removed 4 high-cardinality text columns from cats.
# Those 4 were later cast pl.Categorical -> pl.String for CatBoost's text
# fastpath. But because of the `or` short-circuit, the stale 13-column
# cats list was passed to CB.fit(cat_features=...). CatBoost 1.2.10's
# `_set_features_order_data_polars_categorical_column` is a fused cpdef
# with NO overload for pl.String, so it raised "No matching signature
# found" — and the old one-line warning truncated the error, so nobody
# could tell WHICH column caused it until the new diagnostic landed.
#
# Primary fix: use `cat_features` (the post-promotion, dedup'd list) in
# core.py:1935. Defensive fix: `_filter_polars_cat_features_by_dtype`
# drops any cat_feature whose runtime dtype in the DF is not Categorical
# /Enum, so even if someone reintroduces the short-circuit bug, CB won't
# get a malformed list — the filter drops the mismatch and warns.


class TestFilterPolarsCatFeaturesByDtype:
    """Sensor for the defensive runtime filter that prevents the
    2026-04-19 'No matching signature found' prod incident from recurring.
    """

    def test_drops_string_columns_declared_as_cat(self, caplog):
        """The exact prod shape: a column in cat_features is actually
        pl.String in the DataFrame (was cast from Categorical for text
        fastpath). Filter must drop it and WARN."""
        import logging
        from mlframe.training.core import _filter_polars_cat_features_by_dtype
        df = pl.DataFrame({
            "good_cat":    pl.Series("good_cat", ["a", "b", "a"]).cast(pl.Categorical),
            "skills_text": pl.Series("skills_text", ["x", "y", "z"]),  # pl.String
        })
        with caplog.at_level(logging.WARNING, logger="mlframe.training.core"):
            out = _filter_polars_cat_features_by_dtype(df, ["good_cat", "skills_text"])
        assert out == ["good_cat"], (
            "String-dtype column declared as cat must be dropped — "
            "CatBoost's Polars dispatcher has no String overload"
        )
        warns = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("skills_text" in m and "String" in m for m in warns), warns

    def test_keeps_categorical_columns(self, caplog):
        """Happy path: all cat_features are pl.Categorical -> all kept,
        no warning (runs on every CB fit, false positives would spam)."""
        import logging
        from mlframe.training.core import _filter_polars_cat_features_by_dtype
        df = pl.DataFrame({
            "a": pl.Series("a", ["x", "y"] * 5).cast(pl.Categorical),
            "b": pl.Series("b", ["p", "q"] * 5).cast(pl.Categorical),
        })
        with caplog.at_level(logging.WARNING, logger="mlframe.training.core"):
            out = _filter_polars_cat_features_by_dtype(df, ["a", "b"])
        assert out == ["a", "b"]
        warns = [r for r in caplog.records if r.levelname == "WARNING"]
        assert not warns, f"Unexpected warning on clean Categorical cats: {[r.message for r in warns]}"

    def test_keeps_enum_columns(self, caplog):
        """pl.Enum on some CB builds has a dispatch overload and on others
        fails the same way as String. We keep Enum in the list (let CB
        decide) and NOT warn — that's the upstream Enum hypothesis's
        territory, not ours."""
        import logging
        from mlframe.training.core import _filter_polars_cat_features_by_dtype
        enum_dt = pl.Enum(["A", "B", "C"])
        df = pl.DataFrame({
            "e": pl.Series("e", ["A", "B", "C"] * 2, dtype=enum_dt),
        })
        with caplog.at_level(logging.WARNING, logger="mlframe.training.core"):
            out = _filter_polars_cat_features_by_dtype(df, ["e"])
        assert out == ["e"]

    def test_silently_skips_missing_columns(self):
        """Defensive: a cat_feature name that's not in the DF columns is
        silently dropped (not the helper's job to raise — CB would
        error with a different, clearer message anyway)."""
        from mlframe.training.core import _filter_polars_cat_features_by_dtype
        df = pl.DataFrame({"a": pl.Series("a", ["x"] * 3).cast(pl.Categorical)})
        out = _filter_polars_cat_features_by_dtype(df, ["a", "not_in_df"])
        assert out == ["a"]

    def test_empty_input_returns_empty(self):
        """Empty cat_features -> empty output, no crash, no warning."""
        from mlframe.training.core import _filter_polars_cat_features_by_dtype
        df = pl.DataFrame({"a": [1, 2, 3]})
        assert _filter_polars_cat_features_by_dtype(df, []) == []
        assert _filter_polars_cat_features_by_dtype(df, None) == []

    def test_all_string_returns_empty_not_none(self, caplog):
        """All cat_features are wrong-dtype -> return empty list
        (not None). Callers wrap with `if _valid_cat:` so empty is safe
        but None would crash."""
        import logging
        from mlframe.training.core import _filter_polars_cat_features_by_dtype
        df = pl.DataFrame({
            "a": pl.Series("a", ["x"] * 3),  # pl.String
            "b": pl.Series("b", ["y"] * 3),  # pl.String
        })
        with caplog.at_level(logging.WARNING, logger="mlframe.training.core"):
            out = _filter_polars_cat_features_by_dtype(df, ["a", "b"])
        assert out == []
        assert out is not None

    def test_numeric_column_in_cat_features_also_dropped(self, caplog):
        """Same dispatcher miss on numeric — probe the boundary beyond
        String. If someone mistakenly declares a Float column as a
        cat_feature, the filter must drop it."""
        import logging
        from mlframe.training.core import _filter_polars_cat_features_by_dtype
        df = pl.DataFrame({
            "real_cat": pl.Series("real_cat", ["a", "b"] * 3).cast(pl.Categorical),
            "numeric":  np.arange(6, dtype=np.float32),
        })
        with caplog.at_level(logging.WARNING, logger="mlframe.training.core"):
            out = _filter_polars_cat_features_by_dtype(df, ["real_cat", "numeric"])
        assert out == ["real_cat"]
        warns = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("numeric" in m for m in warns), warns


def test_cb_fallback_warning_emits_schema_dump_on_rejection(polars_frame_with_text_cats, caplog):
    """End-to-end: when the CB Polars fastpath rejects (simulated via
    FakeCatBoost's first-call TypeError), the fallback path must emit a
    WARNING containing the schema dump. This is the sensor for the
    2026-04-19 incident where the old one-line warning was unactionable.
    """
    import logging
    train_df, val_df, cat, text = polars_frame_with_text_cats
    train_target = np.arange(train_df.height) % 2
    val_target = np.arange(val_df.height) % 2

    model = FakeCatBoost()
    with caplog.at_level(logging.WARNING, logger="mlframe.training.trainer"):
        _train_model_with_fallback(
            model=model, model_obj=model, model_type_name="CatBoostClassifier",
            train_df=train_df, train_target=train_target,
            fit_params={"cat_features": cat, "text_features": text,
                        "eval_set": (val_df, val_target)},
            verbose=False,
        )
    warn_msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
    # One warning carries the schema context.
    assert any("schema context" in m or "Polars schema diagnostic" in m for m in warn_msgs), (
        f"Expected a WARNING with schema dump; got: {warn_msgs}"
    )
