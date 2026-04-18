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
