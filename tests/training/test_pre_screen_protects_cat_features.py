"""Sensor: the suite-wide unsupervised pre-screen MUST protect ctx.cat_features
(plus text / embedding / group / ts cols) and MUST protect target columns from
EVERY (target_type, target_name) pair across the suite, not just the first
target's siblings.

Pre-fix shape (stateful side-effects audit P0-2):

  if ctx.cat_features:
      pass                              # &lt;-- DEAD no-op. cat_features unprotected.
  ... compute_unsupervised_drops(protected_columns=_protected)

A near-constant categorical column (e.g. one-hot encoded rare-event flag) was
silently dropped. Multi-target-type suites (regression + binary) also lost
sibling target columns because the protected set only covered the FIRST target's
siblings, not the union across all target_by_type entries.

Post-fix: cat / text / embedding / group / ts cols are explicitly added to the
protected set, AND the protected target set is computed from the full
ctx.target_by_type (suite-wide union).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.pre_screen import compute_unsupervised_drops


def test_compute_unsupervised_drops_respects_protected_set():
    """Baseline: the underlying helper honours the protected set on a numeric column."""
    df = pd.DataFrame(
        {
            "var_col": np.arange(100, dtype=np.float64),  # varying, would be kept
            "const_num": np.full(100, 3.0),  # constant, would normally drop
        }
    )
    drops_unprotected = compute_unsupervised_drops(df, protected_columns=())
    assert "const_num" in drops_unprotected, "baseline: const numeric should be detected"

    # Protect const_num -> stays even though constant.
    drops_protected = compute_unsupervised_drops(df, protected_columns={"const_num"})
    assert "const_num" not in drops_protected, "protected col must survive the screen"


def test_compute_unsupervised_drops_handles_categorical_dtype():
    """REGRESSION: pre-fix np.issubdtype(col.dtype, np.number) raised TypeError on
    pd.CategoricalDtype (it's not a numpy dtype), taking the whole pre-screen pass
    down. Post-fix pd.api.types.is_numeric_dtype handles extension dtypes gracefully."""
    df = pd.DataFrame(
        {
            "var_col": np.arange(100, dtype=np.float64),
            "cat_col": pd.Categorical(["A", "B"] * 50),
            "const_num": np.full(100, 3.0),
        }
    )
    drops = compute_unsupervised_drops(df, protected_columns=())
    assert "const_num" in drops, "const num still detected"
    # Categorical with 2 distinct values is informative, not constant -> kept.
    assert "cat_col" not in drops, "varying Categorical must be kept"


def test_compute_unsupervised_drops_handles_pandas_string_dtype():
    """Pandas StringDtype (extension) must also not crash the pre-screen."""
    df = pd.DataFrame(
        {
            "var_col": np.arange(50, dtype=np.float64),
            "str_col": pd.array(["x", "y"] * 25, dtype=pd.StringDtype()),
        }
    )
    # Must not raise.
    drops = compute_unsupervised_drops(df, protected_columns=())
    # Just sanity that the function returned a list and didn't crash.
    assert isinstance(drops, list)


def test_compute_unsupervised_drops_handles_pandas_nullable_int():
    """Pandas nullable Int extension dtypes: is_numeric_dtype returns True so the
    variance check still runs (and drops a constant Int64 col)."""
    df = pd.DataFrame(
        {
            "const_nullable_int": pd.array([5, 5, 5, 5, 5], dtype="Int64"),
            "var_col": np.arange(5, dtype=np.float64),
        }
    )
    drops = compute_unsupervised_drops(df, protected_columns=())
    assert "const_nullable_int" in drops


class _FSCfg:
    """Groups tests covering f s cfg."""
    pre_screen_unsupervised = True
    pre_screen_variance_threshold = 0.0
    pre_screen_null_fraction_threshold = 0.99


def _make_prescreen_ctx(df, *, cat_features=(), text_features=(), embedding_features=(), target_by_type=None):
    """Minimal ctx exercising the real suite-once pre-screen helper."""
    from types import SimpleNamespace

    ctx = SimpleNamespace(
        feature_selection_config=_FSCfg(),
        _pre_screen_done=False,
        _pre_screen_dropped_cols=None,
        target_by_type=target_by_type or {},
        cat_features=list(cat_features),
        text_features=list(text_features),
        embedding_features=list(embedding_features),
        group_id_col=None,
        ts_field=None,
        extractor=None,
        features_and_targets_extractor=None,
        split_config=None,
        metadata=None,
        verbose=0,
        filtered_train_df=df,
        filtered_val_df=None,
        train_df_pd=df,
        val_df_pd=None,
        test_df_pd=None,
        train_df_polars=None,
        val_df_polars=None,
        test_df_polars=None,
    )
    return ctx


def test_prescreen_protects_cat_text_embedding_and_targets():
    """Run the real suite-once pre-screen helper: near-constant cat/text/embedding columns and
    every suite target column survive, while an unprotected near-constant numeric is dropped.
    Pre-fix the dead ``if ctx.cat_features: pass`` no-op dropped protected categoricals."""
    from mlframe.training.core._phase_train_one_target_pre_screen import (
        _maybe_run_unsupervised_pre_screen,
    )

    df = pd.DataFrame(
        {
            "var_col": np.arange(100, dtype=np.float64),
            "const_num": np.full(100, 3.0),
            "cat_const": pd.Categorical(["X"] * 100),
            "text_const": ["doc"] * 100,
            "emb_const": np.full(100, 7.0),
            "tgt_reg": np.full(100, 1.0),
            "tgt_bin": np.full(100, 0.0),
        }
    )
    ctx = _make_prescreen_ctx(
        df,
        cat_features=["cat_const"],
        text_features=["text_const"],
        embedding_features=["emb_const"],
        target_by_type={"regression": {"tgt_reg": {}}, "binary": {"tgt_bin": {}}},
    )
    _maybe_run_unsupervised_pre_screen(ctx, {})

    dropped = set(ctx._pre_screen_dropped_cols)
    assert "const_num" in dropped, "unprotected near-constant numeric must be dropped"
    for protected in ("cat_const", "text_const", "emb_const", "tgt_reg", "tgt_bin"):
        assert protected not in dropped, f"{protected!r} is protected (cat/text/embedding/suite-target) and must survive the pre-screen"


def test_prescreen_protects_targets_across_all_types_not_just_first():
    """Suite-once pre-screen must protect the target column of EVERY target_type, not just the
    first iterated one (pre-fix used the type-scoped local targets and dropped sibling targets)."""
    from mlframe.training.core._phase_train_one_target_pre_screen import (
        _maybe_run_unsupervised_pre_screen,
    )

    df = pd.DataFrame(
        {
            "var_col": np.arange(50, dtype=np.float64),
            "tgt_a": np.full(50, 1.0),  # near-constant target of type A
            "tgt_b": np.full(50, 0.0),  # near-constant target of type B
        }
    )
    # ``targets`` arg covers only type A; target_by_type union must still protect tgt_b.
    ctx = _make_prescreen_ctx(
        df,
        target_by_type={"regression": {"tgt_a": {}}, "binary": {"tgt_b": {}}},
    )
    _maybe_run_unsupervised_pre_screen(ctx, {"tgt_a": {}})
    dropped = set(ctx._pre_screen_dropped_cols)
    assert "tgt_b" not in dropped, "sibling-type target must be protected via the target_by_type union"
    assert "tgt_a" not in dropped
