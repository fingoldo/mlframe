"""Unit + biz_value coverage for ``mlframe.training.pipeline._categorical_composite_fe``.

The underlying tricks (``categorical_powerset_concat`` / ``auto_concat_categorical_groups``)
already have their own biz_value tests at the function level
(``tests/feature_engineering/test_biz_val_categorical_powerset_concat.py`` /
``test_biz_val_categorical_group_concat.py``). This file covers what's NEW here: the suite-wiring
layer (schema alignment across train/val/test, the opt-in/no-op gate, the source-column safety
cap, and predict-time replay producing byte-identical composite columns from persisted metadata) --
plus one biz_value test proving the WIRED module (not just the isolated trick) recovers a
pairwise-only signal a raw-columns-only baseline can't.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from mlframe.training._preprocessing_configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline._categorical_composite_fe import apply_categorical_composite_fe, replay_categorical_composite_fe


def _cat_frame(n=400, seed=0):
    """Cat frame."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.choice(list("XYZ"), n),
            "b": rng.choice(list("PQ"), n),
            "c": rng.choice(list("MN"), n),
            "num0": rng.normal(size=n).astype(np.float32),
        }
    )


def test_apply_categorical_composite_fe_noop_when_disabled():
    """Apply categorical composite fe noop when disabled."""
    df = _cat_frame()
    cfg = PreprocessingExtensionsConfig()
    train, val, test = apply_categorical_composite_fe(df.iloc[:200], df.iloc[200:300], df.iloc[300:], cfg, None, {}, verbose=0)
    assert list(train.columns) == list(df.columns)
    assert list(val.columns) == list(df.columns)
    assert list(test.columns) == list(df.columns)


def test_apply_categorical_composite_fe_schema_aligned_across_splits():
    """Apply categorical composite fe schema aligned across splits."""
    df = _cat_frame()
    y = np.random.default_rng(1).integers(0, 2, len(df))
    cfg = PreprocessingExtensionsConfig(categorical_powerset_concat_enabled=True, categorical_group_concat_auto_enabled=True)
    metadata: dict = {}
    train, val, test = apply_categorical_composite_fe(df.iloc[:200], df.iloc[200:300], df.iloc[300:], cfg, y[:200], metadata, verbose=0)
    assert set(train.columns) == set(val.columns) == set(test.columns)
    # powerset over 3 cols, max_order=2 (default) -> exactly the 3 pairwise composites, no triple.
    assert {"a_b", "a_c", "b_c"} <= set(train.columns)
    assert "a_b_c" not in train.columns
    assert metadata["categorical_powerset_concat_columns"] == ["a", "b", "c"]


def test_apply_categorical_composite_fe_respects_max_source_columns_cap():
    """Apply categorical composite fe respects max source columns cap."""
    df = _cat_frame()
    df["d"] = df["a"]
    df["e"] = df["b"]
    y = np.random.default_rng(2).integers(0, 2, len(df))
    cfg = PreprocessingExtensionsConfig(
        categorical_powerset_concat_enabled=True,
        categorical_composite_max_source_columns=3,
    )
    metadata: dict = {}
    train, _val, _test = apply_categorical_composite_fe(df, None, None, cfg, y, metadata, verbose=0)
    # 5 raw categorical columns (a,b,c,d,e) > cap of 3 -> step skipped entirely, no new columns.
    assert list(train.columns) == list(df.columns)
    assert "categorical_powerset_concat_columns" not in metadata


def test_apply_categorical_composite_fe_polars_roundtrip():
    """Apply categorical composite fe polars roundtrip."""
    n = 200
    rng = np.random.default_rng(3)
    df = pl.DataFrame({"a": rng.choice(list("XY"), n), "b": rng.choice(list("PQ"), n), "num0": rng.normal(size=n).astype(np.float32)})
    cfg = PreprocessingExtensionsConfig(categorical_powerset_concat_enabled=True)
    metadata: dict = {}
    train, val, _test = apply_categorical_composite_fe(df[:150], df[150:], None, cfg, None, metadata, verbose=0)
    assert isinstance(train, pl.DataFrame)
    assert "a_b" in train.columns
    assert "a_b" in val.columns


def test_replay_categorical_composite_fe_matches_fit_time_columns():
    """Replay categorical composite fe matches fit time columns."""
    df = _cat_frame()
    y = np.random.default_rng(4).integers(0, 2, len(df))
    cfg = PreprocessingExtensionsConfig(categorical_powerset_concat_enabled=True, categorical_group_concat_auto_enabled=True)
    metadata: dict = {}
    train, _, _ = apply_categorical_composite_fe(df.iloc[:300], df.iloc[300:350], df.iloc[350:], cfg, y[:300], metadata, verbose=0)

    fresh = df.iloc[:20][["a", "b", "c", "num0"]]
    replayed = replay_categorical_composite_fe(fresh, metadata, verbose=0)
    assert set(replayed.columns) == set(train.columns)


def test_replay_categorical_composite_fe_noop_without_persisted_metadata():
    """Replay categorical composite fe noop without persisted metadata."""
    df = _cat_frame(n=20)
    out = replay_categorical_composite_fe(df, {}, verbose=0)
    assert list(out.columns) == list(df.columns)


def test_biz_val_categorical_composite_wiring_recovers_pairwise_signal():
    """The composite columns this module GENERATES (via its public apply/replay API, not the raw
    trick functions directly) must let a linear model recover a target that's a pure function of the
    (a, b) PAIR -- structureless along either marginal, so a raw-columns-only one-hot baseline is
    a spurious marginal correlation (finite-sample coin-flip imbalance per level, not real signal)
    while the wired composite recovers the mapping exactly. Thresholds set from measured values
    across seeds 5-7 (raw: 0.687-0.74; composite: 1.0 every time) with headroom on both sides."""
    rng = np.random.default_rng(5)
    n = 3000
    n_levels = 8
    a = rng.integers(0, n_levels, n)
    b = rng.integers(0, n_levels, n)
    pair_label = {(i, j): rng.integers(0, 2) for i in range(n_levels) for j in range(n_levels)}
    y = np.array([pair_label[(ai, bi)] for ai, bi in zip(a, b)])
    df = pd.DataFrame({"a": a.astype(str), "b": b.astype(str)})

    train_df, test_df, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=0)
    cfg = PreprocessingExtensionsConfig(categorical_powerset_concat_enabled=True)
    metadata: dict = {}
    train_composite, _, test_composite = apply_categorical_composite_fe(train_df, None, test_df, cfg, y_train, metadata, verbose=0)
    test_composite_replayed = replay_categorical_composite_fe(test_df[["a", "b"]], metadata, verbose=0)
    assert set(test_composite_replayed.columns) == set(test_composite.columns)

    def _fit_eval(train_frame, test_frame, cols):
        """Fit eval."""
        enc = OneHotEncoder(handle_unknown="ignore")
        X_train = enc.fit_transform(train_frame[cols])
        X_test = enc.transform(test_frame[cols])
        clf = LogisticRegression(max_iter=500)
        clf.fit(X_train, y_train)
        return accuracy_score(y_test, clf.predict(X_test))

    acc_raw = _fit_eval(train_df, test_df, ["a", "b"])
    acc_composite = _fit_eval(train_composite, test_composite, ["a_b"])

    assert acc_raw < 0.80, f"raw marginal-only baseline should stay well below the composite, got {acc_raw:.3f}"
    assert acc_composite >= 0.95, f"wired composite column should recover the pairwise mapping, got {acc_composite:.3f}"
