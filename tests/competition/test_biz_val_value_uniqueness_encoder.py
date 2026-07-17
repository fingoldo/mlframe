"""Unit + biz_value tests for mlframe.competition.value_uniqueness_encoder.

COMPETITION/EXPLORATORY ONLY — see module docstring under src/mlframe/competition/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from mlframe.competition.value_uniqueness_encoder import (
    REPEATS_MIXED_TARGET,
    REPEATS_ONLY_TARGET_1,
    REPEATS_UNKNOWN_TARGET,
    UNIQUE_GLOBALLY,
    value_uniqueness_encoder,
)


def _make_informative_dataset(n_train: int = 4000, n_test: int = 1000, seed: int = 0):
    """Build a dataset where a column's uniqueness pattern is genuinely predictive.

    "informative" values only ever co-occur with target=1 (or only target=0); "noise"
    values repeat with mixed targets or are unique and carry no signal on their own.
    The raw column values are random-looking floats, so a model using only the raw
    numeric feature gets near-zero signal - the encoder must recover the pattern.
    """
    rng = np.random.default_rng(seed)

    n_informative_1 = 60
    n_mixed = 100

    # all value pools share the SAME numeric range so the raw scalar carries no
    # linearly-separable signal on its own - only the (hidden) identity/repetition
    # pattern of the specific value distinguishes them.
    informative_1_vals = rng.uniform(0, 1, n_informative_1)
    informative_0_vals = rng.uniform(0, 1, n_informative_1)
    mixed_vals = rng.uniform(0, 1, n_mixed)

    n_total = n_train + n_test
    col_values = np.empty(n_total, dtype=float)
    target = np.empty(n_total, dtype=int)

    for i in range(n_total):
        r = rng.random()
        if r < 0.30:
            # value that only ever appears with target=1
            col_values[i] = rng.choice(informative_1_vals)
            target[i] = 1
        elif r < 0.60:
            # value that only ever appears with target=0
            col_values[i] = rng.choice(informative_0_vals)
            target[i] = 0
        elif r < 0.85:
            # mixed value: repeats, but with random target (no info)
            col_values[i] = rng.choice(mixed_vals)
            target[i] = int(rng.random() < 0.5)
        else:
            # globally unique noise value, random target
            col_values[i] = rng.uniform(0, 1)
            target[i] = int(rng.random() < 0.5)

    df = pd.DataFrame({"feat": col_values, "target": target})
    train_df = df.iloc[:n_train].reset_index(drop=True)
    test_df = df.iloc[n_train:].reset_index(drop=True)

    y_train = train_df["target"].to_numpy()
    y_test = test_df["target"].to_numpy()
    train_df = train_df.drop(columns=["target"])
    test_df = test_df.drop(columns=["target"])

    return train_df, test_df, y_train, y_test


def test_value_uniqueness_encoder_basic_categories():
    train = pd.DataFrame({"c": [1, 1, 2, 2, 3]})
    test = pd.DataFrame({"c": [1, 4, 4, 5]})
    y_train = np.array([1, 1, 0, 1, 0])  # value 1 -> only target 1; value 2 -> mixed; value 3 -> unique

    out = value_uniqueness_encoder(train, test, real_test_mask=None, y_train=y_train, columns=["c"])

    train_flags = out["c__value_uniqueness"].to_numpy()[: len(train)]
    test_flags = out["c__value_uniqueness"].to_numpy()[len(train) :]

    assert train_flags[0] == REPEATS_ONLY_TARGET_1
    assert train_flags[1] == REPEATS_ONLY_TARGET_1
    assert train_flags[2] == REPEATS_MIXED_TARGET
    assert train_flags[3] == REPEATS_MIXED_TARGET
    assert train_flags[4] == UNIQUE_GLOBALLY

    # value 1 was seen in train (always target=1) -> inherits the train-derived flag
    # value 4 is novel to test and repeats (count 2) -> count-only flag, no target info
    # value 5 is novel to test and unique -> unique_globally
    assert test_flags[0] == REPEATS_ONLY_TARGET_1
    assert test_flags[1] == REPEATS_UNKNOWN_TARGET
    assert test_flags[2] == REPEATS_UNKNOWN_TARGET
    assert test_flags[3] == UNIQUE_GLOBALLY


def test_value_uniqueness_encoder_respects_real_test_mask():
    train = pd.DataFrame({"c": [1, 1]})
    test = pd.DataFrame({"c": [9, 9, 9]})
    real_mask = np.array([True, False, False])  # only first "9" is real; other two are synthetic
    y_train = np.array([1, 0])

    out = value_uniqueness_encoder(train, test, real_test_mask=real_mask, y_train=y_train, columns=["c"])
    test_flags = out["c__value_uniqueness"].to_numpy()[len(train) :]

    # counted only among real rows -> value 9 appears once among real rows -> unique
    assert (test_flags == UNIQUE_GLOBALLY).all()


def test_value_uniqueness_encoder_reuses_train_flag_without_leaking_test_targets():
    """A test-row value seen in train inherits the train-derived flag (not leakage: that flag was
    built purely from train's own y_train; test's own labels are never consulted). A test-row value
    NEVER seen in train must fall back to a count-only flag with no target information."""
    train = pd.DataFrame({"c": [1, 1, 2]})
    test = pd.DataFrame({"c": [1, 1, 3, 3]})
    y_train = np.array([1, 1, 0])

    out = value_uniqueness_encoder(train, test, real_test_mask=None, y_train=y_train, columns=["c"])
    test_flags = out["c__value_uniqueness"].to_numpy()[len(train) :]

    # value 1 was seen in train, always with target=1 -> inherited flag
    assert test_flags[0] == REPEATS_ONLY_TARGET_1
    assert test_flags[1] == REPEATS_ONLY_TARGET_1
    # value 3 was never seen in train -> falls back to test-only count-based flag, no target info
    assert test_flags[2] == REPEATS_UNKNOWN_TARGET
    assert test_flags[3] == REPEATS_UNKNOWN_TARGET


def test_biz_val_value_uniqueness_encoder_auc_improvement_over_raw_feature():
    train, test, y_train, y_test = _make_informative_dataset()

    encoded = value_uniqueness_encoder(train, test, real_test_mask=None, y_train=y_train, columns=["feat"])
    train_encoded_col = encoded["feat__value_uniqueness"].to_numpy()[: len(train)]
    test_encoded_col = encoded["feat__value_uniqueness"].to_numpy()[len(train) :]

    # baseline: raw numeric feature only
    baseline_train_X = train[["feat"]].to_numpy()
    baseline_test_X = test[["feat"]].to_numpy()
    baseline_model = LogisticRegression(max_iter=1000)
    baseline_model.fit(baseline_train_X, y_train)
    baseline_auc = roc_auc_score(y_test, baseline_model.predict_proba(baseline_test_X)[:, 1])

    # encoded: one-hot of the uniqueness flag, plus raw feature
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_ohe = ohe.fit_transform(train_encoded_col.reshape(-1, 1))
    test_ohe = ohe.transform(test_encoded_col.reshape(-1, 1))

    encoded_train_X = np.hstack([baseline_train_X, train_ohe])
    encoded_test_X = np.hstack([baseline_test_X, test_ohe])
    encoded_model = LogisticRegression(max_iter=1000)
    encoded_model.fit(encoded_train_X, y_train)
    encoded_auc = roc_auc_score(y_test, encoded_model.predict_proba(encoded_test_X)[:, 1])

    # measured (seed=0): baseline_auc ~0.559 (raw floats carry near-zero signal by construction),
    # encoded_auc ~0.899 (uniqueness/target-co-occurrence flag recovers the hidden pattern).
    # Thresholds set with margin around the measured values.
    assert baseline_auc < 0.62
    assert encoded_auc >= 0.80
    assert encoded_auc - baseline_auc >= 0.25


def test_biz_val_value_uniqueness_encoder_stratified_split_reproduces_signal():
    """Same informative pattern, but sourced from a single frame + stratified split, to rule out seed luck."""
    train, test, y_train, y_test = _make_informative_dataset(n_train=3000, n_test=800, seed=7)

    full = pd.concat([train, test], ignore_index=True)
    full["target"] = np.concatenate([y_train, y_test])
    tr_idx, te_idx = train_test_split(np.arange(len(full)), test_size=0.25, stratify=full["target"], random_state=1)

    train2 = full.iloc[tr_idx][["feat"]].reset_index(drop=True)
    test2 = full.iloc[te_idx][["feat"]].reset_index(drop=True)
    y_train2 = full.iloc[tr_idx]["target"].to_numpy()
    y_test2 = full.iloc[te_idx]["target"].to_numpy()

    encoded = value_uniqueness_encoder(train2, test2, real_test_mask=None, y_train=y_train2, columns=["feat"])
    train_encoded_col = encoded["feat__value_uniqueness"].to_numpy()[: len(train2)]
    test_encoded_col = encoded["feat__value_uniqueness"].to_numpy()[len(train2) :]

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_ohe = ohe.fit_transform(train_encoded_col.reshape(-1, 1))
    test_ohe = ohe.transform(test_encoded_col.reshape(-1, 1))

    model = LogisticRegression(max_iter=1000)
    model.fit(train_ohe, y_train2)
    auc = roc_auc_score(y_test2, model.predict_proba(test_ohe)[:, 1])

    assert auc >= 0.70
