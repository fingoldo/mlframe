"""biz_value test for ``feature_engineering.boolean_pair_interactions.boolean_pair_interactions``.

The win: XOR has no continuous-arithmetic equivalent already in mlframe's pairwise recipe set (``mul``/
``maxab``/``minab`` coincide with AND/OR for {0,1} inputs by coincidence, but nothing computes XOR). A target
that depends on the XOR of two binary symptom flags (present precisely when EXACTLY ONE of two conditions
holds -- a classic "exclusive or" diagnostic pattern) is UNLEARNABLE by a linear model from the two raw binary
columns alone (their individual coefficients can't express the interaction), but trivially learnable once the
XOR interaction column is added.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from mlframe.feature_engineering.boolean_pair_interactions import boolean_pair_interactions, is_binary_column


def _make_xor_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    symptom_a = rng.integers(0, 2, n)
    symptom_b = rng.integers(0, 2, n)
    noise_flag = rng.integers(0, 2, n)
    y = (symptom_a ^ symptom_b).astype(int)  # positive exactly when exactly one symptom present
    df = pd.DataFrame({"symptom_a": symptom_a, "symptom_b": symptom_b, "noise_flag": noise_flag})
    return df, y


def test_biz_val_xor_interaction_makes_unlearnable_target_learnable():
    df, y = _make_xor_dataset(n=2000, seed=0)

    clf_raw = LogisticRegression(max_iter=500).fit(df, y)
    auc_raw = roc_auc_score(y, clf_raw.predict_proba(df)[:, 1])

    interactions = boolean_pair_interactions(df, columns=["symptom_a", "symptom_b", "noise_flag"], operators=("and", "or", "xor"))
    df_aug = pd.concat([df, interactions], axis=1)
    clf_aug = LogisticRegression(max_iter=500).fit(df_aug, y)
    auc_aug = roc_auc_score(y, clf_aug.predict_proba(df_aug)[:, 1])

    assert auc_raw < 0.6, f"expected the raw binary columns alone to be near-unlearnable for an XOR target, got AUC={auc_raw:.4f}"
    assert auc_aug > 0.98, f"expected the XOR-augmented feature set to make the target trivially learnable, got AUC={auc_aug:.4f}"


def test_boolean_pair_interactions_output_shape_and_values():
    df = pd.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]})
    out = boolean_pair_interactions(df, columns=["a", "b"], operators=("and", "or", "xor"))
    assert list(out.columns) == ["a__and__b", "a__or__b", "a__xor__b"]
    np.testing.assert_array_equal(out["a__and__b"], [0, 0, 0, 1])
    np.testing.assert_array_equal(out["a__or__b"], [0, 1, 1, 1])
    np.testing.assert_array_equal(out["a__xor__b"], [0, 1, 1, 0])


def test_is_binary_column_detects_binary_vs_continuous():
    assert is_binary_column(pd.Series([0, 1, 1, 0]))
    assert is_binary_column(pd.Series([True, False, True]))
    assert not is_binary_column(pd.Series([0, 1, 2]))
    assert not is_binary_column(pd.Series([0.5, 1.0, 0.0]))


def test_boolean_pair_interactions_auto_detects_binary_columns():
    df = pd.DataFrame({"bin_a": [0, 1, 0, 1], "bin_b": [1, 1, 0, 0], "continuous": [0.5, 1.2, 3.4, 2.1]})
    out = boolean_pair_interactions(df)
    assert list(out.columns) == ["bin_a__and__bin_b", "bin_a__or__bin_b", "bin_a__xor__bin_b"]


def _make_pruning_dataset(n: int, n_noise_cols: int, seed: int):
    """A few informative binary flags whose XOR/AND drive the target, plus many independent noise flags.

    With ``n_noise_cols`` independent noise columns, ``n choose 2`` per operator explodes combinatorially
    while only the pairs involving the informative flags carry any target signal -- almost every generated
    AND/OR/XOR column pairs two noise flags together and is indistinguishable from chance.
    """
    rng = np.random.default_rng(seed)
    symptom_a = rng.integers(0, 2, n)
    symptom_b = rng.integers(0, 2, n)
    y = (symptom_a ^ symptom_b).astype(int)
    cols = {"symptom_a": symptom_a, "symptom_b": symptom_b}
    for i in range(n_noise_cols):
        cols[f"noise_{i}"] = rng.integers(0, 2, n)
    df = pd.DataFrame(cols)
    return df, y


def test_biz_val_prune_against_target_keeps_informative_cuts_noise():
    df, y = _make_pruning_dataset(n=3000, n_noise_cols=20, seed=1)

    unpruned = boolean_pair_interactions(df, operators=("and", "or", "xor"))
    pruned = boolean_pair_interactions(df, operators=("and", "or", "xor"), prune_against_target=(y, 0.05))

    n_unpruned, n_pruned = unpruned.shape[1], pruned.shape[1]
    assert n_pruned < 0.2 * n_unpruned, f"expected pruning to cut most of the {n_unpruned} noise columns, kept {n_pruned}"
    assert "symptom_a__xor__symptom_b" in pruned.columns, "the genuinely informative XOR combo must survive pruning"

    # Downstream accuracy must not suffer -- the informative combo alone should still make the target learnable.
    clf_pruned = LogisticRegression(max_iter=500).fit(pd.concat([df[["symptom_a", "symptom_b"]], pruned], axis=1), y)
    auc_pruned = roc_auc_score(y, clf_pruned.predict_proba(pd.concat([df[["symptom_a", "symptom_b"]], pruned], axis=1))[:, 1])
    assert auc_pruned > 0.98, f"expected pruned feature set to keep near-perfect learnability, got AUC={auc_pruned:.4f}"


def test_boolean_pair_interactions_default_unchanged_when_prune_not_supplied():
    df, _y = _make_pruning_dataset(n=500, n_noise_cols=10, seed=2)
    baseline = boolean_pair_interactions(df, operators=("and", "or", "xor"))
    default_call = boolean_pair_interactions(df, operators=("and", "or", "xor"), prune_against_target=None)
    pd.testing.assert_frame_equal(baseline, default_call)
