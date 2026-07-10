"""biz_value test for ``feature_engineering.categorical_group_concat.concat_categorical_group``.

The win (2nd_porto-seguro-safe-driver-prediction.md): a target that depends on the JOINT combination of two
categorical columns (a "combination code" pattern, e.g. specific product-code x region-code pairs carry risk
signal no single column does) is invisible to frequency encoding of EITHER raw column alone -- each column's
individual frequency distribution is uninformative about which specific pair matters. Concatenating the two
columns into a composite categorical BEFORE frequency-encoding recovers that signal, using mlframe's existing
``frequency_encode_fit`` (reused as-is; the concatenator is the only new piece).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from mlframe.feature_engineering.categorical_group_concat import concat_categorical_group
from mlframe.feature_selection.filters._count_freq_interaction_fe import frequency_encode_fit


def _make_joint_rarity_dataset(n: int, seed: int):
    # Frequency encoding's real value is exposing how RARE a specific category value is -- so the realistic
    # target here is "this exact (a, b) PAIR is one of the rare joint combinations" (e.g. fraud more likely
    # for unusual product-code x region-code pairs), not an arbitrary lookup-table target (frequency has no
    # way to recover an arbitrary lookup rule, since occurrence count carries no semantic label information).
    rng = np.random.default_rng(seed)
    n_a, n_b = 8, 8
    combo_ids = np.arange(n_a * n_b)
    probs = np.ones(n_a * n_b)
    rare_mask = rng.random(n_a * n_b) < 0.15
    probs[rare_mask] *= 0.02
    probs /= probs.sum()

    chosen = rng.choice(combo_ids, size=n, p=probs)
    cat_a = (chosen // n_b).astype(str)
    cat_b = (chosen % n_b).astype(str)
    y = rare_mask[chosen].astype(int)  # target: this exact pair is a rare joint combination

    df = pd.DataFrame({"cat_a": cat_a, "cat_b": cat_b})
    return df, y


def test_biz_val_concat_categorical_group_recovers_joint_rarity_signal_marginal_freq_misses():
    df, y = _make_joint_rarity_dataset(n=8000, seed=0)

    freq_a, _ = frequency_encode_fit(df, ["cat_a"])
    freq_b, _ = frequency_encode_fit(df, ["cat_b"])
    marginal_features = pd.concat([freq_a, freq_b], axis=1).to_numpy()
    auc_marginal = cross_val_score(LogisticRegression(max_iter=500), marginal_features, y, cv=5, scoring="roc_auc").mean()

    df_composite = concat_categorical_group(df, ["cat_a", "cat_b"], feature_name="ab_combo")
    freq_composite, _ = frequency_encode_fit(df_composite, ["ab_combo"])
    auc_composite = cross_val_score(LogisticRegression(max_iter=500), freq_composite.to_numpy(), y, cv=5, scoring="roc_auc").mean()

    assert auc_composite > 0.98, f"expected the composite-categorical frequency encoding to near-perfectly recover joint-pair rarity, got AUC={auc_composite:.4f}"
    assert auc_composite > auc_marginal + 0.1, f"expected the composite encoding to materially beat marginal frequency encodings (which only see each column's OWN popularity, not this specific pair's), got composite={auc_composite:.4f} marginal={auc_marginal:.4f}"


def test_concat_categorical_group_exact_values():
    df = pd.DataFrame({"a": ["x", "y"], "b": ["1", "2"], "c": ["p", "q"]})
    out = concat_categorical_group(df, ["a", "b", "c"], separator="-", feature_name="combo")
    assert list(out["combo"]) == ["x-1-p", "y-2-q"]


def test_concat_categorical_group_requires_two_columns():
    import pytest

    df = pd.DataFrame({"a": ["x", "y"]})
    with pytest.raises(ValueError):
        concat_categorical_group(df, ["a"])
