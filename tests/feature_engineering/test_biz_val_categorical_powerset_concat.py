"""biz_value test for ``feature_engineering.categorical_powerset_concat.categorical_powerset_concat``.

Synthetic: each unique (A, B) PAIR maps to an independently-random target class -- no linear function of
A's one-hot and B's one-hot separately can recover it (the mapping has zero structure along either marginal),
but the composite "A_B" categorical identifies each pair uniquely, so a linear model fit on the composite's
one-hot recovers the mapping near-perfectly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from mlframe.feature_engineering.categorical_powerset_concat import categorical_powerset_concat


def _make_pairwise_random_target(n_rows: int, n_levels: int, seed: int):
    """Helper: Make pairwise random target."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n_levels, size=n_rows)
    b = rng.integers(0, n_levels, size=n_rows)
    pair_to_label = {(i, j): rng.integers(0, 2) for i in range(n_levels) for j in range(n_levels)}
    y = np.array([pair_to_label[(ai, bi)] for ai, bi in zip(a, b)])
    df = pd.DataFrame({"A": a.astype(str), "B": b.astype(str)})
    return df, y


def test_biz_val_categorical_powerset_concat_recovers_pairwise_random_mapping():
    """Biz val categorical powerset concat recovers pairwise random mapping."""
    df, y = _make_pairwise_random_target(n_rows=12000, n_levels=12, seed=0)
    df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=0, stratify=y)

    enc_marginal = OneHotEncoder(handle_unknown="ignore")
    X_train_marginal = enc_marginal.fit_transform(df_train[["A", "B"]])
    X_test_marginal = enc_marginal.transform(df_test[["A", "B"]])
    model_marginal = LogisticRegression(max_iter=1000, C=0.1).fit(X_train_marginal, y_train)
    acc_marginal = accuracy_score(y_test, model_marginal.predict(X_test_marginal))

    df_train_composite = categorical_powerset_concat(df_train, columns=["A", "B"])
    df_test_composite = categorical_powerset_concat(df_test, columns=["A", "B"])
    enc_composite = OneHotEncoder(handle_unknown="ignore")
    X_train_composite = enc_composite.fit_transform(df_train_composite[["A_B"]])
    X_test_composite = enc_composite.transform(df_test_composite[["A_B"]])
    model_composite = LogisticRegression(max_iter=1000).fit(X_train_composite, y_train)
    acc_composite = accuracy_score(y_test, model_composite.predict(X_test_composite))

    assert acc_marginal < 0.65, f"expected marginal-only one-hot to stay far below the composite's accuracy on a pairwise-random target, got {acc_marginal:.4f}"
    assert acc_composite > 0.9, f"expected the composite A_B categorical to recover the pairwise mapping, got {acc_composite:.4f}"
    assert acc_composite - acc_marginal > 0.3, (
        f"expected the composite feature to beat marginal-only features by a wide margin, got composite={acc_composite:.4f} marginal={acc_marginal:.4f}"
    )


def test_categorical_powerset_concat_column_count_and_names():
    """Categorical powerset concat column count and names."""
    df = pd.DataFrame({"A": ["a0", "a1"], "B": ["b0", "b1"], "C": ["c0", "c1"]})
    out = categorical_powerset_concat(df, columns=["A", "B", "C"])
    assert set(out.columns) == {"A", "B", "C", "A_B", "A_C", "B_C", "A_B_C"}


def test_categorical_powerset_concat_max_order_caps_to_pairwise():
    """Categorical powerset concat max order caps to pairwise."""
    df = pd.DataFrame({"A": ["a0", "a1"], "B": ["b0", "b1"], "C": ["c0", "c1"]})
    out = categorical_powerset_concat(df, columns=["A", "B", "C"], max_order=2)
    assert "A_B_C" not in out.columns
    assert {"A_B", "A_C", "B_C"}.issubset(out.columns)


def _make_mixed_informative_and_noise_target(n_rows: int, n_levels: int, n_noise_keys: int, seed: int):
    """A xor B (bit-parity) determines y; N additional keys are drawn independently of y.

    ``y = popcount(A xor B) % 2`` is EXACTLY marginal-balanced in both A and B by construction (for any fixed
    A, y is uniform 0/1 as B ranges uniformly -- unlike a random per-pair label table, which only balances
    marginals in expectation and leaves finite-sample bias a lone key can exploit). So A alone, B alone, and
    any composite that doesn't contain BOTH A and B carries zero real signal, while A_B fully determines y.
    ``n_levels`` must be a power of two for the popcount-parity trick to stay uniform.
    """
    assert n_levels > 0 and (n_levels & (n_levels - 1)) == 0, "n_levels must be a power of two for exact marginal balance"
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n_levels, size=n_rows)
    b = rng.integers(0, n_levels, size=n_rows)
    xor_vals = a ^ b
    y = np.array([bin(v).count("1") % 2 for v in xor_vals])

    data = {"A": a.astype(str), "B": b.astype(str)}
    for k in range(n_noise_keys):
        data[f"N{k}"] = rng.integers(0, n_levels, size=n_rows).astype(str)
    df = pd.DataFrame(data)
    return df, y


def test_biz_val_categorical_powerset_concat_prune_against_target_keeps_informative_drops_noise():
    """Biz val categorical powerset concat prune against target keeps informative drops noise."""
    n_noise_keys = 4
    df, y = _make_mixed_informative_and_noise_target(n_rows=15000, n_levels=8, n_noise_keys=n_noise_keys, seed=1)
    columns = ["A", "B"] + [f"N{k}" for k in range(n_noise_keys)]

    out_unpruned = categorical_powerset_concat(df, columns=columns, max_order=2)
    composite_cols = [c for c in out_unpruned.columns if c not in columns]
    informative_cols = {"A_B"}
    noise_cols = set(composite_cols) - informative_cols
    assert len(noise_cols) >= 8, f"expected a clear majority of pairwise composites to be pure noise, got {len(noise_cols)} of {len(composite_cols)}"

    out_pruned = categorical_powerset_concat(df, columns=columns, max_order=2, prune_against_target=(y, 0.05))
    kept_composites = set(out_pruned.columns) - set(columns)

    assert informative_cols.issubset(kept_composites), f"expected the informative A_B composite to survive pruning, kept={kept_composites}"
    kept_noise = kept_composites & noise_cols
    dropped_noise_fraction = 1.0 - (len(kept_noise) / len(noise_cols))
    assert dropped_noise_fraction >= 0.8, (
        f"expected pruning to drop >=80% of the {len(noise_cols)} noise composites, dropped only {dropped_noise_fraction:.2%} (kept {kept_noise})"
    )
