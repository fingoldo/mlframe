"""biz_value + unit tests for mlframe.competition.RoundedNumericCategoricalInteraction.

COMPETITION / EXPLORATORY ONLY — see module docstring in rounded_categorical_interaction.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from mlframe.competition.rounded_categorical_interaction import RoundedNumericCategoricalInteraction


def _make_xor_style_dataset(n: int = 4000, seed: int = 0) -> tuple[pd.Series, pd.Series, npt_array]:
    """Build numeric/categorical columns whose XOR of parity buckets determines the target, invisible to either column alone."""
    rng = np.random.default_rng(seed)
    numeric = rng.uniform(0.0, 10.0, size=n)
    categories = rng.choice(["A", "B", "C", "D"], size=n)

    numeric_bucket = np.round(numeric).astype(int) % 2  # 0/1 parity matching decimals=0 rounding
    cat_bucket = np.isin(categories, ["A", "C"]).astype(int)  # 0/1 grouping of the categorical

    # target is the XOR of the two buckets: neither raw column alone predicts it,
    # but the (rounded numeric, categorical) joint level fully determines it.
    xor_signal = (numeric_bucket ^ cat_bucket).astype(float)
    noise = rng.normal(0.0, 0.15, size=n)
    prob = np.clip(0.5 + (xor_signal - 0.5) * 0.9 + noise, 0.02, 0.98)
    target = rng.binomial(1, prob)

    return pd.Series(numeric, name="numeric_col"), pd.Series(categories, name="cat_col"), target


def _target_mean_encode_oof(composite: pd.Series, target: npt_array, n_splits: int = 5, seed: int = 0) -> npt_array:
    """Out-of-fold target-mean encode a categorical series via K-fold, filling unseen keys with the global mean."""
    encoded = np.zeros(len(composite), dtype=np.float64)
    global_mean = target.mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in kf.split(composite):
        train_df = pd.DataFrame({"key": composite.iloc[train_idx].to_numpy(), "y": target[train_idx]})
        means = train_df.groupby("key")["y"].mean()
        val_keys = composite.iloc[val_idx].to_numpy()
        encoded[val_idx] = pd.Series(val_keys).map(means).fillna(global_mean).to_numpy()
    return encoded


npt_array = np.ndarray  # local alias to keep the tuple annotation above readable


def test_rounded_numeric_categorical_interaction_basic_transform() -> None:
    """Composite key rounds numeric to N decimals, joins with sep, and renders nulls on either side as <NA>."""
    interaction = RoundedNumericCategoricalInteraction(decimals=1, sep="|")
    numeric = pd.Series([1.234, 2.567, np.nan, 3.001])
    categorical = pd.Series(["x", "y", "z", None])

    composite = interaction.transform(numeric, categorical, name="composite")

    assert list(composite) == ["1.2|x", "2.6|y", "<NA>|z", "3.0|<NA>"]
    assert composite.name == "composite"
    assert composite.dtype == object


def test_rounded_numeric_categorical_interaction_rejects_length_mismatch() -> None:
    """Numeric and categorical inputs of differing lengths raise ValueError instead of silently truncating."""
    interaction = RoundedNumericCategoricalInteraction()
    with pytest.raises(ValueError):
        interaction.transform(np.array([1.0, 2.0]), np.array(["a", "b", "c"]))


def test_rounded_numeric_categorical_interaction_rejects_bad_params() -> None:
    """Negative decimals or an empty separator both raise ValueError at construction time."""
    with pytest.raises(ValueError):
        RoundedNumericCategoricalInteraction(decimals=-1)
    with pytest.raises(ValueError):
        RoundedNumericCategoricalInteraction(sep="")


def test_biz_val_rounded_numeric_categorical_interaction_xor_signal_recovery() -> None:
    """Composite categorical (rounded numeric x categorical) recovers XOR-style interaction signal
    that is invisible to either raw column alone, when fed through target-mean encoding."""
    numeric, categorical, target = _make_xor_style_dataset(n=4000, seed=0)

    # Baselines: encode each raw column alone via the same OOF target-mean scheme.
    auc_numeric_alone = roc_auc_score(target, _target_mean_encode_oof(numeric.round(0).astype(str).astype(object), target))
    auc_cat_alone = roc_auc_score(target, _target_mean_encode_oof(categorical.astype(object), target))

    interaction = RoundedNumericCategoricalInteraction(decimals=0, sep="|")
    composite = interaction.transform(numeric, categorical, name="composite")
    encoded_composite = _target_mean_encode_oof(composite, target)
    auc_composite = roc_auc_score(target, encoded_composite)

    assert auc_numeric_alone < 0.58, f"numeric-alone AUC unexpectedly high: {auc_numeric_alone}"
    assert auc_cat_alone < 0.58, f"categorical-alone AUC unexpectedly high: {auc_cat_alone}"
    assert auc_composite >= 0.75, f"composite AUC too low: {auc_composite}"


def test_biz_val_rounded_numeric_categorical_interaction_decimals_matter() -> None:
    """Rounding to 0 decimals (matching the true bucket boundary) beats an overly fine 3-decimal
    rounding that fragments the composite into near-unique keys and destroys the OOF signal."""
    numeric, categorical, target = _make_xor_style_dataset(n=4000, seed=1)

    coarse = RoundedNumericCategoricalInteraction(decimals=0, sep="|").transform(numeric, categorical)
    fine = RoundedNumericCategoricalInteraction(decimals=3, sep="|").transform(numeric, categorical)

    auc_coarse = roc_auc_score(target, _target_mean_encode_oof(coarse, target))
    auc_fine = roc_auc_score(target, _target_mean_encode_oof(fine, target))

    assert auc_coarse >= 0.75, f"coarse composite AUC too low: {auc_coarse}"
    assert auc_coarse > auc_fine, f"expected coarse ({auc_coarse}) to beat fine ({auc_fine}) rounding"
