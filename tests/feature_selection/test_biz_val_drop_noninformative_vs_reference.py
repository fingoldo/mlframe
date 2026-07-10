"""biz_value test for ``feature_selection.drop_noninformative_vs_reference.drop_noninformative_vs_reference``.

The win (3rd_mechanisms-of-action-moa-prediction.md): among features measured for both a reference/control
cohort and the rest (e.g. treated samples), some carry pure batch/instrument noise with NO relationship to
the control-vs-treated distinction, while others genuinely shift between the two groups (the actual signal).
A KS-test p > alpha correctly identifies the noise features (fails to reject "same distribution"), while a
downstream model trained keeping only the KS-significant features should retain (or improve) predictive power
versus keeping everything, since the dropped columns carried no discriminative information about the very
distinction being modeled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from mlframe.feature_selection.drop_noninformative_vs_reference import drop_noninformative_vs_reference


def _make_control_treated_dataset(n: int, n_signal: int, n_noise: int, seed: int):
    rng = np.random.default_rng(seed)
    is_treated = rng.integers(0, 2, n).astype(bool)

    cols = {}
    for i in range(n_signal):
        # genuinely shifts between control and treated
        cols[f"signal{i}"] = rng.normal(loc=np.where(is_treated, 2.0, 0.0), scale=1.0)
    for i in range(n_noise):
        # identical distribution regardless of treated/control -- pure batch noise
        cols[f"noise{i}"] = rng.normal(loc=0.0, scale=1.0, size=n)

    df = pd.DataFrame(cols)
    reference_mask = ~is_treated  # control = reference
    y = is_treated.astype(int)
    return df, reference_mask, y


def test_biz_val_drop_noninformative_vs_reference_correctly_identifies_noise_columns():
    df, reference_mask, y = _make_control_treated_dataset(n=2000, n_signal=5, n_noise=15, seed=0)

    kept = drop_noninformative_vs_reference(df, reference_mask, alpha=0.1)

    n_signal_kept = sum(1 for c in kept if c.startswith("signal"))
    n_noise_kept = sum(1 for c in kept if c.startswith("noise"))
    assert n_signal_kept == 0, f"expected the drop-candidate list to contain NO genuinely-shifting signal columns, got {[c for c in kept if c.startswith('signal')]}"
    assert n_noise_kept >= 12, f"expected most pure-noise columns to be correctly flagged as drop candidates, got {n_noise_kept}/15"


def test_biz_val_dropping_noninformative_columns_preserves_downstream_auc():
    df, reference_mask, y = _make_control_treated_dataset(n=2000, n_signal=5, n_noise=15, seed=1)
    drop_candidates = drop_noninformative_vs_reference(df, reference_mask, alpha=0.1)
    kept_cols = [c for c in df.columns if c not in drop_candidates]

    auc_full = cross_val_score(LogisticRegression(max_iter=500), df, y, cv=5, scoring="roc_auc").mean()
    auc_pruned = cross_val_score(LogisticRegression(max_iter=500), df[kept_cols], y, cv=5, scoring="roc_auc").mean()

    assert len(kept_cols) < df.shape[1], "expected the pruned set to be strictly smaller than the full feature set"
    assert auc_pruned >= auc_full - 0.02, f"expected pruning non-informative columns to not meaningfully hurt downstream AUC, got pruned={auc_pruned:.4f} full={auc_full:.4f}"
