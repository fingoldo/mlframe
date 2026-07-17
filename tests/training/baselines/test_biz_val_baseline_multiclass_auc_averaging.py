"""biz_value: the multiclass baseline headline AUC must use MACRO averaging.

Macro weights every class equally, so a model that ranks the rare minority
classes well scores higher than one that only nails the majority class. Under
imbalance the minority classes are the expensive errors, so the honest headline
must reward the minority-strong model. Weighted averaging is dominated by the
majority class and misranks it (bench: macro 25/25 vs weighted 0/25 rank-correct).

This test calls the REAL production function
``_compute_metrics_table`` and fails if the multiclass AUC default is flipped
to ``average='weighted'`` (B then scores below A and the assertion trips).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.baselines._dummy_metrics_pick_plot import _compute_metrics_table


def _probs(y, n_classes, strong_classes, strength, rng):
    """Builds softmax class probabilities where strong_classes get a much larger separation margin than the rest."""
    n = len(y)
    logits = rng.normal(0.0, 1.0, size=(n, n_classes))
    for c in range(n_classes):
        margin = strength if c in strong_classes else strength * 0.05
        logits[y == c, c] += margin * 4.0
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def test_biz_val_baseline_multiclass_auc_macro_rewards_minority_ranking():
    """Macro-averaged multiclass AUC rewards correctly ranking a minority class, unlike a support-weighted average that would drown it out."""
    rng = np.random.default_rng(1234)
    n, k = 6000, 4
    p = np.array([0.85] + [0.05] * (k - 1))
    y = rng.choice(k, size=n, p=p)
    minorities = set(range(1, k))

    # A: strong on majority class 0 only. B: strong on all minorities.
    pA = _probs(y, k, strong_classes={0}, strength=0.9, rng=rng)
    pB = _probs(y, k, strong_classes=minorities, strength=0.9, rng=rng)

    df, _primary = _compute_metrics_table(
        target_type="multiclass_classification",
        val_preds={"A_majority": pA, "B_minority": pB},
        test_preds={"A_majority": pA, "B_minority": pB},
        val_y=y,
        test_y=y,
        extras={"n_classes": k},
    )

    # _compute_metrics_table returns the baseline name as the frame index.
    auc = df["val_AUC_macro"]
    auc_a = float(auc.loc["A_majority"])
    auc_b = float(auc.loc["B_minority"])

    # Macro must rank the minority-strong model strictly higher. A flip to
    # weighted averaging inverts this (majority class dominates) and fails here.
    assert auc_b > auc_a + 0.05, f"macro AUC must reward minority ranking: B={auc_b:.4f} !> A={auc_a:.4f}; default averaging likely flipped to 'weighted'"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s", "--no-cov"]))
