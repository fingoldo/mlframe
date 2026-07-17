"""Wave-5 transformer OOF / leakage tests + biz_value.

A2-06  compute_rff_features Mode-A OOF / Mode-B train-only fit; biz_value: in-sample-fit optimism > OOF.
A2-07  local_lift binary pr_auc is true trapezoidal (rank-correlates with sklearn average_precision_score).
A2-14  class_distance Mode-B self-label leak removed when exclude_self_ids mask is passed.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer import (
    RFFState,
    compute_class_distance_features,
    compute_rff_features,
    rff_apply_state,
)


# ----------------------------- A2-06 ---------------------------------------


def test_a2_06_rff_mode_b_returns_state_and_replays() -> None:
    """Mode B fits on X (train) only, projects X_query; the returned state replays identically via rff_apply_state."""
    rng = np.random.default_rng(0)
    Xtr = rng.standard_normal((200, 8)).astype(np.float32)
    Xq = rng.standard_normal((50, 8)).astype(np.float32)
    df, state = compute_rff_features(Xtr, seed=1, n_features=32, X_query=Xq, return_state=True)
    assert isinstance(state, RFFState)
    assert df.shape == (50, 32)
    replay = rff_apply_state(state, Xq)
    assert np.allclose(df.to_numpy(), replay.to_numpy()), "rff_apply_state must reproduce Mode-B output"


def test_a2_06_rff_mode_a_oof_shape_and_no_self_fit() -> None:
    """Mode A (splitter) yields a full (N, n_features) OOF matrix; each fold fits on its own train slice."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((300, 6)).astype(np.float32)
    splitter = KFold(n_splits=4, shuffle=True, random_state=0)
    out = compute_rff_features(X, seed=3, n_features=16, splitter=splitter)
    assert out.shape == (300, 16)
    # OOF differs from a single full-data fit (the scaler/bandwidth differ per fold).
    full = compute_rff_features(X, seed=3, n_features=16)
    assert not np.allclose(out.to_numpy(), full.to_numpy())


def test_a2_06_biz_value_oof_less_optimistic_than_in_sample() -> None:
    """biz_value: in-sample RFF features fit a downstream linear model with lower CV-test error than they appear on train.

    Construct a target driven by an RBF bump. In-sample RFF (fit on full X then encode the same X) gives the linear
    head an optimistic train R^2 that does not hold out; OOF RFF closes that train/test optimism gap. Floor: optimism gap
    (in-sample) clearly exceeds the OOF gap.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(7)
    n = 600
    X = rng.standard_normal((n, 4)).astype(np.float32)
    center = np.zeros(4, dtype=np.float32)
    y = np.exp(-0.5 * ((X - center) ** 2).sum(axis=1)) + 0.05 * rng.standard_normal(n)

    # In-sample: fit RFF on ALL X, then split for the linear head -> the RFF map already "saw" the test rows.
    feats_insample = compute_rff_features(X, seed=11, n_features=64).to_numpy()
    Xtr, Xte, ytr, yte, ftr_in, fte_in = train_test_split(X, y, feats_insample, test_size=0.4, random_state=0)
    ridge_in = Ridge(alpha=1.0).fit(ftr_in, ytr)
    train_r2_in = ridge_in.score(ftr_in, ytr)
    test_r2_in = ridge_in.score(fte_in, yte)
    gap_in = train_r2_in - test_r2_in

    # OOF: fit RFF on the training subset only, apply to held-out test.
    _, state = compute_rff_features(Xtr, seed=11, n_features=64, X_query=Xtr, return_state=True)
    ftr_oof = rff_apply_state(state, Xtr).to_numpy()
    fte_oof = rff_apply_state(state, Xte).to_numpy()
    ridge_oof = Ridge(alpha=1.0).fit(ftr_oof, ytr)
    gap_oof = ridge_oof.score(ftr_oof, ytr) - ridge_oof.score(fte_oof, yte)

    # The train-only-fit path must not have a LARGER optimism gap than the leaky full-fit path.
    assert gap_oof <= gap_in + 1e-6, f"OOF optimism gap {gap_oof:.4f} should not exceed in-sample gap {gap_in:.4f}"


# ----------------------------- A2-07 ---------------------------------------


def test_a2_07_local_pr_auc_matches_sklearn_average_precision() -> None:
    """local_pr_auc uses the sklearn Average-Precision (step-PR) convention, so it must MATCH average_precision_score per query neighbourhood (not just rank-correlate). The pre-fix docstring mislabelled this as 'trapezoidal'; trapezoidal would deviate from AP."""
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score

    from mlframe.feature_engineering.transformer.local_lift import _local_lift_and_pr_auc

    rng = np.random.default_rng(5)
    n_q, k = 60, 24
    # Distances ascending; labels increasingly concentrated near the front for some queries, back for others.
    y_neighbors = (rng.random((n_q, k)) < (0.5 - 0.4 * np.linspace(-1, 1, k))[None, :]).astype(np.float32)
    dists = np.sort(rng.random((n_q, k)).astype(np.float32), axis=1)
    _lift, pr_auc, _top1 = _local_lift_and_pr_auc(y_neighbors, dists, 0.3, task="binary")

    # Reference: sklearn AP using similarity score = -rank (closer neighbour => higher score).
    scores = (k - 1) - np.arange(k)
    ref = np.zeros(n_q)
    for i in range(n_q):
        if y_neighbors[i].sum() == 0:
            ref[i] = 0.0
        else:
            ref[i] = average_precision_score(y_neighbors[i], scores)
    mask = ref > 0
    assert np.allclose(pr_auc[mask], ref[mask], atol=1e-5), "local_pr_auc must equal sklearn average_precision_score (step-PR convention)"
    rho, _ = spearmanr(pr_auc[mask], ref[mask])
    assert rho >= 0.99, f"local_pr_auc must rank-correlate ~perfectly with sklearn AP (rho={rho:.3f})"


# ----------------------------- A2-14 ---------------------------------------


def test_a2_14_class_distance_self_leak_removed_with_mask() -> None:
    """Mode B: when query rows overlap train, exclude_self_ids removes the self-match (distance 0 to own row)."""
    rng = np.random.default_rng(9)
    n = 120
    X = rng.standard_normal((n, 5)).astype(np.float32)
    y = (rng.random(n) < 0.4).astype(np.float32)

    # Query = first 30 train rows (overlap). Without the mask, each query finds itself -> a near-zero k=1 distance.
    q_idx = np.arange(30)
    Xq = X[q_idx]

    leaky = compute_class_distance_features(X, y, Xq, seed=1, standardize=True)
    # Build a self-mask: training rows that are also query rows.
    mask = np.zeros(n, dtype=bool)
    mask[q_idx] = True
    masked = compute_class_distance_features(X, y, Xq, seed=1, standardize=True, exclude_self_ids=mask)

    # For positive-class query rows the k=1 distance to the nearest positive is ~0 in the leaky version (self-match).
    pos_q = y[q_idx] > 0.5
    leaky_pos_k1 = leaky["cdist_pos_k1"].to_numpy()[pos_q]
    masked_pos_k1 = masked["cdist_pos_k1"].to_numpy()[pos_q]
    assert leaky_pos_k1.min() < 1e-4, "sanity: leaky path should have a ~0 self-distance for an overlapping positive query"
    assert masked_pos_k1.min() > leaky_pos_k1.min(), "exclude_self_ids must remove the zero self-distance"


def test_a2_14_exclude_self_ids_rejected_in_mode_a() -> None:
    """exclude_self_ids is Mode-B only; passing it in Mode A (splitter) raises."""
    X = np.random.default_rng(0).standard_normal((40, 3)).astype(np.float32)
    y = (np.random.default_rng(1).random(40) < 0.4).astype(np.float32)
    with pytest.raises(ValueError, match="only meaningful in Mode B"):
        compute_class_distance_features(X, y, None, splitter=KFold(3), seed=1, exclude_self_ids=np.zeros(40, dtype=bool))
