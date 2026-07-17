"""Identity-pin regression tests for two algorithmic-complexity optimizations.

CPX2 -- ``anchor.anchor_ewm_features``: O(A^2)-per-segment decayed-sum replaced
by an O(1)-per-step EWMA recurrence (running accumulators). Output must stay
bit-identical up to FP reduction-order (~1e-9).

CPX6 -- ``spatial.knn_label_dispersion_features``: per-query np.bincount /
np.unique Python loop replaced by dense label codes + np.add.at over an
(n_q, k) count matrix. Output must stay bit-identical (~1e-9).

Each test pins the NEW output against an independent reference computation of
the SAME math (not the production code), so it fails if the optimization ever
regresses to wrong numbers. The benches in
``src/mlframe/feature_engineering/_benchmarks/`` additionally A/B against the
real ``git show HEAD`` baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.anchor import anchor_ewm_features
from mlframe.feature_engineering.spatial import knn_label_dispersion_features


# ----------------------------- CPX2: anchor EWMA -----------------------------


def _ewm_reference(label, is_anchor, half_life):
    """Naive O(A^2) reference: recompute decayed sums over all anchors per row."""
    m = label.size
    val_out = np.full(m, np.nan)
    slope_out = np.full(m, np.nan)
    pos = []
    val = []
    for i in range(m):
        if is_anchor[i] and np.isfinite(label[i]):
            pos.append(i)
            val.append(float(label[i]))
        if not pos:
            continue
        xs = np.asarray(pos, float)
        ys = np.asarray(val, float)
        w = 0.5 ** ((i - xs) / half_life)
        w_sum = w.sum() + 1e-12
        w_mean = (ys * w).sum() / w_sum
        val_out[i] = w_mean
        if xs.size >= 2:
            xm = (xs * w).sum() / w_sum
            num = (w * (xs - xm) * (ys - w_mean)).sum()
            den = (w * (xs - xm) ** 2).sum()
            slope_out[i] = num / (den + 1e-12)
    return val_out, slope_out


@pytest.mark.parametrize("half_life", [10.0, 30.0])
def test_cpx2_anchor_ewm_identity(half_life):
    """Cpx2 anchor ewm identity."""
    rng = np.random.default_rng(7)
    n = 3000
    is_anchor = rng.random(n) < 0.2
    label = np.where(is_anchor, np.cumsum(rng.standard_normal(n)) * 0.05, np.nan)

    out = anchor_ewm_features(label, is_anchor, half_life_rows=half_life)
    val_key = f"ewm_anchor_value_H{int(half_life)}"
    slope_key = f"ewm_anchor_slope_H{int(half_life)}"
    ref_val, ref_slope = _ewm_reference(label, is_anchor, half_life)

    for got, ref in ((out[val_key], ref_val), (out[slope_key], ref_slope)):
        assert np.array_equal(np.isnan(got), np.isnan(ref))
        m = np.isfinite(ref)
        assert np.allclose(got[m], ref[m], atol=1e-9, rtol=0)


def test_cpx2_anchor_ewm_grouped_identity():
    """Cpx2 anchor ewm grouped identity."""
    rng = np.random.default_rng(11)
    n = 2500
    is_anchor = rng.random(n) < 0.25
    label = np.where(is_anchor, rng.standard_normal(n), np.nan)
    gids = np.sort(rng.integers(0, 8, size=n))

    out = anchor_ewm_features(label, is_anchor, gids, half_life_rows=20.0)
    # Compute the per-group reference independently.
    val_ref = np.full(n, np.nan)
    slope_ref = np.full(n, np.nan)
    order = np.argsort(gids, kind="stable")
    for g in np.unique(gids):
        seg = order[gids[order] == g]
        v, s = _ewm_reference(label[seg], is_anchor[seg], 20.0)
        val_ref[seg] = v
        slope_ref[seg] = s
    got_v = out["ewm_anchor_value_H20"]
    got_s = out["ewm_anchor_slope_H20"]
    assert np.array_equal(np.isnan(got_v), np.isnan(val_ref))
    mv = np.isfinite(val_ref)
    assert np.allclose(got_v[mv], val_ref[mv], atol=1e-9, rtol=0)
    ms = np.isfinite(slope_ref)
    assert np.allclose(got_s[ms], slope_ref[ms], atol=1e-9, rtol=0)


# -------------------- CPX6: kNN label dispersion vectorize -------------------


def _dispersion_reference(q, ref, labels, k, task, n_bins=8):
    """Per-row loop reference mirroring the prior bincount/unique semantics."""
    from sklearn.neighbors import KDTree

    ref = np.ascontiguousarray(ref, float)
    if task == "regression":
        labels = np.ascontiguousarray(labels, float)
        finite_ref = np.isfinite(ref).all(axis=1) & np.isfinite(labels)
        global_median = float(np.median(labels[np.isfinite(labels)]))
    else:
        labels = np.asarray(labels)
        finite_ref = np.isfinite(ref).all(axis=1)
    ref = ref[finite_ref]
    labels = labels[finite_ref]
    tree = KDTree(ref)
    q_clean = np.where(np.isfinite(q), q, 0.0)
    _, indices = tree.query(q_clean, k=min(k + 1, ref.shape[0]))
    label_arr = labels[indices[:, :k]]
    n_q = q.shape[0]
    entropy = np.full(n_q, np.nan)
    majority = np.full(n_q, np.nan)
    disagree = np.full(n_q, np.nan)
    if task == "regression":
        finite_lab = labels[np.isfinite(labels)]
        edges = np.unique(np.quantile(finite_lab, np.linspace(0, 1, n_bins + 1)))
        binned = np.searchsorted(edges[1:-1], label_arr, side="right")
        for i in range(n_q):
            bc = np.bincount(binned[i], minlength=max(2, edges.size - 1))
            p = bc.astype(float) / bc.sum()
            p_nz = p[p > 0]
            entropy[i] = -np.sum(p_nz * np.log(p_nz))
            majority[i] = p.max()
        opp = (label_arr > global_median) != (label_arr[:, :1] > global_median)
        disagree = opp.mean(axis=1)
    else:
        for i in range(n_q):
            row = label_arr[i]
            _, counts = np.unique(row, return_counts=True)
            p = counts.astype(float) / counts.sum()
            entropy[i] = -np.sum(p * np.log(p))
            majority[i] = p.max()
            disagree[i] = (row != row[0]).mean()
    return entropy, disagree, majority


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_cpx6_knn_dispersion_identity(task):
    """Cpx6 knn dispersion identity."""
    rng = np.random.default_rng(3)
    ref = rng.standard_normal((4000, 2))
    q = rng.standard_normal((1500, 2))
    if task == "regression":
        labels = rng.standard_normal(4000)
    else:
        labels = rng.integers(0, 5, size=4000)

    out = knn_label_dispersion_features(q, ref, labels, k=10, task=task)
    ent_ref, dis_ref, maj_ref = _dispersion_reference(q, ref, labels, 10, task)

    np.testing.assert_allclose(out["local_label_entropy"], ent_ref, atol=1e-9, rtol=0)
    np.testing.assert_allclose(out["local_majority_share"], maj_ref, atol=1e-9, rtol=0)
    np.testing.assert_allclose(out["local_disagreement_ratio"], dis_ref, atol=1e-9, rtol=0)
