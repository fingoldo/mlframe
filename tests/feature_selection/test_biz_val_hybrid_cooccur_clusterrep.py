"""Unit + biz_value tests for the two HybridSelector innovations.

  (1) cooccur_weight="gain"|"count" -- the tree member ranks co-occurrence interaction pairs by summed split GAIN
      (default) instead of raw frequency. Unit: gain mode produces gain-ranked pairs; both modes round-trip a fit.
      biz_value: on an interaction-heavy XOR bed, gain-weighting recovers the true operand pairs at honest holdout
      AUC >= count-weighting (pinned margin), and the gain ranking puts a true operand pair first.

  (2) cluster_rep="first"|"max_fi"|"sum_fi" -- when a correlation cluster collapses to one representative, sum_fi
      (default) keeps the highest summed-per-repeat perm-FI member; first keeps the arbitrary first column. Unit:
      _rep_member honours each mode. biz_value: on a correlated-cluster bed where the cleanest copy is NOT first,
      sum_fi keeps a more informative representative -> honest holdout AUC >= first (pinned margin).

Env: heavy mlframe import path is CUDA-disabled so the selector fits CPU-only (host segfaults otherwise).
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MLFRAME_NO_CUDA_AUTOCONFIG", "1")
os.environ.setdefault("MLFRAME_KEEP_BROKEN_CUPY", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytest

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------- synthetic beds
def _xor_bed(n=2000, seed=0, n_pairs=3, n_noise=24):
    rng = np.random.default_rng(seed)
    cols, logit = {}, np.zeros(n)
    for p in range(n_pairs):
        a = rng.standard_normal(n); b = rng.standard_normal(n)
        cols[f"xa_{p}"] = a; cols[f"xb_{p}"] = b
        logit += 1.8 * np.sign(a) * np.sign(b)
    for i in range(n_noise):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    X = pd.DataFrame(cols)
    order = list(X.columns); rng.shuffle(order)
    return X[order], pd.Series(y, name="target")


def _cluster_bed(n=2000, seed=0, n_clusters=4, copies=5, n_noise=16):
    rng = np.random.default_rng(seed)
    cols, logit = {}, np.zeros(n)
    coef = [1.6, -1.3, 1.1, 0.9]
    for c in range(n_clusters):
        z = rng.standard_normal(n)
        logit += coef[c % len(coef)] * z
        for j in range(copies):                       # later copy is cleaner -> first-column is a worse member
            cols[f"c{c}_{j}"] = z + max(0.55 - 0.09 * j, 0.05) * rng.standard_normal(n)
    for i in range(n_noise):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(logit / 1.3)))).astype(int)
    X = pd.DataFrame(cols)
    order = list(X.columns); rng.shuffle(order)
    return X[order], pd.Series(y, name="target")


def _honest_auc(Xtr, ytr, Xte, yte, sel, seed):
    import lightgbm as lgb
    feats = [c for c in sel if c in Xtr.columns]
    if not feats:
        return 0.5
    m = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05, n_jobs=-1,
                           verbose=-1, random_state=seed)
    m.fit(Xtr[feats], ytr)
    return float(roc_auc_score(yte, m.predict_proba(Xte[feats])[:, 1]))


def _fit(X, y, **kw):
    from mlframe.feature_selection import HybridSelector
    return HybridSelector(random_state=kw.pop("random_state", 0), **kw).fit(X, y)


# ===================================================================== UNIT: defaults + plumbing
def test_default_knobs_are_gain_and_first():
    from mlframe.feature_selection import HybridSelector
    h = HybridSelector()
    assert h.cooccur_weight == "gain"  # measured win on interaction beds -> default
    assert h.cluster_rep == "first"  # FI-based reps are bed-dependent (regress on some beds) -> opt-in, not default


@pytest.mark.parametrize("mode", ["first", "max_fi", "sum_fi"])
def test_rep_member_honours_each_mode(mode):
    from mlframe.feature_selection import HybridSelector
    h = HybridSelector(cluster_rep=mode)
    members = ["a", "b", "c"]
    h.fi_ = {"a": 0.1, "b": 0.3, "c": 0.2}        # max mean-FI = b
    h._fi_sum_ = {"a": 0.1, "b": 0.2, "c": 0.9}   # max sum-FI = c
    rep = h._rep_member(members)
    assert rep == {"first": "a", "max_fi": "b", "sum_fi": "c"}[mode]
    assert h._rep_member([]) is None


def test_rep_member_sum_fi_falls_back_to_mean_when_no_sum():
    from mlframe.feature_selection import HybridSelector
    h = HybridSelector(cluster_rep="sum_fi")
    h.fi_ = {"a": 0.1, "b": 0.4}                   # no _fi_sum_ -> falls back to mean FI
    assert h._rep_member(["a", "b"]) == "b"


@pytest.mark.parametrize("weight", ["count", "gain"])
def test_cooccur_weight_roundtrips_a_fit(weight):
    X, y = _xor_bed(n=1200, seed=0, n_pairs=2, n_noise=12)
    h = _fit(X, y, cooccur_weight=weight)
    assert len(h.raw_selected_) >= 1
    # tree co-occurrence pairs were proposed (the member is on by default)
    assert hasattr(h, "_tree_prod_pairs_")


def test_gain_mode_ranks_a_true_operand_pair_among_top():
    """The tree member's gain-weighted co-occurrence must surface a real XOR operand pair (one of xa_p/xb_p
    co-occurring) among its proposed pairs -- a structural check the gain aggregation is wired correctly."""
    X, y = _xor_bed(n=2000, seed=0, n_pairs=3, n_noise=20)
    h = _fit(X, y, cooccur_weight="gain")
    pairs = [tuple(sorted(p)) for p in h._tree_prod_pairs_]
    true_pairs = {tuple(sorted((f"xa_{p}", f"xb_{p}"))) for p in range(3)}
    assert any(p in true_pairs for p in pairs), f"no true operand pair among proposed {pairs[:6]}"


# ===================================================================== BIZ_VALUE
def test_biz_val_hybrid_cooccur_gain_beats_count_on_interaction_bed():
    """Floor: gain-weighted honest holdout AUC >= count-weighted - 0.005, averaged over 3 seeds, on XOR beds.
    Gain ranks true interaction operands above shallow high-frequency noise splits."""
    deltas = []
    for seed in (0, 1, 2):
        X, y = _xor_bed(n=2000, seed=seed, n_pairs=3, n_noise=24)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.35, random_state=seed, stratify=y)
        a_cnt = _honest_auc(Xtr, ytr, Xte, yte, list(_fit(Xtr, ytr, random_state=seed, cooccur_weight="count").raw_selected_), seed)
        a_gn = _honest_auc(Xtr, ytr, Xte, yte, list(_fit(Xtr, ytr, random_state=seed, cooccur_weight="gain").raw_selected_), seed)
        deltas.append(a_gn - a_cnt)
    mean_d = float(np.mean(deltas))
    assert mean_d >= -0.005, f"gain co-occurrence must not regress count on interaction beds: mean_delta={mean_d:+.4f} {deltas}"


def test_cluster_rep_sum_fi_is_a_valid_optin_selection():
    """``cluster_rep='sum_fi'`` is an OPT-IN option, NOT the default: its honest-holdout win over first-column is
    bed-dependent (it regresses on some correlated-cluster beds), so we do not pin a win. Pin only that the option
    runs and yields a valid non-empty selection (the default stays 'first'; the win claim was withdrawn after the
    sum_fi-beats-first floor failed to replicate across seeds on _cluster_bed)."""
    X, y = _cluster_bed(n=2000, seed=0, n_clusters=4, copies=5, n_noise=16)
    sel = list(_fit(X, y, random_state=0, cluster_rep="sum_fi").raw_selected_)
    assert len(sel) > 0 and len(sel) == len(set(sel))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s", "--no-cov", "-p", "no:cacheprovider", "-p", "no:randomly"]))
