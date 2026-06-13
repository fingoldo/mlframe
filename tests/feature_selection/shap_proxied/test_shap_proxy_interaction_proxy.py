"""Unit + biz_val for ``proxy_mode="interaction"`` (interaction-aware subset proxy).

The additive subset proxy ``base + sum phi_j`` is blind to a non-additive PAIR: at a tight cardinality
cap it cannot tell a competing-XOR pair from a single noise feature (each operand has ~0 marginal). The
interaction proxy ``base + sum phi_j + 2*sum_{i<j} Phi_ij`` (gated to the top-k |phi| features) restores
the joint credit, so the chosen subset's honest-holdout AUC jumps. Default stays additive (bench: only
1/6 beds wins), so these pin the OPT-IN trick's measured win + the no-op contract + the gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")
pytest.importorskip("xgboost")

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_interaction_proxy import (  # noqa: E402
    build_pair_table,
    interaction_proxy_top_n,
)


def _xgb(seed=0):
    import xgboost as xgb

    return xgb.XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.15, subsample=0.9,
                             n_jobs=1, tree_method="hist", verbosity=0, random_state=seed)


def _two_xor_pairs(n=1500, p_noise=30, seed=0):
    """Two competing independent XOR pairs + noise. At cap=2 the additive proxy can only keep ONE
    operand from each pair (recall 0.5, AUC ~chance); the interaction proxy keeps a full pair."""
    rng = np.random.default_rng(seed)
    xa0, xb0, xa1, xb1 = (rng.normal(size=n) for _ in range(4))
    cols = {"xa0": xa0, "xb0": xb0, "xa1": xa1, "xb1": xb1}
    for j in range(p_noise):
        cols[f"nz{j}"] = rng.normal(size=n)
    X = pd.DataFrame(cols)
    logit = 2.5 * (np.sign(xa0 * xb0) + np.sign(xa1 * xb1))
    p = 1.0 / (1.0 + np.exp(-(logit - logit.mean())))
    y = (rng.uniform(size=n) < p).astype(int)
    return X, y


def _shap_and_tensor(X, y, seed):
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import compute_shap_matrix
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_interactions import compute_interaction_tensor

    phi, base, y_phi = compute_shap_matrix(
        _xgb(seed), X, np.asarray(y), classification=True, out_of_fold=True, n_splits=3,
        n_models=1, rng=np.random.default_rng(seed), n_jobs=1)
    Phi, _ibase = compute_interaction_tensor(
        _xgb(seed), X, np.asarray(y), classification=True, rng=np.random.default_rng(seed))
    return phi, base, y_phi, Phi


# ---------------------------------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------------------------------

def test_build_pair_table_gate_and_symmetry():
    rng = np.random.default_rng(0)
    n, P = 50, 8
    phi = rng.normal(size=(n, P))
    # make features 0,1,2 the high-|phi| head; 3..7 low
    phi[:, :3] *= 10.0
    Phi = rng.normal(size=(n, P, P))
    Phi = 0.5 * (Phi + np.transpose(Phi, (0, 2, 1)))  # symmetric per row
    pair_rows, in_gate = build_pair_table(Phi, phi, interaction_top_k=3)
    assert in_gate.sum() == 3
    assert in_gate[0] and in_gate[1] and in_gate[2]
    # gated pair carries per-row values; non-gated pair is zeroed (the top-k gate)
    assert np.allclose(pair_rows[0, 1], Phi[:, 0, 1])
    assert np.allclose(pair_rows[0, 1], pair_rows[1, 0])  # symmetric
    assert np.allclose(pair_rows[3, 4], 0.0)  # both non-gated -> zero
    assert np.allclose(pair_rows[0, 5], 0.0)  # one non-gated -> zero
    # diagonal never set
    assert np.allclose(pair_rows[0, 0], 0.0)


def test_interaction_proxy_per_row_not_collapsed():
    """The interaction term must stay PER-ROW: an XOR pair has row-mean Phi_ij ~0 but a large per-row
    swing. A scalar-mean collapse would erase the signal -> this guards that regression."""
    X, y = _two_xor_pairs(seed=0)
    phi, base, y_phi, Phi = _shap_and_tensor(X, y, 0)
    pair_rows, in_gate = build_pair_table(Phi, phi, interaction_top_k=30)
    # pick the two operands of the first XOR pair by name
    names = list(X.columns)
    ia, ib = names.index("xa0"), names.index("xb0")
    assert in_gate[ia] and in_gate[ib]
    per_row = 2.0 * pair_rows[ia, ib]
    assert np.abs(per_row).mean() > 5 * np.abs(per_row.mean())  # per-row swing dwarfs its mean


def test_interaction_proxy_top_n_smoke_and_candidate_rescore():
    X, y = _two_xor_pairs(seed=0)
    phi, base, y_phi, Phi = _shap_and_tensor(X, y, 0)
    cands = interaction_proxy_top_n(
        phi, Phi, base, y_phi, classification=True, metric="brier",
        min_card=1, max_card=2, top_n=10, interaction_top_k=30,
        candidate_subsets=[(0,), (0, 1)])
    assert cands and all(np.isfinite(l) for l, _ in cands)
    # a true XOR pair should appear among the best candidates
    names = list(X.columns)
    pair = tuple(sorted((names.index("xa0"), names.index("xb0"))))
    top_keys = {tuple(sorted(c)) for _l, c in cands[:5]}
    assert pair in top_keys or tuple(sorted((names.index("xa1"), names.index("xb1")))) in top_keys


# ---------------------------------------------------------------------------------------------------
# biz_value: interaction proxy beats additive on competing-XOR by a pinned REPLICATED margin
# ---------------------------------------------------------------------------------------------------

def _honest_auc(X, y, sel, seed):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    if not sel:
        return 0.5
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    m = _xgb(seed + 1)
    m.fit(Xtr[sel], ytr)
    if len(np.unique(yte)) < 2:
        return 0.5
    return float(roc_auc_score(yte, m.predict_proba(Xte[sel])[:, 1]))


def _best_subset(cands, names):
    return [names[i] for i in cands[0][1]] if cands else []


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_biz_val_interaction_proxy_beats_additive_on_competing_xor(seed):
    """Floor +0.12 honest-holdout AUC; measured ~+0.22..+0.26 replicated across seeds 0/1/2.

    At cap=2 the additive proxy keeps one operand per XOR pair (AUC ~chance); the interaction proxy
    keeps a full pair (AUC ~0.76). A regression in the per-row interaction term drops the ratio to ~0."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import brute_force_top_n

    X, y = _two_xor_pairs(seed=seed)
    # search-split SHAP, exactly the selector machinery, on a disjoint search subset
    from sklearn.model_selection import train_test_split
    Xs, _Xh, ys, _yh = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    Xs = Xs.reset_index(drop=True)
    phi, base, y_phi, Phi = _shap_and_tensor(Xs, ys, seed)
    P = phi.shape[1]
    add_c = brute_force_top_n(phi, base, y_phi, classification=True, metric="brier",
                              min_card=1, max_card=min(2, P), top_n=30, parallel=False)
    int_c = interaction_proxy_top_n(
        phi, Phi, base, y_phi, classification=True, metric="brier",
        min_card=1, max_card=min(2, P), top_n=30, interaction_top_k=30,
        candidate_subsets=[c for _l, c in add_c])
    names = list(Xs.columns)
    a_auc = _honest_auc(X, y, _best_subset(add_c, names), seed)
    i_auc = _honest_auc(X, y, _best_subset(int_c, names), seed)
    assert i_auc - a_auc >= 0.12, f"seed={seed}: interaction {i_auc:.4f} vs additive {a_auc:.4f}"


def test_biz_val_interaction_proxy_no_regression_on_additive_bed():
    """On a purely additive bed the interaction proxy must NOT pick a worse subset than additive.

    Additive features have full marginal |phi|, so the additive search already recovers them; the
    interaction term is small and must not flip the pick. Floor: interaction AUC >= additive - 0.02."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import brute_force_top_n
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(0)
    n = 1500
    cols = {}
    logit = np.zeros(n)
    for k in range(4):
        v = rng.normal(size=n)
        cols[f"ad{k}"] = v
        logit = logit + (1.5 - 0.2 * k) * v
    for j in range(20):
        cols[f"nz{j}"] = rng.normal(size=n)
    X = pd.DataFrame(cols)
    p = 1.0 / (1.0 + np.exp(-(logit - logit.mean())))
    y = (rng.uniform(size=n) < p).astype(int)
    Xs, _Xh, ys, _yh = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    Xs = Xs.reset_index(drop=True)
    phi, base, y_phi, Phi = _shap_and_tensor(Xs, ys, 0)
    P = phi.shape[1]
    add_c = brute_force_top_n(phi, base, y_phi, classification=True, metric="brier",
                              min_card=1, max_card=min(4, P), top_n=30, parallel=False)
    int_c = interaction_proxy_top_n(
        phi, Phi, base, y_phi, classification=True, metric="brier",
        min_card=1, max_card=min(4, P), top_n=30, interaction_top_k=30,
        candidate_subsets=[c for _l, c in add_c])
    names = list(Xs.columns)
    a_auc = _honest_auc(X, y, _best_subset(add_c, names), 0)
    i_auc = _honest_auc(X, y, _best_subset(int_c, names), 0)
    assert i_auc >= a_auc - 0.02, f"interaction regressed additive bed: {i_auc:.4f} vs {a_auc:.4f}"


def test_proxy_mode_param_validation_and_roundtrip():
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    with pytest.raises(ValueError):
        ShapProxiedFS(proxy_mode="bogus")
    s = ShapProxiedFS(proxy_mode="interaction", interaction_proxy_top_k=20)
    # raw value preserved verbatim (no constructor mutation -> sklearn clone identity safe for these params)
    params = s.get_params(deep=False)
    assert params["proxy_mode"] == "interaction" and params["interaction_proxy_top_k"] == 20
    s2 = ShapProxiedFS(**params)
    assert s2.proxy_mode is params["proxy_mode"]  # identity preserved (clone-safe)
    assert ShapProxiedFS().proxy_mode == "additive"  # default unchanged
