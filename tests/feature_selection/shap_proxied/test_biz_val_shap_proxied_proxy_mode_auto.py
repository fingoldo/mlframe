"""biz_val + unit tests for ``proxy_mode="auto"`` (gt_08): the new ShapProxiedFS default.

"auto" ALWAYS runs the cheap su_seeded synergy screen (regardless of the ``su_seeded_interactions``
flag) and lets its permutation-null SNR gate decide the branch: empty ``kept`` -> the additive path
runs BYTE-IDENTICAL to ``proxy_mode="additive"`` (the screen result is discarded); non-empty ``kept``
-> the same operand-rescue + sparse-candidate-augmentation path ``su_seeded_interactions=True``
already ships. See ``research/gt_08_interaction_levers_auto_default.md`` section 3 for the design and
``src/mlframe/feature_selection/_benchmarks/bench_shap_interaction_proxy.py`` for the pre-flip
6-bed x 3-seed bench that validated the gate stays silent on every additive bed.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import train_test_split

pytest.importorskip("shap")
pytest.importorskip("xgboost")

from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS


def _shap_sel(proxy_mode, n_features, seed=0):
    """A shared small-fixture-sized ShapProxiedFS config, varying only proxy_mode and seed."""
    return ShapProxiedFS(
        classification=True,
        n_splits=3,
        top_n=20,
        min_features=8,
        prefilter_top=min(60, n_features),
        prefilter_n_estimators=60,
        oof_shap_n_estimators=60,
        revalidation_n_estimators=60,
        n_revalidation_models=2,
        trust_guard=True,
        trust_guard_n_estimators=20,
        cluster_features="auto",
        within_cluster_refine=True,
        parsimony_tol=0.005,
        proxy_mode=proxy_mode,
        random_state=seed,
        verbose=False,
    )


def _honest_auc(X, y, sel_names, seed=0):
    """AUC of a fresh XGBClassifier trained on sel_names, evaluated on a held-out split of X/y."""
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb

    if not sel_names:
        return 0.5
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    m = xgb.XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.15, subsample=0.9, n_jobs=1, tree_method="hist", verbosity=0, random_state=seed + 1)
    m.fit(Xtr[list(sel_names)], ytr)
    if len(np.unique(yte)) < 2:
        return 0.5
    return float(roc_auc_score(yte, m.predict_proba(Xte[list(sel_names)])[:, 1]))


def _xor_bed(n=2000, p_noise=195, seed=0):
    """4 additive informative (weight 1.0) + 1 XOR pair (weight 1.5, sign(a*b)) + noise."""
    rng = np.random.default_rng(seed)
    add_feats = {f"add{k}": rng.standard_normal(n) for k in range(4)}
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    logit = sum(1.0 * v for v in add_feats.values()) + 1.5 * np.sign(a * b)
    cols = dict(add_feats)
    cols["op_a"] = a
    cols["op_b"] = b
    for i in range(p_noise):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    order = list(X.columns)
    rng.shuffle(order)
    X = X[order]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return X, pd.Series(y, name="target")


def _pure_additive_bed(n=2000, p_noise=194, n_informative=6, seed=0):
    """6 informative additive features + noise, no interactions anywhere."""
    rng = np.random.default_rng(seed)
    cols = {}
    logit = np.zeros(n)
    for k in range(n_informative):
        v = rng.standard_normal(n)
        cols[f"inf{k}"] = v
        logit = logit + (1.6 - 0.15 * k) * v
    for i in range(p_noise):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    order = list(X.columns)
    rng.shuffle(order)
    X = X[order]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return X, pd.Series(y, name="target")


def _noise_buried_interaction_bed(n=2000, p_noise=190, seed=0):
    """A real interaction pair buried in 190 noise cols at low SNR (hard_synth-style): the linear
    signal dominates and the a*b term's contribution is tiny, so synergy stays below the SNR floor."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, 6))
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    logit = base @ np.array([1.4, -1.1, 0.9, 0.0, 0.0, 0.0]) + 0.12 * (a * b)
    cols = {f"base_{i}": base[:, i] for i in range(6)}
    cols["ia"] = a
    cols["ib"] = b
    for i in range(p_noise):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    order = list(X.columns)
    rng.shuffle(order)
    X = X[order]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return X, pd.Series(y, name="target")


# --------------------------------------------------------------------------------------------------
# biz_val 1: gate fires on the XOR bed -- recall lift + honest-AUC lift over additive.
# --------------------------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.timeout(900)
def test_biz_val_proxy_mode_auto_gate_fires_on_xor_bed():
    """auto recovers both XOR operands and beats additive's honest AUC; additive recovers neither."""
    X, y = _xor_bed(n=2000, p_noise=195, seed=0)
    n_features = X.shape[1]
    Xtr, _Xte, ytr, _yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    Xtr = Xtr.reset_index(drop=True)
    ytr = ytr.reset_index(drop=True)

    sel_auto = _shap_sel("auto", n_features)
    sel_auto.fit(Xtr, ytr)
    sel_add = _shap_sel("additive", n_features)
    sel_add.fit(Xtr, ytr)

    resolved = sel_auto.shap_proxy_report_.get("proxy_mode_resolved")
    assert resolved == "interaction(auto:gate-fired)", f"gate did not fire on the designed XOR bed: {resolved}"

    xor_pair = {"op_a", "op_b"}
    auto_recall = len(xor_pair & set(sel_auto.selected_features_))
    add_recall = len(xor_pair & set(sel_add.selected_features_))
    assert auto_recall == 2, f"auto recovered {auto_recall}/2 XOR operands: {sorted(sel_auto.selected_features_)}"
    assert add_recall == 0, f"additive unexpectedly recovered {add_recall}/2 XOR operands (bed premise broken): {sorted(sel_add.selected_features_)}"

    auto_auc = _honest_auc(X, y, sel_auto.selected_features_, seed=0)
    add_auc = _honest_auc(X, y, sel_add.selected_features_, seed=0)
    assert auto_auc >= add_auc + 0.03, f"auto honest AUC {auto_auc:.4f} did not beat additive {add_auc:.4f} by >=0.03"


# --------------------------------------------------------------------------------------------------
# biz_val 2: gate stays silent on a pure additive bed -- byte-identical selection, bounded overhead.
# --------------------------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.timeout(900)
def test_biz_val_proxy_mode_auto_gate_silent_on_additive_bed():
    """auto's selection is byte-identical to additive's and its wall stays within 1.10x on a purely additive bed."""
    X, y = _pure_additive_bed(n=2000, p_noise=194, n_informative=6, seed=0)
    n_features = X.shape[1]

    sel_add = _shap_sel("additive", n_features)
    t0 = time.perf_counter()
    sel_add.fit(X, y)
    add_wall = time.perf_counter() - t0

    sel_auto = _shap_sel("auto", n_features)
    t0 = time.perf_counter()
    sel_auto.fit(X, y)
    auto_wall = time.perf_counter() - t0

    rep = sel_auto.shap_proxy_report_.get("su_seeded_interactions", {})
    assert rep.get("n_kept_pairs", -1) == 0, f"screen unexpectedly kept pairs on a purely additive bed: {rep}"
    assert (
        sel_auto.selected_features_ == sel_add.selected_features_
    ), f"auto's gate-silent selection diverged from additive:\n auto={sel_auto.selected_features_}\n add ={sel_add.selected_features_}"
    assert auto_wall <= 1.10 * add_wall, f"auto wall {auto_wall:.2f}s exceeded 1.10x additive wall {add_wall:.2f}s"


# --------------------------------------------------------------------------------------------------
# biz_val 3: gate stays silent when the interaction is buried below the permutation-null SNR floor.
# --------------------------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.timeout(900)
def test_biz_val_proxy_mode_auto_noop_on_noise_buried_interactions():
    """the permutation-null SNR gate stays silent when the interaction is buried below its floor."""
    X, y = _noise_buried_interaction_bed(n=2000, p_noise=190, seed=3)
    n_features = X.shape[1]

    sel_auto = _shap_sel("auto", n_features)
    sel_auto.fit(X, y)
    sel_add = _shap_sel("additive", n_features)
    sel_add.fit(X, y)

    rep = sel_auto.shap_proxy_report_.get("su_seeded_interactions", {})
    assert rep.get("n_kept_pairs", -1) == 0, f"gate should stay silent on a noise-buried interaction: {rep}"
    assert sel_auto.selected_features_ == sel_add.selected_features_, "gate-silent auto must match additive exactly"


# --------------------------------------------------------------------------------------------------
# Unit tests: validator, default, additive skips the screen entirely, clone() round-trip.
# --------------------------------------------------------------------------------------------------
def test_proxy_mode_validator_accepts_auto_and_legacy_values():
    """the validator accepts "auto" alongside the legacy modes, case-insensitively."""
    for mode in ("additive", "interaction", "auto", "AUTO", "Additive"):
        ShapProxiedFS(proxy_mode=mode)  # must not raise


def test_proxy_mode_validator_rejects_garbage():
    """an unrecognized proxy_mode value raises ValueError."""
    with pytest.raises(ValueError):
        ShapProxiedFS(proxy_mode="bogus")


def test_proxy_mode_default_is_auto():
    """the constructor default for proxy_mode is now "auto", not "additive"."""
    sel = ShapProxiedFS()
    assert sel.proxy_mode == "auto"


@pytest.mark.timeout(120)
def test_proxy_mode_additive_skips_screen_entirely():
    """The legacy escape hatch must never pay even the screen's O(P)+O(K) cost."""
    n = 600
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"f{i}": rng.standard_normal(n) for i in range(20)})
    y = pd.Series((rng.random(n) < 0.5).astype(int))

    sel = _shap_sel("additive", n_features=20)
    sel.fit(X, y)
    assert "su_seeded_interactions" not in sel.shap_proxy_report_
    assert sel.shap_proxy_report_.get("proxy_mode_resolved") == "additive"


def test_proxy_mode_auto_sklearn_clone_roundtrip():
    """sklearn.clone() preserves proxy_mode="auto" and other constructor params, unfitted."""
    sel = ShapProxiedFS(proxy_mode="auto", random_state=5)
    cloned = clone(sel)
    assert cloned.proxy_mode == "auto"
    assert cloned.random_state == 5
    assert not hasattr(cloned, "shap_proxy_report_")
