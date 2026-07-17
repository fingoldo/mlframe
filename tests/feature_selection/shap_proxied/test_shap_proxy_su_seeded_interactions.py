"""Unit + biz_val for the su_seeded_interactions path (lever A4-4).

ShapProxiedFS's additive coalition proxy ``base + sum_{j in S} phi_j`` is BLIND to PURE interactions:
for ``y = sign(a * b)`` each operand has ~0 marginal SHAP/SU, so the informative pair looks like noise
and is dropped by the marginal-importance prefilter/prescreen. The dense ``interaction_aware`` tensor
fixes this but is O(P^2) (gated to phi<=16, a no-op on wide data).

``su_seeded_interactions`` (opt-in, default OFF) is the CHEAP fix: a pairwise-SU SYNERGY screen
    synergy(a, b ; y) = SU(joint_bin(a, b) ; y) - max(SU(a ; y), SU(b ; y))
ranks candidate interaction pairs at O(P)+O(K) (no tensor), a permutation-null SNR gate skips
noise-buried pairs, and the surviving operands are rescued past the prefilter/prescreen + paired by a
sparse interaction objective on ONLY the K survivors.

Three falsifiable checks:
 1. SCREEN: on a designed ``y=sign(a*b)`` bed the screen ranks the TRUE operand pair #0 (premise).
 2. SNR no-op: on pure noise the gate clears nothing -> empty kept-pairs (the documented hard_synth
    behaviour), and ``su_seeded_interactions=True`` then leaves the additive default byte-identical.
 3. FACADE lift (opt-in): the in-class path recovers BOTH operands of a pure interaction the additive
    default misses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------------------------------
# 1. SCREEN unit test -- no model, no SHAP; pure SU synergy ranking + SNR gate.
# --------------------------------------------------------------------------------------------------
def _pure_interaction(n=4000, p_noise=30, seed=0):
    """y = sign(a*b): operands a, b have ~0 MARGINAL signal; ALL signal is the interaction."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)  # one weak linear control so something has a marginal
    logit = 2.2 * np.sign(a * b) + 0.7 * c
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    cols = {"op_a": a, "op_b": b, "lin_c": c}
    for i in range(p_noise):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    order = list(X.columns)
    rng.shuffle(order)
    return X[order], pd.Series(y, name="target")


def test_su_synergy_screen_ranks_true_pair_first():
    """The cheap pairwise-SU synergy screen ranks the true a*b operand pair #0 (and clears the SNR
    gate), on a bed where each operand has ~0 marginal signal -- the whole premise of A4-4."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_interactions import su_synergy_screen

    X, y = _pure_interaction(n=4000, p_noise=30, seed=0)
    kept, info = su_synergy_screen(
        X, y, n_bins=8, top_k=20, max_screen_cols=120, snr_z=3.0, snr_null_quantile=0.99, n_permutations=3, rng=np.random.default_rng(0)
    )

    assert kept, f"screen kept nothing despite a strong pure interaction (info={info})"
    # The TOP-synergy surviving pair must be exactly {op_a, op_b}. su_synergy_screen returns
    # (synergy, joint_su, col_a, col_b) tuples, best-synergy first.
    top_syn, _top_jsu, top_a, top_b = kept[0]
    assert {top_a, top_b} == {"op_a", "op_b"}, f"true operand pair not ranked #0; got {top_a} x {top_b} (info={info})"
    # The true pair's synergy must clear the SNR gate by a wide margin (it is the only real pair).
    assert top_syn >= info["gate"], f"top synergy {top_syn} below SNR gate {info['gate']}"
    assert info["best_synergy"] >= 5 * info["gate"], f"true synergy {info['best_synergy']} not well above the noise floor {info['gate']}"


def test_su_synergy_screen_snr_gate_no_ops_on_noise():
    """On a target with NO real dependence the SNR gate must clear nothing (the hard_synth /
    noise-buried regime) -> empty kept-pairs, so the caller no-ops."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_interactions import su_synergy_screen

    rng = np.random.default_rng(1)
    n = 4000
    X = pd.DataFrame({f"noise_{i}": rng.standard_normal(n) for i in range(30)})
    y = pd.Series((rng.random(n) < 0.5).astype(int), name="target")  # pure noise target

    kept, info = su_synergy_screen(
        X, y, n_bins=8, top_k=8, max_screen_cols=120, snr_z=3.0, snr_null_quantile=0.99, n_permutations=3, rng=np.random.default_rng(0)
    )

    assert kept == [], f"SNR gate seeded noise pairs (kept={kept}, info={info})"
    assert info["best_synergy"] < info["gate"], f"best noise synergy {info['best_synergy']} should sit below the gate {info['gate']}"


# --------------------------------------------------------------------------------------------------
# 2 + 3. FACADE: opt-in lift on a pure interaction + byte-identical no-op when the gate clears nothing.
# --------------------------------------------------------------------------------------------------
pytest.importorskip("shap")
pytest.importorskip("xgboost")


def _shap_sel(su_seeded, n_features):
    """Shap sel."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    return ShapProxiedFS(
        classification=True,
        n_splits=3,
        top_n=20,
        min_features=8,
        prefilter_top=min(40, n_features),
        prefilter_n_estimators=60,
        oof_shap_n_estimators=60,
        revalidation_n_estimators=60,
        n_revalidation_models=2,
        trust_guard=True,
        trust_guard_n_estimators=20,
        cluster_features="auto",
        within_cluster_refine=True,
        parsimony_tol=0.005,
        su_seeded_interactions=su_seeded,
        random_state=0,
        verbose=False,
    )


@pytest.mark.slow
def test_su_seeded_facade_recovers_pure_interaction_operands():
    """OPT-IN lift: on a pure-interaction bed where the operands have ~0 marginal signal, the additive
    default starves and misses the pair while ``su_seeded_interactions=True`` rescues + pairs them and
    recovers BOTH operands. Fitting on a 60% train split (mirroring the recipe bench) keeps the
    additive default starved enough to demonstrate the recall lift."""
    from sklearn.model_selection import train_test_split

    X, y = _pure_interaction(n=5000, p_noise=60, seed=0)
    Xtr, _Xte, ytr, _yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    Xtr = Xtr.reset_index(drop=True)
    ytr = ytr.reset_index(drop=True)
    n_features = Xtr.shape[1]

    sel_off = _shap_sel(False, n_features)
    sel_off.fit(Xtr, ytr)
    sel_on = _shap_sel(True, n_features)
    sel_on.fit(Xtr, ytr)

    rep = sel_on.shap_proxy_report_.get("su_seeded_interactions", {})
    assert rep.get("applied") is True
    # The screen must have kept at least one synergistic pair above the SNR gate (the report's
    # kept_pairs are in PROXY-column space -- clustering renames operands to ``unitK`` -- so the
    # true-operand assertion is made on the final selection, which is in ORIGINAL-name space).
    assert rep.get("n_kept_pairs", 0) >= 1, f"screen kept nothing: {rep}"
    # The opt-in path must recover BOTH operands...
    on_set = set(sel_on.selected_features_)
    assert {"op_a", "op_b"} <= on_set, f"su_seeded did not recover the operand pair: {sorted(on_set)}"
    # ...and beat the starved additive default's operand recall.
    off_set = set(sel_off.selected_features_)
    on_recall = len({"op_a", "op_b"} & on_set)
    off_recall = len({"op_a", "op_b"} & off_set)
    assert on_recall > off_recall, (
        f"su_seeded recall {on_recall} did not beat additive-default recall {off_recall} (off={sorted(off_set)}, on={sorted(on_set)})"
    )


@pytest.mark.slow
def test_su_seeded_no_op_leaves_additive_default_byte_identical():
    """When the SNR gate clears NO pair (noise-buried interaction), the opt-in path must be a true
    no-op: the same selection as ``su_seeded_interactions=False``. The screen uses an ISOLATED rng so
    a no-op never perturbs the downstream stochastic revalidation."""
    # A bed with a real but DEEPLY-buried interaction below the noise floor + plain noise: the gate
    # should clear nothing here (mirrors hard_synth). We build a target driven by a LINEAR signal only,
    # plus an a*b pair whose contribution is dominated by noise so its synergy stays below the floor.
    rng = np.random.default_rng(3)
    n = 5000
    base = rng.standard_normal((n, 6))
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    # linear-dominated target; the a*b term is tiny relative to the linear signal + label noise.
    logit = base @ np.array([1.4, -1.1, 0.9, 0.0, 0.0, 0.0]) + 0.12 * (a * b)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    cols = {f"base_{i}": base[:, i] for i in range(6)}
    cols["ia"] = a
    cols["ib"] = b
    for i in range(40):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    order = list(X.columns)
    rng.shuffle(order)
    X = X[order]
    yser = pd.Series(y, name="target")
    n_features = X.shape[1]

    sel_off = _shap_sel(False, n_features)
    sel_off.fit(X, yser)
    sel_on = _shap_sel(True, n_features)
    sel_on.fit(X, yser)

    rep = sel_on.shap_proxy_report_.get("su_seeded_interactions", {})
    assert rep.get("applied") is True
    # Premise of THIS test: the buried interaction is below the SNR floor -> nothing seeded.
    assert rep.get("n_kept_pairs", 0) == 0, f"expected SNR-gate no-op but kept {rep.get('kept_pairs')}; tune the bed if the floor moved"
    # No-op => byte-identical selection to the additive default.
    assert sel_on.selected_features_ == sel_off.selected_features_, (
        f"no-op path changed the additive default:\n off={sel_off.selected_features_}\n on ={sel_on.selected_features_}"
    )
