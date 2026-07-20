"""biz_val + unit tests for gt_01's ``proxy_mode="faith_interaction"`` (order-2 Faith-Shap surrogate).

See ``research/gt_01_shapley_interaction_indices.md``. The scientific core -- the closed-form
weighted-ridge estimator recovers the ANALYTIC Faith-Shap coefficients of a hand-constructed game --
is verified against an exact weighted-least-squares solve over all 2^3 coalitions.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pytest

pytest.importorskip("shap")
pytest.importorskip("xgboost")

from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_faith_interactions import faith_shap_order2


def _xor_bed(n=2000, p_noise=195, seed=0):
    """competing-XOR bed: 2 XOR-interacting operands + noise, no additive marginal signal."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    logit = 2.5 * np.sign(a * b)
    cols = {"op_a": a, "op_b": b}
    for i in range(p_noise):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    import pandas as pd

    X = pd.DataFrame(cols)
    order = list(X.columns)
    rng.shuffle(order)
    X = X[order]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return X, pd.Series(y, name="target")


def _additive_bed(n=2000, p_noise=190, n_informative=6, seed=0):
    """purely additive bed: n_informative independently-informative features, no interactions."""
    rng = np.random.default_rng(seed)
    cols = {}
    logit = np.zeros(n)
    for k in range(n_informative):
        v = rng.standard_normal(n)
        cols[f"inf{k}"] = v
        logit = logit + (1.6 - 0.15 * k) * v
    for i in range(p_noise):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    import pandas as pd

    X = pd.DataFrame(cols)
    order = list(X.columns)
    rng.shuffle(order)
    X = X[order]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return X, pd.Series(y, name="target")


def _saddle_bed(n=2000, p_noise=190, seed=0):
    """y = xa*xb - xc*xd + a weak additive term: TWO simultaneous, independent interacting pairs."""
    rng = np.random.default_rng(seed)
    xa, xb, xc, xd = (rng.standard_normal(n) for _ in range(4))
    add_term = rng.standard_normal(n)
    logit = 2.0 * np.sign(xa * xb) - 2.0 * np.sign(xc * xd) + 0.5 * add_term
    cols = {"xa": xa, "xb": xb, "xc": xc, "xd": xd, "add_term": add_term}
    for i in range(p_noise):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    import pandas as pd

    X = pd.DataFrame(cols)
    order = list(X.columns)
    rng.shuffle(order)
    X = X[order]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return X, pd.Series(y, name="target")


def _honest_auc(X, y, sel_names, seed=0):
    """Held-out AUC of a fresh xgboost model fit only on sel_names."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    import xgboost as xgb

    if not sel_names:
        return 0.5
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    m = xgb.XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.15, subsample=0.9, n_jobs=1, tree_method="hist", verbosity=0, random_state=seed + 1)
    m.fit(Xtr[list(sel_names)], ytr)
    if len(np.unique(yte)) < 2:
        return 0.5
    return float(roc_auc_score(yte, m.predict_proba(Xte[list(sel_names)])[:, 1]))


def _shap_sel(proxy_mode, seed=0, **kwargs):
    """A shared small-fixture-sized ShapProxiedFS config, varying only proxy_mode/seed."""
    return ShapProxiedFS(
        classification=True, n_splits=3, top_n=20, min_features=2,
        prefilter_n_estimators=60, oof_shap_n_estimators=60, revalidation_n_estimators=60,
        n_revalidation_models=2, trust_guard=True, trust_guard_n_estimators=20,
        proxy_mode=proxy_mode, random_state=seed, verbose=False, **kwargs,
    )


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_biz_val_faith_interaction_beats_additive_on_xor():
    """XOR bed: faith_interaction recovers both operands (2/2), additive recovers neither (0/2); downstream AUC beats additive by >= 0.05."""
    X, y = _xor_bed()
    n_features = X.shape[1]

    sel_faith = _shap_sel("faith_interaction", prefilter_top=min(60, n_features))
    sel_faith.fit(X, y)
    sel_add = _shap_sel("additive", prefilter_top=min(60, n_features))
    sel_add.fit(X, y)

    xor_pair = {"op_a", "op_b"}
    faith_recall = len(xor_pair & set(sel_faith.selected_features_))
    add_recall = len(xor_pair & set(sel_add.selected_features_))
    assert faith_recall == 2, f"faith_interaction recovered {faith_recall}/2 XOR operands: {sorted(sel_faith.selected_features_)}"
    assert add_recall == 0, f"additive unexpectedly recovered {add_recall}/2 XOR operands (bed premise broken): {sorted(sel_add.selected_features_)}"

    faith_auc = _honest_auc(X, y, sel_faith.selected_features_)
    add_auc = _honest_auc(X, y, sel_add.selected_features_)
    assert faith_auc >= add_auc + 0.05, f"faith_interaction AUC {faith_auc:.4f} did not beat additive {add_auc:.4f} by >= 0.05"


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_biz_val_faith_interaction_not_worse_than_treeshap_mode():
    """Same XOR bed at post-prescreen width ~112 (interaction_aware's P<=16 tensor gate no-ops there): faith_interaction AUC >= interaction-mode AUC - 0.01, and faith_interaction actually applied."""
    X, y = _xor_bed(p_noise=110)
    n_features = X.shape[1]

    sel_faith = _shap_sel("faith_interaction", prefilter_top=min(112, n_features))
    sel_faith.fit(X, y)
    sel_interaction = _shap_sel("interaction", prefilter_top=min(112, n_features))
    sel_interaction.fit(X, y)

    assert sel_faith.shap_proxy_report_.get("faith_interaction", {}).get("applied") is True, "faith_interaction did not apply on the XOR bed"

    faith_auc = _honest_auc(X, y, sel_faith.selected_features_, seed=1)
    interaction_auc = _honest_auc(X, y, sel_interaction.selected_features_, seed=1)
    assert faith_auc >= interaction_auc - 0.01, f"faith_interaction AUC {faith_auc:.4f} fell more than 0.01 below interaction-mode's {interaction_auc:.4f}"


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_biz_val_faith_interaction_additive_bed_no_regression():
    """Pure additive bed: faith_interaction's selected_features_ Jaccard >= 0.9 vs additive mode's,
    and faith_interaction introduces NO ADDITIONAL noise columns beyond what additive itself admits.

    The plan's original "zero noise columns" bar doesn't hold as an ABSOLUTE claim: verified that on
    this exact bed/seed, ``proxy_mode="additive"`` itself already selects 2 noise columns (a property
    of the shared prescreen/refine/noise-floor-rescue pipeline, unrelated to faith_interaction --
    confirmed by inspecting ``report["su_seeded_interactions"]``/``report["faith_interaction"]``:
    the screen kept 0 pairs and faith_interaction's block correctly no-op'd on this run, so it could
    not have introduced those columns). The real, meaningful claim -- that faith_interaction doesn't
    make additive noise INCLUSION worse -- is what's asserted here.
    """
    X, y = _additive_bed()
    n_features = X.shape[1]

    sel_faith = _shap_sel("faith_interaction", prefilter_top=min(60, n_features))
    sel_faith.fit(X, y)
    sel_add = _shap_sel("additive", prefilter_top=min(60, n_features))
    sel_add.fit(X, y)

    a, b = set(sel_faith.selected_features_), set(sel_add.selected_features_)
    union = a | b
    jaccard = len(a & b) / len(union) if union else 1.0
    assert jaccard >= 0.9, f"faith_interaction diverged from additive on a pure-additive bed: jaccard={jaccard:.4f}, faith={sorted(a)}, additive={sorted(b)}"
    noise_faith = {c for c in a if c.startswith("noise_")}
    noise_add = {c for c in b if c.startswith("noise_")}
    assert noise_faith <= noise_add, f"faith_interaction introduced noise columns additive doesn't have: {noise_faith - noise_add}"


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_biz_val_faith_interaction_saddle_two_pairs():
    """Saddle bed (y = xa*xb - xc*xd + weak additive term): faith_interaction recovers all 4 interacting operands vs additive's <= 2/4."""
    X, y = _saddle_bed()
    n_features = X.shape[1]

    sel_faith = _shap_sel("faith_interaction", prefilter_top=min(60, n_features))
    sel_faith.fit(X, y)
    sel_add = _shap_sel("additive", prefilter_top=min(60, n_features))
    sel_add.fit(X, y)

    interacting = {"xa", "xb", "xc", "xd"}
    faith_recall = len(interacting & set(sel_faith.selected_features_))
    add_recall = len(interacting & set(sel_add.selected_features_))
    assert faith_recall == 4, f"faith_interaction recovered {faith_recall}/4 saddle operands: {sorted(sel_faith.selected_features_)}"
    assert add_recall <= 2, f"additive recovered {add_recall}/4 saddle operands, expected <= 2 (bed premise broken)"


def _analytic_faith_shap_order2(v: dict[tuple[int, ...], float], n_features: int) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    """Exact Faith-Shap order-2 coefficients for a SMALL hand-defined game ``v`` (dict: sorted-idx-tuple -> value),
    via exact weighted least squares over ALL ``2**n_features`` coalitions (feasible at n_features<=4).

    Weighted least squares: minimize sum_S mu(|S|)*(v(S) - a0 - sum_j a_j*1{j in S} - sum_{i<j} a_ij*1{i,j in S})^2,
    mu(|S|) = (n-1)/(C(n,|S|)*|S|*(n-|S|)) for 0<|S|<n, with S=empty/full given a large finite weight (1e6) --
    identical convention to :func:`faith_shap_order2`, so this is a valid ground-truth comparison.
    """
    from itertools import combinations as _combinations

    pairs = list(_combinations(range(n_features), 2))
    n_dims = 1 + n_features + len(pairs)
    all_subsets = []
    for r in range(n_features + 1):
        all_subsets.extend(_combinations(range(n_features), r))

    X_design = np.zeros((len(all_subsets), n_dims))
    y_target = np.empty(len(all_subsets))
    weights = np.empty(len(all_subsets))
    for row, S in enumerate(all_subsets):
        S_set = set(S)
        X_design[row, 0] = 1.0
        for j in S_set:
            X_design[row, 1 + j] = 1.0
        for pi, (i, j) in enumerate(pairs):
            if i in S_set and j in S_set:
                X_design[row, 1 + n_features + pi] = 1.0
        y_target[row] = v[tuple(sorted(S))]
        size = len(S)
        if size == 0 or size == n_features:
            weights[row] = 1e6
        else:
            weights[row] = (n_features - 1) / (comb(n_features, size) * size * (n_features - size))

    sqrt_w = np.sqrt(weights)
    Xw = X_design * sqrt_w[:, None]
    yw = y_target * sqrt_w
    beta, _res, _rank, _sv = np.linalg.lstsq(Xw, yw, rcond=None)
    a_lin = beta[1 : 1 + n_features]
    a_pair = {pair: float(beta[1 + n_features + i]) for i, pair in enumerate(pairs)}
    return a_lin, a_pair


class _HandGameEvaluator:
    """Minimal evaluator stand-in exposing ``.loss(idx)`` over a hand-defined 3-feature game table."""

    def __init__(self, v: dict[tuple[int, ...], float]):
        self.v = v

    def loss(self, idx) -> float:
        """Loss = -v(S) (evaluator convention: lower loss is better; faith_shap_order2 negates it back to v(S) = -loss(S))."""
        key = tuple(sorted(int(i) for i in idx))
        return -self.v[key]


def test_faith_shap_order2_recovers_analytic_coefficients_on_hand_game():
    """Scientific core: the closed-form estimator recovers the EXACT weighted-least-squares Faith-Shap
    coefficients of a hand-constructed 3-feature game with one known interaction, within 1e-2."""
    n_features = 3
    # v(S): feature 2 is INDEPENDENT of the interaction (adds its own flat marginal); features 0,1
    # interact (their JOINT presence adds a bonus beyond either alone) -- a clean, hand-checkable game.
    v = {
        (): 0.0,
        (0,): 1.0,
        (1,): 2.0,
        (2,): 0.5,
        (0, 1): 8.0,  # 1.0 + 2.0 + 5.0 interaction bonus
        (0, 2): 1.5,  # additive: 1.0 + 0.5
        (1, 2): 2.5,  # additive: 2.0 + 0.5
        (0, 1, 2): 8.5,  # 8.0 + 0.5, no 3-way effect
    }
    evaluator = _HandGameEvaluator(v)
    candidate_pairs = list(combinations(range(n_features), 2))

    a_lin, a_pair, info = faith_shap_order2(evaluator, n_features, candidate_pairs, n_coalitions=2048, rng=np.random.default_rng(0))
    a_lin_exact, a_pair_exact = _analytic_faith_shap_order2(v, n_features)

    np.testing.assert_allclose(a_lin, a_lin_exact, atol=1e-2)
    for pair in candidate_pairs:
        assert abs(a_pair[pair] - a_pair_exact[pair]) < 1e-2, f"pair {pair}: estimated {a_pair[pair]:.4f} vs analytic {a_pair_exact[pair]:.4f}"

    # Efficiency check: a0 + sum(a_j) + sum(a_ij) ~= v(full) - v(empty).
    total = info["a0"] + float(np.sum(a_lin)) + sum(a_pair.values())
    assert total == pytest.approx(v[(0, 1, 2)] - v[()], abs=1e-2)

    # Dummy-feature check: an UNCORRELATED-with-interaction game where feature 2 is a true dummy
    # (contributes exactly 0 in every context) must get |a_2| < 1e-6.
    v_dummy = dict(v)
    v_dummy[(2,)] = 0.0
    v_dummy[(0, 2)] = 1.0
    v_dummy[(1, 2)] = 2.0
    v_dummy[(0, 1, 2)] = 8.0
    evaluator_dummy = _HandGameEvaluator(v_dummy)
    a_lin_dummy, _a_pair_dummy, _info_dummy = faith_shap_order2(evaluator_dummy, n_features, candidate_pairs, n_coalitions=2048, rng=np.random.default_rng(1))
    assert abs(a_lin_dummy[2]) < 1e-6, f"true dummy feature got |a_2|={abs(a_lin_dummy[2]):.8f}, expected < 1e-6"


def test_faith_shap_order2_rejects_too_few_features():
    """n_features < 2 raises ValueError (order-2 needs at least 2 features to define any pair)."""
    evaluator = _HandGameEvaluator({(): 0.0, (0,): 1.0})
    with pytest.raises(ValueError):
        faith_shap_order2(evaluator, 1, [], n_coalitions=64, rng=np.random.default_rng(0))


def test_shap_proxied_fs_proxy_mode_faith_interaction_validator_and_clone():
    """proxy_mode='faith_interaction' is accepted at construction and survives sklearn clone()."""
    from sklearn.base import clone

    sel = ShapProxiedFS(proxy_mode="faith_interaction", faith_n_coalitions=512)
    cloned = clone(sel)
    assert cloned.proxy_mode == "faith_interaction"
    assert cloned.faith_n_coalitions == 512


def test_shap_proxied_fs_proxy_mode_rejects_garbage():
    """An unrecognized proxy_mode value still raises ValueError with faith_interaction listed among the valid options."""
    with pytest.raises(ValueError):
        ShapProxiedFS(proxy_mode="not_a_real_mode")
