"""biz_val for clustered-feature aggregation: prove on synthetic data that aggregating clusters of
noisy reflections of HIDDEN factor(s) recovers those factors better than any single reflection, finds
the right NUMBER of clusters (one per latent factor), and does NO HARM to a downstream model -- across
a regularized linear model, gradient boosting, and a neural net.

Scope (honest, per the docstring caveat + the beam-search lesson): the downstream contract is NO-HARM
across model classes. Trees/boosting already average reflections via splits, so a strict AUC LIFT is
not expected from them; the genuine lift is for capacity-limited linear models. We therefore assert
no-harm uniformly and treat any lift as a bonus.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from tests.feature_selection._biz_val_synth import make_latent_reflections, make_two_latent_groups, as_df
from sklearn.metrics import roc_auc_score


pytestmark = pytest.mark.timeout(
    240
)  # untimed biz_val real-fit tier: hang-detector, not a perf budget. The module-scoped full-mode MRMR fixture fits legitimately run ~75-90s on many-core/contended hosts; 60s killed a progressing fit mid-way. 240s stays well under the coarse 600s global backstop while still surfacing a true hang fast.


def _abs_corr(a, b):
    return abs(np.corrcoef(np.asarray(a, dtype=float), np.asarray(b, dtype=float))[0, 1])


def _make_model(name):
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(penalty="l2", C=0.2, max_iter=1000)
    if name == "boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_iter=120, max_depth=3, learning_rate=0.1, random_state=0)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(hidden_layer_sizes=(16,), max_iter=400, random_state=0, early_stopping=True)
    raise ValueError(name)


DOWNSTREAM_MODELS = ["logreg", "boosting", "mlp"]
_CA_KW = dict(
    verbose=0,
    random_seed=42,
    use_simple_mode=False,
    cluster_aggregate_corr_threshold=0.4,
    cluster_aggregate_homogeneity_tau=0.5,
    cluster_aggregate_min_cluster_size=3,
    cluster_aggregate_methods=("mean_z", "mean_inv_var", "pca_pc1"),
    # Wave 9.1 (mrmr.py:1294) auto-suppresses the post-hoc
    # cluster_aggregate FE step when ``dcd_enable=True`` (new default)
    # + ``dcd_postoc_compose=False`` (default). These tests focus on
    # the cluster_aggregate path itself, so disable DCD.
    dcd_enable=False,
)


# ---------------------------------------------------------------------------
# Fast aggregator-level proofs (no MRMR fit)
# ---------------------------------------------------------------------------


def test_biz_val_aggregate_recovers_hidden_factor_near_theory():
    """S1: standardized mean of k iid-noise reflections recovers the hidden z markedly better than the
    best single reflection, and close to the sigma^2/k theory (corr ~ 1/sqrt(1+sigma^2/k))."""
    from mlframe.feature_selection.filters._cluster_aggregate import _standardize_align, _derive_weights

    k, sigma = 5, 1.2
    X, _y, info = make_latent_reflections(n=6000, loadings=(1.0,) * k, noise_sd=(sigma,) * k, n_noise=0, seed=1)
    z = info["z"]
    refl = X[:, info["reflections"]]
    Z, *_ = _standardize_align(refl, 0)
    agg = Z @ _derive_weights(Z, "mean_z")
    best_member = max(_abs_corr(refl[:, j], z) for j in range(k))
    rc_agg = _abs_corr(agg, z)
    theory = 1.0 / np.sqrt(1.0 + sigma**2 / k)
    assert rc_agg > best_member + 0.05, f"aggregate should beat best member clearly: {rc_agg:.3f} vs {best_member:.3f}"
    assert abs(rc_agg - theory) / theory < 0.05, f"recovery should track sigma^2/k theory {theory:.3f}; got {rc_agg:.3f}"


@pytest.mark.parametrize("regime", ["hetero_loadings", "hetero_noise"])
def test_biz_val_menu_beats_naive_mean_in_heterogeneous_regimes(regime):
    """In heterogeneous regimes the weighted combiners (mean_inv_var / pca_pc1) recover the hidden z
    better than the naive equal-weight mean -- why the menu exists. Empirically mean_inv_var (BLUE-ish)
    is strongest in BOTH heterogeneous regimes; we assert the menu's value, not a single fixed winner."""
    from mlframe.feature_selection.filters._cluster_aggregate import _standardize_align, _derive_weights

    if regime == "hetero_loadings":
        X, y, info = make_latent_reflections(n=6000, loadings=(2.0, 1.5, 0.5, 0.2), noise_sd=(1.0,) * 4, n_noise=0, seed=2)
    else:
        X, _y, info = make_latent_reflections(n=6000, loadings=(1.0,) * 4, noise_sd=(0.3, 0.5, 2.0, 3.0), n_noise=0, seed=3)
    z = info["z"]
    Z, *_ = _standardize_align(X[:, info["reflections"]], 0)
    rc = {m: _abs_corr(Z @ _derive_weights(Z, m), z) for m in ("mean_z", "mean_inv_var", "pca_pc1")}
    assert max(rc["mean_inv_var"], rc["pca_pc1"]) > rc["mean_z"] + 0.02, f"{regime}: a weighted combiner should beat naive mean; got {rc}"
    assert rc["mean_inv_var"] >= rc["pca_pc1"] - 1e-3, f"{regime}: mean_inv_var (BLUE-ish) expected >= pca_pc1; got {rc}"


# ---------------------------------------------------------------------------
# Scenario fixtures: fit MRMR ONCE per scenario (replace + disabled), reused across downstream models.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=["augment", "replace"])
def one_group_fit(request):
    """Single hidden factor + 5 reflections + indep + noise. Parametrized over BOTH dispositions."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    mode = request.param
    X, y, info = make_latent_reflections(n=6000, loadings=(1.0,) * 5, noise_sd=(0.85,) * 5, n_noise=3, seed=10)
    df, _ = as_df(X, y)
    tr, te = (0, 4000), (4000, 6000)
    Xtr, Xte, ytr, yte = df.iloc[tr[0] : tr[1]], df.iloc[te[0] : te[1]], y[tr[0] : tr[1]], y[te[0] : te[1]]
    s_rep = MRMR(cluster_aggregate_enable=True, cluster_aggregate_mode=mode, **_CA_KW).fit(Xtr, ytr)
    s_off = MRMR(cluster_aggregate_enable=False, verbose=0, random_seed=42, use_simple_mode=False).fit(Xtr, ytr)
    return dict(mode=mode, s_rep=s_rep, s_off=s_off, df=df, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, info=info, refl_names=[f"x{i}" for i in info["reflections"]])


@pytest.fixture(scope="module")
def two_group_fit():
    """TWO hidden factors, each with its own reflection group, + a pure-noise group."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y, info = make_two_latent_groups(n=6000, k1=4, k2=4, noise=0.85, n_noise=3, seed=20)
    df, _ = as_df(X, y)
    tr, te = (0, 4000), (4000, 6000)
    Xtr, Xte, ytr, yte = df.iloc[tr[0] : tr[1]], df.iloc[te[0] : te[1]], y[tr[0] : tr[1]], y[te[0] : te[1]]
    s_rep = MRMR(cluster_aggregate_enable=True, cluster_aggregate_mode="replace", **_CA_KW).fit(Xtr, ytr)
    s_off = MRMR(cluster_aggregate_enable=False, verbose=0, random_seed=42, use_simple_mode=False).fit(Xtr, ytr)
    return dict(s_rep=s_rep, s_off=s_off, df=df, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, info=info)


# ---------------------------------------------------------------------------
# Recovery + cluster-count proofs
# ---------------------------------------------------------------------------


def test_biz_val_one_group_integrated_recovery(one_group_fit):
    """REPLACE builds the aggregate; the transformed aggregate column recovers the hidden z better than
    any single raw reflection (the headline ask)."""
    f = one_group_fit
    names = list(f["s_rep"].get_feature_names_out())
    assert any("clusteragg" in c for c in names), f"replace should build an aggregate; got {names}"
    out = f["s_rep"].transform(f["df"])
    agg_col = next(c for c in out.columns if "clusteragg" in c)
    rc_agg = _abs_corr(out[agg_col].to_numpy(), f["info"]["z"])
    rc_best = max(_abs_corr(f["df"][r].to_numpy(), f["info"]["z"]) for r in f["refl_names"])
    assert rc_agg > rc_best + 0.03, f"aggregate must recover hidden z better than best reflection: {rc_agg:.3f} vs {rc_best:.3f}"


def test_biz_val_two_clusters_two_latent_factors(two_group_fit):
    """The algorithm must find EXACTLY TWO clusters (one per hidden factor), each aggregate recovering
    its OWN latent (z1 / z2) and NOT mixing groups or pulling in noise columns."""
    f = two_group_fit
    info = f["info"]
    out = f["s_rep"].transform(f["df"])
    agg_cols = [c for c in out.columns if "clusteragg" in c]
    assert len(agg_cols) == 2, f"expected 2 aggregates (one per latent factor); got {agg_cols}"

    groupA_names = {f"x{i}" for i in info["groupA"]}
    groupB_names = {f"x{i}" for i in info["groupB"]}
    noise_names = {f"x{i}" for i in info["noise"]}
    # Each aggregate must (a) be built from exactly ONE group's members (no cross-group / noise mixing),
    # and (b) recover that group's latent factor better than its best raw member.
    matched = {"A": False, "B": False}
    for r in f["s_rep"]._engineered_recipes_:
        if r.kind != "cluster_aggregate":
            continue
        srcs = set(r.src_names)
        assert not (srcs & noise_names), f"aggregate {r.name} pulled in noise columns: {srcs & noise_names}"
        col = out[r.name].to_numpy()
        if srcs <= groupA_names:
            assert _abs_corr(col, info["z1"]) > max(_abs_corr(f["df"][s].to_numpy(), info["z1"]) for s in srcs) + 0.02
            matched["A"] = True
        elif srcs <= groupB_names:
            assert _abs_corr(col, info["z2"]) > max(_abs_corr(f["df"][s].to_numpy(), info["z2"]) for s in srcs) + 0.02
            matched["B"] = True
        else:
            pytest.fail(f"aggregate {r.name} mixes groups: {srcs}")
    assert matched["A"] and matched["B"], f"both latent-factor groups must be recovered as clusters; got {matched}"


# ---------------------------------------------------------------------------
# Downstream no-harm across model classes (parametrized: linear / boosting / neural net)
# ---------------------------------------------------------------------------


def _auc(sel, model_name, Xtr, ytr, Xte, yte):
    Atr, Ate = sel.transform(Xtr), sel.transform(Xte)
    m = _make_model(model_name).fit(np.asarray(Atr, float), ytr)
    return roc_auc_score(yte, m.predict_proba(np.asarray(Ate, float))[:, 1])


@pytest.mark.parametrize("model", DOWNSTREAM_MODELS)
def test_biz_val_one_group_downstream_no_harm(one_group_fit, model):
    """Replacing the reflection cluster with its denoised aggregate must not materially hurt held-out
    AUC for a linear / boosting / neural-net downstream."""
    f = one_group_fit
    auc_rep = _auc(f["s_rep"], model, f["Xtr"], f["ytr"], f["Xte"], f["yte"])
    auc_off = _auc(f["s_off"], model, f["Xtr"], f["ytr"], f["Xte"], f["yte"])
    assert auc_rep >= auc_off - 0.025, f"[{model}] cluster aggregate hurt AUC: {auc_rep:.4f} vs {auc_off:.4f}"


@pytest.mark.parametrize("model", DOWNSTREAM_MODELS)
def test_biz_val_two_group_downstream_no_harm(two_group_fit, model):
    """Replacing two reflection clusters with two aggregates must not materially hurt held-out AUC."""
    f = two_group_fit
    auc_rep = _auc(f["s_rep"], model, f["Xtr"], f["ytr"], f["Xte"], f["yte"])
    auc_off = _auc(f["s_off"], model, f["Xtr"], f["ytr"], f["Xte"], f["yte"])
    assert auc_rep >= auc_off - 0.025, f"[{model}] two-cluster aggregate hurt AUC: {auc_rep:.4f} vs {auc_off:.4f}"


# ---------------------------------------------------------------------------
# No-harm control: correlated noise (denoising premise violated)
# ---------------------------------------------------------------------------


def test_biz_val_no_harm_correlated_noise():
    """S5: reflection noise is mostly SHARED -> averaging cannot denoise -> the strict MI gate must not
    let an aggregate silently strip all members for no real gain."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y, info = make_latent_reflections(n=5000, loadings=(1.0,) * 4, noise_sd=(1.0,) * 4, n_noise=2, shared_noise=0.9, seed=11)
    df, _ = as_df(X, y)
    s = MRMR(cluster_aggregate_enable=True, cluster_aggregate_mode="replace", **_CA_KW).fit(df.iloc[:3500], y[:3500])
    names = list(s.get_feature_names_out())
    refl_kept = sum(c[1:].isdigit() and int(c[1:]) in info["reflections"] for c in names if c.startswith("x"))
    has_agg = any("clusteragg" in c for c in names)
    assert has_agg or refl_kept >= 1, f"correlated-noise cluster should not strip all members for no gain; got {names}"
