"""Unit tests for clustered-feature aggregation (denoised cluster representative).

Covers: aggregator weight derivation, recipe build/replay round-trip + train/test parity,
sign-alignment, the supervised gate, and the MRMR-level augment/replace/disabled/clone/no-mutation
contracts.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


def _reflection_frame(n=3000, k=4, noise=0.7, seed=0, extra_indep=True):
    """k clean reflections of a latent z (member-member corr ~ 1/(1+noise^2)) + an independent signal."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    cols = {f"refl{i}": z + noise * rng.normal(size=n) for i in range(k)}
    if extra_indep:
        cols["indep"] = rng.normal(size=n)
    X = pd.DataFrame(cols)
    score = z + (0.4 * X["indep"] if extra_indep else 0.0)
    y = pd.Series((score > 0).astype(np.int64), name="y")
    return X, y, z


_MRMR_KW = dict(
    verbose=0, random_seed=42, use_simple_mode=False,
    cluster_aggregate_corr_threshold=0.5,
    # Wave 9.1 (mrmr.py:1294) suppresses the post-hoc cluster_aggregate FE step
    # when ``dcd_enable=True`` (the new default) AND ``dcd_postoc_compose`` is
    # the default False, to avoid double-aggregation. These tests focus on the
    # cluster_aggregate path itself, so disable DCD or let the two compose.
    dcd_enable=False,
)


# ---------------------------------------------------------------------------
# Aggregator weight derivation
# ---------------------------------------------------------------------------


def test_derive_weights_shapes_and_normalization():
    from mlframe.feature_selection.filters._cluster_aggregate import _derive_weights, _standardize_align

    rng = np.random.default_rng(1)
    z = rng.normal(size=2000)
    M = np.column_stack([z + 0.5 * rng.normal(size=2000) for _ in range(4)])
    Z, mean, std, signs = _standardize_align(M, 0)
    assert _derive_weights(Z, "median") is None
    w_mean = _derive_weights(Z, "mean_z")
    assert np.allclose(w_mean, 0.25)
    for method in ("mean_inv_var", "pca_pc1", "factor_score"):
        w = _derive_weights(Z, method)
        assert w.shape == (4,) and np.all(np.isfinite(w))


def test_sign_alignment_flips_anticorrelated_member():
    from mlframe.feature_selection.filters._cluster_aggregate import _standardize_align

    rng = np.random.default_rng(2)
    z = rng.normal(size=2000)
    # member 2 is an ANTI-correlated reflection (-z): must be sign-flipped to +1 alignment.
    M = np.column_stack([z + 0.3 * rng.normal(size=2000), z + 0.3 * rng.normal(size=2000),
                         -z + 0.3 * rng.normal(size=2000)])
    Z, mean, std, signs = _standardize_align(M, 0)
    assert signs[2] == -1.0
    # After alignment all columns positively correlate with the reference.
    for j in range(3):
        assert np.corrcoef(Z[:, j], Z[:, 0])[0, 1] > 0


def test_sign_alignment_matches_corrcoef_reference_incl_constant_column():
    """The vectorised sign step must produce the SAME ``signs`` as the per-column
    np.corrcoef reference, including the edge that decides the optimisation's
    correctness: a CONSTANT (zero-variance) member. corrcoef returns NaN there
    (not finite -> the reference leaves sign +1); the vectorised form gets a
    zero covariance numerator -> sign +1. Both must agree, and a positively
    correlated member must keep +1."""
    from mlframe.feature_selection.filters._cluster_aggregate import _standardize_align

    rng = np.random.default_rng(7)
    z = rng.normal(size=2500)
    M = np.column_stack([
        z + 0.3 * rng.normal(size=2500),   # ref (col 0)
        z + 0.3 * rng.normal(size=2500),   # positively correlated -> +1
        -z + 0.3 * rng.normal(size=2500),  # anti-correlated -> -1
        np.full(2500, 4.2),                # constant -> NaN corr -> +1
    ])
    ref_col = 0
    _Z, _mean, _std, signs = _standardize_align(M, ref_col)

    # Reference: the original per-column corrcoef-sign rule.
    mean = M.mean(axis=0); std = M.std(axis=0)
    Zc = (M - mean) / np.where(std > 0.0, std, 1.0)
    expected = np.ones(M.shape[1])
    for j in range(M.shape[1]):
        if j == ref_col:
            continue
        c = np.corrcoef(Zc[:, j], Zc[:, ref_col])[0, 1]
        if np.isfinite(c) and c < 0:
            expected[j] = -1.0

    assert np.array_equal(signs, expected)
    assert signs[1] == 1.0 and signs[2] == -1.0 and signs[3] == 1.0


# ---------------------------------------------------------------------------
# Recipe build / replay round-trip + train-test parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["mean_z", "mean_inv_var", "median", "pca_pc1", "factor_score"])
def test_recipe_roundtrip_and_parity(method):
    from mlframe.feature_selection.filters._cluster_aggregate import _standardize_align, _derive_weights
    from mlframe.feature_selection.filters.engineered_recipes import build_cluster_aggregate_recipe, apply_recipe

    X, _y, _z = _reflection_frame(n=2000, k=4, noise=0.5, seed=3, extra_indep=False)
    names = list(X.columns)
    M = X[names].to_numpy()
    Z, mean, std, signs = _standardize_align(M, 0)
    weights = _derive_weights(Z, method)
    q = {"nbins": 8, "method": "quantile", "dtype": np.dtype(np.int32).str}
    r = build_cluster_aggregate_recipe(name=f"agg_{method}", src_names=tuple(names), method=method,
                                       member_mean=mean, member_std=std, signs=signs, weights=weights, quantization=q)
    c1 = apply_recipe(r, X)
    c2 = apply_recipe(r, X)
    assert np.array_equal(c1, c2)  # deterministic
    # Replay uses STORED TRAIN stats (not test-refit): the recipe carries the train mean/std verbatim,
    # and replay on a disjoint frame returns the right length. (Quantile binning is monotone-invariant,
    # so a binned-output diff cannot probe stat usage; assert the stored stats instead.)
    assert np.allclose(np.asarray(r.extra["member_mean"]), M.mean(axis=0))
    assert np.allclose(np.asarray(r.extra["member_std"]), M.std(axis=0))
    ct = apply_recipe(r, X.iloc[:500])
    assert len(ct) == 500


def test_recipe_extra_pickle_eq_roundtrip():
    import pickle
    from mlframe.feature_selection.filters.engineered_recipes import build_cluster_aggregate_recipe

    r = build_cluster_aggregate_recipe(
        name="agg", src_names=("a", "b", "c"), method="pca_pc1",
        member_mean=np.array([0.1, 0.2, 0.3]), member_std=np.array([1.0, 1.1, 0.9]),
        signs=np.array([1.0, -1.0, 1.0]), weights=np.array([0.5, 0.3, 0.4]),
        quantization={"nbins": 8, "method": "quantile", "dtype": "<i4"},
    )
    r2 = pickle.loads(pickle.dumps(r))
    assert r == r2  # __eq__ walks ndarray extra via _extra_equal


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def test_gate_rejects_when_no_mi_gain():
    """When the 'cluster' is actually independent noise (no shared latent), the aggregate cannot beat
    the best member's MI -> gate rejects -> no recipe."""
    from mlframe.feature_selection.filters._cluster_aggregate import run_cluster_aggregate_step
    from mlframe.feature_selection.filters.discretization import discretize_array

    rng = np.random.default_rng(4)
    n = 3000
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(4)})  # 4 INDEPENDENT noise features
    y = rng.integers(0, 2, size=n)
    cols = list(X.columns) + ["y"]
    NB = 8
    binned = [discretize_array(arr=X[c].to_numpy(), n_bins=NB, method="quantile", dtype=np.int32) for c in X.columns]
    binned.append(y.astype(np.int32))
    data = np.column_stack(binned).astype(np.int32)
    nbins = np.array([NB] * 4 + [2], dtype=np.int64)
    recipes = {}
    *_, n_added, removed, added_idx, _summary = run_cluster_aggregate_step(
        data=data, cols=cols, nbins=nbins, X=X, target_indices=(4,), feature_names_in_=list(X.columns),
        categorical_idx=(), cached_MIs={}, engineered_recipes=recipes, quantization_nbins=NB,
        quantization_method="quantile", quantization_dtype=np.int32, methods=("mean_z", "pca_pc1"),
        mi_prevalence=1.0, corr_threshold=0.3, min_cluster_size=3, verbose=0,
    )
    assert n_added == 0 and not recipes  # independent noise -> no denoising gain -> rejected


# ---------------------------------------------------------------------------
# MRMR-level contracts
# ---------------------------------------------------------------------------


def test_mrmr_augment_selects_aggregate_and_transforms():
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y, _z = _reflection_frame(seed=5)
    Xtr, Xte, ytr = X.iloc[:2000], X.iloc[2000:], y.iloc[:2000]
    s = MRMR(cluster_aggregate_enable=True, cluster_aggregate_mode="augment",
             cluster_aggregate_methods=("mean_z", "pca_pc1"), **_MRMR_KW).fit(Xtr, ytr)
    names = list(s.get_feature_names_out())
    assert any("clusteragg" in c for c in names), f"augment should select an aggregate; got {names}"
    out = s.transform(Xte)
    assert list(out.columns) == names and out.shape[0] == len(Xte)


def test_mrmr_replace_substitutes_members():
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y, _z = _reflection_frame(seed=6)
    Xtr, Xte, ytr = X.iloc[:2000], X.iloc[2000:], y.iloc[:2000]
    s = MRMR(cluster_aggregate_enable=True, cluster_aggregate_mode="replace",
             cluster_aggregate_methods=("mean_z", "pca_pc1"), **_MRMR_KW).fit(Xtr, ytr)
    names = list(s.get_feature_names_out())
    assert any("clusteragg" in c for c in names)
    # All reflection members are replaced by the aggregate.
    assert not any(c.startswith("refl") for c in names), f"replace should drop members; got {names}"
    s.transform(Xte)  # must not raise


def test_explicit_disable_adds_no_aggregate():
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y, _z = _reflection_frame(seed=7)
    s = MRMR(cluster_aggregate_enable=False, **_MRMR_KW).fit(X.iloc[:2000], y.iloc[:2000])
    assert not any("clusteragg" in c for c in s.get_feature_names_out())


def test_enabled_by_default():
    """cluster_aggregate is ON by default and fires on a clean reflection cluster."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    assert MRMR().get_params()["cluster_aggregate_enable"] is True
    X, y, _z = _reflection_frame(seed=7)
    # dcd_enable=False so DCD doesn't auto-suppress the post-hoc cluster
    # aggregate FE step (mrmr.py:1294); the gate fires the OLD default.
    s = MRMR(verbose=0, random_seed=42, use_simple_mode=False, cluster_aggregate_corr_threshold=0.5, dcd_enable=False).fit(X.iloc[:2000], y.iloc[:2000])
    assert any("clusteragg" in c for c in s.get_feature_names_out())
    # The fitted summary is populated for meta_info.
    assert getattr(s, "cluster_aggregate_", None) and s.cluster_aggregate_[0]["method"] in {"mean_z", "mean_inv_var", "median", "pca_pc1", "factor_score"}


def test_fit_does_not_mutate_caller_X():
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y, _z = _reflection_frame(seed=8)
    Xtr = X.iloc[:2000].copy()
    cols_before = list(Xtr.columns)
    MRMR(cluster_aggregate_enable=True, cluster_aggregate_mode="augment",
         cluster_aggregate_methods=("mean_z",), **_MRMR_KW).fit(Xtr, y.iloc[:2000])
    assert list(Xtr.columns) == cols_before, "fit must not add engineered columns to the caller's frame"


@pytest.mark.parametrize("method", ["mean_z", "mean_inv_var", "median", "pca_pc1", "factor_score"])
def test_each_method_builds_aggregate_recovering_latent(method):
    """Every aggregator in the menu, used as the sole method, builds an aggregate on a clean reflection
    cluster whose column recovers the hidden latent better than its best raw member."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y, z = _reflection_frame(n=3000, k=4, noise=0.6, seed=20, extra_indep=True)
    s = MRMR(verbose=0, random_seed=42, use_simple_mode=False, cluster_aggregate_mode="replace",
             cluster_aggregate_corr_threshold=0.5, cluster_aggregate_methods=(method,),
             dcd_enable=False).fit(X.iloc[:2000], y.iloc[:2000])
    aggs = [r for r in s._engineered_recipes_ if r.kind == "cluster_aggregate"]
    assert aggs and aggs[0].extra["method"] == method, f"method {method} should build an aggregate"
    out = s.transform(X)
    agg_col = [c for c in out.columns if "clusteragg" in c][0]
    rc_agg = abs(np.corrcoef(out[agg_col].to_numpy(), z)[0, 1])
    rc_best = max(abs(np.corrcoef(X[f"refl{i}"].to_numpy(), z)[0, 1]) for i in range(4))
    assert rc_agg > rc_best, f"[{method}] aggregate must recover z better than best member: {rc_agg:.3f} vs {rc_best:.3f}"


def test_cluster_aggregate_feeds_further_fe_step(monkeypatch):
    """PROOF that a cluster aggregate built in FE step 1 is a first-class feature available to FE step 2:
    spy on _run_fe_step and assert the aggregate column (created inside step 1) is present in the
    `cols` and `selected_vars` handed to the step-2 call."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    calls = []
    orig = MRMR._run_fe_step

    def _spy(self, **kw):
        calls.append({"cols": list(kw["cols"]), "selected_vars": [int(v) for v in kw["selected_vars"]]})
        return orig(self, **kw)

    monkeypatch.setattr(MRMR, "_run_fe_step", _spy)

    X, y, _z = _reflection_frame(n=3000, k=4, noise=0.6, seed=21, extra_indep=True)
    MRMR(verbose=0, random_seed=42, use_simple_mode=False, cluster_aggregate_corr_threshold=0.5,
         cluster_aggregate_methods=("mean_z",), fe_max_steps=2, dcd_enable=False).fit(X.iloc[:2000], y.iloc[:2000])

    assert len(calls) >= 2, f"expected >=2 FE steps (fe_max_steps=2); got {len(calls)}"
    step2 = calls[1]
    agg_idx = [i for i, c in enumerate(step2["cols"]) if "clusteragg" in c]
    assert agg_idx, f"the cluster aggregate must be in the cols passed to FE step 2; got {step2['cols']}"
    # It is also a selected candidate the step-2 pair search will consider (numeric_vars_to_consider = selected_vars).
    assert any(i in step2["selected_vars"] for i in agg_idx), "the aggregate must be a selected candidate at FE step 2"


def test_cluster_aggregate_surfaces_in_meta_info():
    """The fitted cluster-aggregate summary flows into the per-model feature-selection report that the
    training suite stamps onto meta_info (mirrors the friend_graph block)."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from mlframe.training.core._phase_train_one_target_helpers import _build_feature_selection_report

    X, y, _z = _reflection_frame(seed=30)
    s = MRMR(verbose=0, random_seed=42, use_simple_mode=False, cluster_aggregate_corr_threshold=0.5, dcd_enable=False).fit(X.iloc[:2000], y.iloc[:2000])
    assert s.cluster_aggregate_, "fit should populate cluster_aggregate_"
    report = _build_feature_selection_report(s, "MRMR", list(X.columns), list(s.get_feature_names_out()))
    assert "cluster_aggregate" in report, f"report should carry the cluster_aggregate summary; got {list(report)}"
    rec = report["cluster_aggregate"][0]
    assert {"name", "method", "members", "aggregate_mi", "best_member_mi", "mi_gain"} <= set(rec)


def test_param_plumbing_clone():
    from sklearn.base import clone
    from mlframe.feature_selection.filters.mrmr import MRMR

    e = MRMR(cluster_aggregate_enable=True, cluster_aggregate_mode="replace", cluster_aggregate_methods=("mean_z", "pca_pc1"))
    p = e.get_params()
    assert p["cluster_aggregate_enable"] is True and p["cluster_aggregate_mode"] == "replace"
    assert clone(e).get_params()["cluster_aggregate_methods"] == ("mean_z", "pca_pc1")


def test_invalid_mode_and_method_raise():
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y, _z = _reflection_frame(n=400, seed=9)
    with pytest.raises(ValueError):
        MRMR(cluster_aggregate_enable=True, cluster_aggregate_mode="bogus").fit(X, y)
    with pytest.raises(ValueError):
        MRMR(cluster_aggregate_enable=True, cluster_aggregate_methods=("bogus",)).fit(X, y)
