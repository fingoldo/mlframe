"""Cross-axis fuzz blind-spot sensors (W11C extension of the F1-F7 series).

Each sensor pins a cross-axis combo that the pairwise sampler may not always reach but is known to stress a
distinct production code path. Sensors are intentionally narrow: build a FuzzCombo with the axis overrides and
assert it constructs + the fields survive canonicalisation. Full-suite execution stays under ``--run-fuzz``.

Findings:

- C1: ``inject_all_nan_col=True`` x ``use_mrmr_fs=True`` x ``mrmr_nan_strategy_cfg`` -- an all-NaN column under
  MRMR previously crashed the MI estimator on divide-by-zero when fillna_zero produced a constant-0 post-fill
  column (B1 F4 tests-expand.md). Pin a 3-axis combo so the canonicaliser cannot collapse this away.
- C2: ``recurrent_model_cfg in {lstm, gru, transformer}`` x ``weight_schemas=("recency",)`` -- recurrent models
  have their own sample-weight handling and recency-only schemas (no uniform fallback) are a likely null-weight
  trap on empty mini-batches (B1 F3 tests-expand.md).
- C3: ``composite_discovery_enabled_cfg=True`` x ``outlier_detection in {lof, ocsvm}`` x ``imbalance_ratio !=
  balanced`` -- the F6 2-axis combo extended with a 3rd axis to catch the documented "four layered 0-row val
  tolerances" cluster (CLAUDE.md) where outlier removal + imbalance shift + composite-target discovery stack to
  collapse val to 0 rows silently.
"""
from __future__ import annotations

import pytest

from tests.training._fuzz_combo import AXES, _build_combo


def _make_combo(**overrides):
    axes = {name: values[0] for name, values in AXES.items()}
    axes.update(overrides)
    return _build_combo(models=("cb",), axes=axes, seed=0)


# ---------------------------------------------------------------------------
# C1: inject_all_nan_col x use_mrmr_fs x mrmr_nan_strategy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nan_strategy", ["separate_bin", "fillna_zero"])
def test_C1_all_nan_col_x_mrmr_x_nan_strategy_reachable(nan_strategy):
    """3-axis cross-combo must construct without canon-collapsing inject_all_nan_col or use_mrmr_fs to False.

    The MRMR NaN-handling strategy axis was added 2026-04 (line 766 _fuzz_combo) specifically because fillna_zero
    on an all-NaN column produces a constant-0 column and the MI estimator can divide-by-zero. Sensor pins the
    reachability of the bug-surface combo so future canon edits cannot quietly hide it."""
    combo = _make_combo(
        inject_all_nan_col=True,
        use_mrmr_fs=True,
        mrmr_nan_strategy_cfg=nan_strategy,
        target_type="regression",
    )
    assert combo.inject_all_nan_col is True
    assert combo.use_mrmr_fs is True
    # Distinct canonical key vs the use_mrmr_fs=False variant (canon should NOT collapse this 3-axis bug-surface).
    combo_no_mrmr = _make_combo(
        inject_all_nan_col=True,
        use_mrmr_fs=False,
        mrmr_nan_strategy_cfg=nan_strategy,
        target_type="regression",
    )
    assert combo.canonical_key() != combo_no_mrmr.canonical_key(), (
        "C1: use_mrmr_fs True/False must produce distinct canonical keys under inject_all_nan_col"
    )


# ---------------------------------------------------------------------------
# C2: recurrent_model_cfg x weight_schemas=("recency",)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rec", ["lstm", "gru", "transformer"])
def test_C2_recurrent_x_recency_only_weights_reachable(rec):
    """``recurrent_model_cfg in {lstm,gru,transformer}`` x ``weight_schemas=("recency",)`` must construct and
    keep both axis values intact.

    Recency-only schemas (no uniform fallback) are iter150-new and recurrent models handle sample weights through
    a distinct PyTorch-Lightning path. Sensor pins that the bug-surface combo is reachable; the actual fit
    behaviour is asserted by the suite when the combo runs under --run-fuzz."""
    combo = _make_combo(
        recurrent_model_cfg=rec,
        weight_schemas=("recency",),
        target_type="regression",
        n_rows=1000,
    )
    assert combo.recurrent_model_cfg == rec
    assert tuple(combo.weight_schemas) == ("recency",)
    # Distinct vs the uniform-only baseline (catches accidental canon collapse of the recency-only schema).
    combo_uniform = _make_combo(
        recurrent_model_cfg=rec,
        weight_schemas=("uniform",),
        target_type="regression",
        n_rows=1000,
    )
    assert combo.canonical_key() != combo_uniform.canonical_key(), (
        "C2: weight_schemas=(recency,) must NOT canonicalise to (uniform,) under recurrent_model_cfg"
    )


# ---------------------------------------------------------------------------
# C3: composite_discovery x outlier_detection x imbalance_ratio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("od", ["lof", "ocsvm"])
@pytest.mark.parametrize("imb", ["rare_5pct", "rare_1pct"])
def test_C3_composite_discovery_x_outlier_x_imbalance_reachable(od, imb):
    """3-axis combo extends F6 (composite_discovery x outlier_detection) with imbalance_ratio so the documented
    "four layered 0-row val tolerances" cluster (CLAUDE.md) is fuzz-reachable: outlier removal + imbalance shift
    + composite-target discovery stack to collapse val to 0 rows in the trainer.

    Sensor only verifies combo construction; the suite-level run under --run-fuzz asserts the trainer raises a
    clear OutlierEmptyValError instead of silently producing an empty val frame."""
    combo = _make_combo(
        composite_discovery_enabled_cfg=True,
        outlier_detection=od,
        imbalance_ratio=imb,
        target_type="regression",
    )
    assert combo.composite_discovery_enabled_cfg is True
    assert combo.outlier_detection == od
    assert combo.imbalance_ratio == imb


# ---------------------------------------------------------------------------
# iter373: LTR + no native ranker = unrunnable combo filter
# ---------------------------------------------------------------------------


def test_iter373_no_ltr_combos_without_native_ranker():
    """iter373 regression: enumerator must never emit LTR combos whose model
    subset has zero native rankers (cb/xgb/lgb/mlp). Pre-fix the random pick
    surfaced c0120_0a4f6506 with models=('linear',) target=LTR which crashed
    at fit-time with NotImplementedError. The filter rejects such subsets
    in all three enumeration phases (initial powerset, pairwise greedy,
    random fill)."""
    from tests.training._fuzz_combo import (
        _LTR_NATIVE_RANKERS, enumerate_combos,
    )
    combos = enumerate_combos(target=150, master_seed=20260422)
    unrunnable = [
        c for c in combos
        if c.target_type == "learning_to_rank"
        and not any(m in _LTR_NATIVE_RANKERS for m in c.models)
    ]
    assert not unrunnable, (
        f"enumerator emitted {len(unrunnable)} LTR combos with no native ranker: "
        f"{[c.short_id() for c in unrunnable[:5]]}"
    )


# ---------------------------------------------------------------------------
# iter466: MRMR friend-graph + cluster-aggregate axes
# ---------------------------------------------------------------------------


def test_iter466_mrmr_friend_graph_cluster_axes_flow_to_kwargs():
    """iter466: the recent mrmr.py friend-graph + cluster-aggregate features
    must be (a) present as fuzz axes, (b) varied across MRMR-on combos, and
    (c) threaded into the mrmr_kwargs dict the suite passes to the MRMR
    constructor. Non-MRMR combos canonicalise the 4 axes to their mrmr.py
    defaults so they don't gain phantom variation."""
    from tests.training._fuzz_combo import (
        AXES, enumerate_combos, build_mrmr_kwargs,
    )
    for ax in (
        "mrmr_build_friend_graph_cfg", "mrmr_friend_graph_prune_cfg",
        "mrmr_cluster_aggregate_enable_cfg", "mrmr_cluster_aggregate_mode_cfg",
    ):
        assert ax in AXES, f"missing fuzz axis {ax}"

    combos = enumerate_combos(target=150, master_seed=20260422)
    mrmr_combos = [c for c in combos if c.use_mrmr_fs]
    assert mrmr_combos, "expected at least one MRMR-on combo in the suite"

    # kwargs carry all 4 keys (names match the MRMR constructor params).
    kw = build_mrmr_kwargs(mrmr_combos[0])
    for k in (
        "build_friend_graph", "friend_graph_prune",
        "cluster_aggregate_enable", "cluster_aggregate_mode",
    ):
        assert k in kw, f"mrmr_kwargs missing {k}"

    # The mode axis must actually vary across MRMR combos (both augment+replace
    # reachable) so the dedup pass didn't collapse the new variation away.
    modes = {c.mrmr_cluster_aggregate_mode_cfg for c in mrmr_combos}
    assert modes == {"augment", "replace"}, f"mode variation lost: {modes}"

    # Non-MRMR combos must canonicalise prune to its default False (gated on
    # both use_mrmr_fs AND build_friend_graph) so they can't gain variation.
    non_mrmr = [c for c in combos if not c.use_mrmr_fs]
    if non_mrmr:
        kw_off = build_mrmr_kwargs(non_mrmr[0])
        assert kw_off is None, "use_mrmr_fs=False must yield None mrmr_kwargs"


# ---------------------------------------------------------------------------
# ShapProxiedFS axes (SHAP-coalition-proxy selector)
# ---------------------------------------------------------------------------


def test_shap_proxied_fs_axes_flow_to_kwargs():
    """ShapProxiedFS (feature_selection/shap_proxied_fs.py, registry name
    "ShapProxiedFS") must be (a) present as fuzz axes, (b) varied across
    shap-proxied-on combos, and (c) threaded into the kwargs dict the suite
    would pass to ShapProxiedFS.__init__ (param names match exactly).
    Shap-proxied-off combos canonicalise the sub-knobs to their
    ShapProxiedFS.__init__ defaults + yield None kwargs so they gain no
    phantom variation. Mirrors test_iter466_mrmr_friend_graph_cluster_axes."""
    from tests.training._fuzz_combo import (
        AXES, _build_combo, enumerate_combos, build_shap_proxied_fs_kwargs,
    )
    for ax in (
        "use_shap_proxied_fs", "shap_proxied_optimizer_cfg",
        "shap_proxied_revalidate_cfg", "shap_proxied_trust_guard_cfg",
        "shap_proxied_interaction_aware_cfg", "shap_proxied_cluster_features_cfg",
    ):
        assert ax in AXES, f"missing fuzz axis {ax}"

    combos = enumerate_combos(target=150, master_seed=20260422)
    shap_combos = [c for c in combos if c.use_shap_proxied_fs]
    assert shap_combos, "expected at least one ShapProxiedFS-on combo in the suite"

    # kwargs carry all 5 knobs (names match the ShapProxiedFS constructor params).
    kw = build_shap_proxied_fs_kwargs(shap_combos[0])
    assert kw is not None
    for k in (
        "optimizer", "revalidate", "trust_guard",
        "interaction_aware", "cluster_features",
    ):
        assert k in kw, f"shap_proxied_fs_kwargs missing {k}"

    # An explicitly-enabled combo with non-default sub-knobs threads them through
    # verbatim (the suite would forward these straight to ShapProxiedFS.__init__).
    base_axes = {name: values[0] for name, values in AXES.items()}
    base_axes.update(
        use_shap_proxied_fs=True,
        shap_proxied_optimizer_cfg="greedy_forward",
        shap_proxied_revalidate_cfg=False,
        shap_proxied_trust_guard_cfg=False,
        shap_proxied_interaction_aware_cfg=True,
        shap_proxied_cluster_features_cfg=False,
    )
    on = _build_combo(models=("cb",), axes=base_axes, seed=0)
    kw_on = build_shap_proxied_fs_kwargs(on)
    # Original 5 sub-knobs thread through verbatim. 2026-05-28 ext axes
    # (active_learning + prefilter_method) also flow through here; the
    # iter-NNN cross-axis test below pins their values explicitly.
    for k, v in {
        "optimizer": "greedy_forward",
        "revalidate": False,
        "trust_guard": False,
        "interaction_aware": True,
        "cluster_features": False,
    }.items():
        assert kw_on[k] == v, f"shap_proxied kwargs[{k}] = {kw_on[k]!r}, expected {v!r}"

    # Shap-proxied-off: sub-knobs collapse to ShapProxiedFS.__init__ defaults in
    # canonical_key, and the kwargs builder yields None (no-op FS step). Toggling
    # only the sub-knobs while OFF must NOT change the canonical key.
    off_a = dict(base_axes)
    off_a.update(use_shap_proxied_fs=False)
    off_b = dict(off_a)
    off_b.update(
        shap_proxied_optimizer_cfg="auto",
        shap_proxied_revalidate_cfg=True,
        shap_proxied_trust_guard_cfg=True,
        shap_proxied_interaction_aware_cfg=False,
        shap_proxied_cluster_features_cfg="auto",
    )
    combo_off_a = _build_combo(models=("cb",), axes=off_a, seed=0)
    combo_off_b = _build_combo(models=("cb",), axes=off_b, seed=0)
    assert build_shap_proxied_fs_kwargs(combo_off_a) is None
    assert combo_off_a.canonical_key() == combo_off_b.canonical_key(), (
        "shap-proxied sub-knobs must collapse to defaults when the enable flag is off"
    )

    # The enable axis must actually vary the canonical key when toggled with the
    # same sub-knobs (so dedup keeps both branches reachable).
    assert on.canonical_key() != combo_off_a.canonical_key()


# ---------------------------------------------------------------------------
# 2026-05-28 new coverage axes (11 axes: 4 HIGH + 7 MED from coverage audit).
# ---------------------------------------------------------------------------


def test_iter501_new_coverage_axes_flow_to_kwargs():
    """11 new fuzz axes (4 HIGH + 7 MED, coverage-audit batch 2026-05-28) must
    be (a) present in AXES, (b) reachable in enumerate_combos at non-default
    values when the gating axis is ON, (c) canonical_key-collapsed when the
    gating axis is OFF (no phantom variation), (d) threaded verbatim through
    the relevant config builder.

    Mirrors test_shap_proxied_fs_axes_flow_to_kwargs + the iter466 MRMR
    friend-graph/cluster test. One test covers all 11 because each axis
    follows the same wiring template; per-axis sensors live in the
    pairwise sampler itself.

    Axes:
      HIGH:
        - fhc_text_min_cardinality_cfg (TextDetectionConfig.text_min_cardinality)
        - composite_auto_skip_on_baseline_optimal_cfg (CompositeTargetDiscoveryConfig.auto_skip_on_baseline_optimal)
        - shap_proxied_active_learning_cfg (ShapProxiedFS.active_learning)
        - shap_proxied_prefilter_method_cfg (ShapProxiedFS.prefilter_method)
      MED:
        - composite_mi_n_neighbors_cfg
        - composite_auto_base_null_perms_cfg
        - composite_multi_base_max_k_cfg
        - extreme_ar_group_aware_skip_models_cfg (TrainingBehaviorConfig.extreme_ar_group_aware_skip_models)
        - fs_pre_screen_null_fraction_threshold_cfg (FeatureSelectionConfig.pre_screen_null_fraction_threshold)
        - linear_l1_ratio_cfg (LinearModelConfig.l1_ratio)
        - recurrent_hidden_size_cfg (RecurrentConfig.hidden_size)
    """
    from tests.training._fuzz_combo import (
        AXES, _build_combo, enumerate_combos,
        build_shap_proxied_fs_kwargs, build_composite_discovery_config,
    )

    new_axes = (
        "fhc_text_min_cardinality_cfg",
        "composite_auto_skip_on_baseline_optimal_cfg",
        "composite_mi_n_neighbors_cfg",
        "composite_auto_base_null_perms_cfg",
        "composite_multi_base_max_k_cfg",
        "shap_proxied_active_learning_cfg",
        "shap_proxied_prefilter_method_cfg",
        "extreme_ar_group_aware_skip_models_cfg",
        "fs_pre_screen_null_fraction_threshold_cfg",
        "linear_l1_ratio_cfg",
        "recurrent_hidden_size_cfg",
    )
    # (a) all 11 axes present in AXES.
    for ax in new_axes:
        assert ax in AXES, f"missing fuzz axis {ax}"
        assert len(AXES[ax]) >= 2, f"axis {ax} must offer at least 2 values"

    # (b) enumerate_combos still reaches 150 combos with the new axes wired.
    combos = enumerate_combos(target=150, master_seed=20260422)
    assert len(combos) == 150, f"enumerate_combos lost combos: {len(combos)}"

    # (c) Per-axis: pin a base combo with the gating axis ON + the new axis at
    # a non-default value; assert the canonical_key collapses to the default
    # when the gating axis is OFF.
    base_axes = {name: values[0] for name, values in AXES.items()}

    # SHAP-proxied extension axes: gate on use_shap_proxied_fs.
    on_axes = dict(base_axes)
    on_axes.update(
        use_shap_proxied_fs=True,
        shap_proxied_active_learning_cfg=True,
        shap_proxied_prefilter_method_cfg="univariate",
    )
    on = _build_combo(models=("cb",), axes=on_axes, seed=0)
    kw_on = build_shap_proxied_fs_kwargs(on)
    assert kw_on is not None
    assert kw_on["active_learning"] is True
    assert kw_on["prefilter_method"] == "univariate"
    # OFF: kwargs is None + canonical key collapses sub-knobs to defaults.
    off_a = dict(on_axes); off_a["use_shap_proxied_fs"] = False
    off_b = dict(off_a)
    off_b.update(shap_proxied_active_learning_cfg=False, shap_proxied_prefilter_method_cfg="auto")
    combo_off_a = _build_combo(models=("cb",), axes=off_a, seed=0)
    combo_off_b = _build_combo(models=("cb",), axes=off_b, seed=0)
    assert build_shap_proxied_fs_kwargs(combo_off_a) is None
    assert combo_off_a.canonical_key() == combo_off_b.canonical_key(), (
        "shap_proxied ext sub-knobs must collapse when use_shap_proxied_fs is off"
    )

    # FHC text_min_cardinality: gate on enable_feature_handling_config_cfg.
    on_fhc = dict(base_axes)
    on_fhc.update(enable_feature_handling_config_cfg=True, fhc_text_min_cardinality_cfg=50)
    off_fhc_a = dict(on_fhc); off_fhc_a["enable_feature_handling_config_cfg"] = False
    off_fhc_b = dict(off_fhc_a); off_fhc_b["fhc_text_min_cardinality_cfg"] = 300
    c_on = _build_combo(models=("cb",), axes=on_fhc, seed=0)
    c_off_a = _build_combo(models=("cb",), axes=off_fhc_a, seed=0)
    c_off_b = _build_combo(models=("cb",), axes=off_fhc_b, seed=0)
    assert c_on.fhc_text_min_cardinality_cfg == 50
    assert c_off_a.canonical_key() == c_off_b.canonical_key(), (
        "fhc_text_min_cardinality_cfg must collapse when FHC is disabled"
    )

    # Composite deep knobs: gate on composite_discovery_enabled_cfg + regression.
    on_comp = dict(base_axes)
    on_comp.update(
        composite_discovery_enabled_cfg=True,
        target_type="regression",
        baseline_diagnostics_enabled_cfg=True,
        composite_mi_estimator_cfg="knn",
        composite_auto_skip_on_baseline_optimal_cfg=True,
        composite_mi_n_neighbors_cfg=10,
        composite_auto_base_null_perms_cfg=50,
        composite_multi_base_max_k_cfg=5,
    )
    c_comp_on = _build_combo(models=("cb",), axes=on_comp, seed=0)
    cfg_on = build_composite_discovery_config(c_comp_on)
    # Builder threads all 4 deep knobs verbatim (names match the dataclass fields).
    assert getattr(cfg_on, "auto_skip_on_baseline_optimal", None) is True
    assert getattr(cfg_on, "mi_n_neighbors", None) == 10
    assert getattr(cfg_on, "auto_base_null_perms", None) == 50
    assert getattr(cfg_on, "multi_base_max_k", None) == 5

    # OFF: composite disabled -> all 4 deep knobs canon to defaults in
    # canonical_key. We pick non-default values in off_a and library
    # defaults in off_b; both must produce the SAME canonical key.
    off_comp_a = dict(on_comp); off_comp_a["composite_discovery_enabled_cfg"] = False
    off_comp_b = dict(off_comp_a)
    off_comp_b.update(
        composite_auto_skip_on_baseline_optimal_cfg=False,
        composite_mi_n_neighbors_cfg=3,
        composite_auto_base_null_perms_cfg=20,
        composite_multi_base_max_k_cfg=3,
    )
    c_comp_off_a = _build_combo(models=("cb",), axes=off_comp_a, seed=0)
    c_comp_off_b = _build_combo(models=("cb",), axes=off_comp_b, seed=0)
    assert c_comp_off_a.canonical_key() == c_comp_off_b.canonical_key(), (
        "composite deep knobs must collapse when composite_discovery_enabled_cfg is off"
    )

    # extreme_ar_group_aware_skip_models: gate on mlp_extreme_ar_group_aware_skip + 'mlp' in models.
    on_ear = dict(base_axes)
    on_ear.update(
        mlp_extreme_ar_group_aware_skip_cfg=True,
        extreme_ar_group_aware_skip_models_cfg="include_linear",
    )
    on_ear_combo = _build_combo(models=("mlp",), axes=on_ear, seed=0)
    assert on_ear_combo.extreme_ar_group_aware_skip_models_cfg == "include_linear"
    # When MLP-extreme-AR-skip is OFF, the skip-list axis must canon away.
    off_ear_a = dict(on_ear); off_ear_a["mlp_extreme_ar_group_aware_skip_cfg"] = False
    off_ear_b = dict(off_ear_a); off_ear_b["extreme_ar_group_aware_skip_models_cfg"] = "default_neural"
    c_ear_a = _build_combo(models=("mlp",), axes=off_ear_a, seed=0)
    c_ear_b = _build_combo(models=("mlp",), axes=off_ear_b, seed=0)
    assert c_ear_a.canonical_key() == c_ear_b.canonical_key(), (
        "extreme_ar_group_aware_skip_models_cfg must collapse when MLP-extreme-AR-skip is off"
    )

    # fs_pre_screen_null_fraction_threshold: gates on use_mrmr_fs OR
    # rfecv_estimator_cfg OR use_boruta_shap_cfg (mirrors the existing
    # fs_pre_screen_variance_threshold_cfg sibling gating exactly).
    on_fs = dict(base_axes)
    on_fs.update(use_mrmr_fs=True, fs_pre_screen_null_fraction_threshold_cfg=0.5)
    off_fs_a = dict(on_fs)
    off_fs_a.update(use_mrmr_fs=False, rfecv_estimator_cfg=None, use_boruta_shap_cfg=False)
    off_fs_b = dict(off_fs_a); off_fs_b["fs_pre_screen_null_fraction_threshold_cfg"] = 0.99
    c_fs_on = _build_combo(models=("cb",), axes=on_fs, seed=0)
    c_fs_off_a = _build_combo(models=("cb",), axes=off_fs_a, seed=0)
    c_fs_off_b = _build_combo(models=("cb",), axes=off_fs_b, seed=0)
    assert c_fs_on.fs_pre_screen_null_fraction_threshold_cfg == 0.5
    assert c_fs_off_a.canonical_key() == c_fs_off_b.canonical_key(), (
        "fs_pre_screen_null_fraction_threshold_cfg must collapse when no FS method is active"
    )

    # linear_l1_ratio_cfg: gate on 'linear' in models AND linear_solver_cfg='saga'.
    on_lin = dict(base_axes)
    on_lin.update(linear_solver_cfg="saga", linear_l1_ratio_cfg=1.0)
    c_lin_on = _build_combo(models=("linear",), axes=on_lin, seed=0)
    assert c_lin_on.linear_l1_ratio_cfg == 1.0
    # Solver != saga -> canon to 0.0 in canonical_key (avoids sklearn ValueError path).
    off_lin_a = dict(on_lin); off_lin_a["linear_solver_cfg"] = "lbfgs"
    off_lin_b = dict(off_lin_a); off_lin_b["linear_l1_ratio_cfg"] = 0.0
    c_lin_a = _build_combo(models=("linear",), axes=off_lin_a, seed=0)
    c_lin_b = _build_combo(models=("linear",), axes=off_lin_b, seed=0)
    assert c_lin_a.canonical_key() == c_lin_b.canonical_key(), (
        "linear_l1_ratio_cfg must collapse to 0.0 when linear_solver_cfg != saga"
    )

    # recurrent_hidden_size_cfg: gate on _canonical_recurrent_model() != None.
    on_rec = dict(base_axes)
    on_rec.update(
        recurrent_model_cfg="lstm",
        target_type="regression",
        n_rows=1000,
        text_col_count=0,
        embedding_col_count=0,
        recurrent_hidden_size_cfg=32,
    )
    c_rec_on = _build_combo(models=("cb",), axes=on_rec, seed=0)
    assert c_rec_on._canonical_recurrent_model() is not None
    assert c_rec_on.recurrent_hidden_size_cfg == 32
    # No recurrent (recurrent_model_cfg=None) -> canon to library default 128.
    off_rec_a = dict(on_rec); off_rec_a["recurrent_model_cfg"] = None
    off_rec_b = dict(off_rec_a); off_rec_b["recurrent_hidden_size_cfg"] = 128
    c_rec_a = _build_combo(models=("cb",), axes=off_rec_a, seed=0)
    c_rec_b = _build_combo(models=("cb",), axes=off_rec_b, seed=0)
    assert c_rec_a.canonical_key() == c_rec_b.canonical_key(), (
        "recurrent_hidden_size_cfg must collapse when no recurrent model is requested"
    )

    # (d) enable-axis varies the canonical key when toggled (the dedup keeps
    # both branches reachable). Spot-check one new axis per gating family.
    assert c_on.canonical_key() != c_off_a.canonical_key()  # FHC text gate
    assert c_comp_on.canonical_key() != c_comp_off_a.canonical_key()  # composite gate
    assert on.canonical_key() != combo_off_a.canonical_key()  # shap-proxied gate (re-asserted)


# ---------------------------------------------------------------------------
# 2026-05-28 audit-pass-2: 10 deeper axes (4 PART-A coverage gaps + 6 ShapProxiedFS
# deep extension knobs). Mirrors test_iter501_new_coverage_axes_flow_to_kwargs.
# ---------------------------------------------------------------------------


def test_iter502_audit_pass_2_axes_flow_to_kwargs():
    """10 audit-pass-2 fuzz axes must:
      (a) be present in AXES with >=2 candidate values,
      (b) be present in the FuzzCombo dataclass with the library default,
      (c) thread verbatim through their downstream consumer when ON,
      (d) canonical_key-collapse to defaults when the gating axis is OFF
          (no phantom variation in pairwise dedup).

    Axes (all from D:/Temp/AUDIT_PASS_2_DONE.md):
      PART A (4 LOW-tier coverage-gap axes, deferred from W11C wave):
        - ensembling_degenerate_class_ratio_cfg (EnsemblingConfig.degenerate_class_ratio)
        - behavior_use_flaml_zeroshot_cfg (TrainingBehaviorConfig.use_flaml_zeroshot)
        - target_temporal_audit_granularity_cfg (TrainingBehaviorConfig.target_temporal_audit_granularity)
        - prep_ext_dim_n_components_cfg (PreprocessingExtensionsConfig.dim_n_components)
      PART B (6 ShapProxiedFS deep extension knobs):
        - shap_proxied_config_jitter_cfg (ShapProxiedFS.config_jitter)
        - shap_proxied_uncertainty_penalty_cfg (ShapProxiedFS.uncertainty_penalty)
        - shap_proxied_within_cluster_refine_cfg (ShapProxiedFS.within_cluster_refine)
        - shap_proxied_use_bias_corrector_cfg (ShapProxiedFS.use_bias_corrector)
        - shap_proxied_refine_n_estimators_cfg (ShapProxiedFS.refine_n_estimators)
        - shap_proxied_trust_guard_n_estimators_cfg (ShapProxiedFS.trust_guard_n_estimators)
    """
    from tests.training._fuzz_combo import (
        AXES, _build_combo, enumerate_combos, build_shap_proxied_fs_kwargs,
    )

    new_axes = (
        # PART A
        "ensembling_degenerate_class_ratio_cfg",
        "behavior_use_flaml_zeroshot_cfg",
        "target_temporal_audit_granularity_cfg",
        "prep_ext_dim_n_components_cfg",
        # PART B
        "shap_proxied_config_jitter_cfg",
        "shap_proxied_uncertainty_penalty_cfg",
        "shap_proxied_within_cluster_refine_cfg",
        "shap_proxied_use_bias_corrector_cfg",
        "shap_proxied_refine_n_estimators_cfg",
        "shap_proxied_trust_guard_n_estimators_cfg",
    )
    # (a) Presence in AXES with >=2 candidates.
    for ax in new_axes:
        assert ax in AXES, f"missing fuzz axis {ax}"
        assert len(AXES[ax]) >= 2, f"axis {ax} must offer at least 2 values"

    # (b) FuzzCombo dataclass defaults match the library defaults.
    base_axes = {name: values[0] for name, values in AXES.items()}
    c_default = _build_combo(models=("cb",), axes=base_axes, seed=0)
    assert c_default.ensembling_degenerate_class_ratio_cfg == 0.01
    assert c_default.behavior_use_flaml_zeroshot_cfg is False
    assert c_default.target_temporal_audit_granularity_cfg == "auto"
    assert c_default.prep_ext_dim_n_components_cfg == 50
    assert c_default.shap_proxied_config_jitter_cfg is False
    assert c_default.shap_proxied_uncertainty_penalty_cfg == 0.0
    assert c_default.shap_proxied_within_cluster_refine_cfg is True
    assert c_default.shap_proxied_use_bias_corrector_cfg is True
    assert c_default.shap_proxied_refine_n_estimators_cfg == 100
    assert c_default.shap_proxied_trust_guard_n_estimators_cfg == 100

    # (c) enumerate_combos still reaches 150 combos with the new axes wired.
    combos = enumerate_combos(target=150, master_seed=20260422)
    assert len(combos) == 150, f"enumerate_combos lost combos: {len(combos)}"

    # -----------------------------------------------------------------
    # PART B: 6 ShapProxiedFS deep knobs thread into build_shap_proxied_fs_kwargs
    # verbatim when use_shap_proxied_fs=True (param names match ShapProxiedFS.__init__).
    # -----------------------------------------------------------------
    on_axes_b = dict(base_axes)
    on_axes_b.update(
        use_shap_proxied_fs=True,
        shap_proxied_cluster_features_cfg="auto",  # keep cluster on so within_cluster_refine survives
        shap_proxied_config_jitter_cfg=True,
        shap_proxied_uncertainty_penalty_cfg=0.5,
        shap_proxied_within_cluster_refine_cfg=False,
        shap_proxied_use_bias_corrector_cfg=False,
        shap_proxied_refine_n_estimators_cfg=None,
        shap_proxied_trust_guard_n_estimators_cfg=None,
    )
    on_b = _build_combo(models=("cb",), axes=on_axes_b, seed=0)
    kw_b = build_shap_proxied_fs_kwargs(on_b)
    assert kw_b is not None
    assert kw_b["config_jitter"] is True
    assert kw_b["uncertainty_penalty"] == 0.5
    assert kw_b["within_cluster_refine"] is False
    assert kw_b["use_bias_corrector"] is False
    assert kw_b["refine_n_estimators"] is None
    assert kw_b["trust_guard_n_estimators"] is None

    # OFF: kwargs is None, canonical_key collapses sub-knobs to defaults.
    off_b_a = dict(on_axes_b); off_b_a["use_shap_proxied_fs"] = False
    off_b_b = dict(off_b_a)
    off_b_b.update(
        shap_proxied_config_jitter_cfg=False,
        shap_proxied_uncertainty_penalty_cfg=0.0,
        shap_proxied_within_cluster_refine_cfg=True,
        shap_proxied_use_bias_corrector_cfg=True,
        shap_proxied_refine_n_estimators_cfg=100,
        shap_proxied_trust_guard_n_estimators_cfg=100,
    )
    combo_off_b_a = _build_combo(models=("cb",), axes=off_b_a, seed=0)
    combo_off_b_b = _build_combo(models=("cb",), axes=off_b_b, seed=0)
    assert build_shap_proxied_fs_kwargs(combo_off_b_a) is None
    assert combo_off_b_a.canonical_key() == combo_off_b_b.canonical_key(), (
        "ShapProxiedFS deep knobs must collapse when use_shap_proxied_fs is off"
    )

    # within_cluster_refine has a SECONDARY gate on cluster_features != False.
    # When cluster_features is literally False the refine flag becomes a no-op
    # and canonical_key must collapse it to the True default regardless.
    cluster_off_a = dict(base_axes)
    cluster_off_a.update(
        use_shap_proxied_fs=True,
        shap_proxied_cluster_features_cfg=False,
        shap_proxied_within_cluster_refine_cfg=False,
    )
    cluster_off_b = dict(cluster_off_a)
    cluster_off_b["shap_proxied_within_cluster_refine_cfg"] = True
    c_clu_a = _build_combo(models=("cb",), axes=cluster_off_a, seed=0)
    c_clu_b = _build_combo(models=("cb",), axes=cluster_off_b, seed=0)
    assert c_clu_a.canonical_key() == c_clu_b.canonical_key(), (
        "shap_proxied_within_cluster_refine_cfg must collapse when cluster_features is literal False"
    )

    # -----------------------------------------------------------------
    # PART A: 4 coverage-gap axes canon-collapse when their gates are off.
    # -----------------------------------------------------------------

    # A1. ensembling_degenerate_class_ratio_cfg: gate on use_ensembles AND
    # target_type in (binary/multilabel)_classification.
    on_a1 = dict(base_axes)
    on_a1.update(
        use_ensembles=True,
        target_type="binary_classification",
        ensembling_degenerate_class_ratio_cfg=0.05,
    )
    c_a1_on = _build_combo(models=("cb",), axes=on_a1, seed=0)
    assert c_a1_on.ensembling_degenerate_class_ratio_cfg == 0.05
    # Regression -> canon to default 0.01 (degenerate-subset gate is binary-only).
    off_a1_a = dict(on_a1); off_a1_a["target_type"] = "regression"
    off_a1_b = dict(off_a1_a); off_a1_b["ensembling_degenerate_class_ratio_cfg"] = 0.01
    c_a1_off_a = _build_combo(models=("cb",), axes=off_a1_a, seed=0)
    c_a1_off_b = _build_combo(models=("cb",), axes=off_a1_b, seed=0)
    assert c_a1_off_a.canonical_key() == c_a1_off_b.canonical_key(), (
        "ensembling_degenerate_class_ratio_cfg must collapse for non-classification targets"
    )

    # A2. behavior_use_flaml_zeroshot_cfg: gate on xgb OR lgb in models.
    # NOTE: the from_axes canon also drops True->False when `flaml` is not
    # importable, so on systems without flaml the axis collapses to False
    # at construction time -- the canon test below only fires when flaml IS
    # present. Use the as-constructed value so the test is environment-agnostic.
    on_a2 = dict(base_axes)
    on_a2.update(behavior_use_flaml_zeroshot_cfg=True)
    c_a2_xgb = _build_combo(models=("xgb",), axes=on_a2, seed=0)
    c_a2_cb = _build_combo(models=("cb",), axes=on_a2, seed=0)
    # When neither xgb nor lgb is in models, canon to False regardless of
    # the requested value -> distinct canonical key only when flaml is present
    # AND the request reaches the dataclass intact.
    if c_a2_xgb.behavior_use_flaml_zeroshot_cfg is True:
        assert c_a2_xgb.canonical_key() != c_a2_cb.canonical_key(), (
            "behavior_use_flaml_zeroshot_cfg must surface in canon when xgb is in scope (flaml present)"
        )

    # A3. target_temporal_audit_granularity_cfg: gate on with_datetime_col=True
    # AND target_temporal_audit_column_cfg='ts_col'.
    on_a3 = dict(base_axes)
    on_a3.update(
        with_datetime_col=True,
        target_temporal_audit_column_cfg="ts_col",
        target_temporal_audit_granularity_cfg="day",
    )
    c_a3_on = _build_combo(models=("cb",), axes=on_a3, seed=0)
    assert c_a3_on.target_temporal_audit_granularity_cfg == "day"
    # No datetime column -> canon away.
    off_a3_a = dict(on_a3); off_a3_a["with_datetime_col"] = False
    off_a3_b = dict(off_a3_a); off_a3_b["target_temporal_audit_granularity_cfg"] = "auto"
    c_a3_off_a = _build_combo(models=("cb",), axes=off_a3_a, seed=0)
    c_a3_off_b = _build_combo(models=("cb",), axes=off_a3_b, seed=0)
    assert c_a3_off_a.canonical_key() == c_a3_off_b.canonical_key(), (
        "target_temporal_audit_granularity_cfg must collapse when no datetime column"
    )

    # A4. prep_ext_dim_n_components_cfg: gate on prep_ext_dim_reducer_cfg in (PCA, TruncatedSVD).
    on_a4 = dict(base_axes)
    on_a4.update(
        prep_ext_dim_reducer_cfg="PCA",
        prep_ext_dim_n_components_cfg=10,
    )
    c_a4_on = _build_combo(models=("cb",), axes=on_a4, seed=0)
    assert c_a4_on.prep_ext_dim_n_components_cfg == 10
    # No dim_reducer -> canon to default 50.
    off_a4_a = dict(on_a4); off_a4_a["prep_ext_dim_reducer_cfg"] = None
    off_a4_b = dict(off_a4_a); off_a4_b["prep_ext_dim_n_components_cfg"] = 50
    c_a4_off_a = _build_combo(models=("cb",), axes=off_a4_a, seed=0)
    c_a4_off_b = _build_combo(models=("cb",), axes=off_a4_b, seed=0)
    assert c_a4_off_a.canonical_key() == c_a4_off_b.canonical_key(), (
        "prep_ext_dim_n_components_cfg must collapse when no dim_reducer is picked"
    )

    # (d) Each PART-A gate ALSO produces distinct canonical keys when the gate
    # is on with a non-default value vs. the canonical default branch -- so the
    # pairwise sampler reaches both branches.
    assert c_a1_on.canonical_key() != c_a1_off_a.canonical_key()
    assert c_a3_on.canonical_key() != c_a3_off_a.canonical_key()
    assert c_a4_on.canonical_key() != c_a4_off_a.canonical_key()


# ---------------------------------------------------------------------------
# 2026-05-28 audit-pass-3 W3: 4 ShapProxiedFS deep extension knobs.
# Mirrors test_iter502_audit_pass_2_axes_flow_to_kwargs.
# ---------------------------------------------------------------------------


def test_iter503_audit_pass_3_axes_flow_to_kwargs():
    """4 audit-pass-3 W3 fuzz axes must:
      (a) be present in AXES with >=2 candidate values,
      (b) be present in the FuzzCombo dataclass with the library default,
      (c) thread verbatim through build_shap_proxied_fs_kwargs when ON,
      (d) canonical_key-collapse to defaults when the gating axis is OFF.

    Axes (all source-verified against
    src/mlframe/feature_selection/shap_proxied_fs.py:69-79):
      - shap_proxied_cluster_weighting_cfg (ShapProxiedFS.cluster_weighting,
        default "pca_pc1"; secondary gate on cluster_features != False)
      - shap_proxied_max_interaction_features_cfg
        (ShapProxiedFS.max_interaction_features, default 16; secondary gate on
        interaction_aware=True -- the 64 alternate exercises the
        interaction-tensor wide-fan branch the default 16 cap would otherwise
        starve)
      - shap_proxied_prefilter_top_cfg (ShapProxiedFS.prefilter_top, default
        2000; None lifts the cap)
      - shap_proxied_prefilter_n_estimators_cfg
        (ShapProxiedFS.prefilter_n_estimators, default 100; None disables the
        booster tree cap)
    """
    from tests.training._fuzz_combo import (
        AXES, _build_combo, enumerate_combos, build_shap_proxied_fs_kwargs,
    )

    new_axes = (
        "shap_proxied_cluster_weighting_cfg",
        "shap_proxied_max_interaction_features_cfg",
        "shap_proxied_prefilter_top_cfg",
        "shap_proxied_prefilter_n_estimators_cfg",
    )
    # (a) Presence in AXES with >=2 candidates.
    for ax in new_axes:
        assert ax in AXES, f"missing fuzz axis {ax}"
        assert len(AXES[ax]) >= 2, f"axis {ax} must offer at least 2 values"

    # (b) FuzzCombo dataclass defaults match the library defaults.
    base_axes = {name: values[0] for name, values in AXES.items()}
    c_default = _build_combo(models=("cb",), axes=base_axes, seed=0)
    assert c_default.shap_proxied_cluster_weighting_cfg == "pca_pc1"
    assert c_default.shap_proxied_max_interaction_features_cfg == 16
    assert c_default.shap_proxied_prefilter_top_cfg == 2000
    assert c_default.shap_proxied_prefilter_n_estimators_cfg == 100

    # (c) enumerate_combos still reaches 150 combos with the new axes wired.
    combos = enumerate_combos(target=150, master_seed=20260422)
    assert len(combos) == 150, f"enumerate_combos lost combos: {len(combos)}"

    # -----------------------------------------------------------------
    # (c) Threading: 4 W3 axes flow into build_shap_proxied_fs_kwargs verbatim
    # when use_shap_proxied_fs=True (param names match ShapProxiedFS.__init__).
    # Keep cluster_features ON and interaction_aware=True so both
    # secondary-gated axes survive into the kwargs dict.
    # -----------------------------------------------------------------
    on_axes = dict(base_axes)
    on_axes.update(
        use_shap_proxied_fs=True,
        shap_proxied_cluster_features_cfg="auto",
        shap_proxied_interaction_aware_cfg=True,
        shap_proxied_cluster_weighting_cfg="factor_score",
        shap_proxied_max_interaction_features_cfg=64,
        shap_proxied_prefilter_top_cfg=None,
        shap_proxied_prefilter_n_estimators_cfg=None,
    )
    on = _build_combo(models=("cb",), axes=on_axes, seed=0)
    kw = build_shap_proxied_fs_kwargs(on)
    assert kw is not None
    assert kw["cluster_weighting"] == "factor_score"
    assert kw["max_interaction_features"] == 64
    assert kw["prefilter_top"] is None
    assert kw["prefilter_n_estimators"] is None

    # (d) When use_shap_proxied_fs=False, kwargs is None AND canonical_key
    # collapses all 4 axes to their library defaults.
    off_a = dict(on_axes); off_a["use_shap_proxied_fs"] = False
    off_b = dict(off_a)
    off_b.update(
        shap_proxied_cluster_weighting_cfg="pca_pc1",
        shap_proxied_max_interaction_features_cfg=16,
        shap_proxied_prefilter_top_cfg=2000,
        shap_proxied_prefilter_n_estimators_cfg=100,
    )
    c_off_a = _build_combo(models=("cb",), axes=off_a, seed=0)
    c_off_b = _build_combo(models=("cb",), axes=off_b, seed=0)
    assert build_shap_proxied_fs_kwargs(c_off_a) is None
    assert c_off_a.canonical_key() == c_off_b.canonical_key(), (
        "W3 ShapProxiedFS axes must collapse when use_shap_proxied_fs is off"
    )

    # cluster_weighting has a SECONDARY gate on cluster_features != False.
    # When cluster_features is literal False the weighting head becomes a
    # no-op and canonical_key must collapse it to "pca_pc1" regardless.
    cluster_off_a = dict(base_axes)
    cluster_off_a.update(
        use_shap_proxied_fs=True,
        shap_proxied_cluster_features_cfg=False,
        shap_proxied_cluster_weighting_cfg="factor_score",
    )
    cluster_off_b = dict(cluster_off_a)
    cluster_off_b["shap_proxied_cluster_weighting_cfg"] = "pca_pc1"
    c_clu_a = _build_combo(models=("cb",), axes=cluster_off_a, seed=0)
    c_clu_b = _build_combo(models=("cb",), axes=cluster_off_b, seed=0)
    assert c_clu_a.canonical_key() == c_clu_b.canonical_key(), (
        "shap_proxied_cluster_weighting_cfg must collapse when cluster_features is literal False"
    )

    # max_interaction_features has a SECONDARY gate on interaction_aware=True.
    # When interactions are off the cap is consumed by no code path; canon
    # must collapse it to 16 regardless of the stored value.
    inter_off_a = dict(base_axes)
    inter_off_a.update(
        use_shap_proxied_fs=True,
        shap_proxied_interaction_aware_cfg=False,
        shap_proxied_max_interaction_features_cfg=64,
    )
    inter_off_b = dict(inter_off_a)
    inter_off_b["shap_proxied_max_interaction_features_cfg"] = 16
    c_int_a = _build_combo(models=("cb",), axes=inter_off_a, seed=0)
    c_int_b = _build_combo(models=("cb",), axes=inter_off_b, seed=0)
    assert c_int_a.canonical_key() == c_int_b.canonical_key(), (
        "shap_proxied_max_interaction_features_cfg must collapse when interaction_aware=False"
    )

    # When interaction_aware=True the 64 alternate MUST reach kwargs (the
    # audit's worry: default 16 cap was starving the interaction-aware
    # combos so the interaction tensor never fired with the wider fan).
    inter_on_axes = dict(base_axes)
    inter_on_axes.update(
        use_shap_proxied_fs=True,
        shap_proxied_interaction_aware_cfg=True,
        shap_proxied_max_interaction_features_cfg=64,
    )
    c_int_on = _build_combo(models=("cb",), axes=inter_on_axes, seed=0)
    kw_int_on = build_shap_proxied_fs_kwargs(c_int_on)
    assert kw_int_on is not None
    assert kw_int_on["max_interaction_features"] == 64, (
        "interaction_aware=True must let max_interaction_features=64 reach the kwargs"
    )
    assert kw_int_on["interaction_aware"] is True

    # The 64-with-interactions canonical key must differ from the 16-default
    # canonical key (so the pairwise sampler actually reaches the wide-fan
    # interaction branch).
    inter_on_default = dict(inter_on_axes)
    inter_on_default["shap_proxied_max_interaction_features_cfg"] = 16
    c_int_def = _build_combo(models=("cb",), axes=inter_on_default, seed=0)
    assert c_int_on.canonical_key() != c_int_def.canonical_key(), (
        "max_interaction_features=64 vs 16 must produce distinct canon keys when interactions on"
    )


# ---------------------------------------------------------------------------
# iter554: short_id() determinism across import-order / env state
# ---------------------------------------------------------------------------


def test_iter554_short_id_independent_of_haflaml_env_state():
    """FuzzCombo.short_id() must be a pure function of the axis values.

    Prior to the fix, ``_canon_use_flaml_zeroshot(axes[...]) `` was applied at
    ``from_axes`` resolution time, dropping ``True`` -> ``False`` when the
    optional ``flaml`` dep was not importable. flaml's import success can
    flip on transitive sys.path / module-cache state across processes
    (matplotlib import order observably toggles it on Windows in this repo),
    which made the same logical combo emit different short_ids across
    "picker" scripts and ``profile_one_combo.py``. The fix stores the
    LOGICAL requested value in the FuzzCombo dataclass; the fit-time
    contract remains the consumer of ``_HAS_FLAML`` for skip / xfail
    decisions.

    This test pins:
      (1) Two combos differing ONLY in behavior_use_flaml_zeroshot_cfg
          have distinct canonical_keys and distinct short_ids regardless
          of ``_HAS_FLAML`` -- the canon never collapses True -> False.
      (2) When the requested value is True, the dataclass field is True.
    """
    base_axes = {name: values[0] for name, values in AXES.items()}
    # xgb in models so the canonical_key's flaml gate ("xgb in models or
    # lgb in models") doesn't short-circuit the axis.
    axes_false = dict(base_axes, behavior_use_flaml_zeroshot_cfg=False)
    axes_true = dict(base_axes, behavior_use_flaml_zeroshot_cfg=True)

    c_false = _build_combo(models=("xgb",), axes=axes_false, seed=0)
    c_true = _build_combo(models=("xgb",), axes=axes_true, seed=0)

    assert c_true.behavior_use_flaml_zeroshot_cfg is True, (
        "FuzzCombo must store the LOGICAL requested value (True), regardless of _HAS_FLAML env state"
    )
    assert c_false.behavior_use_flaml_zeroshot_cfg is False

    assert c_true.canonical_key() != c_false.canonical_key(), (
        "True and False must produce distinct canonical_keys; canon must NOT collapse to False"
    )
    assert c_true.short_id() != c_false.short_id(), (
        "True and False must produce distinct short_ids; reproducibility across environments"
    )


# ---------------------------------------------------------------------------
# 2026-05-28 audit-pass-4 SAFE-subset (W4): 8 canon-only axes.
# Mirrors test_iter503_audit_pass_3_axes_flow_to_kwargs but for canon-only
# axes (no downstream consumer-builder helper) so the assertions cover
# (a) AXES presence, (b) FuzzCombo dataclass defaults source-verified,
# (c) enumerate_combos still reaches 150 combos, (d) canon-collapse under
# each gate. The slice_stable_es_* family is deferred to a separate batch
# pending the parallel SliceStableESConfig refactor.
# ---------------------------------------------------------------------------


def test_iter556_audit_pass_4_safe_axes_flow_to_kwargs():
    """8 audit-pass-4 SAFE-subset (W4) fuzz axes must:
      (a) be present in AXES with >=2 candidate values,
      (b) be present in the FuzzCombo dataclass with the SOURCE-verified
          library default (drift corrections from the audit are listed in
          FUZZ_AXES_W4_IMPL_DONE.json's drift_corrections array),
      (c) leave enumerate_combos at exactly 150 combos with the new axes wired,
      (d) canonical_key-collapse under each documented gate.

    Axes (all source-verified pre-edit):
      - calibration_policy_auto_pick_cfg
        (CalibrationConfig.policy_auto_pick at calibration/policy.py:464;
        default True -- audit said False, drift!)
      - calibration_n_bootstrap_cfg
        (CalibrationConfig.n_bootstrap at policy.py:467 via DEFAULT_N_BOOTSTRAP;
        default 1000 -- audit said 200, drift!)
      - calibration_candidates_cfg
        (CalibrationConfig.candidates at policy.py:469;
        default None -- audit said tuple, drift!)
      - pipeline_cache_ram_budget_fraction_cfg
        (TrainingBehaviorConfig.pipeline_cache_ram_budget_fraction at
        _model_configs.py:641; default 0.4 -- matches audit)
      - reporting_compute_trainset_metrics_cfg
        (ReportingConfig.compute_trainset_metrics at _reporting_configs.py:96;
        default False -- audit said True, drift!)
      - reporting_mase_seasonality_cfg
        (ReportingConfig.mase_seasonality at _reporting_configs.py:140;
        default 1 -- audit said None/sequence, drift! it's an int)
      - recurrent_use_stratified_sampler_cfg
        (RecurrentConfig.use_stratified_sampler at neural/_recurrent_config.py:90;
        default True -- audit said False/True, source pins True)
      - behavior_model_file_hash_suffix_cfg
        (TrainingBehaviorConfig.model_file_hash_suffix at _model_configs.py:547;
        default True -- audit said str|None default None, MAJOR drift! it's a bool)

    All 8 axes are canon-only (no downstream consumer-builder helper) --
    matches the ensembling_degenerate_class_ratio_cfg pattern from W2.
    """
    from tests.training._fuzz_combo import (
        AXES, _build_combo, enumerate_combos,
    )

    new_axes = (
        "calibration_policy_auto_pick_cfg",
        "calibration_n_bootstrap_cfg",
        "calibration_candidates_cfg",
        "pipeline_cache_ram_budget_fraction_cfg",
        "reporting_compute_trainset_metrics_cfg",
        "reporting_mase_seasonality_cfg",
        "recurrent_use_stratified_sampler_cfg",
        "behavior_model_file_hash_suffix_cfg",
    )
    # (a) Presence in AXES with >=2 candidates.
    for ax in new_axes:
        assert ax in AXES, f"missing fuzz axis {ax}"
        assert len(AXES[ax]) >= 2, f"axis {ax} must offer at least 2 values"

    # (b) FuzzCombo dataclass defaults match the source-verified library defaults.
    base_axes = {name: values[0] for name, values in AXES.items()}
    c_default = _build_combo(models=("cb",), axes=base_axes, seed=0)
    assert c_default.calibration_policy_auto_pick_cfg is True
    assert c_default.calibration_n_bootstrap_cfg == 1000
    assert c_default.calibration_candidates_cfg is None
    assert c_default.pipeline_cache_ram_budget_fraction_cfg == 0.4
    assert c_default.reporting_compute_trainset_metrics_cfg is False
    assert c_default.reporting_mase_seasonality_cfg == 1
    assert c_default.recurrent_use_stratified_sampler_cfg is True
    assert c_default.behavior_model_file_hash_suffix_cfg is True

    # (c) enumerate_combos still reaches 150 combos with the new axes wired.
    combos = enumerate_combos(target=150, master_seed=20260422)
    assert len(combos) == 150, f"enumerate_combos lost combos: {len(combos)}"

    # -----------------------------------------------------------------
    # (d) Canon collapse: gating axes drive the per-axis collapse to
    # source defaults so non-applicable combos don't gain phantom canonical
    # keys.
    # -----------------------------------------------------------------

    # Calibration trio: only meaningful on classification targets AND
    # policy_auto_pick=True (n_bootstrap / candidates collapse on EITHER
    # condition off). Use regression target_type to confirm the calibration
    # trio collapses regardless of stored values.
    reg_axes_a = dict(base_axes)
    reg_axes_a.update(
        target_type="regression",
        calibration_policy_auto_pick_cfg=False,
        calibration_n_bootstrap_cfg=100,
        calibration_candidates_cfg=("Sigmoid", "Isotonic"),
    )
    reg_axes_b = dict(base_axes)
    reg_axes_b.update(
        target_type="regression",
        calibration_policy_auto_pick_cfg=True,
        calibration_n_bootstrap_cfg=1000,
        calibration_candidates_cfg=None,
    )
    c_reg_a = _build_combo(models=("cb",), axes=reg_axes_a, seed=0)
    c_reg_b = _build_combo(models=("cb",), axes=reg_axes_b, seed=0)
    assert c_reg_a.canonical_key() == c_reg_b.canonical_key(), (
        "calibration trio must collapse to defaults when target_type=regression"
    )

    # Classification + policy_auto_pick=False: n_bootstrap + candidates must
    # collapse to defaults (1000, None) regardless of stored values.
    cls_off_a = dict(base_axes)
    cls_off_a.update(
        target_type="binary_classification",
        calibration_policy_auto_pick_cfg=False,
        calibration_n_bootstrap_cfg=100,
        calibration_candidates_cfg=("Sigmoid", "Isotonic"),
    )
    cls_off_b = dict(cls_off_a)
    cls_off_b.update(
        calibration_n_bootstrap_cfg=1000,
        calibration_candidates_cfg=None,
    )
    c_cls_off_a = _build_combo(models=("cb",), axes=cls_off_a, seed=0)
    c_cls_off_b = _build_combo(models=("cb",), axes=cls_off_b, seed=0)
    assert c_cls_off_a.canonical_key() == c_cls_off_b.canonical_key(), (
        "n_bootstrap+candidates must collapse when policy_auto_pick=False"
    )

    # Classification + policy_auto_pick=True: the n_bootstrap + candidates
    # values now WIDEN the canonical key (the gate is open).
    cls_on_a = dict(cls_off_a)
    cls_on_a["calibration_policy_auto_pick_cfg"] = True
    cls_on_b = dict(cls_on_a)
    cls_on_b.update(
        calibration_n_bootstrap_cfg=1000,
        calibration_candidates_cfg=None,
    )
    c_cls_on_a = _build_combo(models=("cb",), axes=cls_on_a, seed=0)
    c_cls_on_b = _build_combo(models=("cb",), axes=cls_on_b, seed=0)
    assert c_cls_on_a.canonical_key() != c_cls_on_b.canonical_key(), (
        "n_bootstrap+candidates must fork the canon key when policy_auto_pick=True"
    )

    # reporting_mase_seasonality: collapses to default 1 for non-regression.
    mase_cls_a = dict(base_axes)
    mase_cls_a.update(
        target_type="binary_classification",
        reporting_mase_seasonality_cfg=12,
    )
    mase_cls_b = dict(mase_cls_a)
    mase_cls_b["reporting_mase_seasonality_cfg"] = 1
    c_mase_cls_a = _build_combo(models=("cb",), axes=mase_cls_a, seed=0)
    c_mase_cls_b = _build_combo(models=("cb",), axes=mase_cls_b, seed=0)
    assert c_mase_cls_a.canonical_key() == c_mase_cls_b.canonical_key(), (
        "mase_seasonality must collapse to 1 for non-regression target_type"
    )

    # reporting_mase_seasonality: forks the canon key for regression.
    mase_reg_a = dict(base_axes)
    mase_reg_a.update(
        target_type="regression",
        reporting_mase_seasonality_cfg=12,
    )
    mase_reg_b = dict(mase_reg_a)
    mase_reg_b["reporting_mase_seasonality_cfg"] = 1
    c_mase_reg_a = _build_combo(models=("cb",), axes=mase_reg_a, seed=0)
    c_mase_reg_b = _build_combo(models=("cb",), axes=mase_reg_b, seed=0)
    assert c_mase_reg_a.canonical_key() != c_mase_reg_b.canonical_key(), (
        "mase_seasonality must fork the canon key for regression"
    )

    # recurrent_use_stratified_sampler: collapses when no recurrent model.
    rec_off_a = dict(base_axes)
    rec_off_a.update(
        recurrent_model_cfg=None,
        recurrent_use_stratified_sampler_cfg=False,
    )
    rec_off_b = dict(rec_off_a)
    rec_off_b["recurrent_use_stratified_sampler_cfg"] = True
    c_rec_off_a = _build_combo(models=("cb",), axes=rec_off_a, seed=0)
    c_rec_off_b = _build_combo(models=("cb",), axes=rec_off_b, seed=0)
    assert c_rec_off_a.canonical_key() == c_rec_off_b.canonical_key(), (
        "use_stratified_sampler must collapse when recurrent_model_cfg=None"
    )

    # pipeline_cache_ram_budget_fraction and behavior_model_file_hash_suffix
    # are ungated (always meaningful). Any value flip MUST fork the canon key.
    rambud_a = dict(base_axes)
    rambud_a["pipeline_cache_ram_budget_fraction_cfg"] = 0.4
    rambud_b = dict(base_axes)
    rambud_b["pipeline_cache_ram_budget_fraction_cfg"] = 0.1
    c_rambud_a = _build_combo(models=("cb",), axes=rambud_a, seed=0)
    c_rambud_b = _build_combo(models=("cb",), axes=rambud_b, seed=0)
    assert c_rambud_a.canonical_key() != c_rambud_b.canonical_key(), (
        "pipeline_cache_ram_budget_fraction must fork the canon key (ungated)"
    )

    hashsuf_a = dict(base_axes)
    hashsuf_a["behavior_model_file_hash_suffix_cfg"] = True
    hashsuf_b = dict(base_axes)
    hashsuf_b["behavior_model_file_hash_suffix_cfg"] = False
    c_hashsuf_a = _build_combo(models=("cb",), axes=hashsuf_a, seed=0)
    c_hashsuf_b = _build_combo(models=("cb",), axes=hashsuf_b, seed=0)
    assert c_hashsuf_a.canonical_key() != c_hashsuf_b.canonical_key(), (
        "behavior_model_file_hash_suffix must fork the canon key (ungated)"
    )

    # reporting_compute_trainset_metrics is ungated.
    tsm_a = dict(base_axes)
    tsm_a["reporting_compute_trainset_metrics_cfg"] = False
    tsm_b = dict(base_axes)
    tsm_b["reporting_compute_trainset_metrics_cfg"] = True
    c_tsm_a = _build_combo(models=("cb",), axes=tsm_a, seed=0)
    c_tsm_b = _build_combo(models=("cb",), axes=tsm_b, seed=0)
    assert c_tsm_a.canonical_key() != c_tsm_b.canonical_key(), (
        "reporting_compute_trainset_metrics must fork the canon key (ungated)"
    )


# ---------------------------------------------------------------------------
# iter558: audit-pass-5 (W5) ShapProxiedFS trust-guard / fidelity axes
# ---------------------------------------------------------------------------


def test_iter558_audit_pass_5_axes_flow_to_kwargs():
    """8 audit-pass-5 SAFE-subset (W5) ShapProxiedFS fuzz axes must:
      (a) be present in AXES with >=2 candidate values,
      (b) be present in the FuzzCombo dataclass with the SOURCE-verified
          ShapProxiedFS.__init__ defaults (verified pre-edit against
          feature_selection/shap_proxied_fs.py:62, 78, 89-94),
      (c) flow through build_shap_proxied_fs_kwargs into the kwargs dict
          when use_shap_proxied_fs=True (with the right secondary gate on),
      (d) canon-collapse correctly under the documented secondary gates:
            - uniform_tail_frac collapses to 0.2 when stratified_anchors=False
            - zipf_alpha collapses to 0.25 when cardinality_dist != "zipf"

    Axes (all source-verified pre-edit):
      - shap_proxied_trust_guard_stratified_anchors_cfg
        (ShapProxiedFS.trust_guard_stratified_anchors, default False at :89)
      - shap_proxied_trust_guard_uniform_tail_frac_cfg
        (ShapProxiedFS.trust_guard_uniform_tail_frac, default 0.2 at :90)
      - shap_proxied_trust_guard_cardinality_dist_cfg
        (ShapProxiedFS.trust_guard_cardinality_dist, default "zipf" at :91)
      - shap_proxied_trust_guard_zipf_alpha_cfg
        (ShapProxiedFS.trust_guard_zipf_alpha, default 0.25 at :92)
      - shap_proxied_trust_guard_fidelity_weights_cfg
        (ShapProxiedFS.trust_guard_fidelity_weights, default (0.6, 0.4) at :93)
      - shap_proxied_trust_guard_metric_cfg
        (ShapProxiedFS.trust_guard_metric, default "proxy_fidelity_score" at :94)
      - shap_proxied_fidelity_floor_cfg
        (ShapProxiedFS.fidelity_floor, default 0.5 at :62)
      - shap_proxied_oof_shap_n_estimators_cfg
        (ShapProxiedFS.oof_shap_n_estimators, default 100 at :78)
    """
    from tests.training._fuzz_combo import (
        AXES, _build_combo, build_shap_proxied_fs_kwargs,
    )

    new_axes = (
        "shap_proxied_trust_guard_stratified_anchors_cfg",
        "shap_proxied_trust_guard_uniform_tail_frac_cfg",
        "shap_proxied_trust_guard_cardinality_dist_cfg",
        "shap_proxied_trust_guard_zipf_alpha_cfg",
        "shap_proxied_trust_guard_fidelity_weights_cfg",
        "shap_proxied_trust_guard_metric_cfg",
        "shap_proxied_fidelity_floor_cfg",
        "shap_proxied_oof_shap_n_estimators_cfg",
    )
    # (a) Presence in AXES with >=2 candidates.
    for ax in new_axes:
        assert ax in AXES, f"missing fuzz axis {ax}"
        assert len(AXES[ax]) >= 2, f"axis {ax} must offer at least 2 values"

    # (b) FuzzCombo dataclass defaults match SOURCE defaults verified against
    # ShapProxiedFS.__init__ (feature_selection/shap_proxied_fs.py).
    base_axes = {name: values[0] for name, values in AXES.items()}
    c_default = _build_combo(models=("cb",), axes=base_axes, seed=0)
    assert c_default.shap_proxied_trust_guard_stratified_anchors_cfg is False
    assert c_default.shap_proxied_trust_guard_uniform_tail_frac_cfg == 0.2
    assert c_default.shap_proxied_trust_guard_cardinality_dist_cfg == "zipf"
    assert c_default.shap_proxied_trust_guard_zipf_alpha_cfg == 0.25
    assert c_default.shap_proxied_trust_guard_fidelity_weights_cfg == (0.6, 0.4)
    assert c_default.shap_proxied_trust_guard_metric_cfg == "proxy_fidelity_score"
    assert c_default.shap_proxied_fidelity_floor_cfg == 0.5
    assert c_default.shap_proxied_oof_shap_n_estimators_cfg == 100

    # (c) build_shap_proxied_fs_kwargs returns the new kwargs with non-default
    # values when use_shap_proxied_fs=True (the gates inside the kwargs
    # builder do not filter; canon-gates only live in canonical_key).
    on_axes = dict(base_axes)
    on_axes.update(
        use_shap_proxied_fs=True,
        shap_proxied_trust_guard_cfg=True,
        shap_proxied_prefilter_method_cfg="univariate",
        shap_proxied_trust_guard_stratified_anchors_cfg=True,
        shap_proxied_trust_guard_uniform_tail_frac_cfg=0.0,
        shap_proxied_trust_guard_cardinality_dist_cfg="uniform",
        shap_proxied_trust_guard_zipf_alpha_cfg=1.0,
        shap_proxied_trust_guard_fidelity_weights_cfg=(0.5, 0.5),
        shap_proxied_trust_guard_metric_cfg="spearman",
        shap_proxied_fidelity_floor_cfg=0.7,
        shap_proxied_oof_shap_n_estimators_cfg=None,
    )
    c_on = _build_combo(models=("cb",), axes=on_axes, seed=0)
    kw = build_shap_proxied_fs_kwargs(c_on)
    assert kw is not None, "build_shap_proxied_fs_kwargs must return dict when use_shap_proxied_fs=True"
    assert kw["trust_guard_stratified_anchors"] is True
    assert kw["trust_guard_uniform_tail_frac"] == 0.0
    assert kw["trust_guard_cardinality_dist"] == "uniform"
    assert kw["trust_guard_zipf_alpha"] == 1.0
    assert kw["trust_guard_fidelity_weights"] == (0.5, 0.5)
    assert kw["trust_guard_metric"] == "spearman"
    assert kw["fidelity_floor"] == 0.7
    assert kw["oof_shap_n_estimators"] is None

    # (d) Canon-collapse: uniform_tail_frac collapses to 0.2 when
    # stratified_anchors=False (no anchor weights -> tail_frac is a no-op).
    utf_a = dict(base_axes)
    utf_a.update(
        use_shap_proxied_fs=True,
        shap_proxied_trust_guard_cfg=True,
        shap_proxied_prefilter_method_cfg="univariate",
        shap_proxied_trust_guard_stratified_anchors_cfg=False,
        shap_proxied_trust_guard_uniform_tail_frac_cfg=0.0,
    )
    utf_b = dict(utf_a)
    utf_b["shap_proxied_trust_guard_uniform_tail_frac_cfg"] = 0.2
    c_utf_a = _build_combo(models=("cb",), axes=utf_a, seed=0)
    c_utf_b = _build_combo(models=("cb",), axes=utf_b, seed=0)
    assert c_utf_a.canonical_key() == c_utf_b.canonical_key(), (
        "uniform_tail_frac must collapse to 0.2 when stratified_anchors=False"
    )

    # zipf_alpha collapses to 0.25 when cardinality_dist != "zipf"
    # (alpha is unused on the uniform branch).
    za_a = dict(base_axes)
    za_a.update(
        use_shap_proxied_fs=True,
        shap_proxied_trust_guard_cardinality_dist_cfg="uniform",
        shap_proxied_trust_guard_zipf_alpha_cfg=1.0,
    )
    za_b = dict(za_a)
    za_b["shap_proxied_trust_guard_zipf_alpha_cfg"] = 0.25
    c_za_a = _build_combo(models=("cb",), axes=za_a, seed=0)
    c_za_b = _build_combo(models=("cb",), axes=za_b, seed=0)
    assert c_za_a.canonical_key() == c_za_b.canonical_key(), (
        "zipf_alpha must collapse to 0.25 when cardinality_dist != 'zipf'"
    )


def test_iter569_audit_pass_6_axes_flow_to_kwargs():
    """15 audit-pass-6 (W6) fuzz axes must:
      (a) be present in AXES with >=2 candidate values,
      (b) carry the SOURCE-verified library defaults in the FuzzCombo
          dataclass (verified pre-edit against
          src/mlframe/training/_training_runtime_configs.py:42-95 for
          SliceStableESConfig, _model_configs.py:505 for
          early_stop_on_worsening, filters/mrmr.py:224-302,589 for
          Wave 7/8/9, and _composite_target_discovery_config.py:117
          for cv_selector_mode),
      (c) collapse correctly under the documented secondary gates:
            - mrmr_* axes (8) collapse to defaults when use_mrmr_fs=False,
            - cv_selector_mode_cfg collapses to "mean" when
              composite_discovery_enabled_cfg=False OR target != regression,
            - slice_stable_es sub-knobs (4) collapse to SliceStableESConfig
              defaults when slice_stable_es_enabled_cfg=False,
      (d) thread through the relevant config builder where applicable:
            - MRMR axes flow into ``build_mrmr_kwargs`` dict
              (consumed by FeatureSelectionConfig.mrmr_kwargs),
            - cv_selector_mode flows through
              ``build_composite_discovery_config_from_flat`` kwargs
              (consumed by CompositeTargetDiscoveryConfig.cv_selector_mode),
            - slice_stable_es + early_stop_on_worsening are CANON-ONLY:
              the fuzz suite does not currently propagate
              ``SliceStableESConfig`` or ``TrainingBehaviorConfig.early_stop_on_worsening``
              into ``train_mlframe_models_suite`` (the suite-level kwarg is
              not exposed); only the canon dedup is exercised by fuzz.
              Pinned by canonical_key collapse assertions below.

    Drift corrections caught pre-edit (audit-pass-6 vs source-of-truth):
      - audit said ``slice_stable_es_source_cfg`` default "random" -->
        SOURCE _training_runtime_configs.py:78 says "temporal". Using source.
      - audit said ``slice_stable_es_aggregate_cfg`` default "t_lcb" -->
        SOURCE _training_runtime_configs.py:89 says "mean". Using source.
      - audit said ``early_stop_on_worsening_cfg`` default unspecified;
        SOURCE _model_configs.py:505 says True. Using source.
    """
    from tests.training._fuzz_combo import (
        AXES, _build_combo, build_mrmr_kwargs,
        build_composite_discovery_config_from_flat,
    )

    new_axes = (
        "slice_stable_es_enabled_cfg",
        "slice_stable_es_aggregate_cfg",
        "slice_stable_es_source_cfg",
        "slice_stable_es_pareto_best_iter_selection_cfg",
        "slice_stable_es_diagnostic_only_cfg",
        "early_stop_on_worsening_cfg",
        "mrmr_nbins_strategy_cfg",
        "mrmr_mi_correction_cfg",
        "mrmr_redundancy_aggregator_cfg",
        "mrmr_bur_lambda_cfg",
        "mrmr_cmi_perm_stop_cfg",
        "mrmr_stability_selection_method_cfg",
        "mrmr_mi_normalization_cfg",
        "mrmr_dcd_enable_cfg",
        "cv_selector_mode_cfg",
    )

    # (a) Presence in AXES with >=2 candidates.
    for ax in new_axes:
        assert ax in AXES, f"missing fuzz axis {ax}"
        assert len(AXES[ax]) >= 2, f"axis {ax} must offer at least 2 values"

    # (b) FuzzCombo dataclass defaults match SOURCE defaults verified pre-edit.
    base_axes = {name: values[0] for name, values in AXES.items()}
    c_default = _build_combo(models=("cb",), axes=base_axes, seed=0)
    assert c_default.slice_stable_es_enabled_cfg is False
    assert c_default.slice_stable_es_aggregate_cfg == "mean"
    assert c_default.slice_stable_es_source_cfg == "temporal"
    assert c_default.slice_stable_es_pareto_best_iter_selection_cfg is False
    assert c_default.slice_stable_es_diagnostic_only_cfg is False
    assert c_default.early_stop_on_worsening_cfg is True
    assert c_default.mrmr_nbins_strategy_cfg == "mdlp"
    assert c_default.mrmr_mi_correction_cfg == "none"
    assert c_default.mrmr_redundancy_aggregator_cfg is None
    assert c_default.mrmr_bur_lambda_cfg == 0.0
    assert c_default.mrmr_cmi_perm_stop_cfg is False
    assert c_default.mrmr_stability_selection_method_cfg == "classic"
    assert c_default.mrmr_mi_normalization_cfg == "none"
    assert c_default.mrmr_dcd_enable_cfg is False
    assert c_default.cv_selector_mode_cfg == "mean"

    # (c) Canon-collapse sanity. Build two combos that differ ONLY on
    # the gated-off knob and verify canonical_key() is identical.
    # (c.1) MRMR axes collapse when use_mrmr_fs=False.
    mrmr_a = dict(base_axes)
    mrmr_a.update(use_mrmr_fs=False)
    mrmr_b = dict(mrmr_a)
    mrmr_b.update(
        mrmr_nbins_strategy_cfg="quantile",
        mrmr_mi_correction_cfg="chao_shen",
        mrmr_redundancy_aggregator_cfg="jmim",
        mrmr_bur_lambda_cfg=0.5,
        mrmr_cmi_perm_stop_cfg=True,
        mrmr_stability_selection_method_cfg="cluster",
        mrmr_mi_normalization_cfg="su",
        mrmr_dcd_enable_cfg=True,
    )
    c_mrmr_a = _build_combo(models=("cb",), axes=mrmr_a, seed=0)
    c_mrmr_b = _build_combo(models=("cb",), axes=mrmr_b, seed=0)
    assert c_mrmr_a.canonical_key() == c_mrmr_b.canonical_key(), (
        "all 8 MRMR Wave 7/8/9 axes must collapse to MRMR defaults when "
        "use_mrmr_fs=False so dedup absorbs the disabled-branch combos"
    )

    # (c.2) cv_selector_mode_cfg collapses to "mean" when discovery is off.
    cv_a = dict(base_axes)
    cv_a.update(
        composite_discovery_enabled_cfg=False,
        cv_selector_mode_cfg="t_lcb",
    )
    cv_b = dict(cv_a)
    cv_b["cv_selector_mode_cfg"] = "mean"
    c_cv_a = _build_combo(models=("cb",), axes=cv_a, seed=0)
    c_cv_b = _build_combo(models=("cb",), axes=cv_b, seed=0)
    assert c_cv_a.canonical_key() == c_cv_b.canonical_key(), (
        "cv_selector_mode_cfg must collapse to 'mean' when "
        "composite_discovery_enabled_cfg=False"
    )

    # (c.3) slice_stable_es sub-knobs collapse to SliceStableESConfig
    # defaults when slice_stable_es_enabled_cfg=False.
    sse_a = dict(base_axes)
    sse_a.update(slice_stable_es_enabled_cfg=False)
    sse_b = dict(sse_a)
    sse_b.update(
        slice_stable_es_aggregate_cfg="t_lcb",
        slice_stable_es_source_cfg="random",
        slice_stable_es_pareto_best_iter_selection_cfg=True,
        slice_stable_es_diagnostic_only_cfg=True,
    )
    c_sse_a = _build_combo(models=("cb",), axes=sse_a, seed=0)
    c_sse_b = _build_combo(models=("cb",), axes=sse_b, seed=0)
    assert c_sse_a.canonical_key() == c_sse_b.canonical_key(), (
        "slice_stable_es sub-knobs must collapse to source defaults when "
        "slice_stable_es_enabled_cfg=False"
    )

    # (d.1) MRMR axes flow through build_mrmr_kwargs into the dict that
    # FeatureSelectionConfig.mrmr_kwargs consumes. Names match MRMR.__init__
    # exactly (filters/mrmr.py:224-302, 589).
    on_axes = dict(base_axes)
    on_axes.update(
        use_mrmr_fs=True,
        mrmr_nbins_strategy_cfg="quantile",
        mrmr_mi_correction_cfg="chao_shen",
        mrmr_redundancy_aggregator_cfg="jmim",
        mrmr_bur_lambda_cfg=0.5,
        mrmr_cmi_perm_stop_cfg=True,
        mrmr_stability_selection_method_cfg="cluster",
        mrmr_mi_normalization_cfg="su",
        mrmr_dcd_enable_cfg=True,
    )
    c_on = _build_combo(models=("cb",), axes=on_axes, seed=0)
    kw = build_mrmr_kwargs(c_on)
    assert kw is not None, "build_mrmr_kwargs must return dict when use_mrmr_fs=True"
    assert kw["nbins_strategy"] == "quantile"
    assert kw["mi_correction"] == "chao_shen"
    assert kw["redundancy_aggregator"] == "jmim"
    assert kw["bur_lambda"] == 0.5
    assert kw["cmi_perm_stop"] is True
    assert kw["stability_selection_method"] == "cluster"
    assert kw["mi_normalization"] == "su"
    assert kw["dcd_enable"] is True

    # (d.2) cv_selector_mode flows through
    # build_composite_discovery_config_from_flat into the
    # CompositeTargetDiscoveryConfig kwargs.
    cfg = build_composite_discovery_config_from_flat(
        enabled=True, cv_selector_mode="t_lcb",
    )
    assert cfg is not None
    # Source field name verified at _composite_target_discovery_config.py:117.
    assert getattr(cfg, "cv_selector_mode", None) == "t_lcb"

    # (d.3) slice_stable_es + early_stop_on_worsening are CANON-ONLY:
    # the suite does not currently expose SliceStableESConfig nor the
    # early_stop_on_worsening behaviour flag as a top-level kwarg to
    # train_mlframe_models_suite; canon dedup is exercised above (c.3)
    # and the master enable/disable axis below.
    e_a = dict(base_axes)
    e_a.update(slice_stable_es_enabled_cfg=True)
    e_b = dict(base_axes)
    e_b.update(slice_stable_es_enabled_cfg=False)
    c_e_a = _build_combo(models=("cb",), axes=e_a, seed=0)
    c_e_b = _build_combo(models=("cb",), axes=e_b, seed=0)
    assert c_e_a.canonical_key() != c_e_b.canonical_key(), (
        "slice_stable_es_enabled_cfg=True must NOT canon-collapse to False "
        "(master toggle is always meaningful)"
    )

    # early_stop_on_worsening is unconditionally meaningful (no gate),
    # so flipping it must yield a distinct canonical_key.
    w_a = dict(base_axes)
    w_a.update(early_stop_on_worsening_cfg=True)
    w_b = dict(base_axes)
    w_b.update(early_stop_on_worsening_cfg=False)
    c_w_a = _build_combo(models=("cb",), axes=w_a, seed=0)
    c_w_b = _build_combo(models=("cb",), axes=w_b, seed=0)
    assert c_w_a.canonical_key() != c_w_b.canonical_key(), (
        "early_stop_on_worsening_cfg must differentiate combos (no canon gate)"
    )


def test_iter576_audit_pass_6_low_tier_axes_flow_to_kwargs():
    """28 audit-pass-6 LOW-tier (W6 LOW) deferred fuzz axes must:
      (a) be present in AXES with >=2 candidate values,
      (b) carry the SOURCE-verified library defaults in the FuzzCombo
          dataclass (verified pre-edit against
          src/mlframe/feature_selection/shap_proxied_fs.py:79-113 for
          ShapProxiedFS S1-S18, _model_configs.py:506-507 for
          early_stop_on_worsening_{coeff,min_iters},
          filters/mrmr.py:241,249,252,265 for Wave 8 S32/S34/S35/S37,
          _composite_target_discovery_config.py:127-130 for S41-S44),
      (c) collapse correctly under the documented secondary gates:
            - 18 shap_proxied_* axes collapse to source defaults when
              use_shap_proxied_fs=False (with refine_ucb sub-knobs also
              gated by within_cluster_refine; revalidation_ucb sub-knobs
              also gated by revalidate_cfg),
            - 2 early_stop_on_worsening_{coeff,min_iters} collapse when
              early_stop_on_worsening_cfg=False,
            - 4 mrmr_* LOW scalars collapse when use_mrmr_fs=False,
            - 4 cv_selector_* LOW scalars collapse when
              composite_discovery_enabled_cfg=False OR target != regression.

    S27 dropped: audit said `auto_wrap_partial_fit_es_force_off_cfg` is
    "Not a knob today"; grep confirmed no source ctor param exists, so
    the axis was not wired (29 -> 28 axes).

    Drift corrections caught pre-edit (audit-vs-source):
      - audit S15/S16/S17 (revalidation_ucb): defaults all None per
        source shap_proxied_fs.py:102-104 -- confirmed.
      - audit S13 (revalidation_n_estimators): default 100 per
        shap_proxied_fs.py:100 -- confirmed.
    """
    from tests.training._fuzz_combo import (
        AXES, _build_combo, build_mrmr_kwargs,
        build_shap_proxied_fs_kwargs,
        build_composite_discovery_config_from_flat,
    )

    new_axes = (
        # ShapProxiedFS Stage-A (S1-S8).
        "shap_proxied_prefilter_stage1_keep_cfg",
        "shap_proxied_prefilter_univariate_batch_size_cfg",
        "shap_proxied_shap_prefilter_enabled_cfg",
        "shap_proxied_shap_prefilter_safety_factor_cfg",
        "shap_proxied_shap_prefilter_min_features_cfg",
        "shap_proxied_shap_aware_stage1_keep_cfg",
        "shap_proxied_shap_aware_stage1_cushion_cfg",
        "shap_proxied_shap_aware_stage1_floor_cfg",
        # ShapProxiedFS Refine UCB (S9-S12).
        "shap_proxied_refine_ucb_enabled_cfg",
        "shap_proxied_refine_ucb_min_eval_size_cfg",
        "shap_proxied_refine_ucb_slack_cfg",
        "shap_proxied_refine_ucb_stdev_multiplier_cfg",
        # ShapProxiedFS Revalidation (S13-S17).
        "shap_proxied_revalidation_n_estimators_cfg",
        "shap_proxied_revalidation_ucb_enabled_cfg",
        "shap_proxied_revalidation_ucb_min_eval_size_cfg",
        "shap_proxied_revalidation_ucb_slack_cfg",
        "shap_proxied_revalidation_ucb_stdev_multiplier_cfg",
        # ShapProxiedFS Threading (S18).
        "shap_proxied_inner_n_jobs_cap_cfg",
        # Curve-shape ES (S25, S26).
        "early_stop_on_worsening_coeff_cfg",
        "early_stop_on_worsening_min_iters_cfg",
        # MRMR Wave 8 LOW (S32, S34, S35, S37).
        "mrmr_relaxmrmr_alpha_cfg",
        "mrmr_uaed_auto_size_cfg",
        "mrmr_cpt_test_cfg",
        "mrmr_pid_synergy_bonus_cfg",
        # CV-selector LOW (S41-S44).
        "cv_selector_alpha_cfg",
        "cv_selector_confidence_cfg",
        "cv_selector_quantile_level_cfg",
        "cv_persist_fold_scores_cfg",
    )

    # (a) Presence in AXES with >=2 candidates.
    assert len(new_axes) == 28, f"expected 28 W6 LOW axes, got {len(new_axes)}"
    for ax in new_axes:
        assert ax in AXES, f"missing fuzz axis {ax}"
        assert len(AXES[ax]) >= 2, f"axis {ax} must offer at least 2 values"

    # (b) FuzzCombo dataclass defaults match SOURCE defaults verified pre-edit.
    base_axes = {name: values[0] for name, values in AXES.items()}
    c_default = _build_combo(models=("cb",), axes=base_axes, seed=0)
    # ShapProxiedFS Stage-A: shap_proxied_fs.py:79-87.
    assert c_default.shap_proxied_prefilter_stage1_keep_cfg is None
    assert c_default.shap_proxied_prefilter_univariate_batch_size_cfg is None
    assert c_default.shap_proxied_shap_prefilter_enabled_cfg is True
    assert c_default.shap_proxied_shap_prefilter_safety_factor_cfg == 4
    assert c_default.shap_proxied_shap_prefilter_min_features_cfg == 40
    assert c_default.shap_proxied_shap_aware_stage1_keep_cfg is True
    assert c_default.shap_proxied_shap_aware_stage1_cushion_cfg == 8
    assert c_default.shap_proxied_shap_aware_stage1_floor_cfg == 200
    # ShapProxiedFS Refine UCB: shap_proxied_fs.py:96-99.
    assert c_default.shap_proxied_refine_ucb_enabled_cfg is True
    assert c_default.shap_proxied_refine_ucb_min_eval_size_cfg is None
    assert c_default.shap_proxied_refine_ucb_slack_cfg is None
    assert c_default.shap_proxied_refine_ucb_stdev_multiplier_cfg == 1.0
    # ShapProxiedFS Revalidation: shap_proxied_fs.py:100-104.
    assert c_default.shap_proxied_revalidation_n_estimators_cfg == 100
    assert c_default.shap_proxied_revalidation_ucb_enabled_cfg is True
    assert c_default.shap_proxied_revalidation_ucb_min_eval_size_cfg is None
    assert c_default.shap_proxied_revalidation_ucb_slack_cfg is None
    assert c_default.shap_proxied_revalidation_ucb_stdev_multiplier_cfg is None
    # ShapProxiedFS Threading: shap_proxied_fs.py:113.
    assert c_default.shap_proxied_inner_n_jobs_cap_cfg is False
    # Curve-shape ES: _model_configs.py:506-507.
    assert c_default.early_stop_on_worsening_coeff_cfg == 5
    assert c_default.early_stop_on_worsening_min_iters_cfg == 5
    # MRMR Wave 8 scalars: filters/mrmr.py:241,249,252,265.
    assert c_default.mrmr_relaxmrmr_alpha_cfg == 0.0
    assert c_default.mrmr_uaed_auto_size_cfg is False
    assert c_default.mrmr_cpt_test_cfg is False
    assert c_default.mrmr_pid_synergy_bonus_cfg == 0.0
    # CV-selector scalars: _composite_target_discovery_config.py:127-130.
    assert c_default.cv_selector_alpha_cfg == 1.0
    assert c_default.cv_selector_confidence_cfg == 0.9
    assert c_default.cv_selector_quantile_level_cfg == 0.9
    assert c_default.cv_persist_fold_scores_cfg is False

    # (c.1) All 18 shap_proxied_* axes collapse when use_shap_proxied_fs=False.
    sp_a = dict(base_axes)
    sp_a.update(use_shap_proxied_fs=False)
    sp_b = dict(sp_a)
    sp_b.update(
        shap_proxied_prefilter_stage1_keep_cfg=200,
        shap_proxied_prefilter_univariate_batch_size_cfg=256,
        shap_proxied_shap_prefilter_enabled_cfg=False,
        shap_proxied_shap_prefilter_safety_factor_cfg=8,
        shap_proxied_shap_prefilter_min_features_cfg=80,
        shap_proxied_shap_aware_stage1_keep_cfg=False,
        shap_proxied_shap_aware_stage1_cushion_cfg=4,
        shap_proxied_shap_aware_stage1_floor_cfg=500,
        shap_proxied_refine_ucb_enabled_cfg=False,
        shap_proxied_refine_ucb_min_eval_size_cfg=8,
        shap_proxied_refine_ucb_slack_cfg=0.0,
        shap_proxied_refine_ucb_stdev_multiplier_cfg=0.5,
        shap_proxied_revalidation_n_estimators_cfg=None,
        shap_proxied_revalidation_ucb_enabled_cfg=False,
        shap_proxied_revalidation_ucb_min_eval_size_cfg=3,
        shap_proxied_revalidation_ucb_slack_cfg=0.0,
        shap_proxied_revalidation_ucb_stdev_multiplier_cfg=1.0,
        shap_proxied_inner_n_jobs_cap_cfg=True,
    )
    c_sp_a = _build_combo(models=("cb",), axes=sp_a, seed=0)
    c_sp_b = _build_combo(models=("cb",), axes=sp_b, seed=0)
    assert c_sp_a.canonical_key() == c_sp_b.canonical_key(), (
        "all 18 shap_proxied_* LOW axes must collapse to ShapProxiedFS "
        "source defaults when use_shap_proxied_fs=False"
    )

    # (c.2) Curve-shape ES scalars collapse when detector is disabled.
    es_a = dict(base_axes)
    es_a.update(early_stop_on_worsening_cfg=False)
    es_b = dict(es_a)
    es_b.update(
        early_stop_on_worsening_coeff_cfg=7,
        early_stop_on_worsening_min_iters_cfg=10,
    )
    c_es_a = _build_combo(models=("cb",), axes=es_a, seed=0)
    c_es_b = _build_combo(models=("cb",), axes=es_b, seed=0)
    assert c_es_a.canonical_key() == c_es_b.canonical_key(), (
        "early_stop_on_worsening_{coeff,min_iters} must collapse to source "
        "defaults when early_stop_on_worsening_cfg=False"
    )

    # (c.3) MRMR LOW scalars collapse when use_mrmr_fs=False.
    mr_a = dict(base_axes)
    mr_a.update(use_mrmr_fs=False)
    mr_b = dict(mr_a)
    mr_b.update(
        mrmr_relaxmrmr_alpha_cfg=0.1,
        mrmr_uaed_auto_size_cfg=True,
        mrmr_cpt_test_cfg=True,
        mrmr_pid_synergy_bonus_cfg=0.1,
    )
    c_mr_a = _build_combo(models=("cb",), axes=mr_a, seed=0)
    c_mr_b = _build_combo(models=("cb",), axes=mr_b, seed=0)
    assert c_mr_a.canonical_key() == c_mr_b.canonical_key(), (
        "4 MRMR Wave 8 LOW scalars must collapse to MRMR defaults when "
        "use_mrmr_fs=False"
    )

    # (c.4) CV-selector LOW scalars collapse when discovery is off.
    cv_a = dict(base_axes)
    cv_a.update(composite_discovery_enabled_cfg=False)
    cv_b = dict(cv_a)
    cv_b.update(
        cv_selector_alpha_cfg=1.5,
        cv_selector_confidence_cfg=0.99,
        cv_selector_quantile_level_cfg=0.95,
        cv_persist_fold_scores_cfg=True,
    )
    c_cv_a = _build_combo(models=("cb",), axes=cv_a, seed=0)
    c_cv_b = _build_combo(models=("cb",), axes=cv_b, seed=0)
    assert c_cv_a.canonical_key() == c_cv_b.canonical_key(), (
        "4 CV-selector LOW scalars must collapse to discovery defaults when "
        "composite_discovery_enabled_cfg=False"
    )

    # (d.1) ShapProxiedFS LOW knobs flow through
    # build_shap_proxied_fs_kwargs into the dict consumed by
    # ShapProxiedFS.__init__. Names match the ctor verbatim
    # (feature_selection/shap_proxied_fs.py:79-113).
    on_axes = dict(base_axes)
    on_axes.update(
        use_shap_proxied_fs=True,
        shap_proxied_prefilter_stage1_keep_cfg=200,
        shap_proxied_prefilter_univariate_batch_size_cfg=256,
        shap_proxied_shap_prefilter_enabled_cfg=False,
        shap_proxied_shap_prefilter_safety_factor_cfg=8,
        shap_proxied_shap_prefilter_min_features_cfg=80,
        shap_proxied_shap_aware_stage1_keep_cfg=False,
        shap_proxied_shap_aware_stage1_cushion_cfg=4,
        shap_proxied_shap_aware_stage1_floor_cfg=500,
        shap_proxied_refine_ucb_enabled_cfg=False,
        shap_proxied_refine_ucb_min_eval_size_cfg=8,
        shap_proxied_refine_ucb_slack_cfg=0.0,
        shap_proxied_refine_ucb_stdev_multiplier_cfg=0.5,
        shap_proxied_revalidation_n_estimators_cfg=None,
        shap_proxied_revalidation_ucb_enabled_cfg=False,
        shap_proxied_revalidation_ucb_min_eval_size_cfg=3,
        shap_proxied_revalidation_ucb_slack_cfg=0.0,
        shap_proxied_revalidation_ucb_stdev_multiplier_cfg=1.0,
        shap_proxied_inner_n_jobs_cap_cfg=True,
    )
    c_on = _build_combo(models=("cb",), axes=on_axes, seed=0)
    kw_sp = build_shap_proxied_fs_kwargs(c_on)
    assert kw_sp is not None, (
        "build_shap_proxied_fs_kwargs must return dict when use_shap_proxied_fs=True"
    )
    assert kw_sp["prefilter_stage1_keep"] == 200
    assert kw_sp["prefilter_univariate_batch_size"] == 256
    assert kw_sp["shap_prefilter_enabled"] is False
    assert kw_sp["shap_prefilter_safety_factor"] == 8
    assert kw_sp["shap_prefilter_min_features"] == 80
    assert kw_sp["shap_aware_stage1_keep"] is False
    assert kw_sp["shap_aware_stage1_cushion"] == 4
    assert kw_sp["shap_aware_stage1_floor"] == 500
    assert kw_sp["refine_ucb_enabled"] is False
    assert kw_sp["refine_ucb_min_eval_size"] == 8
    assert kw_sp["refine_ucb_slack"] == 0.0
    assert kw_sp["refine_ucb_stdev_multiplier"] == 0.5
    assert kw_sp["revalidation_n_estimators"] is None
    assert kw_sp["revalidation_ucb_enabled"] is False
    assert kw_sp["revalidation_ucb_min_eval_size"] == 3
    assert kw_sp["revalidation_ucb_slack"] == 0.0
    assert kw_sp["revalidation_ucb_stdev_multiplier"] == 1.0
    assert kw_sp["inner_n_jobs_cap"] is True

    # (d.2) MRMR LOW knobs flow through build_mrmr_kwargs.
    mr_on_axes = dict(base_axes)
    mr_on_axes.update(
        use_mrmr_fs=True,
        mrmr_relaxmrmr_alpha_cfg=0.1,
        mrmr_uaed_auto_size_cfg=True,
        mrmr_cpt_test_cfg=True,
        mrmr_pid_synergy_bonus_cfg=0.1,
    )
    c_mr_on = _build_combo(models=("cb",), axes=mr_on_axes, seed=0)
    kw_mr = build_mrmr_kwargs(c_mr_on)
    assert kw_mr is not None
    assert kw_mr["relaxmrmr_alpha"] == 0.1
    assert kw_mr["uaed_auto_size"] is True
    assert kw_mr["cpt_test"] is True
    assert kw_mr["pid_synergy_bonus"] == 0.1

    # (d.3) CV-selector LOW knobs flow through
    # build_composite_discovery_config_from_flat into the
    # CompositeTargetDiscoveryConfig.
    cfg = build_composite_discovery_config_from_flat(
        enabled=True,
        cv_selector_alpha=1.5,
        cv_selector_confidence=0.99,
        cv_selector_quantile_level=0.95,
        cv_persist_fold_scores=True,
    )
    assert cfg is not None
    # Source field names verified at _composite_target_discovery_config.py:127-130.
    assert getattr(cfg, "cv_selector_alpha", None) == 1.5
    assert getattr(cfg, "cv_selector_confidence", None) == 0.99
    assert getattr(cfg, "cv_selector_quantile_level", None) == 0.95
    assert getattr(cfg, "cv_persist_fold_scores", None) is True


def test_iter582_close_remaining_axes_flow_to_kwargs():
    """Pin S27 ``auto_wrap_partial_fit_es_force_off_cfg`` + 5 ``slice_stable_es_*``
    axes flowing into their target config fields.

    Closes the two remaining items from the session summary:

      S27: ``TrainingBehaviorConfig.auto_wrap_partial_fit_es`` was added as a
        real ctor param (default True preserves prior behaviour) and the gate
        landed at ``_trainer_train_and_evaluate.py:551``. Fuzz axis is inverted
        (``auto_wrap_partial_fit_es_force_off_cfg=True`` -> ctor field False).

      slice_stable_es_*: 5 axes wired canon-only in commit 8d38bf20 are now
        threaded through ``build_slice_stable_es_config(combo)`` into a real
        ``SliceStableESConfig`` -- field-name mapping verified against
        ``_training_runtime_configs.py:42-95`` (no audit-vs-source drift).
    """
    from tests.training._fuzz_combo import (
        AXES, _build_combo, build_slice_stable_es_config,
    )

    # ----- S27 axis presence + dataclass default -----
    assert "auto_wrap_partial_fit_es_force_off_cfg" in AXES
    assert AXES["auto_wrap_partial_fit_es_force_off_cfg"] == (False, True)

    base_axes = {name: values[0] for name, values in AXES.items()}
    c_default = _build_combo(models=("linear",), axes=base_axes, seed=0)
    assert c_default.auto_wrap_partial_fit_es_force_off_cfg is False

    # S27 inversion: force_off=True -> auto_wrap_partial_fit_es=False
    # (mirrors the inline TrainingBehaviorConfig kwarg wiring at
    # test_fuzz_suite.py inside the behavior_kwargs dict).
    s27_axes = dict(base_axes)
    s27_axes.update(auto_wrap_partial_fit_es_force_off_cfg=True)
    c_force_off = _build_combo(models=("linear",), axes=s27_axes, seed=0)
    _auto_wrap = not c_force_off.auto_wrap_partial_fit_es_force_off_cfg
    assert _auto_wrap is False, (
        "force_off=True must invert into auto_wrap_partial_fit_es=False"
    )
    # And the default case: force_off=False -> auto_wrap=True (no behaviour change).
    _auto_wrap_default = not c_default.auto_wrap_partial_fit_es_force_off_cfg
    assert _auto_wrap_default is True

    # ----- TrainingBehaviorConfig field ordering / type -----
    from mlframe.training.configs import TrainingBehaviorConfig
    assert "auto_wrap_partial_fit_es" in TrainingBehaviorConfig.model_fields
    _fi = TrainingBehaviorConfig.model_fields["auto_wrap_partial_fit_es"]
    # default must preserve current behaviour
    assert _fi.default is True
    # ctor sanity: explicit False round-trips.
    _beh = TrainingBehaviorConfig(auto_wrap_partial_fit_es=False)
    assert _beh.auto_wrap_partial_fit_es is False

    # ----- 5 slice_stable_es_* axes flow into SliceStableESConfig -----
    sl_on = dict(base_axes)
    sl_on.update(
        slice_stable_es_enabled_cfg=True,
        slice_stable_es_aggregate_cfg="t_lcb",
        slice_stable_es_source_cfg="random",
        slice_stable_es_pareto_best_iter_selection_cfg=True,
        slice_stable_es_diagnostic_only_cfg=True,
    )
    c_on = _build_combo(models=("cb",), axes=sl_on, seed=0)
    cfg = build_slice_stable_es_config(c_on)
    assert cfg is not None
    # Field-name mapping (verified against SliceStableESConfig at
    # _training_runtime_configs.py:42-95 -- no audit drift; all 5 SOURCE
    # field names match the audit's audit-vs-source mapping).
    assert cfg.enabled is True
    assert cfg.aggregate == "t_lcb"
    assert cfg.source == "random"
    assert cfg.pareto_best_iter_selection is True
    assert cfg.diagnostic_only is True

    # Default-axes path: defaults flow through unchanged.
    cfg_default = build_slice_stable_es_config(c_default)
    assert cfg_default.enabled is False
    assert cfg_default.aggregate == "mean"
    assert cfg_default.source == "temporal"
    assert cfg_default.pareto_best_iter_selection is False
    assert cfg_default.diagnostic_only is False


# ---------------------------------------------------------------------------
# 2026-05-30 audit-pass-7 (4 axes): MRMR DCD-default flip + 3 perm/binning knobs.
# ---------------------------------------------------------------------------


def test_iter594_audit_pass_7_axes_flow_to_kwargs():
    """4 audit-pass-7 axes must:
      (a) FuzzCombo dataclass default for mrmr_dcd_enable_cfg mirrors the
          flipped source default (True; mrmr.py:596). Pre-flip the dataclass
          and the source disagreed -- bare ``MRMR()`` callers were taking the
          DCD branch the fuzz default never exercised.
      (b) Three NEW axes (baseline_npermutations / low_card_cap /
          collapsed_fallback_nbins) are present with >=2 values + their
          dataclass defaults match the source defaults verbatim.
      (c) When use_mrmr_fs=False every axis canon-collapses to the source
          default so the dedup pass cannot keep phantom MRMR-knob variation
          alive on non-MRMR combos.
      (d) When use_mrmr_fs=True the axes flow into build_mrmr_kwargs verbatim:
          baseline_npermutations as a top-level kwarg, low_card_cap +
          collapsed_fallback_nbins via the nbins_strategy_kwargs subdict
          (the kwargs path the MRMR ctor forwards into _mrmr_fit_impl ->
          categorize_dataset -> per_feature_edges).

    Axes (all source-verified pre-edit):
      #1 mrmr_dcd_enable_cfg (mrmr.py:596 default True; was False)
      #2 mrmr_baseline_npermutations_cfg (mrmr.py:309 default 2)
      #3 mrmr_low_card_cap_cfg (_adaptive_nbins.py:511 default 32)
      #4 mrmr_collapsed_fallback_nbins_cfg (_adaptive_nbins.py:586 default 5)
    """
    from tests.training._fuzz_combo import (
        AXES, FuzzCombo, _build_combo, enumerate_combos, build_mrmr_kwargs,
    )

    # (a) #1: dataclass FIELD default mirrors the flipped source default.
    # We check the dataclass default directly (NOT via _build_combo + base_axes,
    # because base_axes selects AXES[ax][0] which is False for the dcd axis --
    # the dataclass default is what bare ``FuzzCombo()`` callers get when no
    # axis dict is supplied).
    assert FuzzCombo.__dataclass_fields__["mrmr_dcd_enable_cfg"].default is True, (
        "audit-pass-7 #1: dataclass default must mirror mrmr.py:596 (True)"
    )
    base_axes = {name: values[0] for name, values in AXES.items()}

    # (b) Presence + dataclass defaults for the 3 new axes.
    for ax, expected_first in (
        ("mrmr_baseline_npermutations_cfg", 2),
        ("mrmr_low_card_cap_cfg", 2),  # AXES pair is (2, 32) so AXES[ax][0] == 2
        ("mrmr_collapsed_fallback_nbins_cfg", 3),  # AXES pair (3, 10)
    ):
        assert ax in AXES, f"missing fuzz axis {ax}"
        assert len(AXES[ax]) >= 2, f"axis {ax} must offer at least 2 values"
        # Sanity: the AXES pair's first element matches the value the base
        # canonical-axes dict produces (base_axes[ax] is AXES[ax][0]).
        assert base_axes[ax] == expected_first, (
            f"base_axes[{ax}] = {base_axes[ax]!r}, expected {expected_first!r}"
        )

    # Dataclass defaults match SOURCE defaults, not the first AXES element.
    # AXES first element happens to differ from source default for #3 and #4
    # (the audit's "stress" value); the dataclass default mirrors mrmr.py /
    # _adaptive_nbins.py so a default-fuzz combo runs the source-default
    # algorithm path.
    assert FuzzCombo.__dataclass_fields__["mrmr_baseline_npermutations_cfg"].default == 2
    assert FuzzCombo.__dataclass_fields__["mrmr_low_card_cap_cfg"].default == 32
    assert FuzzCombo.__dataclass_fields__["mrmr_collapsed_fallback_nbins_cfg"].default == 5

    # (c) enumerate_combos still hits 150 with 3 new axes wired.
    combos = enumerate_combos(target=150, master_seed=20260422)
    assert len(combos) == 150, f"enumerate_combos lost combos: {len(combos)}"

    # (c-bis) #1 canon-collapse: when use_mrmr_fs=False, dcd_enable axis
    # collapses to the (now True) default regardless of the stored value.
    # The audit-pass-7 #1 fallback flip ensures legacy/no-MRMR combos share a
    # canonical key with the post-flip baseline.
    off_dcd_a = dict(base_axes); off_dcd_a.update(use_mrmr_fs=False, mrmr_dcd_enable_cfg=False)
    off_dcd_b = dict(off_dcd_a); off_dcd_b["mrmr_dcd_enable_cfg"] = True
    c_off_dcd_a = _build_combo(models=("cb",), axes=off_dcd_a, seed=0)
    c_off_dcd_b = _build_combo(models=("cb",), axes=off_dcd_b, seed=0)
    assert c_off_dcd_a.canonical_key() == c_off_dcd_b.canonical_key(), (
        "audit-pass-7 #1: mrmr_dcd_enable_cfg must canon-collapse when use_mrmr_fs=False"
    )

    # (c) Per-axis canon-collapse for #2/#3/#4 under use_mrmr_fs=False.
    off_axes_a = dict(base_axes)
    off_axes_a.update(
        use_mrmr_fs=False,
        mrmr_baseline_npermutations_cfg=8,
        mrmr_low_card_cap_cfg=32,
        mrmr_collapsed_fallback_nbins_cfg=10,
    )
    off_axes_b = dict(off_axes_a)
    off_axes_b.update(
        mrmr_baseline_npermutations_cfg=2,
        mrmr_low_card_cap_cfg=2,  # AXES first element; would matter only under use_mrmr_fs=True
        mrmr_collapsed_fallback_nbins_cfg=3,
    )
    c_off_a = _build_combo(models=("cb",), axes=off_axes_a, seed=0)
    c_off_b = _build_combo(models=("cb",), axes=off_axes_b, seed=0)
    assert build_mrmr_kwargs(c_off_a) is None, (
        "use_mrmr_fs=False must yield None mrmr_kwargs"
    )
    assert c_off_a.canonical_key() == c_off_b.canonical_key(), (
        "audit-pass-7 #2/#3/#4: MRMR knob axes must canon-collapse when use_mrmr_fs=False"
    )

    # (c-quad) #4 has a COMPOUND gate: collapsed_fallback_nbins only fires
    # when nbins_strategy is one of the supervised methods that can actually
    # collapse a column. With nbins_strategy='quantile' (unsupervised) the
    # fallback is never reached even with use_mrmr_fs=True, so the axis must
    # canon-collapse there too.
    quantile_a = dict(base_axes)
    quantile_a.update(
        use_mrmr_fs=True,
        mrmr_nbins_strategy_cfg="quantile",
        mrmr_collapsed_fallback_nbins_cfg=3,
    )
    quantile_b = dict(quantile_a)
    quantile_b["mrmr_collapsed_fallback_nbins_cfg"] = 10
    c_q_a = _build_combo(models=("cb",), axes=quantile_a, seed=0)
    c_q_b = _build_combo(models=("cb",), axes=quantile_b, seed=0)
    assert c_q_a.canonical_key() == c_q_b.canonical_key(), (
        "audit-pass-7 #4: collapsed_fallback_nbins must canon-collapse when "
        "nbins_strategy is not one of {mdlp, fayyad_irani}"
    )

    # (d) Threading: when use_mrmr_fs=True the axes flow into build_mrmr_kwargs.
    on_axes = dict(base_axes)
    on_axes.update(
        use_mrmr_fs=True,
        mrmr_nbins_strategy_cfg="mdlp",  # keep #4 reachable through the gate
        mrmr_baseline_npermutations_cfg=8,
        mrmr_low_card_cap_cfg=2,
        mrmr_collapsed_fallback_nbins_cfg=10,
    )
    c_on = _build_combo(models=("cb",), axes=on_axes, seed=0)
    kw = build_mrmr_kwargs(c_on)
    assert kw is not None
    # #2 top-level kwarg (name matches MRMR.__init__ exactly).
    assert kw["baseline_npermutations"] == 8, (
        f"#2 baseline_npermutations did not thread: {kw.get('baseline_npermutations')!r}"
    )
    # #3 + #4 ride in the nbins_strategy_kwargs subdict forwarded to
    # per_feature_edges via categorize_dataset (mrmr.py:225 -> _mrmr_fit_impl:341).
    sub = kw.get("nbins_strategy_kwargs")
    assert sub is not None, "nbins_strategy_kwargs subdict missing"
    assert sub.get("low_card_cap") == 2, (
        f"#3 low_card_cap did not thread into nbins_strategy_kwargs: {sub!r}"
    )
    assert sub.get("collapsed_fallback_nbins") == 10, (
        f"#4 collapsed_fallback_nbins did not thread into nbins_strategy_kwargs: {sub!r}"
    )

    # When axes #3 + #4 are at source defaults, the nbins_strategy_kwargs
    # subdict should NOT be injected (so existing caller-supplied dicts are
    # not silently shadowed with empty overrides). Reset #2 to the source
    # default too so we can also pin its threading at the default value.
    on_defaults = dict(on_axes)
    on_defaults.update(
        mrmr_baseline_npermutations_cfg=2,
        mrmr_low_card_cap_cfg=32,
        mrmr_collapsed_fallback_nbins_cfg=5,
    )
    c_on_def = _build_combo(models=("cb",), axes=on_defaults, seed=0)
    kw_def = build_mrmr_kwargs(c_on_def)
    assert kw_def is not None
    assert "nbins_strategy_kwargs" not in kw_def, (
        "nbins_strategy_kwargs must NOT be injected when #3 + #4 are at source defaults; "
        f"got {kw_def.get('nbins_strategy_kwargs')!r}"
    )
    # #2 still threads even at the source default (always a real kwarg).
    assert kw_def["baseline_npermutations"] == 2

    # (d-bis) Distinct canonical_keys when the non-default values are set
    # under use_mrmr_fs=True (so the pairwise sampler reaches both branches).
    assert c_on.canonical_key() != c_on_def.canonical_key(), (
        "audit-pass-7 #2/#3/#4: non-default values must produce distinct canonical "
        "keys under use_mrmr_fs=True so the dedup keeps both branches reachable"
    )


def test_iter606_audit_pass_8_axes_flow_to_kwargs():
    """4 audit-pass-8 HIGH axes must:
      (a) Dataclass defaults mirror HEAD source verbatim:
            #1 mrmr_cardinality_bias_correction_cfg = True
               (filters/mrmr.py:334; audit-cited line matches HEAD)
            #2 mrmr_min_relevance_gain_relative_to_first_cfg = 0.05
               (filters/mrmr.py:326; audit-cited line matches HEAD)
            #3 mlp_random_state_cfg = None
               (training/neural/base.py:217; audit-cited line matches HEAD)
            #4 mlp_class_weight_cfg = None
               (training/neural/base.py:218; audit-cited line matches HEAD)
      (b) Each axis pair is present in AXES with the values listed in the audit.
      (c) Canon-collapse rules hold:
            #1/#2 collapse to source default under use_mrmr_fs=False.
            #3 collapses to None when 'mlp' NOT in models AND recurrent_model_cfg is None.
            #4 collapses to None outside the compound gate
               (mlp AND binary/multiclass AND rare_5pct/rare_1pct imbalance).
      (d) Threading produces the expected kwargs subdicts:
            #1/#2 flow via build_mrmr_kwargs as top-level MRMR ctor kwargs.
            #3/#4 flow via build_mlp_kwargs into the MLP estimator kwargs dict.
    """
    from tests.training._fuzz_combo import (
        AXES, FuzzCombo, _build_combo, enumerate_combos,
        build_mrmr_kwargs, build_mlp_kwargs,
    )

    # (a) Dataclass defaults mirror HEAD source.
    fields = FuzzCombo.__dataclass_fields__
    assert fields["mrmr_cardinality_bias_correction_cfg"].default is True, (
        "audit-pass-8 #1: default must mirror filters/mrmr.py:334 (True)"
    )
    assert fields["mrmr_min_relevance_gain_relative_to_first_cfg"].default == 0.05, (
        "audit-pass-8 #2: default must mirror filters/mrmr.py:326 (0.05)"
    )
    assert fields["mlp_random_state_cfg"].default is None, (
        "audit-pass-8 #3: default must mirror training/neural/base.py:217 (None)"
    )
    assert fields["mlp_class_weight_cfg"].default is None, (
        "audit-pass-8 #4: default must mirror training/neural/base.py:218 (None)"
    )

    # (b) AXES presence with the audit-listed pairs.
    assert AXES["mrmr_cardinality_bias_correction_cfg"] == (True, False)
    assert AXES["mrmr_min_relevance_gain_relative_to_first_cfg"] == (0.05, 0.0)
    assert AXES["mlp_random_state_cfg"] == (None, 42)
    assert AXES["mlp_class_weight_cfg"] == (None, "balanced")

    base_axes = {name: values[0] for name, values in AXES.items()}

    # (c) enumerate_combos still hits 150 with 4 new axes wired.
    combos = enumerate_combos(target=150, master_seed=20260601)
    assert len(combos) == 150, f"enumerate_combos lost combos: {len(combos)}"

    # (c-i) #1 canon-collapse: when use_mrmr_fs=False every value collapses
    # to the source default True.
    off1_a = dict(base_axes); off1_a.update(
        use_mrmr_fs=False, mrmr_cardinality_bias_correction_cfg=False,
    )
    off1_b = dict(off1_a); off1_b["mrmr_cardinality_bias_correction_cfg"] = True
    c_off1_a = _build_combo(models=("cb",), axes=off1_a, seed=0)
    c_off1_b = _build_combo(models=("cb",), axes=off1_b, seed=0)
    assert c_off1_a.canonical_key() == c_off1_b.canonical_key(), (
        "audit-pass-8 #1: must canon-collapse under use_mrmr_fs=False"
    )

    # (c-ii) #2 canon-collapse: when use_mrmr_fs=False, values collapse
    # to source default 0.05.
    off2_a = dict(base_axes); off2_a.update(
        use_mrmr_fs=False, mrmr_min_relevance_gain_relative_to_first_cfg=0.0,
    )
    off2_b = dict(off2_a); off2_b["mrmr_min_relevance_gain_relative_to_first_cfg"] = 0.05
    c_off2_a = _build_combo(models=("cb",), axes=off2_a, seed=0)
    c_off2_b = _build_combo(models=("cb",), axes=off2_b, seed=0)
    assert c_off2_a.canonical_key() == c_off2_b.canonical_key(), (
        "audit-pass-8 #2: must canon-collapse under use_mrmr_fs=False"
    )

    # (c-iii) #3 canon-collapse: with no MLP and no recurrent, every value
    # collapses to None. Use a non-mlp model + recurrent_model_cfg=None.
    off3_a = dict(base_axes); off3_a.update(
        recurrent_model_cfg=None, mlp_random_state_cfg=42,
    )
    off3_b = dict(off3_a); off3_b["mlp_random_state_cfg"] = None
    c_off3_a = _build_combo(models=("cb",), axes=off3_a, seed=0)
    c_off3_b = _build_combo(models=("cb",), axes=off3_b, seed=0)
    assert c_off3_a.canonical_key() == c_off3_b.canonical_key(), (
        "audit-pass-8 #3: must canon-collapse when 'mlp' not in models AND "
        "recurrent_model_cfg is None"
    )

    # (c-iv) #4 canon-collapse: outside the compound gate every value
    # collapses to None. Test (a) no MLP, (b) MLP but regression target,
    # (c) MLP + binary but balanced imbalance.
    for ax_override in (
        # (a) no MLP -> compound gate fails on models check.
        {"target_type": "binary_classification", "imbalance_ratio": "rare_5pct",
         "recurrent_model_cfg": None},
        # (c) MLP + binary but balanced -> compound gate fails on imbalance.
        # (kept models default to base_axes, which puts cb only; we test
        #  the "balanced" leg by overriding _build_combo's models below).
    ):
        off4_a = dict(base_axes); off4_a.update(ax_override)
        off4_a["mlp_class_weight_cfg"] = "balanced"
        off4_b = dict(off4_a); off4_b["mlp_class_weight_cfg"] = None
        c_off4_a = _build_combo(models=("cb",), axes=off4_a, seed=0)
        c_off4_b = _build_combo(models=("cb",), axes=off4_b, seed=0)
        assert c_off4_a.canonical_key() == c_off4_b.canonical_key(), (
            f"audit-pass-8 #4: must canon-collapse outside the compound gate "
            f"with override {ax_override!r}"
        )

    # (c-iv-bis) MLP + binary + balanced -> still outside the gate, collapses.
    off4_bal_a = dict(base_axes); off4_bal_a.update(
        target_type="binary_classification",
        imbalance_ratio="balanced",
        recurrent_model_cfg=None,
        mlp_class_weight_cfg="balanced",
    )
    off4_bal_b = dict(off4_bal_a); off4_bal_b["mlp_class_weight_cfg"] = None
    c_off4_bal_a = _build_combo(models=("mlp",), axes=off4_bal_a, seed=0)
    c_off4_bal_b = _build_combo(models=("mlp",), axes=off4_bal_b, seed=0)
    assert c_off4_bal_a.canonical_key() == c_off4_bal_b.canonical_key(), (
        "audit-pass-8 #4: balanced imbalance must canon-collapse class_weight"
    )

    # (d) Threading: #1/#2 flow into build_mrmr_kwargs when use_mrmr_fs=True.
    on_mrmr = dict(base_axes)
    on_mrmr.update(
        use_mrmr_fs=True,
        mrmr_cardinality_bias_correction_cfg=False,
        mrmr_min_relevance_gain_relative_to_first_cfg=0.0,
    )
    c_on_mrmr = _build_combo(models=("cb",), axes=on_mrmr, seed=0)
    kw_mrmr = build_mrmr_kwargs(c_on_mrmr)
    assert kw_mrmr is not None
    assert kw_mrmr["cardinality_bias_correction"] is False, (
        f"#1 did not thread into mrmr_kwargs: {kw_mrmr.get('cardinality_bias_correction')!r}"
    )
    assert kw_mrmr["min_relevance_gain_relative_to_first"] == 0.0, (
        f"#2 did not thread into mrmr_kwargs: "
        f"{kw_mrmr.get('min_relevance_gain_relative_to_first')!r}"
    )

    # MRMR off -> kwargs is None (no threading).
    off_mrmr = dict(base_axes); off_mrmr.update(
        use_mrmr_fs=False,
        mrmr_cardinality_bias_correction_cfg=False,
        mrmr_min_relevance_gain_relative_to_first_cfg=0.0,
    )
    c_off_mrmr = _build_combo(models=("cb",), axes=off_mrmr, seed=0)
    assert build_mrmr_kwargs(c_off_mrmr) is None

    # (d-bis) #3/#4 flow into build_mlp_kwargs when MLP+rare classification.
    on_mlp = dict(base_axes)
    on_mlp.update(
        target_type="binary_classification",
        imbalance_ratio="rare_5pct",
        recurrent_model_cfg=None,
        mlp_random_state_cfg=42,
        mlp_class_weight_cfg="balanced",
    )
    c_on_mlp = _build_combo(models=("mlp",), axes=on_mlp, seed=0)
    kw_mlp = build_mlp_kwargs(c_on_mlp)
    assert kw_mlp is not None
    assert kw_mlp.get("random_state") == 42, (
        f"#3 did not thread into mlp_kwargs: {kw_mlp.get('random_state')!r}"
    )
    assert kw_mlp.get("class_weight") == "balanced", (
        f"#4 did not thread into mlp_kwargs: {kw_mlp.get('class_weight')!r}"
    )

    # No MLP / no recurrent -> builder returns None.
    no_neural = dict(base_axes); no_neural.update(
        recurrent_model_cfg=None,
        mlp_random_state_cfg=42,
        mlp_class_weight_cfg="balanced",
    )
    c_no_neural = _build_combo(models=("cb",), axes=no_neural, seed=0)
    assert build_mlp_kwargs(c_no_neural) is None, (
        "build_mlp_kwargs must return None when neither MLP nor recurrent fire"
    )

    # MLP but regression -> class_weight is dropped (#4 gate fails), but
    # random_state still threads (#3 gate only requires MLP-or-recurrent).
    mlp_reg = dict(base_axes); mlp_reg.update(
        target_type="regression",
        imbalance_ratio="balanced",
        recurrent_model_cfg=None,
        mlp_random_state_cfg=42,
        mlp_class_weight_cfg="balanced",
    )
    c_mlp_reg = _build_combo(models=("mlp",), axes=mlp_reg, seed=0)
    kw_mlp_reg = build_mlp_kwargs(c_mlp_reg)
    assert kw_mlp_reg is not None
    assert kw_mlp_reg.get("random_state") == 42
    assert "class_weight" not in kw_mlp_reg, (
        "audit-pass-8 #4: class_weight must NOT thread on regression target"
    )

    # Distinct canonical_keys: on-MRMR + non-default #1/#2 vs source-defaults.
    on_def = dict(on_mrmr)
    on_def.update(
        mrmr_cardinality_bias_correction_cfg=True,
        mrmr_min_relevance_gain_relative_to_first_cfg=0.05,
    )
    c_on_def = _build_combo(models=("cb",), axes=on_def, seed=0)
    assert c_on_mrmr.canonical_key() != c_on_def.canonical_key(), (
        "audit-pass-8 #1/#2: non-default values must produce distinct canonical "
        "keys under use_mrmr_fs=True so dedup keeps both branches reachable"
    )

    # Distinct canonical_keys for #3/#4 under the compound gate.
    on_mlp_def = dict(on_mlp)
    on_mlp_def.update(mlp_random_state_cfg=None, mlp_class_weight_cfg=None)
    c_on_mlp_def = _build_combo(models=("mlp",), axes=on_mlp_def, seed=0)
    assert c_on_mlp.canonical_key() != c_on_mlp_def.canonical_key(), (
        "audit-pass-8 #3/#4: non-default values must produce distinct canonical "
        "keys when 'mlp' in models AND rare classification holds"
    )


def test_iter613_audit_pass_8_med_axes_flow_to_kwargs():
    """5 audit-pass-8 MED + LOW->MED axes (#5/#7/#8/#9/#10) must:
      (a) Dataclass defaults mirror HEAD source verbatim:
            #5 shap_proxied_adaptive_prescreen_by_stability_cfg = False
               (feature_selection/shap_proxied_fs.py:208; verified)
            #7 mlp_use_layernorm_cfg = False
               (training/neural/flat.py:205; audit-cited line 145 was a
                docstring -- the real signature default lives at :205, drift
                logged in dataclass field comment)
            #8 mlp_l1_alpha_cfg = 0.0
               (library default at MLPTorchModel hparams; the BN/LN/GN-
                excluded L1 branch at _flat_torch_module.py:272-301 only
                fires when l1_alpha > 0)
            #9 mlp_inject_zero_sample_weight_batch_cfg = False
               (regression-prevention gate at _flat_torch_module.py:233-256)
            #10 inject_xor_synergy_pair_cfg = False
                (fleuret-mode conditional-MI gate at
                 feature_selection/filters/evaluation.py:596)
      (b) Each axis pair is present in AXES with the values listed in the
          audit.
      (c) Canon-collapse rules hold:
            #5 collapses to False under use_shap_proxied_fs=False.
            #7 collapses to False outside ('mlp' AND regression).
            #8 collapses to 0.0 outside 'mlp' in models.
            #9 collapses to False outside ('mlp' AND weight_schemas !=
               ("uniform",)).
            #10 collapses to False outside (use_mrmr_fs AND
                interactions_max_order >= 2).
      (d) Threading produces the expected kwargs subdicts:
            #5 flows via build_shap_proxied_fs_kwargs as the
               ``adaptive_prescreen_by_stability`` ctor key.
            #7/#8 flow via build_mlp_kwargs as ``use_layernorm`` /
                  ``l1_alpha`` hparams.
            #9/#10 flow via build_frame_for_combo (assert builder output
                   contains the injected columns / stale-ts block).
    """
    from tests.training._fuzz_combo import (
        AXES, FuzzCombo, _build_combo, enumerate_combos,
        build_shap_proxied_fs_kwargs, build_mlp_kwargs,
        build_frame_for_combo,
    )

    # (a) Dataclass defaults mirror HEAD source.
    fields = FuzzCombo.__dataclass_fields__
    assert fields["shap_proxied_adaptive_prescreen_by_stability_cfg"].default is False, (
        "audit-pass-8 #5: default must mirror "
        "feature_selection/shap_proxied_fs.py:208 (False)"
    )
    assert fields["mlp_use_layernorm_cfg"].default is False, (
        "audit-pass-8 #7: default must mirror training/neural/flat.py:205 (False)"
    )
    assert fields["mlp_l1_alpha_cfg"].default == 0.0, (
        "audit-pass-8 #8: default must mirror library default 0.0"
    )
    assert fields["mlp_inject_zero_sample_weight_batch_cfg"].default is False, (
        "audit-pass-8 #9: default must mirror False (no injection)"
    )
    assert fields["inject_xor_synergy_pair_cfg"].default is False, (
        "audit-pass-8 #10: default must mirror False (no injection)"
    )

    # (b) AXES presence with the audit-listed pairs.
    assert AXES["shap_proxied_adaptive_prescreen_by_stability_cfg"] == (False, True)
    assert AXES["mlp_use_layernorm_cfg"] == (False, True)
    assert AXES["mlp_l1_alpha_cfg"] == (0.0, 0.001)
    assert AXES["mlp_inject_zero_sample_weight_batch_cfg"] == (False, True)
    assert AXES["inject_xor_synergy_pair_cfg"] == (False, True)

    base_axes = {name: values[0] for name, values in AXES.items()}

    # enumerate_combos still hits 150 with 5 new axes wired.
    combos = enumerate_combos(target=150, master_seed=20260601)
    assert len(combos) == 150, f"enumerate_combos lost combos: {len(combos)}"

    # (c-i) #5 canon-collapse: when use_shap_proxied_fs=False, both values
    # collapse to source default False.
    off5_a = dict(base_axes); off5_a.update(
        use_shap_proxied_fs=False,
        shap_proxied_adaptive_prescreen_by_stability_cfg=True,
    )
    off5_b = dict(off5_a); off5_b["shap_proxied_adaptive_prescreen_by_stability_cfg"] = False
    c_off5_a = _build_combo(models=("cb",), axes=off5_a, seed=0)
    c_off5_b = _build_combo(models=("cb",), axes=off5_b, seed=0)
    assert c_off5_a.canonical_key() == c_off5_b.canonical_key(), (
        "audit-pass-8 #5: must canon-collapse under use_shap_proxied_fs=False"
    )

    # (c-ii) #7 canon-collapse: classification target or no MLP -> collapses.
    off7_a = dict(base_axes); off7_a.update(
        target_type="binary_classification",
        mlp_use_layernorm_cfg=True,
    )
    off7_b = dict(off7_a); off7_b["mlp_use_layernorm_cfg"] = False
    c_off7_a = _build_combo(models=("mlp",), axes=off7_a, seed=0)
    c_off7_b = _build_combo(models=("mlp",), axes=off7_b, seed=0)
    assert c_off7_a.canonical_key() == c_off7_b.canonical_key(), (
        "audit-pass-8 #7: must canon-collapse under classification target "
        "even when mlp is in models"
    )
    # And no MLP at all -> collapses regardless of target.
    off7_c = dict(base_axes); off7_c.update(
        target_type="regression",
        mlp_use_layernorm_cfg=True,
    )
    off7_d = dict(off7_c); off7_d["mlp_use_layernorm_cfg"] = False
    c_off7_c = _build_combo(models=("cb",), axes=off7_c, seed=0)
    c_off7_d = _build_combo(models=("cb",), axes=off7_d, seed=0)
    assert c_off7_c.canonical_key() == c_off7_d.canonical_key(), (
        "audit-pass-8 #7: must canon-collapse when MLP not in models"
    )

    # (c-iii) #8 canon-collapse: no MLP -> collapses to 0.0.
    off8_a = dict(base_axes); off8_a["mlp_l1_alpha_cfg"] = 0.001
    off8_b = dict(off8_a); off8_b["mlp_l1_alpha_cfg"] = 0.0
    c_off8_a = _build_combo(models=("cb",), axes=off8_a, seed=0)
    c_off8_b = _build_combo(models=("cb",), axes=off8_b, seed=0)
    assert c_off8_a.canonical_key() == c_off8_b.canonical_key(), (
        "audit-pass-8 #8: must canon-collapse when 'mlp' not in models"
    )

    # (c-iv) #9 canon-collapse: weight_schemas=("uniform",) or no MLP -> False.
    off9_a = dict(base_axes); off9_a.update(
        weight_schemas=("uniform",),
        mlp_inject_zero_sample_weight_batch_cfg=True,
    )
    off9_b = dict(off9_a); off9_b["mlp_inject_zero_sample_weight_batch_cfg"] = False
    c_off9_a = _build_combo(models=("mlp",), axes=off9_a, seed=0)
    c_off9_b = _build_combo(models=("mlp",), axes=off9_b, seed=0)
    assert c_off9_a.canonical_key() == c_off9_b.canonical_key(), (
        "audit-pass-8 #9: must canon-collapse under weight_schemas=('uniform',)"
    )
    # No MLP -> collapses regardless of weight_schemas.
    off9_c = dict(base_axes); off9_c.update(
        weight_schemas=("recency",),
        mlp_inject_zero_sample_weight_batch_cfg=True,
    )
    off9_d = dict(off9_c); off9_d["mlp_inject_zero_sample_weight_batch_cfg"] = False
    c_off9_c = _build_combo(models=("cb",), axes=off9_c, seed=0)
    c_off9_d = _build_combo(models=("cb",), axes=off9_d, seed=0)
    assert c_off9_c.canonical_key() == c_off9_d.canonical_key(), (
        "audit-pass-8 #9: must canon-collapse when 'mlp' not in models"
    )

    # (c-v) #10 canon-collapse: use_mrmr_fs=False -> False; or interactions
    # order < 2 -> False.
    off10_a = dict(base_axes); off10_a.update(
        use_mrmr_fs=False,
        mrmr_interactions_max_order_cfg=2,
        inject_xor_synergy_pair_cfg=True,
    )
    off10_b = dict(off10_a); off10_b["inject_xor_synergy_pair_cfg"] = False
    c_off10_a = _build_combo(models=("cb",), axes=off10_a, seed=0)
    c_off10_b = _build_combo(models=("cb",), axes=off10_b, seed=0)
    assert c_off10_a.canonical_key() == c_off10_b.canonical_key(), (
        "audit-pass-8 #10: must canon-collapse under use_mrmr_fs=False"
    )
    off10_c = dict(base_axes); off10_c.update(
        use_mrmr_fs=True,
        mrmr_interactions_max_order_cfg=1,
        inject_xor_synergy_pair_cfg=True,
    )
    off10_d = dict(off10_c); off10_d["inject_xor_synergy_pair_cfg"] = False
    c_off10_c = _build_combo(models=("cb",), axes=off10_c, seed=0)
    c_off10_d = _build_combo(models=("cb",), axes=off10_d, seed=0)
    assert c_off10_c.canonical_key() == c_off10_d.canonical_key(), (
        "audit-pass-8 #10: must canon-collapse under interactions_max_order < 2"
    )

    # (d-i) #5 threading: when use_shap_proxied_fs=True, builder emits the
    # adaptive_prescreen_by_stability key with the axis value.
    on5 = dict(base_axes); on5.update(
        use_shap_proxied_fs=True,
        shap_proxied_adaptive_prescreen_by_stability_cfg=True,
    )
    c_on5 = _build_combo(models=("cb",), axes=on5, seed=0)
    kw_shap = build_shap_proxied_fs_kwargs(c_on5)
    assert kw_shap is not None
    assert kw_shap["adaptive_prescreen_by_stability"] is True, (
        f"#5 did not thread into shap_proxied_fs_kwargs: "
        f"{kw_shap.get('adaptive_prescreen_by_stability')!r}"
    )
    # ShapProxiedFS off -> kwargs None.
    off5_full = dict(base_axes); off5_full.update(
        use_shap_proxied_fs=False,
        shap_proxied_adaptive_prescreen_by_stability_cfg=True,
    )
    assert build_shap_proxied_fs_kwargs(
        _build_combo(models=("cb",), axes=off5_full, seed=0)
    ) is None

    # (d-ii) #7/#8 threading: MLP + regression -> use_layernorm flows;
    # MLP active -> l1_alpha flows.
    on78 = dict(base_axes); on78.update(
        target_type="regression",
        imbalance_ratio="balanced",
        recurrent_model_cfg=None,
        mlp_use_layernorm_cfg=True,
        mlp_l1_alpha_cfg=0.001,
    )
    c_on78 = _build_combo(models=("mlp",), axes=on78, seed=0)
    kw_mlp = build_mlp_kwargs(c_on78)
    assert kw_mlp is not None
    assert kw_mlp.get("use_layernorm") is True, (
        f"#7 did not thread into mlp_kwargs: {kw_mlp.get('use_layernorm')!r}"
    )
    assert kw_mlp.get("l1_alpha") == 0.001, (
        f"#8 did not thread into mlp_kwargs: {kw_mlp.get('l1_alpha')!r}"
    )

    # MLP + classification -> use_layernorm dropped (#7 gate fails),
    # l1_alpha still threads (#8 gate only needs MLP in models).
    on78_cls = dict(base_axes); on78_cls.update(
        target_type="binary_classification",
        imbalance_ratio="balanced",
        recurrent_model_cfg=None,
        mlp_use_layernorm_cfg=True,
        mlp_l1_alpha_cfg=0.001,
    )
    c_on78_cls = _build_combo(models=("mlp",), axes=on78_cls, seed=0)
    kw_mlp_cls = build_mlp_kwargs(c_on78_cls)
    assert kw_mlp_cls is not None
    assert "use_layernorm" not in kw_mlp_cls, (
        "audit-pass-8 #7: use_layernorm must NOT thread on classification target"
    )
    assert kw_mlp_cls.get("l1_alpha") == 0.001, (
        f"#8 must still thread under classification: "
        f"{kw_mlp_cls.get('l1_alpha')!r}"
    )

    # No MLP + no recurrent -> builder returns None entirely (neither
    # threads).
    no_neural = dict(base_axes); no_neural.update(
        recurrent_model_cfg=None,
        mlp_use_layernorm_cfg=True,
        mlp_l1_alpha_cfg=0.001,
    )
    c_no_neural = _build_combo(models=("cb",), axes=no_neural, seed=0)
    assert build_mlp_kwargs(c_no_neural) is None, (
        "build_mlp_kwargs must return None when neither MLP nor recurrent fire"
    )

    # (d-iii) #10 threading: frame builder emits num_xor_a / num_xor_b
    # when the axis is on (small-n combo so the test is cheap).
    on10 = dict(base_axes); on10.update(
        n_rows=200,
        input_type="pandas",
        target_type="binary_classification",
        imbalance_ratio="balanced",
        recurrent_model_cfg=None,
        use_mrmr_fs=True,
        mrmr_interactions_max_order_cfg=2,
        inject_xor_synergy_pair_cfg=True,
        # Pin axes that would otherwise inject pathological / heavy data.
        cat_feature_count=0,
        text_col_count=0,
        embedding_col_count=0,
        outlier_detection=None,
        inject_inf_nan=False,
        inject_degenerate_cols=False,
        inject_zero_col=False,
        inject_rank_deficient=False,
        inject_all_nan_col=False,
        inject_label_leak=False,
        with_datetime_col=False,
        inject_test_drift=None,
        mlp_inject_zero_sample_weight_batch_cfg=False,
    )
    c_on10 = _build_combo(models=("cb",), axes=on10, seed=0)
    df_on10, _target_col, _cat_names = build_frame_for_combo(c_on10)
    assert "num_xor_a" in df_on10.columns, (
        "#10 frame-builder did not emit num_xor_a under "
        "inject_xor_synergy_pair_cfg=True"
    )
    assert "num_xor_b" in df_on10.columns, (
        "#10 frame-builder did not emit num_xor_b under "
        "inject_xor_synergy_pair_cfg=True"
    )

    # Off -> no XOR columns.
    off10_full = dict(on10); off10_full["inject_xor_synergy_pair_cfg"] = False
    c_off10_full = _build_combo(models=("cb",), axes=off10_full, seed=0)
    df_off10, _, _ = build_frame_for_combo(c_off10_full)
    assert "num_xor_a" not in df_off10.columns
    assert "num_xor_b" not in df_off10.columns

    # (d-iv) #9 threading: frame builder emits a ts column with a far-past
    # tail block when the axis is on AND the gate holds.
    on9 = dict(base_axes); on9.update(
        n_rows=200,
        input_type="pandas",
        target_type="regression",
        imbalance_ratio="balanced",
        recurrent_model_cfg=None,
        weight_schemas=("recency",),
        mlp_inject_zero_sample_weight_batch_cfg=True,
        # Pin pathological-data axes off for the assertion to focus on ts.
        cat_feature_count=0,
        text_col_count=0,
        embedding_col_count=0,
        outlier_detection=None,
        inject_inf_nan=False,
        inject_degenerate_cols=False,
        inject_zero_col=False,
        inject_rank_deficient=False,
        inject_all_nan_col=False,
        inject_label_leak=False,
        with_datetime_col=False,
        inject_test_drift=None,
        inject_xor_synergy_pair_cfg=False,
        use_mrmr_fs=False,
    )
    c_on9 = _build_combo(models=("mlp",), axes=on9, seed=0)
    df_on9, _, _ = build_frame_for_combo(c_on9)
    assert "ts" in df_on9.columns, (
        "#9 frame-builder did not emit ts column under "
        "mlp_inject_zero_sample_weight_batch_cfg=True"
    )
    # Last 20% of rows must be far-past (year 1900); first 80% must be 2026+.
    import pandas as pd
    ts_series = df_on9["ts"]
    n_rows_on9 = len(df_on9)
    tail = max(1, int(n_rows_on9 * 0.2))
    assert (ts_series.iloc[-tail:] < pd.Timestamp("1950-01-01")).all(), (
        "#9: last tail must be far-past (year 1900) so recency weights -> 0"
    )
    assert (ts_series.iloc[: n_rows_on9 - tail] > pd.Timestamp("2025-01-01")).all(), (
        "#9: leading rows must be recent (2026+) so recency weights are positive"
    )
    # Off -> no ts injection (gate fails on weight_schemas=uniform).
    off9_full = dict(on9); off9_full.update(
        weight_schemas=("uniform",),
        mlp_inject_zero_sample_weight_batch_cfg=True,
    )
    c_off9_full = _build_combo(models=("mlp",), axes=off9_full, seed=0)
    df_off9, _, _ = build_frame_for_combo(c_off9_full)
    assert "ts" not in df_off9.columns, (
        "#9: gate must drop ts injection under weight_schemas=('uniform',)"
    )

    # (e) Distinct canonical_keys: on-axis values must NOT collapse to the
    # source default under their compound-gate-on configuration.
    # #5 distinct under use_shap_proxied_fs=True.
    on5_def = dict(on5); on5_def["shap_proxied_adaptive_prescreen_by_stability_cfg"] = False
    c_on5_def = _build_combo(models=("cb",), axes=on5_def, seed=0)
    assert c_on5.canonical_key() != c_on5_def.canonical_key(), (
        "audit-pass-8 #5: non-default value must produce distinct canonical "
        "key under use_shap_proxied_fs=True"
    )
    # #7 distinct under mlp+regression.
    on7_def = dict(on78); on7_def["mlp_use_layernorm_cfg"] = False
    c_on7_def = _build_combo(models=("mlp",), axes=on7_def, seed=0)
    # #8 distinct under mlp in models.
    on8_def = dict(on78); on8_def["mlp_l1_alpha_cfg"] = 0.0
    c_on8_def = _build_combo(models=("mlp",), axes=on8_def, seed=0)
    assert c_on78.canonical_key() != c_on7_def.canonical_key() or (
        c_on78.canonical_key() != c_on8_def.canonical_key()
    ), (
        "audit-pass-8 #7/#8: non-default values must change canonical key "
        "under mlp+regression"
    )
    # #9 distinct under mlp + non-uniform weights.
    on9_def = dict(on9); on9_def["mlp_inject_zero_sample_weight_batch_cfg"] = False
    c_on9_def = _build_combo(models=("mlp",), axes=on9_def, seed=0)
    assert c_on9.canonical_key() != c_on9_def.canonical_key(), (
        "audit-pass-8 #9: non-default value must produce distinct canonical "
        "key under mlp + non-uniform weights"
    )
    # #10 distinct under use_mrmr_fs + interactions >= 2.
    on10_def = dict(on10); on10_def["inject_xor_synergy_pair_cfg"] = False
    c_on10_def = _build_combo(models=("cb",), axes=on10_def, seed=0)
    assert c_on10.canonical_key() != c_on10_def.canonical_key(), (
        "audit-pass-8 #10: non-default value must produce distinct canonical "
        "key under use_mrmr_fs + interactions >= 2"
    )
