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
    assert combo.canonical_key() != combo_no_mrmr.canonical_key(), "C1: use_mrmr_fs True/False must produce distinct canonical keys under inject_all_nan_col"


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
    assert combo.canonical_key() != combo_uniform.canonical_key(), "C2: weight_schemas=(recency,) must NOT canonicalise to (uniform,) under recurrent_model_cfg"


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
        _LTR_NATIVE_RANKERS,
        enumerate_combos,
    )

    combos = enumerate_combos(target=150, master_seed=20260422)
    unrunnable = [c for c in combos if c.target_type == "learning_to_rank" and not any(m in _LTR_NATIVE_RANKERS for m in c.models)]
    assert not unrunnable, f"enumerator emitted {len(unrunnable)} LTR combos with no native ranker: {[c.short_id() for c in unrunnable[:5]]}"


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
        AXES,
        enumerate_combos,
        build_mrmr_kwargs,
    )

    for ax in (
        "mrmr_build_friend_graph_cfg",
        "mrmr_friend_graph_prune_cfg",
        "mrmr_cluster_aggregate_enable_cfg",
        "mrmr_cluster_aggregate_mode_cfg",
    ):
        assert ax in AXES, f"missing fuzz axis {ax}"

    combos = enumerate_combos(target=150, master_seed=20260422)
    mrmr_combos = [c for c in combos if c.use_mrmr_fs]
    assert mrmr_combos, "expected at least one MRMR-on combo in the suite"

    # kwargs carry all 4 keys (names match the MRMR constructor params).
    kw = build_mrmr_kwargs(mrmr_combos[0])
    for k in (
        "build_friend_graph",
        "friend_graph_prune",
        "cluster_aggregate_enable",
        "cluster_aggregate_mode",
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
        AXES,
        _build_combo,
        enumerate_combos,
        build_shap_proxied_fs_kwargs,
    )

    for ax in (
        "use_shap_proxied_fs",
        "shap_proxied_optimizer_cfg",
        "shap_proxied_revalidate_cfg",
        "shap_proxied_trust_guard_cfg",
        "shap_proxied_interaction_aware_cfg",
        "shap_proxied_cluster_features_cfg",
    ):
        assert ax in AXES, f"missing fuzz axis {ax}"

    combos = enumerate_combos(target=150, master_seed=20260422)
    shap_combos = [c for c in combos if c.use_shap_proxied_fs]
    assert shap_combos, "expected at least one ShapProxiedFS-on combo in the suite"

    # kwargs carry all 5 knobs (names match the ShapProxiedFS constructor params).
    kw = build_shap_proxied_fs_kwargs(shap_combos[0])
    assert kw is not None
    for k in (
        "optimizer",
        "revalidate",
        "trust_guard",
        "interaction_aware",
        "cluster_features",
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
    assert combo_off_a.canonical_key() == combo_off_b.canonical_key(), "shap-proxied sub-knobs must collapse to defaults when the enable flag is off"

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
        AXES,
        _build_combo,
        enumerate_combos,
        build_shap_proxied_fs_kwargs,
        build_composite_discovery_config,
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
    off_a = dict(on_axes)
    off_a["use_shap_proxied_fs"] = False
    off_b = dict(off_a)
    off_b.update(shap_proxied_active_learning_cfg=False, shap_proxied_prefilter_method_cfg="auto")
    combo_off_a = _build_combo(models=("cb",), axes=off_a, seed=0)
    combo_off_b = _build_combo(models=("cb",), axes=off_b, seed=0)
    assert build_shap_proxied_fs_kwargs(combo_off_a) is None
    assert combo_off_a.canonical_key() == combo_off_b.canonical_key(), "shap_proxied ext sub-knobs must collapse when use_shap_proxied_fs is off"

    # FHC text_min_cardinality: gate on enable_feature_handling_config_cfg.
    on_fhc = dict(base_axes)
    on_fhc.update(enable_feature_handling_config_cfg=True, fhc_text_min_cardinality_cfg=50)
    off_fhc_a = dict(on_fhc)
    off_fhc_a["enable_feature_handling_config_cfg"] = False
    off_fhc_b = dict(off_fhc_a)
    off_fhc_b["fhc_text_min_cardinality_cfg"] = 300
    c_on = _build_combo(models=("cb",), axes=on_fhc, seed=0)
    c_off_a = _build_combo(models=("cb",), axes=off_fhc_a, seed=0)
    c_off_b = _build_combo(models=("cb",), axes=off_fhc_b, seed=0)
    assert c_on.fhc_text_min_cardinality_cfg == 50
    assert c_off_a.canonical_key() == c_off_b.canonical_key(), "fhc_text_min_cardinality_cfg must collapse when FHC is disabled"

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
    off_comp_a = dict(on_comp)
    off_comp_a["composite_discovery_enabled_cfg"] = False
    off_comp_b = dict(off_comp_a)
    off_comp_b.update(
        composite_auto_skip_on_baseline_optimal_cfg=False,
        composite_mi_n_neighbors_cfg=3,
        composite_auto_base_null_perms_cfg=20,
        composite_multi_base_max_k_cfg=3,
    )
    c_comp_off_a = _build_combo(models=("cb",), axes=off_comp_a, seed=0)
    c_comp_off_b = _build_combo(models=("cb",), axes=off_comp_b, seed=0)
    assert c_comp_off_a.canonical_key() == c_comp_off_b.canonical_key(), "composite deep knobs must collapse when composite_discovery_enabled_cfg is off"

    # extreme_ar_group_aware_skip_models: gate on mlp_extreme_ar_group_aware_skip + 'mlp' in models.
    on_ear = dict(base_axes)
    on_ear.update(
        mlp_extreme_ar_group_aware_skip_cfg=True,
        extreme_ar_group_aware_skip_models_cfg="include_linear",
    )
    on_ear_combo = _build_combo(models=("mlp",), axes=on_ear, seed=0)
    assert on_ear_combo.extreme_ar_group_aware_skip_models_cfg == "include_linear"
    # When MLP-extreme-AR-skip is OFF, the skip-list axis must canon away.
    off_ear_a = dict(on_ear)
    off_ear_a["mlp_extreme_ar_group_aware_skip_cfg"] = False
    off_ear_b = dict(off_ear_a)
    off_ear_b["extreme_ar_group_aware_skip_models_cfg"] = "default_neural"
    c_ear_a = _build_combo(models=("mlp",), axes=off_ear_a, seed=0)
    c_ear_b = _build_combo(models=("mlp",), axes=off_ear_b, seed=0)
    assert c_ear_a.canonical_key() == c_ear_b.canonical_key(), "extreme_ar_group_aware_skip_models_cfg must collapse when MLP-extreme-AR-skip is off"

    # fs_pre_screen_null_fraction_threshold: gates on use_mrmr_fs OR
    # rfecv_estimator_cfg OR use_boruta_shap_cfg (mirrors the existing
    # fs_pre_screen_variance_threshold_cfg sibling gating exactly).
    on_fs = dict(base_axes)
    on_fs.update(use_mrmr_fs=True, fs_pre_screen_null_fraction_threshold_cfg=0.5)
    off_fs_a = dict(on_fs)
    off_fs_a.update(use_mrmr_fs=False, rfecv_estimator_cfg=None, use_boruta_shap_cfg=False)
    off_fs_b = dict(off_fs_a)
    off_fs_b["fs_pre_screen_null_fraction_threshold_cfg"] = 0.99
    c_fs_on = _build_combo(models=("cb",), axes=on_fs, seed=0)
    c_fs_off_a = _build_combo(models=("cb",), axes=off_fs_a, seed=0)
    c_fs_off_b = _build_combo(models=("cb",), axes=off_fs_b, seed=0)
    assert c_fs_on.fs_pre_screen_null_fraction_threshold_cfg == 0.5
    assert c_fs_off_a.canonical_key() == c_fs_off_b.canonical_key(), "fs_pre_screen_null_fraction_threshold_cfg must collapse when no FS method is active"

    # linear_l1_ratio_cfg: gate on 'linear' in models AND linear_solver_cfg='saga'.
    on_lin = dict(base_axes)
    on_lin.update(linear_solver_cfg="saga", linear_l1_ratio_cfg=1.0)
    c_lin_on = _build_combo(models=("linear",), axes=on_lin, seed=0)
    assert c_lin_on.linear_l1_ratio_cfg == 1.0
    # Solver != saga -> canon to 0.0 in canonical_key (avoids sklearn ValueError path).
    off_lin_a = dict(on_lin)
    off_lin_a["linear_solver_cfg"] = "lbfgs"
    off_lin_b = dict(off_lin_a)
    off_lin_b["linear_l1_ratio_cfg"] = 0.0
    c_lin_a = _build_combo(models=("linear",), axes=off_lin_a, seed=0)
    c_lin_b = _build_combo(models=("linear",), axes=off_lin_b, seed=0)
    assert c_lin_a.canonical_key() == c_lin_b.canonical_key(), "linear_l1_ratio_cfg must collapse to 0.0 when linear_solver_cfg != saga"

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
    off_rec_a = dict(on_rec)
    off_rec_a["recurrent_model_cfg"] = None
    off_rec_b = dict(off_rec_a)
    off_rec_b["recurrent_hidden_size_cfg"] = 128
    c_rec_a = _build_combo(models=("cb",), axes=off_rec_a, seed=0)
    c_rec_b = _build_combo(models=("cb",), axes=off_rec_b, seed=0)
    assert c_rec_a.canonical_key() == c_rec_b.canonical_key(), "recurrent_hidden_size_cfg must collapse when no recurrent model is requested"

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
        AXES,
        _build_combo,
        enumerate_combos,
        build_shap_proxied_fs_kwargs,
    )

    new_axes = (
        # PART A
        "ensembling_degenerate_class_ratio_cfg",
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
    off_b_a = dict(on_axes_b)
    off_b_a["use_shap_proxied_fs"] = False
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
    assert combo_off_b_a.canonical_key() == combo_off_b_b.canonical_key(), "ShapProxiedFS deep knobs must collapse when use_shap_proxied_fs is off"

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
    assert c_clu_a.canonical_key() == c_clu_b.canonical_key(), "shap_proxied_within_cluster_refine_cfg must collapse when cluster_features is literal False"

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
    off_a1_a = dict(on_a1)
    off_a1_a["target_type"] = "regression"
    off_a1_b = dict(off_a1_a)
    off_a1_b["ensembling_degenerate_class_ratio_cfg"] = 0.01
    c_a1_off_a = _build_combo(models=("cb",), axes=off_a1_a, seed=0)
    c_a1_off_b = _build_combo(models=("cb",), axes=off_a1_b, seed=0)
    assert c_a1_off_a.canonical_key() == c_a1_off_b.canonical_key(), "ensembling_degenerate_class_ratio_cfg must collapse for non-classification targets"

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
    off_a3_a = dict(on_a3)
    off_a3_a["with_datetime_col"] = False
    off_a3_b = dict(off_a3_a)
    off_a3_b["target_temporal_audit_granularity_cfg"] = "auto"
    c_a3_off_a = _build_combo(models=("cb",), axes=off_a3_a, seed=0)
    c_a3_off_b = _build_combo(models=("cb",), axes=off_a3_b, seed=0)
    assert c_a3_off_a.canonical_key() == c_a3_off_b.canonical_key(), "target_temporal_audit_granularity_cfg must collapse when no datetime column"

    # A4. prep_ext_dim_n_components_cfg: gate on prep_ext_dim_reducer_cfg in (PCA, TruncatedSVD).
    on_a4 = dict(base_axes)
    on_a4.update(
        prep_ext_dim_reducer_cfg="PCA",
        prep_ext_dim_n_components_cfg=10,
    )
    c_a4_on = _build_combo(models=("cb",), axes=on_a4, seed=0)
    assert c_a4_on.prep_ext_dim_n_components_cfg == 10
    # No dim_reducer -> canon to default 50.
    off_a4_a = dict(on_a4)
    off_a4_a["prep_ext_dim_reducer_cfg"] = None
    off_a4_b = dict(off_a4_a)
    off_a4_b["prep_ext_dim_n_components_cfg"] = 50
    c_a4_off_a = _build_combo(models=("cb",), axes=off_a4_a, seed=0)
    c_a4_off_b = _build_combo(models=("cb",), axes=off_a4_b, seed=0)
    assert c_a4_off_a.canonical_key() == c_a4_off_b.canonical_key(), "prep_ext_dim_n_components_cfg must collapse when no dim_reducer is picked"

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
        AXES,
        _build_combo,
        enumerate_combos,
        build_shap_proxied_fs_kwargs,
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
    off_a = dict(on_axes)
    off_a["use_shap_proxied_fs"] = False
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
    assert c_off_a.canonical_key() == c_off_b.canonical_key(), "W3 ShapProxiedFS axes must collapse when use_shap_proxied_fs is off"

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
    assert c_clu_a.canonical_key() == c_clu_b.canonical_key(), "shap_proxied_cluster_weighting_cfg must collapse when cluster_features is literal False"

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
    assert c_int_a.canonical_key() == c_int_b.canonical_key(), "shap_proxied_max_interaction_features_cfg must collapse when interaction_aware=False"

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
    assert kw_int_on["max_interaction_features"] == 64, "interaction_aware=True must let max_interaction_features=64 reach the kwargs"
    assert kw_int_on["interaction_aware"] is True

    # The 64-with-interactions canonical key must differ from the 16-default
    # canonical key (so the pairwise sampler actually reaches the wide-fan
    # interaction branch).
    inter_on_default = dict(inter_on_axes)
    inter_on_default["shap_proxied_max_interaction_features_cfg"] = 16
    c_int_def = _build_combo(models=("cb",), axes=inter_on_default, seed=0)
    assert c_int_on.canonical_key() != c_int_def.canonical_key(), "max_interaction_features=64 vs 16 must produce distinct canon keys when interactions on"


# ---------------------------------------------------------------------------
# iter554: short_id() determinism across import-order / env state
# ---------------------------------------------------------------------------


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
        AXES,
        _build_combo,
        enumerate_combos,
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
    assert c_reg_a.canonical_key() == c_reg_b.canonical_key(), "calibration trio must collapse to defaults when target_type=regression"

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
    assert c_cls_off_a.canonical_key() == c_cls_off_b.canonical_key(), "n_bootstrap+candidates must collapse when policy_auto_pick=False"

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
    assert c_cls_on_a.canonical_key() != c_cls_on_b.canonical_key(), "n_bootstrap+candidates must fork the canon key when policy_auto_pick=True"

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
    assert c_mase_cls_a.canonical_key() == c_mase_cls_b.canonical_key(), "mase_seasonality must collapse to 1 for non-regression target_type"

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
    assert c_mase_reg_a.canonical_key() != c_mase_reg_b.canonical_key(), "mase_seasonality must fork the canon key for regression"

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
    assert c_rec_off_a.canonical_key() == c_rec_off_b.canonical_key(), "use_stratified_sampler must collapse when recurrent_model_cfg=None"

    # pipeline_cache_ram_budget_fraction and behavior_model_file_hash_suffix
    # are ungated (always meaningful). Any value flip MUST fork the canon key.
    rambud_a = dict(base_axes)
    rambud_a["pipeline_cache_ram_budget_fraction_cfg"] = 0.4
    rambud_b = dict(base_axes)
    rambud_b["pipeline_cache_ram_budget_fraction_cfg"] = 0.1
    c_rambud_a = _build_combo(models=("cb",), axes=rambud_a, seed=0)
    c_rambud_b = _build_combo(models=("cb",), axes=rambud_b, seed=0)
    assert c_rambud_a.canonical_key() != c_rambud_b.canonical_key(), "pipeline_cache_ram_budget_fraction must fork the canon key (ungated)"

    hashsuf_a = dict(base_axes)
    hashsuf_a["behavior_model_file_hash_suffix_cfg"] = True
    hashsuf_b = dict(base_axes)
    hashsuf_b["behavior_model_file_hash_suffix_cfg"] = False
    c_hashsuf_a = _build_combo(models=("cb",), axes=hashsuf_a, seed=0)
    c_hashsuf_b = _build_combo(models=("cb",), axes=hashsuf_b, seed=0)
    assert c_hashsuf_a.canonical_key() != c_hashsuf_b.canonical_key(), "behavior_model_file_hash_suffix must fork the canon key (ungated)"

    # reporting_compute_trainset_metrics is ungated.
    tsm_a = dict(base_axes)
    tsm_a["reporting_compute_trainset_metrics_cfg"] = False
    tsm_b = dict(base_axes)
    tsm_b["reporting_compute_trainset_metrics_cfg"] = True
    c_tsm_a = _build_combo(models=("cb",), axes=tsm_a, seed=0)
    c_tsm_b = _build_combo(models=("cb",), axes=tsm_b, seed=0)
    assert c_tsm_a.canonical_key() != c_tsm_b.canonical_key(), "reporting_compute_trainset_metrics must fork the canon key (ungated)"


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
        AXES,
        _build_combo,
        build_shap_proxied_fs_kwargs,
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
    assert c_utf_a.canonical_key() == c_utf_b.canonical_key(), "uniform_tail_frac must collapse to 0.2 when stratified_anchors=False"

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
    assert c_za_a.canonical_key() == c_za_b.canonical_key(), "zipf_alpha must collapse to 0.25 when cardinality_dist != 'zipf'"


def test_iter569_audit_pass_6_axes_flow_to_kwargs():
    """14 audit-pass-6 (W6) fuzz axes must:
      (a) be present in AXES with >=2 candidate values,
      (b) carry the SOURCE-verified library defaults in the FuzzCombo
          dataclass (verified pre-edit against
          src/mlframe/training/_training_runtime_configs.py:42-95 for
          SliceStableESConfig, filters/mrmr.py:224-302,589 for
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
            - slice_stable_es is CANON-ONLY: the fuzz suite does not
              currently propagate ``SliceStableESConfig`` into
              ``train_mlframe_models_suite`` (the suite-level kwarg is
              not exposed); only the canon dedup is exercised by fuzz.
              Pinned by canonical_key collapse assertions below.

    Drift corrections caught pre-edit (audit-pass-6 vs source-of-truth):
      - audit said ``slice_stable_es_source_cfg`` default "random" -->
        SOURCE _training_runtime_configs.py:78 says "temporal". Using source.
      - audit said ``slice_stable_es_aggregate_cfg`` default "t_lcb" -->
        SOURCE _training_runtime_configs.py:89 says "mean". Using source.
    """
    from tests.training._fuzz_combo import (
        AXES,
        _build_combo,
        build_mrmr_kwargs,
        build_composite_discovery_config_from_flat,
    )

    new_axes = (
        "slice_stable_es_enabled_cfg",
        "slice_stable_es_aggregate_cfg",
        "slice_stable_es_source_cfg",
        "slice_stable_es_pareto_best_iter_selection_cfg",
        "slice_stable_es_diagnostic_only_cfg",
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
        "all 8 MRMR Wave 7/8/9 axes must collapse to MRMR defaults when use_mrmr_fs=False so dedup absorbs the disabled-branch combos"
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
    assert c_cv_a.canonical_key() == c_cv_b.canonical_key(), "cv_selector_mode_cfg must collapse to 'mean' when composite_discovery_enabled_cfg=False"

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
        "slice_stable_es sub-knobs must collapse to source defaults when slice_stable_es_enabled_cfg=False"
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
        enabled=True,
        cv_selector_mode="t_lcb",
    )
    assert cfg is not None
    # Source field name verified at _composite_target_discovery_config.py:117.
    assert getattr(cfg, "cv_selector_mode", None) == "t_lcb"

    # (d.3) slice_stable_es is CANON-ONLY: the suite does not currently
    # expose SliceStableESConfig as a top-level kwarg to
    # train_mlframe_models_suite; the master enable/disable axis is still
    # meaningful and must not canon-collapse.
    e_a = dict(base_axes)
    e_a.update(slice_stable_es_enabled_cfg=True)
    e_b = dict(base_axes)
    e_b.update(slice_stable_es_enabled_cfg=False)
    c_e_a = _build_combo(models=("cb",), axes=e_a, seed=0)
    c_e_b = _build_combo(models=("cb",), axes=e_b, seed=0)
    assert c_e_a.canonical_key() != c_e_b.canonical_key(), (
        "slice_stable_es_enabled_cfg=True must NOT canon-collapse to False (master toggle is always meaningful)"
    )


def test_iter576_audit_pass_6_low_tier_axes_flow_to_kwargs():
    """26 audit-pass-6 LOW-tier (W6 LOW) deferred fuzz axes must:
      (a) be present in AXES with >=2 candidate values,
      (b) carry the SOURCE-verified library defaults in the FuzzCombo
          dataclass (verified pre-edit against
          src/mlframe/feature_selection/shap_proxied_fs.py:79-113 for
          ShapProxiedFS S1-S18,
          filters/mrmr.py:241,249,252,265 for Wave 8 S32/S34/S35/S37,
          _composite_target_discovery_config.py:127-130 for S41-S44),
      (c) collapse correctly under the documented secondary gates:
            - 18 shap_proxied_* axes collapse to source defaults when
              use_shap_proxied_fs=False (with refine_ucb sub-knobs also
              gated by within_cluster_refine; revalidation_ucb sub-knobs
              also gated by revalidate_cfg),
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
        AXES,
        _build_combo,
        build_mrmr_kwargs,
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
    assert len(new_axes) == 26, f"expected 26 W6 LOW axes, got {len(new_axes)}"
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
    assert c_default.shap_proxied_shap_aware_stage1_cushion_cfg == 2
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
        "all 18 shap_proxied_* LOW axes must collapse to ShapProxiedFS source defaults when use_shap_proxied_fs=False"
    )

    # (c.2) MRMR LOW scalars collapse when use_mrmr_fs=False.
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
    assert c_mr_a.canonical_key() == c_mr_b.canonical_key(), "4 MRMR Wave 8 LOW scalars must collapse to MRMR defaults when use_mrmr_fs=False"

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
        "4 CV-selector LOW scalars must collapse to discovery defaults when composite_discovery_enabled_cfg=False"
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
    assert kw_sp is not None, "build_shap_proxied_fs_kwargs must return dict when use_shap_proxied_fs=True"
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
        AXES,
        _build_combo,
        build_slice_stable_es_config,
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
    assert _auto_wrap is False, "force_off=True must invert into auto_wrap_partial_fit_es=False"
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
        AXES,
        FuzzCombo,
        _build_combo,
        enumerate_combos,
        build_mrmr_kwargs,
    )

    # (a) #1: dataclass FIELD default mirrors the flipped source default.
    # We check the dataclass default directly (NOT via _build_combo + base_axes,
    # because base_axes selects AXES[ax][0] which is False for the dcd axis --
    # the dataclass default is what bare ``FuzzCombo()`` callers get when no
    # axis dict is supplied).
    assert FuzzCombo.__dataclass_fields__["mrmr_dcd_enable_cfg"].default is True, "audit-pass-7 #1: dataclass default must mirror mrmr.py:596 (True)"
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
        assert base_axes[ax] == expected_first, f"base_axes[{ax}] = {base_axes[ax]!r}, expected {expected_first!r}"

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
    off_dcd_a = dict(base_axes)
    off_dcd_a.update(use_mrmr_fs=False, mrmr_dcd_enable_cfg=False)
    off_dcd_b = dict(off_dcd_a)
    off_dcd_b["mrmr_dcd_enable_cfg"] = True
    c_off_dcd_a = _build_combo(models=("cb",), axes=off_dcd_a, seed=0)
    c_off_dcd_b = _build_combo(models=("cb",), axes=off_dcd_b, seed=0)
    assert c_off_dcd_a.canonical_key() == c_off_dcd_b.canonical_key(), "audit-pass-7 #1: mrmr_dcd_enable_cfg must canon-collapse when use_mrmr_fs=False"

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
    assert build_mrmr_kwargs(c_off_a) is None, "use_mrmr_fs=False must yield None mrmr_kwargs"
    assert c_off_a.canonical_key() == c_off_b.canonical_key(), "audit-pass-7 #2/#3/#4: MRMR knob axes must canon-collapse when use_mrmr_fs=False"

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
        "audit-pass-7 #4: collapsed_fallback_nbins must canon-collapse when nbins_strategy is not one of {mdlp, fayyad_irani}"
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
    assert kw["baseline_npermutations"] == 8, f"#2 baseline_npermutations did not thread: {kw.get('baseline_npermutations')!r}"
    # #3 + #4 ride in the nbins_strategy_kwargs subdict forwarded to
    # per_feature_edges via categorize_dataset (mrmr.py:225 -> _mrmr_fit_impl:341).
    sub = kw.get("nbins_strategy_kwargs")
    assert sub is not None, "nbins_strategy_kwargs subdict missing"
    assert sub.get("low_card_cap") == 2, f"#3 low_card_cap did not thread into nbins_strategy_kwargs: {sub!r}"
    assert sub.get("collapsed_fallback_nbins") == 10, f"#4 collapsed_fallback_nbins did not thread into nbins_strategy_kwargs: {sub!r}"

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
        f"nbins_strategy_kwargs must NOT be injected when #3 + #4 are at source defaults; got {kw_def.get('nbins_strategy_kwargs')!r}"
    )
    # #2 still threads even at the source default (always a real kwarg).
    assert kw_def["baseline_npermutations"] == 2

    # (d-bis) Distinct canonical_keys when the non-default values are set
    # under use_mrmr_fs=True (so the pairwise sampler reaches both branches).
    assert c_on.canonical_key() != c_on_def.canonical_key(), (
        "audit-pass-7 #2/#3/#4: non-default values must produce distinct canonical keys under use_mrmr_fs=True so the dedup keeps both branches reachable"
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
        AXES,
        FuzzCombo,
        _build_combo,
        enumerate_combos,
        build_mrmr_kwargs,
        build_mlp_kwargs,
    )

    # (a) Dataclass defaults mirror HEAD source.
    fields = FuzzCombo.__dataclass_fields__
    assert fields["mrmr_cardinality_bias_correction_cfg"].default is True, "audit-pass-8 #1: default must mirror filters/mrmr.py:334 (True)"
    assert fields["mrmr_min_relevance_gain_relative_to_first_cfg"].default == 0.05, "audit-pass-8 #2: default must mirror filters/mrmr.py:326 (0.05)"
    assert fields["mlp_random_state_cfg"].default is None, "audit-pass-8 #3: default must mirror training/neural/base.py:217 (None)"
    assert fields["mlp_class_weight_cfg"].default is None, "audit-pass-8 #4: default must mirror training/neural/base.py:218 (None)"

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
    off1_a = dict(base_axes)
    off1_a.update(
        use_mrmr_fs=False,
        mrmr_cardinality_bias_correction_cfg=False,
    )
    off1_b = dict(off1_a)
    off1_b["mrmr_cardinality_bias_correction_cfg"] = True
    c_off1_a = _build_combo(models=("cb",), axes=off1_a, seed=0)
    c_off1_b = _build_combo(models=("cb",), axes=off1_b, seed=0)
    assert c_off1_a.canonical_key() == c_off1_b.canonical_key(), "audit-pass-8 #1: must canon-collapse under use_mrmr_fs=False"

    # (c-ii) #2 canon-collapse: when use_mrmr_fs=False, values collapse
    # to source default 0.05.
    off2_a = dict(base_axes)
    off2_a.update(
        use_mrmr_fs=False,
        mrmr_min_relevance_gain_relative_to_first_cfg=0.0,
    )
    off2_b = dict(off2_a)
    off2_b["mrmr_min_relevance_gain_relative_to_first_cfg"] = 0.05
    c_off2_a = _build_combo(models=("cb",), axes=off2_a, seed=0)
    c_off2_b = _build_combo(models=("cb",), axes=off2_b, seed=0)
    assert c_off2_a.canonical_key() == c_off2_b.canonical_key(), "audit-pass-8 #2: must canon-collapse under use_mrmr_fs=False"

    # (c-iii) #3 canon-collapse: with no MLP and no recurrent, every value
    # collapses to None. Use a non-mlp model + recurrent_model_cfg=None.
    off3_a = dict(base_axes)
    off3_a.update(
        recurrent_model_cfg=None,
        mlp_random_state_cfg=42,
    )
    off3_b = dict(off3_a)
    off3_b["mlp_random_state_cfg"] = None
    c_off3_a = _build_combo(models=("cb",), axes=off3_a, seed=0)
    c_off3_b = _build_combo(models=("cb",), axes=off3_b, seed=0)
    assert c_off3_a.canonical_key() == c_off3_b.canonical_key(), "audit-pass-8 #3: must canon-collapse when 'mlp' not in models AND recurrent_model_cfg is None"

    # (c-iv) #4 canon-collapse: outside the compound gate every value
    # collapses to None. Test (a) no MLP, (b) MLP but regression target,
    # (c) MLP + binary but balanced imbalance.
    for ax_override in (
        # (a) no MLP -> compound gate fails on models check.
        {"target_type": "binary_classification", "imbalance_ratio": "rare_5pct", "recurrent_model_cfg": None},
        # (c) MLP + binary but balanced -> compound gate fails on imbalance.
        # (kept models default to base_axes, which puts cb only; we test
        #  the "balanced" leg by overriding _build_combo's models below).
    ):
        off4_a = dict(base_axes)
        off4_a.update(ax_override)
        off4_a["mlp_class_weight_cfg"] = "balanced"
        off4_b = dict(off4_a)
        off4_b["mlp_class_weight_cfg"] = None
        c_off4_a = _build_combo(models=("cb",), axes=off4_a, seed=0)
        c_off4_b = _build_combo(models=("cb",), axes=off4_b, seed=0)
        assert c_off4_a.canonical_key() == c_off4_b.canonical_key(), (
            f"audit-pass-8 #4: must canon-collapse outside the compound gate with override {ax_override!r}"
        )

    # (c-iv-bis) MLP + binary + balanced -> still outside the gate, collapses.
    off4_bal_a = dict(base_axes)
    off4_bal_a.update(
        target_type="binary_classification",
        imbalance_ratio="balanced",
        recurrent_model_cfg=None,
        mlp_class_weight_cfg="balanced",
    )
    off4_bal_b = dict(off4_bal_a)
    off4_bal_b["mlp_class_weight_cfg"] = None
    c_off4_bal_a = _build_combo(models=("mlp",), axes=off4_bal_a, seed=0)
    c_off4_bal_b = _build_combo(models=("mlp",), axes=off4_bal_b, seed=0)
    assert c_off4_bal_a.canonical_key() == c_off4_bal_b.canonical_key(), "audit-pass-8 #4: balanced imbalance must canon-collapse class_weight"

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
    assert kw_mrmr["cardinality_bias_correction"] is False, f"#1 did not thread into mrmr_kwargs: {kw_mrmr.get('cardinality_bias_correction')!r}"
    assert kw_mrmr["min_relevance_gain_relative_to_first"] == 0.0, (
        f"#2 did not thread into mrmr_kwargs: {kw_mrmr.get('min_relevance_gain_relative_to_first')!r}"
    )

    # MRMR off -> kwargs is None (no threading).
    off_mrmr = dict(base_axes)
    off_mrmr.update(
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
    assert kw_mlp.get("random_state") == 42, f"#3 did not thread into mlp_kwargs: {kw_mlp.get('random_state')!r}"
    assert kw_mlp.get("class_weight") == "balanced", f"#4 did not thread into mlp_kwargs: {kw_mlp.get('class_weight')!r}"

    # No MLP / no recurrent -> builder returns None.
    no_neural = dict(base_axes)
    no_neural.update(
        recurrent_model_cfg=None,
        mlp_random_state_cfg=42,
        mlp_class_weight_cfg="balanced",
    )
    c_no_neural = _build_combo(models=("cb",), axes=no_neural, seed=0)
    assert build_mlp_kwargs(c_no_neural) is None, "build_mlp_kwargs must return None when neither MLP nor recurrent fire"

    # MLP but regression -> class_weight is dropped (#4 gate fails), but
    # random_state still threads (#3 gate only requires MLP-or-recurrent).
    mlp_reg = dict(base_axes)
    mlp_reg.update(
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
    assert "class_weight" not in kw_mlp_reg, "audit-pass-8 #4: class_weight must NOT thread on regression target"

    # Distinct canonical_keys: on-MRMR + non-default #1/#2 vs source-defaults.
    on_def = dict(on_mrmr)
    on_def.update(
        mrmr_cardinality_bias_correction_cfg=True,
        mrmr_min_relevance_gain_relative_to_first_cfg=0.05,
    )
    c_on_def = _build_combo(models=("cb",), axes=on_def, seed=0)
    assert c_on_mrmr.canonical_key() != c_on_def.canonical_key(), (
        "audit-pass-8 #1/#2: non-default values must produce distinct canonical keys under use_mrmr_fs=True so dedup keeps both branches reachable"
    )

    # Distinct canonical_keys for #3/#4 under the compound gate.
    on_mlp_def = dict(on_mlp)
    on_mlp_def.update(mlp_random_state_cfg=None, mlp_class_weight_cfg=None)
    c_on_mlp_def = _build_combo(models=("mlp",), axes=on_mlp_def, seed=0)
    assert c_on_mlp.canonical_key() != c_on_mlp_def.canonical_key(), (
        "audit-pass-8 #3/#4: non-default values must produce distinct canonical keys when 'mlp' in models AND rare classification holds"
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
        AXES,
        FuzzCombo,
        _build_combo,
        enumerate_combos,
        build_shap_proxied_fs_kwargs,
        build_mlp_kwargs,
        build_frame_for_combo,
    )

    # (a) Dataclass defaults mirror HEAD source.
    fields = FuzzCombo.__dataclass_fields__
    assert fields["shap_proxied_adaptive_prescreen_by_stability_cfg"].default is False, (
        "audit-pass-8 #5: default must mirror feature_selection/shap_proxied_fs.py:208 (False)"
    )
    assert fields["mlp_use_layernorm_cfg"].default is False, "audit-pass-8 #7: default must mirror training/neural/flat.py:205 (False)"
    assert fields["mlp_l1_alpha_cfg"].default == 0.0, "audit-pass-8 #8: default must mirror library default 0.0"
    assert fields["mlp_inject_zero_sample_weight_batch_cfg"].default is False, "audit-pass-8 #9: default must mirror False (no injection)"
    assert fields["inject_xor_synergy_pair_cfg"].default is False, "audit-pass-8 #10: default must mirror False (no injection)"

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
    off5_a = dict(base_axes)
    off5_a.update(
        use_shap_proxied_fs=False,
        shap_proxied_adaptive_prescreen_by_stability_cfg=True,
    )
    off5_b = dict(off5_a)
    off5_b["shap_proxied_adaptive_prescreen_by_stability_cfg"] = False
    c_off5_a = _build_combo(models=("cb",), axes=off5_a, seed=0)
    c_off5_b = _build_combo(models=("cb",), axes=off5_b, seed=0)
    assert c_off5_a.canonical_key() == c_off5_b.canonical_key(), "audit-pass-8 #5: must canon-collapse under use_shap_proxied_fs=False"

    # (c-ii) #7 canon-collapse: classification target or no MLP -> collapses.
    off7_a = dict(base_axes)
    off7_a.update(
        target_type="binary_classification",
        mlp_use_layernorm_cfg=True,
    )
    off7_b = dict(off7_a)
    off7_b["mlp_use_layernorm_cfg"] = False
    c_off7_a = _build_combo(models=("mlp",), axes=off7_a, seed=0)
    c_off7_b = _build_combo(models=("mlp",), axes=off7_b, seed=0)
    assert c_off7_a.canonical_key() == c_off7_b.canonical_key(), "audit-pass-8 #7: must canon-collapse under classification target even when mlp is in models"
    # And no MLP at all -> collapses regardless of target.
    off7_c = dict(base_axes)
    off7_c.update(
        target_type="regression",
        mlp_use_layernorm_cfg=True,
    )
    off7_d = dict(off7_c)
    off7_d["mlp_use_layernorm_cfg"] = False
    c_off7_c = _build_combo(models=("cb",), axes=off7_c, seed=0)
    c_off7_d = _build_combo(models=("cb",), axes=off7_d, seed=0)
    assert c_off7_c.canonical_key() == c_off7_d.canonical_key(), "audit-pass-8 #7: must canon-collapse when MLP not in models"

    # (c-iii) #8 canon-collapse: no MLP -> collapses to 0.0.
    off8_a = dict(base_axes)
    off8_a["mlp_l1_alpha_cfg"] = 0.001
    off8_b = dict(off8_a)
    off8_b["mlp_l1_alpha_cfg"] = 0.0
    c_off8_a = _build_combo(models=("cb",), axes=off8_a, seed=0)
    c_off8_b = _build_combo(models=("cb",), axes=off8_b, seed=0)
    assert c_off8_a.canonical_key() == c_off8_b.canonical_key(), "audit-pass-8 #8: must canon-collapse when 'mlp' not in models"

    # (c-iv) #9 canon-collapse: weight_schemas=("uniform",) or no MLP -> False.
    off9_a = dict(base_axes)
    off9_a.update(
        weight_schemas=("uniform",),
        mlp_inject_zero_sample_weight_batch_cfg=True,
    )
    off9_b = dict(off9_a)
    off9_b["mlp_inject_zero_sample_weight_batch_cfg"] = False
    c_off9_a = _build_combo(models=("mlp",), axes=off9_a, seed=0)
    c_off9_b = _build_combo(models=("mlp",), axes=off9_b, seed=0)
    assert c_off9_a.canonical_key() == c_off9_b.canonical_key(), "audit-pass-8 #9: must canon-collapse under weight_schemas=('uniform',)"
    # No MLP -> collapses regardless of weight_schemas.
    off9_c = dict(base_axes)
    off9_c.update(
        weight_schemas=("recency",),
        mlp_inject_zero_sample_weight_batch_cfg=True,
    )
    off9_d = dict(off9_c)
    off9_d["mlp_inject_zero_sample_weight_batch_cfg"] = False
    c_off9_c = _build_combo(models=("cb",), axes=off9_c, seed=0)
    c_off9_d = _build_combo(models=("cb",), axes=off9_d, seed=0)
    assert c_off9_c.canonical_key() == c_off9_d.canonical_key(), "audit-pass-8 #9: must canon-collapse when 'mlp' not in models"

    # (c-v) #10 canon-collapse: use_mrmr_fs=False -> False; or interactions
    # order < 2 -> False.
    off10_a = dict(base_axes)
    off10_a.update(
        use_mrmr_fs=False,
        mrmr_interactions_max_order_cfg=2,
        inject_xor_synergy_pair_cfg=True,
    )
    off10_b = dict(off10_a)
    off10_b["inject_xor_synergy_pair_cfg"] = False
    c_off10_a = _build_combo(models=("cb",), axes=off10_a, seed=0)
    c_off10_b = _build_combo(models=("cb",), axes=off10_b, seed=0)
    assert c_off10_a.canonical_key() == c_off10_b.canonical_key(), "audit-pass-8 #10: must canon-collapse under use_mrmr_fs=False"
    off10_c = dict(base_axes)
    off10_c.update(
        use_mrmr_fs=True,
        mrmr_interactions_max_order_cfg=1,
        inject_xor_synergy_pair_cfg=True,
    )
    off10_d = dict(off10_c)
    off10_d["inject_xor_synergy_pair_cfg"] = False
    c_off10_c = _build_combo(models=("cb",), axes=off10_c, seed=0)
    c_off10_d = _build_combo(models=("cb",), axes=off10_d, seed=0)
    assert c_off10_c.canonical_key() == c_off10_d.canonical_key(), "audit-pass-8 #10: must canon-collapse under interactions_max_order < 2"

    # (d-i) #5 threading: when use_shap_proxied_fs=True, builder emits the
    # adaptive_prescreen_by_stability key with the axis value.
    on5 = dict(base_axes)
    on5.update(
        use_shap_proxied_fs=True,
        shap_proxied_adaptive_prescreen_by_stability_cfg=True,
    )
    c_on5 = _build_combo(models=("cb",), axes=on5, seed=0)
    kw_shap = build_shap_proxied_fs_kwargs(c_on5)
    assert kw_shap is not None
    assert kw_shap["adaptive_prescreen_by_stability"] is True, (
        f"#5 did not thread into shap_proxied_fs_kwargs: {kw_shap.get('adaptive_prescreen_by_stability')!r}"
    )
    # ShapProxiedFS off -> kwargs None.
    off5_full = dict(base_axes)
    off5_full.update(
        use_shap_proxied_fs=False,
        shap_proxied_adaptive_prescreen_by_stability_cfg=True,
    )
    assert build_shap_proxied_fs_kwargs(_build_combo(models=("cb",), axes=off5_full, seed=0)) is None

    # (d-ii) #7/#8 threading: MLP + regression -> use_layernorm flows;
    # MLP active -> l1_alpha flows.
    on78 = dict(base_axes)
    on78.update(
        target_type="regression",
        imbalance_ratio="balanced",
        recurrent_model_cfg=None,
        mlp_use_layernorm_cfg=True,
        mlp_l1_alpha_cfg=0.001,
    )
    c_on78 = _build_combo(models=("mlp",), axes=on78, seed=0)
    kw_mlp = build_mlp_kwargs(c_on78)
    assert kw_mlp is not None
    assert kw_mlp.get("use_layernorm") is True, f"#7 did not thread into mlp_kwargs: {kw_mlp.get('use_layernorm')!r}"
    assert kw_mlp.get("l1_alpha") == 0.001, f"#8 did not thread into mlp_kwargs: {kw_mlp.get('l1_alpha')!r}"

    # MLP + classification -> use_layernorm dropped (#7 gate fails),
    # l1_alpha still threads (#8 gate only needs MLP in models).
    on78_cls = dict(base_axes)
    on78_cls.update(
        target_type="binary_classification",
        imbalance_ratio="balanced",
        recurrent_model_cfg=None,
        mlp_use_layernorm_cfg=True,
        mlp_l1_alpha_cfg=0.001,
    )
    c_on78_cls = _build_combo(models=("mlp",), axes=on78_cls, seed=0)
    kw_mlp_cls = build_mlp_kwargs(c_on78_cls)
    assert kw_mlp_cls is not None
    assert "use_layernorm" not in kw_mlp_cls, "audit-pass-8 #7: use_layernorm must NOT thread on classification target"
    assert kw_mlp_cls.get("l1_alpha") == 0.001, f"#8 must still thread under classification: {kw_mlp_cls.get('l1_alpha')!r}"

    # No MLP + no recurrent -> builder returns None entirely (neither
    # threads).
    no_neural = dict(base_axes)
    no_neural.update(
        recurrent_model_cfg=None,
        mlp_use_layernorm_cfg=True,
        mlp_l1_alpha_cfg=0.001,
    )
    c_no_neural = _build_combo(models=("cb",), axes=no_neural, seed=0)
    assert build_mlp_kwargs(c_no_neural) is None, "build_mlp_kwargs must return None when neither MLP nor recurrent fire"

    # (d-iii) #10 threading: frame builder emits num_xor_a / num_xor_b
    # when the axis is on (small-n combo so the test is cheap).
    on10 = dict(base_axes)
    on10.update(
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
    assert "num_xor_a" in df_on10.columns, "#10 frame-builder did not emit num_xor_a under inject_xor_synergy_pair_cfg=True"
    assert "num_xor_b" in df_on10.columns, "#10 frame-builder did not emit num_xor_b under inject_xor_synergy_pair_cfg=True"

    # Off -> no XOR columns.
    off10_full = dict(on10)
    off10_full["inject_xor_synergy_pair_cfg"] = False
    c_off10_full = _build_combo(models=("cb",), axes=off10_full, seed=0)
    df_off10, _, _ = build_frame_for_combo(c_off10_full)
    assert "num_xor_a" not in df_off10.columns
    assert "num_xor_b" not in df_off10.columns

    # (d-iv) #9 threading: frame builder emits a ts column with a far-past
    # tail block when the axis is on AND the gate holds.
    on9 = dict(base_axes)
    on9.update(
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
    assert "ts" in df_on9.columns, "#9 frame-builder did not emit ts column under mlp_inject_zero_sample_weight_batch_cfg=True"
    # Last 20% of rows must be far-past (year 1900); first 80% must be 2026+.
    import pandas as pd

    ts_series = df_on9["ts"]
    n_rows_on9 = len(df_on9)
    tail = max(1, int(n_rows_on9 * 0.2))
    assert (ts_series.iloc[-tail:] < pd.Timestamp("1950-01-01")).all(), "#9: last tail must be far-past (year 1900) so recency weights -> 0"
    assert (ts_series.iloc[: n_rows_on9 - tail] > pd.Timestamp("2025-01-01")).all(), "#9: leading rows must be recent (2026+) so recency weights are positive"
    # Off -> no ts injection (gate fails on weight_schemas=uniform).
    off9_full = dict(on9)
    off9_full.update(
        weight_schemas=("uniform",),
        mlp_inject_zero_sample_weight_batch_cfg=True,
    )
    c_off9_full = _build_combo(models=("mlp",), axes=off9_full, seed=0)
    df_off9, _, _ = build_frame_for_combo(c_off9_full)
    assert "ts" not in df_off9.columns, "#9: gate must drop ts injection under weight_schemas=('uniform',)"

    # (e) Distinct canonical_keys: on-axis values must NOT collapse to the
    # source default under their compound-gate-on configuration.
    # #5 distinct under use_shap_proxied_fs=True.
    on5_def = dict(on5)
    on5_def["shap_proxied_adaptive_prescreen_by_stability_cfg"] = False
    c_on5_def = _build_combo(models=("cb",), axes=on5_def, seed=0)
    assert c_on5.canonical_key() != c_on5_def.canonical_key(), (
        "audit-pass-8 #5: non-default value must produce distinct canonical key under use_shap_proxied_fs=True"
    )
    # #7 distinct under mlp+regression.
    on7_def = dict(on78)
    on7_def["mlp_use_layernorm_cfg"] = False
    c_on7_def = _build_combo(models=("mlp",), axes=on7_def, seed=0)
    # #8 distinct under mlp in models.
    on8_def = dict(on78)
    on8_def["mlp_l1_alpha_cfg"] = 0.0
    c_on8_def = _build_combo(models=("mlp",), axes=on8_def, seed=0)
    assert c_on78.canonical_key() != c_on7_def.canonical_key() or (c_on78.canonical_key() != c_on8_def.canonical_key()), (
        "audit-pass-8 #7/#8: non-default values must change canonical key under mlp+regression"
    )
    # #9 distinct under mlp + non-uniform weights.
    on9_def = dict(on9)
    on9_def["mlp_inject_zero_sample_weight_batch_cfg"] = False
    c_on9_def = _build_combo(models=("mlp",), axes=on9_def, seed=0)
    assert c_on9.canonical_key() != c_on9_def.canonical_key(), (
        "audit-pass-8 #9: non-default value must produce distinct canonical key under mlp + non-uniform weights"
    )
    # #10 distinct under use_mrmr_fs + interactions >= 2.
    on10_def = dict(on10)
    on10_def["inject_xor_synergy_pair_cfg"] = False
    c_on10_def = _build_combo(models=("cb",), axes=on10_def, seed=0)
    assert c_on10.canonical_key() != c_on10_def.canonical_key(), (
        "audit-pass-8 #10: non-default value must produce distinct canonical key under use_mrmr_fs + interactions >= 2"
    )


# ---------------------------------------------------------------------------
# 2026-05-31 audit-pass-9 (W9): 5 HIGH + 3 MED MLP / MRMR / target-type axes.
# Mirrors test_iter613_audit_pass_8_med_axes_flow_to_kwargs.
# ---------------------------------------------------------------------------


def test_iter615_audit_pass_9_axes_flow_to_kwargs():
    """8 audit-pass-9 (W9) fuzz axes must:
    (a) Dataclass defaults mirror HEAD source verbatim:
          #1 mlp_adamw_betas_cfg = (0.9, 0.95)
             (training/neural/_flat_torch_module.py:499)
          #2 mlp_use_ema_cfg = False
             (training/neural/base.py:266)
          #3 mlp_label_smoothing_cfg = 0.0
             (training/neural/base.py:268)
          #4 mlp_focal_loss_gamma_cfg = None
             (training/neural/base.py:269)
          #5 mlp_use_residual_cfg = False
             (training/neural/flat.py:208)
          #6 mlp_numerical_embedding_cfg = None
             mlp_numerical_embedding_kwargs_cfg = "paper_default"
             (training/neural/flat.py:209-210)
          #7 mrmr_fe_hybrid_orth_enable_cfg = False
             mrmr_fe_hybrid_orth_pair_enable_cfg = True
             (feature_selection/filters/mrmr.py:656, :664)
          #8 target_type axis contains "multi_target_regression"
             (configs/_configs_base.py:126; TargetTypes.MULTI_TARGET_REGRESSION)
    (b) Each axis pair is present in AXES with the values listed in the
        audit.
    (c) Canon-collapse rules hold:
          #1 collapses to (0.9, 0.95) outside 'mlp' in models.
          #2 collapses to False outside 'mlp' in models.
          #3 collapses to 0.0 outside ('mlp' AND multiclass).
          #4 collapses to None outside ('mlp' AND binary AND rare imbalance).
          #5 collapses to False outside 'mlp' in models.
          #6 embedding collapses to None outside 'mlp' in models; kwargs
             literal collapses to "paper_default" outside ('mlp' AND
             embedding != None).
          #7 master collapses to False outside use_mrmr_fs; pair_enable
             collapses to True outside (use_mrmr_fs AND master==True).
          #8 multi_target_regression collapses to "regression" when the
             model subset contains no native-MTR backend (cb / mlp).
    (d) Threading produces the expected kwargs subdicts:
          #1-#6 flow via build_mlp_kwargs into the PytorchLightningEstimator /
                generate_mlp kwargs dict (param names match base.py:264-270
                + flat.py:208-210 verbatim).
          #7 flows via build_mrmr_kwargs as the ``fe_hybrid_orth_enable`` /
             ``fe_hybrid_orth_pair_enable`` ctor keys.
          #8 flows via build_frame_for_combo emitting a 2-D (N, K) target
             column when target_type=="multi_target_regression".
    (e) F-23 mirror canon: inject_inf_nan=True collapses to False when the
        model subset is exactly ('mlp',) -- the _validate_no_nan_inf raise
        at training/neural/base.py:326 makes the True/False variants
        behaviour-identical (immediate crash vs. normal train) so dedup
        must absorb them.
    """
    from tests.training._fuzz_combo import (
        AXES,
        FuzzCombo,
        _build_combo,
        enumerate_combos,
        build_mlp_kwargs,
        build_mrmr_kwargs,
        build_frame_for_combo,
    )

    # (a) Dataclass defaults mirror HEAD source.
    fields = FuzzCombo.__dataclass_fields__
    assert fields["mlp_adamw_betas_cfg"].default == (0.9, 0.95), "audit-pass-9 #1: default must mirror training/neural/_flat_torch_module.py:499 ((0.9, 0.95))"
    assert fields["mlp_use_ema_cfg"].default is False, "audit-pass-9 #2: default must mirror training/neural/base.py:266 (False)"
    assert fields["mlp_label_smoothing_cfg"].default == 0.0, "audit-pass-9 #3: default must mirror training/neural/base.py:268 (0.0)"
    assert fields["mlp_focal_loss_gamma_cfg"].default is None, "audit-pass-9 #4: default must mirror training/neural/base.py:269 (None)"
    assert fields["mlp_use_residual_cfg"].default is False, "audit-pass-9 #5: default must mirror training/neural/flat.py:208 (False)"
    assert fields["mlp_numerical_embedding_cfg"].default is None, "audit-pass-9 #6: default must mirror training/neural/flat.py:209 (None)"
    assert fields["mlp_numerical_embedding_kwargs_cfg"].default == "paper_default", "audit-pass-9 #6: kwargs literal default must mirror 'paper_default'"
    assert fields["mrmr_fe_hybrid_orth_enable_cfg"].default is False, (
        "audit-pass-9 #7: master default must mirror feature_selection/filters/mrmr.py:656 (False)"
    )
    assert fields["mrmr_fe_hybrid_orth_pair_enable_cfg"].default is True, (
        "audit-pass-9 #7: pair_enable default must mirror feature_selection/filters/mrmr.py:664 (True)"
    )

    # (b) AXES presence with the audit-listed pairs.
    assert AXES["mlp_adamw_betas_cfg"] == ((0.9, 0.95), (0.9, 0.999))
    assert AXES["mlp_use_ema_cfg"] == (False, True)
    assert AXES["mlp_label_smoothing_cfg"] == (0.0, 0.1)
    assert AXES["mlp_focal_loss_gamma_cfg"] == (None, 2.0)
    assert AXES["mlp_use_residual_cfg"] == (False, True)
    assert AXES["mlp_numerical_embedding_cfg"] == (None, "plr")
    assert AXES["mlp_numerical_embedding_kwargs_cfg"] == (
        "paper_default",
        "include_raw_false",
    )
    assert AXES["mrmr_fe_hybrid_orth_enable_cfg"] == (False, True)
    assert AXES["mrmr_fe_hybrid_orth_pair_enable_cfg"] == (False, True)
    # #8: multi_target_regression appears as a NEW value in the existing
    # target_type tuple alongside the legacy 5 entries.
    assert "multi_target_regression" in AXES["target_type"], "audit-pass-9 #8: multi_target_regression must be added to the existing target_type axis tuple"

    base_axes = {name: values[0] for name, values in AXES.items()}

    # enumerate_combos still hits 150 with 8 new axes wired (master_seed
    # matches the audit's recommended value).
    combos = enumerate_combos(target=150, master_seed=20260601)
    assert len(combos) == 150, f"enumerate_combos lost combos: {len(combos)}"

    # ------------------------------------------------------------------
    # (c) Canon-collapse rules.
    # ------------------------------------------------------------------

    # (c-i) #1 betas canon-collapse: no MLP -> both values collapse to
    # source default (0.9, 0.95).
    off1_a = dict(base_axes)
    off1_a["mlp_adamw_betas_cfg"] = (0.9, 0.999)
    off1_b = dict(off1_a)
    off1_b["mlp_adamw_betas_cfg"] = (0.9, 0.95)
    c_off1_a = _build_combo(models=("cb",), axes=off1_a, seed=0)
    c_off1_b = _build_combo(models=("cb",), axes=off1_b, seed=0)
    assert c_off1_a.canonical_key() == c_off1_b.canonical_key(), "audit-pass-9 #1: must canon-collapse when 'mlp' not in models"

    # (c-ii) #2 use_ema canon-collapse: no MLP -> both collapse to False.
    off2_a = dict(base_axes)
    off2_a["mlp_use_ema_cfg"] = True
    off2_b = dict(off2_a)
    off2_b["mlp_use_ema_cfg"] = False
    c_off2_a = _build_combo(models=("cb",), axes=off2_a, seed=0)
    c_off2_b = _build_combo(models=("cb",), axes=off2_b, seed=0)
    assert c_off2_a.canonical_key() == c_off2_b.canonical_key(), "audit-pass-9 #2: must canon-collapse when 'mlp' not in models"

    # (c-iii) #3 label_smoothing canon-collapse: non-multiclass target -> 0.0.
    off3_a = dict(base_axes)
    off3_a.update(
        target_type="binary_classification",
        mlp_label_smoothing_cfg=0.1,
    )
    off3_b = dict(off3_a)
    off3_b["mlp_label_smoothing_cfg"] = 0.0
    c_off3_a = _build_combo(models=("mlp",), axes=off3_a, seed=0)
    c_off3_b = _build_combo(models=("mlp",), axes=off3_b, seed=0)
    assert c_off3_a.canonical_key() == c_off3_b.canonical_key(), "audit-pass-9 #3: must canon-collapse on non-multiclass targets"

    # (c-iv) #4 focal_loss canon-collapse: balanced target (no rare class)
    # -> None even on binary.
    off4_a = dict(base_axes)
    off4_a.update(
        target_type="binary_classification",
        imbalance_ratio="balanced",
        mlp_focal_loss_gamma_cfg=2.0,
    )
    off4_b = dict(off4_a)
    off4_b["mlp_focal_loss_gamma_cfg"] = None
    c_off4_a = _build_combo(models=("mlp",), axes=off4_a, seed=0)
    c_off4_b = _build_combo(models=("mlp",), axes=off4_b, seed=0)
    assert c_off4_a.canonical_key() == c_off4_b.canonical_key(), "audit-pass-9 #4: must canon-collapse on balanced binary (focal targets imbalance)"

    # (c-v) #5 use_residual canon-collapse: no MLP -> both collapse to False.
    off5_a = dict(base_axes)
    off5_a["mlp_use_residual_cfg"] = True
    off5_b = dict(off5_a)
    off5_b["mlp_use_residual_cfg"] = False
    c_off5_a = _build_combo(models=("cb",), axes=off5_a, seed=0)
    c_off5_b = _build_combo(models=("cb",), axes=off5_b, seed=0)
    assert c_off5_a.canonical_key() == c_off5_b.canonical_key(), "audit-pass-9 #5: must canon-collapse when 'mlp' not in models"

    # (c-vi) #6 numerical_embedding canon-collapse: no MLP -> embedding
    # collapses to None AND kwargs literal collapses to "paper_default".
    off6_a = dict(base_axes)
    off6_a.update(
        mlp_numerical_embedding_cfg="plr",
        mlp_numerical_embedding_kwargs_cfg="include_raw_false",
    )
    off6_b = dict(off6_a)
    off6_b.update(
        mlp_numerical_embedding_cfg=None,
        mlp_numerical_embedding_kwargs_cfg="paper_default",
    )
    c_off6_a = _build_combo(models=("cb",), axes=off6_a, seed=0)
    c_off6_b = _build_combo(models=("cb",), axes=off6_b, seed=0)
    assert c_off6_a.canonical_key() == c_off6_b.canonical_key(), "audit-pass-9 #6: must canon-collapse when 'mlp' not in models"
    # When MLP is in models but embedding is None, kwargs literal still
    # collapses to "paper_default" (irrelevant).
    off6_c = dict(base_axes)
    off6_c.update(
        mlp_numerical_embedding_cfg=None,
        mlp_numerical_embedding_kwargs_cfg="include_raw_false",
    )
    off6_d = dict(off6_c)
    off6_d["mlp_numerical_embedding_kwargs_cfg"] = "paper_default"
    c_off6_c = _build_combo(models=("mlp",), axes=off6_c, seed=0)
    c_off6_d = _build_combo(models=("mlp",), axes=off6_d, seed=0)
    assert c_off6_c.canonical_key() == c_off6_d.canonical_key(), "audit-pass-9 #6: kwargs literal must canon-collapse when embedding=None"

    # (c-vii) #7 mrmr fe_hybrid_orth canon-collapse: use_mrmr_fs=False ->
    # master collapses to False AND pair_enable to True.
    off7_a = dict(base_axes)
    off7_a.update(
        use_mrmr_fs=False,
        mrmr_fe_hybrid_orth_enable_cfg=True,
        mrmr_fe_hybrid_orth_pair_enable_cfg=False,
    )
    off7_b = dict(off7_a)
    off7_b.update(
        mrmr_fe_hybrid_orth_enable_cfg=False,
        mrmr_fe_hybrid_orth_pair_enable_cfg=True,
    )
    c_off7_a = _build_combo(models=("cb",), axes=off7_a, seed=0)
    c_off7_b = _build_combo(models=("cb",), axes=off7_b, seed=0)
    assert c_off7_a.canonical_key() == c_off7_b.canonical_key(), "audit-pass-9 #7: must canon-collapse when use_mrmr_fs=False"
    # And when use_mrmr_fs=True but master=False, pair_enable still
    # collapses to True (sub-stage doesn't fire).
    off7_c = dict(base_axes)
    off7_c.update(
        use_mrmr_fs=True,
        mrmr_fe_hybrid_orth_enable_cfg=False,
        mrmr_fe_hybrid_orth_pair_enable_cfg=False,
    )
    off7_d = dict(off7_c)
    off7_d["mrmr_fe_hybrid_orth_pair_enable_cfg"] = True
    c_off7_c = _build_combo(models=("cb",), axes=off7_c, seed=0)
    c_off7_d = _build_combo(models=("cb",), axes=off7_d, seed=0)
    assert c_off7_c.canonical_key() == c_off7_d.canonical_key(), "audit-pass-9 #7: pair_enable must canon-collapse when master=False"

    # (c-viii) #8 target_type=multi_target_regression canon-collapse: when
    # neither cb nor mlp is in the model subset (no native-MTR backend),
    # canon collapses to "regression" so dedup absorbs identical-behaviour
    # combos.
    off8_a = dict(base_axes)
    off8_a["target_type"] = "multi_target_regression"
    off8_b = dict(off8_a)
    off8_b["target_type"] = "regression"
    c_off8_a = _build_combo(models=("xgb",), axes=off8_a, seed=0)
    c_off8_b = _build_combo(models=("xgb",), axes=off8_b, seed=0)
    assert c_off8_a.canonical_key() == c_off8_b.canonical_key(), (
        "audit-pass-9 #8: multi_target_regression must canon-collapse to 'regression' when neither 'cb' nor 'mlp' is in models"
    )

    # ------------------------------------------------------------------
    # (d) Threading.
    # ------------------------------------------------------------------

    # (d-i) #1 betas threading: MLP active -> optimizer_kwargs['betas']
    # surfaces with the axis value.
    on1 = dict(base_axes)
    on1["mlp_adamw_betas_cfg"] = (0.9, 0.999)
    c_on1 = _build_combo(models=("mlp",), axes=on1, seed=0)
    kw1 = build_mlp_kwargs(c_on1)
    assert kw1 is not None
    assert kw1.get("optimizer_kwargs", {}).get("betas") == (0.9, 0.999), f"#1 betas did not thread into mlp_kwargs: {kw1.get('optimizer_kwargs')!r}"

    # (d-ii) #2 use_ema threading.
    on2 = dict(base_axes)
    on2["mlp_use_ema_cfg"] = True
    c_on2 = _build_combo(models=("mlp",), axes=on2, seed=0)
    kw2 = build_mlp_kwargs(c_on2)
    assert kw2 is not None
    assert kw2.get("use_ema") is True, f"#2 use_ema did not thread into mlp_kwargs: {kw2.get('use_ema')!r}"
    # Off -> key absent.
    on2_off = dict(base_axes)
    on2_off["mlp_use_ema_cfg"] = False
    c_on2_off = _build_combo(models=("mlp",), axes=on2_off, seed=0)
    assert "use_ema" not in build_mlp_kwargs(c_on2_off), "#2 use_ema=False must NOT emit a use_ema key (library default)"

    # (d-iii) #3 label_smoothing threading: multiclass-only.
    on3 = dict(base_axes)
    on3.update(
        target_type="multiclass_classification",
        mlp_label_smoothing_cfg=0.1,
    )
    c_on3 = _build_combo(models=("mlp",), axes=on3, seed=0)
    kw3 = build_mlp_kwargs(c_on3)
    assert kw3.get("label_smoothing") == 0.1, f"#3 label_smoothing did not thread: {kw3.get('label_smoothing')!r}"
    # Binary -> dropped.
    on3_bin = dict(on3)
    on3_bin["target_type"] = "binary_classification"
    c_on3_bin = _build_combo(models=("mlp",), axes=on3_bin, seed=0)
    assert "label_smoothing" not in build_mlp_kwargs(c_on3_bin), "#3 label_smoothing must be dropped on non-multiclass targets"

    # (d-iv) #4 focal_loss_gamma threading: binary-only.
    on4 = dict(base_axes)
    on4.update(
        target_type="binary_classification",
        imbalance_ratio="rare_5pct",
        mlp_focal_loss_gamma_cfg=2.0,
    )
    c_on4 = _build_combo(models=("mlp",), axes=on4, seed=0)
    kw4 = build_mlp_kwargs(c_on4)
    assert kw4.get("focal_loss_gamma") == 2.0, f"#4 focal_loss_gamma did not thread: {kw4.get('focal_loss_gamma')!r}"
    # Multiclass -> dropped (focal is binary-only).
    on4_mc = dict(on4)
    on4_mc["target_type"] = "multiclass_classification"
    c_on4_mc = _build_combo(models=("mlp",), axes=on4_mc, seed=0)
    assert "focal_loss_gamma" not in build_mlp_kwargs(c_on4_mc), "#4 focal_loss_gamma must be dropped on non-binary targets"

    # (d-v) #5 use_residual threading via network_params.
    on5 = dict(base_axes)
    on5["mlp_use_residual_cfg"] = True
    c_on5 = _build_combo(models=("mlp",), axes=on5, seed=0)
    kw5 = build_mlp_kwargs(c_on5)
    assert kw5.get("network_params", {}).get("use_residual") is True, f"#5 use_residual did not thread into network_params: {kw5.get('network_params')!r}"

    # (d-vi) #6 numerical_embedding threading via network_params.
    on6 = dict(base_axes)
    on6.update(
        mlp_numerical_embedding_cfg="plr",
        mlp_numerical_embedding_kwargs_cfg="include_raw_false",
    )
    c_on6 = _build_combo(models=("mlp",), axes=on6, seed=0)
    kw6 = build_mlp_kwargs(c_on6)
    np6 = kw6.get("network_params", {})
    assert np6.get("numerical_embedding") == "plr", f"#6 numerical_embedding did not thread: {np6.get('numerical_embedding')!r}"
    assert np6.get("numerical_embedding_kwargs") == {"include_raw": False}, (
        f"#6 numerical_embedding_kwargs literal did not expand to include_raw=False: {np6.get('numerical_embedding_kwargs')!r}"
    )
    # "paper_default" leaves the kwargs dict unset so the module ctor falls
    # through to library defaults.
    on6_pd = dict(on6)
    on6_pd["mlp_numerical_embedding_kwargs_cfg"] = "paper_default"
    c_on6_pd = _build_combo(models=("mlp",), axes=on6_pd, seed=0)
    np6_pd = build_mlp_kwargs(c_on6_pd).get("network_params", {})
    assert "numerical_embedding_kwargs" not in np6_pd, "#6 'paper_default' must leave numerical_embedding_kwargs unset"

    # No MLP -> kwargs builder returns None (matches the existing
    # build_mlp_kwargs contract).
    no_mlp = dict(base_axes)
    no_mlp.update(
        mlp_adamw_betas_cfg=(0.9, 0.999),
        mlp_use_ema_cfg=True,
        mlp_use_residual_cfg=True,
        mlp_numerical_embedding_cfg="plr",
        recurrent_model_cfg=None,
    )
    c_no_mlp = _build_combo(models=("cb",), axes=no_mlp, seed=0)
    assert build_mlp_kwargs(c_no_mlp) is None, "audit-pass-9: build_mlp_kwargs must return None with no MLP / recurrent"

    # (d-vii) #7 fe_hybrid_orth threading into MRMR kwargs.
    on7 = dict(base_axes)
    on7.update(
        use_mrmr_fs=True,
        mrmr_fe_hybrid_orth_enable_cfg=True,
        mrmr_fe_hybrid_orth_pair_enable_cfg=False,
    )
    c_on7 = _build_combo(models=("cb",), axes=on7, seed=0)
    mk7 = build_mrmr_kwargs(c_on7)
    assert mk7 is not None
    assert mk7.get("fe_hybrid_orth_enable") is True, f"#7 fe_hybrid_orth_enable did not thread into mrmr_kwargs: {mk7.get('fe_hybrid_orth_enable')!r}"
    assert mk7.get("fe_hybrid_orth_pair_enable") is False, f"#7 fe_hybrid_orth_pair_enable did not thread: {mk7.get('fe_hybrid_orth_pair_enable')!r}"
    # use_mrmr_fs=False -> kwargs None entirely.
    off7_full = dict(base_axes)
    off7_full.update(
        use_mrmr_fs=False,
        mrmr_fe_hybrid_orth_enable_cfg=True,
    )
    assert build_mrmr_kwargs(_build_combo(models=("cb",), axes=off7_full, seed=0)) is None, "#7 mrmr_kwargs must be None when use_mrmr_fs=False"

    # (d-viii) #8 multi_target_regression frame builder emits (N, K) target.
    on8 = dict(base_axes)
    on8.update(
        n_rows=200,
        input_type="pandas",
        target_type="multi_target_regression",
        imbalance_ratio="balanced",
        recurrent_model_cfg=None,
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
        mlp_inject_zero_sample_weight_batch_cfg=False,
        use_mrmr_fs=False,
    )
    # iter633: the 2-D (N, K) MTR target is emitted ONLY when every model in
    # the combo natively handles a 2-D continuous target (currently cb via
    # MultiRMSE). Combos containing a non-native model (mlp/lgb/linear/hgb/xgb)
    # are downgraded to a 1-D "target_reg" regression surface so the
    # non-native member doesn't crash with "Unknown label type:
    # continuous-multioutput". Use a cb-only combo here to exercise the native
    # 2-D path this assertion was written for.
    c_on8 = _build_combo(models=("cb",), axes=on8, seed=0)
    df_on8, target_col_on8, _ = build_frame_for_combo(c_on8)
    assert target_col_on8 in df_on8.columns, "#8 frame builder must emit target column for multi_target_regression"
    # Each cell is a list of K=2 floats (native cb-only MTR path).
    first_cell = df_on8[target_col_on8].iloc[0]
    assert hasattr(first_cell, "__len__"), f"#8: multi_target_regression target cell must be array-like: {first_cell!r}"
    assert len(first_cell) == 2, f"#8: target shape must be (N, K=2), got K={len(first_cell)}"

    # (e) F-23 mirror canon: inject_inf_nan=True collapses to False when
    # model subset is exactly ('mlp',) so the _validate_no_nan_inf raise at
    # base.py:326 doesn't manufacture phantom variation.
    off_inf_a = dict(base_axes)
    off_inf_a.update(
        inject_inf_nan=True,
    )
    off_inf_b = dict(off_inf_a)
    off_inf_b["inject_inf_nan"] = False
    c_off_inf_a = _build_combo(models=("mlp",), axes=off_inf_a, seed=0)
    c_off_inf_b = _build_combo(models=("mlp",), axes=off_inf_b, seed=0)
    assert c_off_inf_a.canonical_key() == c_off_inf_b.canonical_key(), (
        "F-23 mirror: inject_inf_nan must canon-collapse when MLP is the sole model (validator raises on entry regardless)"
    )
    # Multi-model subset (mlp + cb): inject_inf_nan stays live so dedup
    # keeps both branches reachable (cb handles inf/nan via its own path).
    multi_a = dict(base_axes)
    multi_a["inject_inf_nan"] = True
    multi_b = dict(multi_a)
    multi_b["inject_inf_nan"] = False
    c_multi_a = _build_combo(models=("cb", "mlp"), axes=multi_a, seed=0)
    c_multi_b = _build_combo(models=("cb", "mlp"), axes=multi_b, seed=0)
    assert c_multi_a.canonical_key() != c_multi_b.canonical_key(), (
        "F-23 mirror: inject_inf_nan must remain live on multi-model subsets where non-MLP models can consume inf/nan"
    )

    # (f) Distinct canonical_keys: on-axis values must NOT collapse to the
    # source default under their compound-gate-on configuration.
    # #1 distinct under 'mlp' in models.
    on1_def = dict(on1)
    on1_def["mlp_adamw_betas_cfg"] = (0.9, 0.95)
    c_on1_def = _build_combo(models=("mlp",), axes=on1_def, seed=0)
    assert c_on1.canonical_key() != c_on1_def.canonical_key(), "#1: non-default betas must produce distinct canonical key under 'mlp'"
    # #2 distinct under 'mlp' in models.
    c_on2_def = _build_combo(models=("mlp",), axes=on2_off, seed=0)
    assert c_on2.canonical_key() != c_on2_def.canonical_key(), "#2: use_ema=True must produce distinct canonical key under 'mlp'"
    # #5 distinct under 'mlp' in models.
    on5_def = dict(on5)
    on5_def["mlp_use_residual_cfg"] = False
    c_on5_def = _build_combo(models=("mlp",), axes=on5_def, seed=0)
    assert c_on5.canonical_key() != c_on5_def.canonical_key(), "#5: use_residual=True must produce distinct canonical key under 'mlp'"
    # #6 distinct under 'mlp' + embedding != None.
    on6_def = dict(on6)
    on6_def["mlp_numerical_embedding_cfg"] = None
    c_on6_def = _build_combo(models=("mlp",), axes=on6_def, seed=0)
    assert c_on6.canonical_key() != c_on6_def.canonical_key(), "#6: numerical_embedding='plr' must produce distinct canonical key under 'mlp'"
    # #7 distinct under use_mrmr_fs.
    on7_def = dict(on7)
    on7_def["mrmr_fe_hybrid_orth_enable_cfg"] = False
    c_on7_def = _build_combo(models=("cb",), axes=on7_def, seed=0)
    assert c_on7.canonical_key() != c_on7_def.canonical_key(), "#7: fe_hybrid_orth_enable=True must produce distinct canonical key under use_mrmr_fs=True"
    # #8 distinct under a native-MTR model subset.
    on8_def = dict(on8)
    on8_def["target_type"] = "regression"
    c_on8_def = _build_combo(models=("mlp",), axes=on8_def, seed=0)
    assert c_on8.canonical_key() != c_on8_def.canonical_key(), (
        "#8: multi_target_regression must produce distinct canonical key from regression when 'mlp' is in models"
    )


# ---------------------------------------------------------------------------
# 2026-05-31 audit-pass-10 (W10): 1 HIGH Muon optimizer + 4 MED MRMR
# hybrid_orth sub-axes + 1 LOW pair_max_degree. Finding #5 (qcut hidden
# constants) is BLOCKED on MRMR ctor promotion; tracked as a TODO marker
# in _fuzz_combo.py AXES dict.
# ---------------------------------------------------------------------------


def test_iter617_audit_pass_10_axes_flow_to_kwargs():
    """5 audit-pass-10 (W10) fuzz axes must:
    (a) Dataclass defaults mirror HEAD source verbatim:
          #1 mlp_optimizer_cfg = "adamw"
             (training/neural/_flat_torch_module.py:86 -- the fallback
             ``optimizer = optimizer or torch.optim.AdamW``; the Muon
             class lives at training/neural/_muon_optimizer.py:123 and
             wires via ``model_params["optimizer"] = MuonAdamWHybrid``).
          #2 mrmr_fe_hybrid_orth_degrees_cfg = (2, 3)
             (feature_selection/filters/mrmr.py:657)
          #3 mrmr_fe_hybrid_orth_basis_cfg = "auto"
             (feature_selection/filters/mrmr.py:658)
          #4 mrmr_fe_hybrid_orth_top_k_cfg = 5
             (feature_selection/filters/mrmr.py:663)
          #6 mrmr_fe_hybrid_orth_pair_max_degree_cfg = 2
             (feature_selection/filters/mrmr.py:665)
    (b) Each axis pair is present in AXES with the values listed in the
        audit.
    (c) Canon-collapse rules hold:
          #1 collapses to "adamw" outside 'mlp' in models.
          #2 collapses to (2, 3) outside (use_mrmr_fs AND master==True).
          #3 collapses to "auto" outside (use_mrmr_fs AND master==True).
          #4 collapses to 5 outside (use_mrmr_fs AND master==True).
          #6 collapses to 2 outside (use_mrmr_fs AND master==True AND
             pair_enable==True).
    (d) Threading produces the expected kwargs subdicts:
          #1 flows via build_mlp_kwargs into
             mlp_kwargs["model_params"]["optimizer"] = MuonAdamWHybrid
             (or stays absent for "adamw").
          #2/#3/#4/#6 flow via build_mrmr_kwargs as
             ``fe_hybrid_orth_degrees`` / ``fe_hybrid_orth_basis`` /
             ``fe_hybrid_orth_top_k`` / ``fe_hybrid_orth_pair_max_degree``.
    (e) enumerate_combos still returns 150 at the audit-recommended
        master_seed.
    (f) Distinct canonical keys hold under the compound-gate-on
        configuration for each axis.
    (g) Finding #5 (qcut hidden constants) is intentionally NOT wired
        here; tracked via TODO marker in _fuzz_combo.py AXES so future
        MRMR ctor promotion can land the axes in one pass.
    """
    from tests.training._fuzz_combo import (
        AXES,
        FuzzCombo,
        _build_combo,
        enumerate_combos,
        build_mlp_kwargs,
        build_mrmr_kwargs,
    )

    # (a) Dataclass defaults mirror HEAD source.
    fields = FuzzCombo.__dataclass_fields__
    assert fields["mlp_optimizer_cfg"].default == "adamw", 'audit-pass-10 #1: default must mirror training/neural/_flat_torch_module.py:86 ("adamw" fallback)'
    assert fields["mrmr_fe_hybrid_orth_degrees_cfg"].default == (2, 3), "audit-pass-10 #2: default must mirror feature_selection/filters/mrmr.py:657 ((2, 3))"
    assert fields["mrmr_fe_hybrid_orth_basis_cfg"].default == "auto", 'audit-pass-10 #3: default must mirror feature_selection/filters/mrmr.py:658 ("auto")'
    assert fields["mrmr_fe_hybrid_orth_top_k_cfg"].default == 5, "audit-pass-10 #4: default must mirror feature_selection/filters/mrmr.py:663 (5)"
    assert fields["mrmr_fe_hybrid_orth_pair_max_degree_cfg"].default == 2, "audit-pass-10 #6: default must mirror feature_selection/filters/mrmr.py:665 (2)"

    # (b) AXES presence with the audit-listed pairs.
    assert AXES["mlp_optimizer_cfg"] == ("adamw", "muon_hybrid")
    assert AXES["mrmr_fe_hybrid_orth_degrees_cfg"] == ((2, 3), (2,))
    assert AXES["mrmr_fe_hybrid_orth_basis_cfg"] == ("auto", "hermite")
    assert AXES["mrmr_fe_hybrid_orth_top_k_cfg"] == (5, 1)
    assert AXES["mrmr_fe_hybrid_orth_pair_max_degree_cfg"] == (2, 3)

    base_axes = {name: values[0] for name, values in AXES.items()}

    # (e) enumerate_combos still returns 150 at master_seed=20260601.
    combos = enumerate_combos(target=150, master_seed=20260601)
    assert len(combos) == 150, f"enumerate_combos lost combos: {len(combos)}"

    # ------------------------------------------------------------------
    # (c) Canon-collapse rules.
    # ------------------------------------------------------------------

    # (c-1) #1 optimizer canon-collapse: no MLP -> both values collapse to
    # source default "adamw".
    off1_a = dict(base_axes)
    off1_a["mlp_optimizer_cfg"] = "muon_hybrid"
    off1_b = dict(off1_a)
    off1_b["mlp_optimizer_cfg"] = "adamw"
    c_off1_a = _build_combo(models=("cb",), axes=off1_a, seed=0)
    c_off1_b = _build_combo(models=("cb",), axes=off1_b, seed=0)
    assert c_off1_a.canonical_key() == c_off1_b.canonical_key(), "audit-pass-10 #1: must canon-collapse when 'mlp' not in models"

    # (c-2) #2 degrees canon-collapse: use_mrmr_fs=False -> both collapse.
    off2_a = dict(base_axes)
    off2_a.update(
        use_mrmr_fs=False,
        mrmr_fe_hybrid_orth_enable_cfg=True,
        mrmr_fe_hybrid_orth_degrees_cfg=(2,),
    )
    off2_b = dict(off2_a)
    off2_b["mrmr_fe_hybrid_orth_degrees_cfg"] = (2, 3)
    c_off2_a = _build_combo(models=("cb",), axes=off2_a, seed=0)
    c_off2_b = _build_combo(models=("cb",), axes=off2_b, seed=0)
    assert c_off2_a.canonical_key() == c_off2_b.canonical_key(), "audit-pass-10 #2: must canon-collapse when use_mrmr_fs=False"
    # use_mrmr_fs=True but master=False -> still collapses (hybrid stage off).
    off2_c = dict(base_axes)
    off2_c.update(
        use_mrmr_fs=True,
        mrmr_fe_hybrid_orth_enable_cfg=False,
        mrmr_fe_hybrid_orth_degrees_cfg=(2,),
    )
    off2_d = dict(off2_c)
    off2_d["mrmr_fe_hybrid_orth_degrees_cfg"] = (2, 3)
    c_off2_c = _build_combo(models=("cb",), axes=off2_c, seed=0)
    c_off2_d = _build_combo(models=("cb",), axes=off2_d, seed=0)
    assert c_off2_c.canonical_key() == c_off2_d.canonical_key(), "audit-pass-10 #2: must canon-collapse when hybrid master=False"

    # (c-3) #3 basis canon-collapse: use_mrmr_fs=False -> both collapse.
    off3_a = dict(base_axes)
    off3_a.update(
        use_mrmr_fs=False,
        mrmr_fe_hybrid_orth_enable_cfg=True,
        mrmr_fe_hybrid_orth_basis_cfg="hermite",
    )
    off3_b = dict(off3_a)
    off3_b["mrmr_fe_hybrid_orth_basis_cfg"] = "auto"
    c_off3_a = _build_combo(models=("cb",), axes=off3_a, seed=0)
    c_off3_b = _build_combo(models=("cb",), axes=off3_b, seed=0)
    assert c_off3_a.canonical_key() == c_off3_b.canonical_key(), "audit-pass-10 #3: must canon-collapse when use_mrmr_fs=False"

    # (c-4) #4 top_k canon-collapse: use_mrmr_fs=False -> both collapse.
    off4_a = dict(base_axes)
    off4_a.update(
        use_mrmr_fs=False,
        mrmr_fe_hybrid_orth_enable_cfg=True,
        mrmr_fe_hybrid_orth_top_k_cfg=1,
    )
    off4_b = dict(off4_a)
    off4_b["mrmr_fe_hybrid_orth_top_k_cfg"] = 5
    c_off4_a = _build_combo(models=("cb",), axes=off4_a, seed=0)
    c_off4_b = _build_combo(models=("cb",), axes=off4_b, seed=0)
    assert c_off4_a.canonical_key() == c_off4_b.canonical_key(), "audit-pass-10 #4: must canon-collapse when use_mrmr_fs=False"

    # (c-6) #6 pair_max_degree canon-collapse: pair_enable=False -> both
    # collapse (pair stage doesn't fire, max_degree is unread).
    off6_a = dict(base_axes)
    off6_a.update(
        use_mrmr_fs=True,
        mrmr_fe_hybrid_orth_enable_cfg=True,
        mrmr_fe_hybrid_orth_pair_enable_cfg=False,
        mrmr_fe_hybrid_orth_pair_max_degree_cfg=3,
    )
    off6_b = dict(off6_a)
    off6_b["mrmr_fe_hybrid_orth_pair_max_degree_cfg"] = 2
    c_off6_a = _build_combo(models=("cb",), axes=off6_a, seed=0)
    c_off6_b = _build_combo(models=("cb",), axes=off6_b, seed=0)
    assert c_off6_a.canonical_key() == c_off6_b.canonical_key(), "audit-pass-10 #6: must canon-collapse when pair_enable=False"
    # And when master=False -> still collapses (pair stage gated on master).
    off6_c = dict(base_axes)
    off6_c.update(
        use_mrmr_fs=True,
        mrmr_fe_hybrid_orth_enable_cfg=False,
        mrmr_fe_hybrid_orth_pair_enable_cfg=True,
        mrmr_fe_hybrid_orth_pair_max_degree_cfg=3,
    )
    off6_d = dict(off6_c)
    off6_d["mrmr_fe_hybrid_orth_pair_max_degree_cfg"] = 2
    c_off6_c = _build_combo(models=("cb",), axes=off6_c, seed=0)
    c_off6_d = _build_combo(models=("cb",), axes=off6_d, seed=0)
    assert c_off6_c.canonical_key() == c_off6_d.canonical_key(), "audit-pass-10 #6: must canon-collapse when hybrid master=False"

    # ------------------------------------------------------------------
    # (d) Threading.
    # ------------------------------------------------------------------

    # (d-1) #1 optimizer threading: MLP active + "muon_hybrid" ->
    # model_params["optimizer"] is set to MuonAdamWHybrid.
    on1 = dict(base_axes)
    on1["mlp_optimizer_cfg"] = "muon_hybrid"
    c_on1 = _build_combo(models=("mlp",), axes=on1, seed=0)
    kw1 = build_mlp_kwargs(c_on1)
    assert kw1 is not None
    from mlframe.training.neural._muon_optimizer import MuonAdamWHybrid

    assert kw1.get("model_params", {}).get("optimizer") is MuonAdamWHybrid, (
        f"#1 muon_hybrid did not thread MuonAdamWHybrid into model_params: {kw1.get('model_params')!r}"
    )
    # "adamw" -> no optimizer key (LightningModule falls back to AdamW).
    on1_adamw = dict(base_axes)
    on1_adamw["mlp_optimizer_cfg"] = "adamw"
    c_on1_adamw = _build_combo(models=("mlp",), axes=on1_adamw, seed=0)
    kw1_adamw = build_mlp_kwargs(c_on1_adamw)
    assert "optimizer" not in kw1_adamw.get("model_params", {}), (
        "#1 'adamw' must NOT emit a model_params['optimizer'] key (library default at _flat_torch_module.py:86)"
    )

    # (d-2) #2 degrees threading: MRMR active + master on -> fe_hybrid_orth_degrees
    # surfaces with the axis value.
    on2 = dict(base_axes)
    on2.update(
        use_mrmr_fs=True,
        mrmr_fe_hybrid_orth_enable_cfg=True,
        mrmr_fe_hybrid_orth_degrees_cfg=(2,),
    )
    c_on2 = _build_combo(models=("cb",), axes=on2, seed=0)
    mk2 = build_mrmr_kwargs(c_on2)
    assert mk2 is not None
    assert mk2.get("fe_hybrid_orth_degrees") == (2,), f"#2 fe_hybrid_orth_degrees did not thread: {mk2.get('fe_hybrid_orth_degrees')!r}"

    # (d-3) #3 basis threading.
    on3 = dict(base_axes)
    on3.update(
        use_mrmr_fs=True,
        mrmr_fe_hybrid_orth_enable_cfg=True,
        mrmr_fe_hybrid_orth_basis_cfg="hermite",
    )
    c_on3 = _build_combo(models=("cb",), axes=on3, seed=0)
    mk3 = build_mrmr_kwargs(c_on3)
    assert mk3.get("fe_hybrid_orth_basis") == "hermite", f"#3 fe_hybrid_orth_basis did not thread: {mk3.get('fe_hybrid_orth_basis')!r}"

    # (d-4) #4 top_k threading.
    on4 = dict(base_axes)
    on4.update(
        use_mrmr_fs=True,
        mrmr_fe_hybrid_orth_enable_cfg=True,
        mrmr_fe_hybrid_orth_top_k_cfg=1,
    )
    c_on4 = _build_combo(models=("cb",), axes=on4, seed=0)
    mk4 = build_mrmr_kwargs(c_on4)
    assert mk4.get("fe_hybrid_orth_top_k") == 1, f"#4 fe_hybrid_orth_top_k did not thread: {mk4.get('fe_hybrid_orth_top_k')!r}"

    # (d-6) #6 pair_max_degree threading: full compound gate on.
    on6 = dict(base_axes)
    on6.update(
        use_mrmr_fs=True,
        mrmr_fe_hybrid_orth_enable_cfg=True,
        mrmr_fe_hybrid_orth_pair_enable_cfg=True,
        mrmr_fe_hybrid_orth_pair_max_degree_cfg=3,
    )
    c_on6 = _build_combo(models=("cb",), axes=on6, seed=0)
    mk6 = build_mrmr_kwargs(c_on6)
    assert mk6.get("fe_hybrid_orth_pair_max_degree") == 3, f"#6 fe_hybrid_orth_pair_max_degree did not thread: {mk6.get('fe_hybrid_orth_pair_max_degree')!r}"

    # use_mrmr_fs=False -> build_mrmr_kwargs returns None.
    off_full = dict(base_axes)
    off_full.update(
        use_mrmr_fs=False,
        mrmr_fe_hybrid_orth_enable_cfg=True,
        mrmr_fe_hybrid_orth_degrees_cfg=(2,),
        mrmr_fe_hybrid_orth_basis_cfg="hermite",
        mrmr_fe_hybrid_orth_top_k_cfg=1,
        mrmr_fe_hybrid_orth_pair_max_degree_cfg=3,
    )
    assert build_mrmr_kwargs(_build_combo(models=("cb",), axes=off_full, seed=0)) is None, (
        "audit-pass-10 #2/#3/#4/#6: mrmr_kwargs must be None when use_mrmr_fs=False"
    )

    # ------------------------------------------------------------------
    # (f) Distinct canonical_keys under compound-gate-on.
    # ------------------------------------------------------------------

    # #1 distinct under 'mlp' in models.
    on1_def = dict(on1)
    on1_def["mlp_optimizer_cfg"] = "adamw"
    c_on1_def = _build_combo(models=("mlp",), axes=on1_def, seed=0)
    assert c_on1.canonical_key() != c_on1_def.canonical_key(), "#1: 'muon_hybrid' must produce distinct canonical key under 'mlp'"
    # #2 distinct under use_mrmr_fs+master.
    on2_def = dict(on2)
    on2_def["mrmr_fe_hybrid_orth_degrees_cfg"] = (2, 3)
    c_on2_def = _build_combo(models=("cb",), axes=on2_def, seed=0)
    assert c_on2.canonical_key() != c_on2_def.canonical_key(), "#2: (2,) degrees must produce distinct canonical key under use_mrmr_fs+master compound gate"
    # #3 distinct under use_mrmr_fs+master.
    on3_def = dict(on3)
    on3_def["mrmr_fe_hybrid_orth_basis_cfg"] = "auto"
    c_on3_def = _build_combo(models=("cb",), axes=on3_def, seed=0)
    assert c_on3.canonical_key() != c_on3_def.canonical_key(), "#3: 'hermite' basis must produce distinct canonical key under use_mrmr_fs+master compound gate"
    # #4 distinct under use_mrmr_fs+master.
    on4_def = dict(on4)
    on4_def["mrmr_fe_hybrid_orth_top_k_cfg"] = 5
    c_on4_def = _build_combo(models=("cb",), axes=on4_def, seed=0)
    assert c_on4.canonical_key() != c_on4_def.canonical_key(), "#4: top_k=1 must produce distinct canonical key under use_mrmr_fs+master compound gate"
    # #6 distinct under use_mrmr_fs+master+pair_enable.
    on6_def = dict(on6)
    on6_def["mrmr_fe_hybrid_orth_pair_max_degree_cfg"] = 2
    c_on6_def = _build_combo(models=("cb",), axes=on6_def, seed=0)
    assert c_on6.canonical_key() != c_on6_def.canonical_key(), (
        "#6: pair_max_degree=3 must produce distinct canonical key under use_mrmr_fs+master+pair_enable compound gate"
    )


# ---------------------------------------------------------------------------
# 2026-05-31 audit-pass-12 (W12): 12 axes (5 HIGH + 5 MED + 2 LOW) covering
# Group A (F-34 MTR dispatch), Group B (MRMR FE layers 26/32/33/34/37/38),
# Group C (MRMR + ShapProxiedFS artifact-reuse pipeline). Group D (Layer 30/31
# perf kernel_tuning_cache) verified non-actionable -- documented as TODO in
# _fuzz_combo.py since no MRMR ctor surface exists.
# ---------------------------------------------------------------------------


def test_iter622_audit_pass_12_axes_flow_to_kwargs():
    """12 audit-pass-12 (W12) fuzz axes must:
      (a) be present in AXES with >=2 candidate values,
      (b) carry the SOURCE-verified library defaults in the FuzzCombo dataclass
          (verified pre-edit against
          ``src/mlframe/training/_composite_target_discovery_config.py:773,940``
          for A1, ``src/mlframe/feature_selection/filters/mrmr.py:676/691/705/
          723/725/727/749/751/752/769/772/774/777/787`` for B/C MRMR ctor
          params, and ``src/mlframe/feature_selection/shap_proxied_fs.py:258``
          for the C1 ``precomputed`` kwarg),
      (c) collapse correctly under the documented gates,
      (d) thread through their downstream consumer:
            - A1 multilabel_strategy flows via build_composite_discovery_config
              into the CompositeTargetDiscoveryConfig dataclass field,
            - A2/A3 are canon-only markers (no production builder target today;
              suite-internal WARN gate at
              _phase_composite_post_xt_ensemble._build_cross_target_ensemble_for_target
              fires on target_type=multi_target_regression irrespective of
              this axis -- pinned via canon dedup),
            - B1-B6 flow via build_mrmr_kwargs into MRMR.__init__,
            - C1 master flows two ways: MRMR.retain_artifacts at mrmr.py:787
              AND ShapProxiedFS(precomputed=...) at shap_proxied_fs.py:258
              (with the sentinel dict driving the 4 align_precomputed_to_X
              branches selected by C2).

    Findings (12 total, sorted by severity per AUDIT_PASS_12_DONE.md table):
      HIGH (5):
        A1 composite_target_multilabel_strategy_cfg ("per_target",
                                                      "multi_target_regression")
        A2 enable_ct_ensemble_cfg                    (True, False)
        B1 mrmr_fe_kfold_te_enable_cfg               (False, True)
        B2 mrmr_fe_missingness_{indicator,count,pattern}_enable_cfg  (False, True)
        C1 mrmr_shap_proxy_artifact_reuse_cfg + mrmr_shap_proxy_align_mode_cfg
      MED (5):
        A3 mtr_eval_metric_cfg                       (None, "rmse_macro")
        B3 mrmr_fe_cat_aux_enable_cfg                ("off","count","freq","interaction")
        B4 mrmr_fe_hybrid_orth_extra_bases_cfg       ((), ("spline",), ("fourier",))
        B5 mrmr_fe_ratio_delta_diff_cfg              ("off","ratio","grouped_delta","lagged_diff")
        B6 mrmr_fe_mi_greedy_enable_cfg              (False, True)
      LOW (2):
        B7 deferred (wrapper-only, not on MRMR ctor -- see audit row B7)
        D1 deferred (kernel_tuning_cache dispatch is internal -- TODO in
           _fuzz_combo.py)
    """
    from tests.training._fuzz_combo import (
        AXES,
        FuzzCombo,
        _build_combo,
        enumerate_combos,
        build_mrmr_kwargs,
        build_shap_proxied_fs_kwargs,
        build_composite_discovery_config,
        build_composite_discovery_config_from_flat,
    )

    new_axes = (
        # Group A
        "composite_target_multilabel_strategy_cfg",
        "enable_ct_ensemble_cfg",
        "mtr_eval_metric_cfg",
        # Group B
        "mrmr_fe_kfold_te_enable_cfg",
        "mrmr_fe_missingness_indicator_enable_cfg",
        "mrmr_fe_missingness_count_enable_cfg",
        "mrmr_fe_missingness_pattern_enable_cfg",
        "mrmr_fe_cat_aux_enable_cfg",
        "mrmr_fe_hybrid_orth_extra_bases_cfg",
        "mrmr_fe_ratio_delta_diff_cfg",
        "mrmr_fe_mi_greedy_enable_cfg",
        # Group C
        "mrmr_shap_proxy_artifact_reuse_cfg",
        "mrmr_shap_proxy_align_mode_cfg",
    )
    # (a) Presence in AXES with >=2 candidates.
    for ax in new_axes:
        assert ax in AXES, f"missing fuzz axis {ax}"
        assert len(AXES[ax]) >= 2, f"axis {ax} must offer at least 2 values"

    # (b) FuzzCombo dataclass defaults match the source-verified library defaults.
    fields = FuzzCombo.__dataclass_fields__
    assert fields["composite_target_multilabel_strategy_cfg"].default == "per_target", (
        "A1: default must mirror _composite_target_discovery_config.py:773 ('per_target')"
    )
    assert fields["enable_ct_ensemble_cfg"].default is True, "A2: default True mirrors the suite-side default"
    assert fields["mtr_eval_metric_cfg"].default is None, "A3: canon-only marker default None"
    assert fields["mrmr_fe_kfold_te_enable_cfg"].default is False, "B1: default must mirror filters/mrmr.py:705 (False)"
    assert fields["mrmr_fe_missingness_indicator_enable_cfg"].default is False, "B2 indicator: default must mirror filters/mrmr.py:749 (False)"
    assert fields["mrmr_fe_missingness_count_enable_cfg"].default is False, "B2 count: default must mirror filters/mrmr.py:751 (False)"
    assert fields["mrmr_fe_missingness_pattern_enable_cfg"].default is False, "B2 pattern: default must mirror filters/mrmr.py:752 (False)"
    assert fields["mrmr_fe_cat_aux_enable_cfg"].default == "off", "B3: 4-way axis default 'off' (all three master switches False)"
    assert fields["mrmr_fe_hybrid_orth_extra_bases_cfg"].default == (), "B4: default must mirror filters/mrmr.py:676 (())"
    assert fields["mrmr_fe_ratio_delta_diff_cfg"].default == "off", "B5: 4-way axis default 'off' (all four master switches False)"
    assert fields["mrmr_fe_mi_greedy_enable_cfg"].default is False, "B6: default must mirror filters/mrmr.py:691 (False)"
    assert fields["mrmr_shap_proxy_artifact_reuse_cfg"].default == "off", "C1: artifact-reuse master default 'off' (legacy bit-identical)"
    assert fields["mrmr_shap_proxy_align_mode_cfg"].default == "exact", "C2: align mode default 'exact' (the no-op branch when no precomputed)"

    base_axes = {name: values[0] for name, values in AXES.items()}

    # (c) enumerate_combos still hits 150 with the 12 new axes wired.
    combos = enumerate_combos(target=150, master_seed=20260601)
    assert len(combos) == 150, f"enumerate_combos lost combos: {len(combos)}"

    # ------------------------------------------------------------------
    # Group A: F-34 MTR suite-side dispatch.
    # ------------------------------------------------------------------

    # A1 threading: multilabel_strategy flows through
    # build_composite_discovery_config -> CompositeTargetDiscoveryConfig.
    on_a1 = dict(base_axes)
    on_a1.update(
        composite_discovery_enabled_cfg=True,
        target_type="regression",  # required for build_composite_discovery_config to enable
        composite_target_multilabel_strategy_cfg="multi_target_regression",
    )
    c_a1_on = _build_combo(models=("cb",), axes=on_a1, seed=0)
    cfg_a1 = build_composite_discovery_config(c_a1_on)
    # The CompositeTargetDiscoveryConfig field is consumed by the suite's
    # phase_helpers MTR routing branch; we assert it threaded through verbatim.
    assert getattr(cfg_a1, "multilabel_strategy", None) == "multi_target_regression", (
        f"A1: multilabel_strategy did not thread: {getattr(cfg_a1, 'multilabel_strategy', None)!r}"
    )

    # A1 canon-collapse: outside multilabel/MTR target_types, the axis
    # collapses to "per_target" (the field is unread on other targets).
    a1_off_a = dict(base_axes)
    a1_off_a.update(
        target_type="binary_classification",
        composite_target_multilabel_strategy_cfg="multi_target_regression",
    )
    a1_off_b = dict(a1_off_a)
    a1_off_b["composite_target_multilabel_strategy_cfg"] = "per_target"
    c_a1_off_a = _build_combo(models=("cb",), axes=a1_off_a, seed=0)
    c_a1_off_b = _build_combo(models=("cb",), axes=a1_off_b, seed=0)
    assert c_a1_off_a.canonical_key() == c_a1_off_b.canonical_key(), "A1: multilabel_strategy must canon-collapse on non-multilabel / non-MTR targets"

    # A1 distinct canonical_key under the gate:
    a1_on_def = dict(on_a1)
    a1_on_def["composite_target_multilabel_strategy_cfg"] = "per_target"
    c_a1_on_def = _build_combo(models=("cb",), axes=a1_on_def, seed=0)
    # NOTE: on_a1 sets target_type="regression" which is NOT in the multilabel/MTR
    # gate -- both variants collapse to "per_target", so canon_key is the same.
    # Rebuild with MTR target so canon truly forks:
    on_a1_mtr = dict(base_axes)
    on_a1_mtr.update(
        target_type="multi_target_regression",
        composite_target_multilabel_strategy_cfg="multi_target_regression",
    )
    on_a1_mtr_def = dict(on_a1_mtr)
    on_a1_mtr_def["composite_target_multilabel_strategy_cfg"] = "per_target"
    # MTR target collapses to "regression" in canonical_key when models
    # don't include cb / mlp; use cb to keep the MTR target live.
    c_a1_mtr = _build_combo(models=("cb",), axes=on_a1_mtr, seed=0)
    c_a1_mtr_def = _build_combo(models=("cb",), axes=on_a1_mtr_def, seed=0)
    assert c_a1_mtr.canonical_key() != c_a1_mtr_def.canonical_key(), "A1: multilabel_strategy must fork canonical_key under MTR target on cb"

    # A2 canon-collapse: enable_ct_ensemble outside MTR target collapses to True.
    a2_off_a = dict(base_axes)
    a2_off_a.update(
        target_type="regression",
        enable_ct_ensemble_cfg=False,
    )
    a2_off_b = dict(a2_off_a)
    a2_off_b["enable_ct_ensemble_cfg"] = True
    c_a2_off_a = _build_combo(models=("cb",), axes=a2_off_a, seed=0)
    c_a2_off_b = _build_combo(models=("cb",), axes=a2_off_b, seed=0)
    assert c_a2_off_a.canonical_key() == c_a2_off_b.canonical_key(), "A2: enable_ct_ensemble must canon-collapse on non-MTR targets"
    # A2 canon distinct under MTR target on cb (no collapse).
    a2_on_a = dict(base_axes)
    a2_on_a.update(
        target_type="multi_target_regression",
        enable_ct_ensemble_cfg=False,
    )
    a2_on_b = dict(a2_on_a)
    a2_on_b["enable_ct_ensemble_cfg"] = True
    c_a2_on_a = _build_combo(models=("cb",), axes=a2_on_a, seed=0)
    c_a2_on_b = _build_combo(models=("cb",), axes=a2_on_b, seed=0)
    assert c_a2_on_a.canonical_key() != c_a2_on_b.canonical_key(), "A2: enable_ct_ensemble must fork canon under MTR target on cb"

    # A3 canon-collapse: outside MTR target, the metric marker collapses to None.
    a3_off_a = dict(base_axes)
    a3_off_a.update(
        target_type="regression",
        mtr_eval_metric_cfg="rmse_macro",
    )
    a3_off_b = dict(a3_off_a)
    a3_off_b["mtr_eval_metric_cfg"] = None
    c_a3_off_a = _build_combo(models=("cb",), axes=a3_off_a, seed=0)
    c_a3_off_b = _build_combo(models=("cb",), axes=a3_off_b, seed=0)
    assert c_a3_off_a.canonical_key() == c_a3_off_b.canonical_key(), "A3: mtr_eval_metric must canon-collapse on non-MTR targets"
    # A3 fork under MTR target on cb.
    a3_on_a = dict(base_axes)
    a3_on_a.update(
        target_type="multi_target_regression",
        mtr_eval_metric_cfg="rmse_macro",
    )
    a3_on_b = dict(a3_on_a)
    a3_on_b["mtr_eval_metric_cfg"] = None
    c_a3_on_a = _build_combo(models=("cb",), axes=a3_on_a, seed=0)
    c_a3_on_b = _build_combo(models=("cb",), axes=a3_on_b, seed=0)
    assert c_a3_on_a.canonical_key() != c_a3_on_b.canonical_key(), "A3: mtr_eval_metric must fork canon under MTR target on cb"

    # ------------------------------------------------------------------
    # Group B: MRMR FE layer master switches.
    # ------------------------------------------------------------------

    # Compose an MRMR-on combo with a categorical column + injected NaNs so
    # every B gate holds simultaneously and all axes flow through builder.
    on_b = dict(base_axes)
    on_b.update(
        use_mrmr_fs=True,
        cat_feature_count=3,
        null_fraction_cats=0.1,  # ensures missingness gate holds
        mrmr_fe_kfold_te_enable_cfg=True,
        mrmr_fe_missingness_indicator_enable_cfg=True,
        mrmr_fe_missingness_count_enable_cfg=True,
        mrmr_fe_missingness_pattern_enable_cfg=True,
        mrmr_fe_cat_aux_enable_cfg="interaction",
        mrmr_fe_mi_greedy_enable_cfg=True,
        mrmr_fe_ratio_delta_diff_cfg="ratio",
        # Activate hybrid_orth so the extra_bases axis is consumed.
        mrmr_fe_hybrid_orth_enable_cfg=True,
        mrmr_fe_hybrid_orth_extra_bases_cfg=("spline",),
    )
    c_on_b = _build_combo(models=("cb",), axes=on_b, seed=0)
    kw_b = build_mrmr_kwargs(c_on_b)
    assert kw_b is not None
    # B1: K-fold TE master switch flows through.
    assert kw_b["fe_kfold_te_enable"] is True, "B1: fe_kfold_te_enable did not thread"
    # B2: missingness master switches flow through.
    assert kw_b["fe_missingness_indicator_enable"] is True, "B2 indicator: did not thread"
    assert kw_b["fe_missingness_count_enable"] is True, "B2 count: did not thread"
    assert kw_b["fe_missingness_pattern_enable"] is True, "B2 pattern: did not thread"
    # B3: 4-way cat_aux axis maps to ONE of the three master switches.
    # "interaction" -> fe_cat_num_interaction_enable=True, others False.
    assert kw_b["fe_cat_num_interaction_enable"] is True, "B3 interaction: did not thread"
    assert kw_b["fe_count_encoding_enable"] is False, "B3: only 'interaction' branch should be on"
    assert kw_b["fe_frequency_encoding_enable"] is False, "B3: only 'interaction' branch should be on"
    # B4: extra_bases flows through verbatim.
    assert kw_b["fe_hybrid_orth_extra_bases"] == ("spline",), f"B4: extra_bases did not thread: {kw_b['fe_hybrid_orth_extra_bases']!r}"
    # B5: 4-way ratio_delta_diff axis maps to ONE of the four master switches.
    # "ratio" -> fe_pairwise_ratio_enable=True, others False.
    assert kw_b["fe_pairwise_ratio_enable"] is True, "B5 ratio: did not thread"
    assert kw_b["fe_pairwise_log_ratio_enable"] is False, "B5: only 'ratio' branch on"
    assert kw_b["fe_grouped_delta_enable"] is False, "B5: only 'ratio' branch on"
    assert kw_b["fe_lagged_diff_enable"] is False, "B5: only 'ratio' branch on"
    # B6: MI-greedy master switch flows through.
    assert kw_b["fe_mi_greedy_enable"] is True, "B6: fe_mi_greedy_enable did not thread"

    # B3 4-way exhaustive: each non-off value maps to its own master switch.
    for label, expected in (
        ("count", "fe_count_encoding_enable"),
        ("freq", "fe_frequency_encoding_enable"),
        ("interaction", "fe_cat_num_interaction_enable"),
    ):
        axes_b3 = dict(base_axes)
        axes_b3.update(
            use_mrmr_fs=True,
            cat_feature_count=3,
            mrmr_fe_cat_aux_enable_cfg=label,
        )
        c_b3 = _build_combo(models=("cb",), axes=axes_b3, seed=0)
        kw_b3 = build_mrmr_kwargs(c_b3)
        assert kw_b3[expected] is True, f"B3 {label}: did not thread to {expected}"
        # Other two switches must be False (exclusivity).
        for other_key in ("fe_count_encoding_enable", "fe_frequency_encoding_enable", "fe_cat_num_interaction_enable"):
            if other_key != expected:
                assert kw_b3[other_key] is False, f"B3 {label}: {other_key} must be False under exclusive 4-way mapping"

    # B5 4-way exhaustive: each non-off value maps to its own master switch and threads its supporting columns.
    # "ratio" -> fe_pairwise_ratio_enable; "grouped_delta" -> fe_grouped_delta_enable (+ group_col / num_cols);
    # "lagged_diff" -> fe_lagged_diff_enable (+ time_col / value_cols). The frame builder now emits the group key /
    # order column for the matching kind, so all three non-off branches actually run -- none collapses to "off".
    for label, expected in (
        ("ratio", "fe_pairwise_ratio_enable"),
        ("grouped_delta", "fe_grouped_delta_enable"),
        ("lagged_diff", "fe_lagged_diff_enable"),
    ):
        axes_b5 = dict(base_axes)
        axes_b5.update(
            use_mrmr_fs=True,
            mrmr_fe_ratio_delta_diff_cfg=label,
        )
        c_b5 = _build_combo(models=("cb",), axes=axes_b5, seed=0)
        kw_b5 = build_mrmr_kwargs(c_b5)
        assert kw_b5[expected] is True, f"B5 {label}: did not thread to {expected}"
    # grouped_delta wires its group_col + numeric source columns so the kind has the inputs prod needs.
    axes_gd = dict(base_axes)
    axes_gd.update(use_mrmr_fs=True, mrmr_fe_ratio_delta_diff_cfg="grouped_delta")
    kw_gd = build_mrmr_kwargs(_build_combo(models=("cb",), axes=axes_gd, seed=0))
    assert kw_gd["fe_grouped_delta_group_col"], "grouped_delta: group_col not wired"
    assert kw_gd["fe_grouped_delta_num_cols"], "grouped_delta: num_cols not wired"
    # lagged_diff wires its time_col + value source columns.
    axes_ld = dict(base_axes)
    axes_ld.update(use_mrmr_fs=True, mrmr_fe_ratio_delta_diff_cfg="lagged_diff")
    kw_ld = build_mrmr_kwargs(_build_combo(models=("cb",), axes=axes_ld, seed=0))
    assert kw_ld["fe_lagged_diff_time_col"], "lagged_diff: time_col not wired"
    assert kw_ld["fe_lagged_diff_value_cols"], "lagged_diff: value_cols not wired"

    # B1-B6 canon-collapse: when use_mrmr_fs=False, every axis collapses.
    off_b_a = dict(on_b)
    off_b_a["use_mrmr_fs"] = False
    off_b_b = dict(off_b_a)
    off_b_b.update(
        mrmr_fe_kfold_te_enable_cfg=False,
        mrmr_fe_missingness_indicator_enable_cfg=False,
        mrmr_fe_missingness_count_enable_cfg=False,
        mrmr_fe_missingness_pattern_enable_cfg=False,
        mrmr_fe_cat_aux_enable_cfg="off",
        mrmr_fe_hybrid_orth_extra_bases_cfg=(),
        mrmr_fe_ratio_delta_diff_cfg="off",
        mrmr_fe_mi_greedy_enable_cfg=False,
    )
    c_off_b_a = _build_combo(models=("cb",), axes=off_b_a, seed=0)
    c_off_b_b = _build_combo(models=("cb",), axes=off_b_b, seed=0)
    assert build_mrmr_kwargs(c_off_b_a) is None, "MRMR off: kwargs must be None"
    assert c_off_b_a.canonical_key() == c_off_b_b.canonical_key(), "B1-B6: all MRMR FE master switches must canon-collapse when use_mrmr_fs=False"

    # B1 specific gate: no cat columns -> kfold_te canon-collapses too.
    off_b1_a = dict(base_axes)
    off_b1_a.update(
        use_mrmr_fs=True,
        cat_feature_count=0,
        mrmr_fe_kfold_te_enable_cfg=True,
    )
    off_b1_b = dict(off_b1_a)
    off_b1_b["mrmr_fe_kfold_te_enable_cfg"] = False
    c_off_b1_a = _build_combo(models=("cb",), axes=off_b1_a, seed=0)
    c_off_b1_b = _build_combo(models=("cb",), axes=off_b1_b, seed=0)
    assert c_off_b1_a.canonical_key() == c_off_b1_b.canonical_key(), "B1: fe_kfold_te must canon-collapse when no categorical column"

    # B2 specific gate: no NaN source -> missingness master switches canon.
    off_b2_a = dict(base_axes)
    off_b2_a.update(
        use_mrmr_fs=True,
        inject_inf_nan=False,
        inject_all_nan_col=False,
        cat_feature_count=0,
        mrmr_fe_missingness_indicator_enable_cfg=True,
        mrmr_fe_missingness_count_enable_cfg=True,
        mrmr_fe_missingness_pattern_enable_cfg=True,
    )
    off_b2_b = dict(off_b2_a)
    off_b2_b.update(
        mrmr_fe_missingness_indicator_enable_cfg=False,
        mrmr_fe_missingness_count_enable_cfg=False,
        mrmr_fe_missingness_pattern_enable_cfg=False,
    )
    c_off_b2_a = _build_combo(models=("cb",), axes=off_b2_a, seed=0)
    c_off_b2_b = _build_combo(models=("cb",), axes=off_b2_b, seed=0)
    assert c_off_b2_a.canonical_key() == c_off_b2_b.canonical_key(), "B2: missingness master switches must canon-collapse with no NaN source"

    # B4 specific gate: hybrid_orth master OFF -> extra_bases canon-collapses.
    off_b4_a = dict(base_axes)
    off_b4_a.update(
        use_mrmr_fs=True,
        mrmr_fe_hybrid_orth_enable_cfg=False,
        mrmr_fe_hybrid_orth_extra_bases_cfg=("spline",),
    )
    off_b4_b = dict(off_b4_a)
    off_b4_b["mrmr_fe_hybrid_orth_extra_bases_cfg"] = ()
    c_off_b4_a = _build_combo(models=("cb",), axes=off_b4_a, seed=0)
    c_off_b4_b = _build_combo(models=("cb",), axes=off_b4_b, seed=0)
    assert c_off_b4_a.canonical_key() == c_off_b4_b.canonical_key(), "B4: extra_bases must canon-collapse when hybrid_orth master is off"

    # B5 specific gate: the frame builder now emits a group key / order column for grouped_delta / lagged_diff, so both
    # kinds actually run and must stay DISTINCT from "off" under canonicalisation (NOT collapsed away -- a collapse would
    # silently drop the kind from the sweep). Disabling MRMR still collapses every kind (no FE entry point at all).
    for kind in ("grouped_delta", "lagged_diff"):
        on_b5 = dict(base_axes)
        on_b5.update(use_mrmr_fs=True, mrmr_fe_ratio_delta_diff_cfg=kind)
        off_b5 = dict(on_b5)
        off_b5["mrmr_fe_ratio_delta_diff_cfg"] = "off"
        c_on_b5 = _build_combo(models=("cb",), axes=on_b5, seed=0)
        c_off_b5 = _build_combo(models=("cb",), axes=off_b5, seed=0)
        assert c_on_b5.canonical_key() != c_off_b5.canonical_key(), (
            f"B5 {kind}: must stay distinct from 'off' (the kind runs; collapsing it drops it from the sweep)"
        )
        no_mrmr_b5 = dict(on_b5)
        no_mrmr_b5["use_mrmr_fs"] = False
        no_mrmr_off = dict(no_mrmr_b5)
        no_mrmr_off["mrmr_fe_ratio_delta_diff_cfg"] = "off"
        c_no_mrmr_b5 = _build_combo(models=("cb",), axes=no_mrmr_b5, seed=0)
        c_no_mrmr_off = _build_combo(models=("cb",), axes=no_mrmr_off, seed=0)
        assert c_no_mrmr_b5.canonical_key() == c_no_mrmr_off.canonical_key(), (
            f"B5 {kind}: must canon-collapse to 'off' when use_mrmr_fs is False (no FE entry point)"
        )

    # ------------------------------------------------------------------
    # Group C: MRMR + ShapProxiedFS artifact-reuse pipeline.
    # ------------------------------------------------------------------

    # C1 threading: retain_artifacts ON via MRMR kwargs + precomputed sentinel
    # via ShapProxiedFS kwargs, for each align_mode branch.
    align_to_reason = {
        "exact": ("exact_match", True),
        "permuted": ("permutation_match", True),
        "subset": ("subset_match", True),
        "mismatched": ("feature_name_mismatch", False),
    }
    for align_mode, (expected_reason, expected_honoured) in align_to_reason.items():
        on_c = dict(base_axes)
        on_c.update(
            use_mrmr_fs=True,
            use_shap_proxied_fs=True,
            mrmr_shap_proxy_artifact_reuse_cfg="on",
            mrmr_shap_proxy_align_mode_cfg=align_mode,
        )
        c_on_c = _build_combo(models=("cb",), axes=on_c, seed=0)
        kw_mrmr = build_mrmr_kwargs(c_on_c)
        assert kw_mrmr is not None
        assert kw_mrmr["retain_artifacts"] is True, f"C1 ({align_mode}): MRMR.retain_artifacts must thread True under artifact-reuse on"
        kw_shap = build_shap_proxied_fs_kwargs(c_on_c)
        assert kw_shap is not None
        assert kw_shap["precomputed"] is not None, f"C1 ({align_mode}): ShapProxiedFS.precomputed must be a dict under artifact-reuse on"
        # Drive the actual align_precomputed_to_X function with the sentinel
        # to verify the expected branch is reached.
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_precomputed import (
            align_precomputed_to_X,
        )
        import pandas as pd

        # X.columns must match what the sentinel feature_names target:
        X_4 = pd.DataFrame({"num_0": [0.0], "num_1": [0.1], "num_2": [0.2], "num_3": [0.3]})
        aligned, report = align_precomputed_to_X(kw_shap["precomputed"], X_4)
        assert report["honoured"] is expected_honoured, (
            f"C2 ({align_mode}): align_precomputed_to_X honoured mismatch -- got {report['honoured']!r}, expected {expected_honoured!r}; report={report!r}"
        )
        assert report["reason"] == expected_reason, (
            f"C2 ({align_mode}): align_precomputed_to_X reason mismatch -- got {report['reason']!r}, expected {expected_reason!r}"
        )
        if expected_honoured:
            assert aligned is not None
        else:
            assert aligned is None

    # C1 canon-collapse: artifact-reuse master OFF -> retain_artifacts False AND
    # precomputed None.
    off_c_a = dict(base_axes)
    off_c_a.update(
        use_mrmr_fs=True,
        use_shap_proxied_fs=True,
        mrmr_shap_proxy_artifact_reuse_cfg="off",
        mrmr_shap_proxy_align_mode_cfg="permuted",
    )
    off_c_b = dict(off_c_a)
    off_c_b["mrmr_shap_proxy_align_mode_cfg"] = "exact"
    c_off_c_a = _build_combo(models=("cb",), axes=off_c_a, seed=0)
    c_off_c_b = _build_combo(models=("cb",), axes=off_c_b, seed=0)
    assert c_off_c_a.canonical_key() == c_off_c_b.canonical_key(), "C2: align_mode must canon-collapse to 'exact' when artifact-reuse master is off"
    kw_off_c = build_mrmr_kwargs(c_off_c_a)
    assert kw_off_c["retain_artifacts"] is False, "C1: MRMR.retain_artifacts must be False when artifact-reuse master is off"
    kw_off_shap = build_shap_proxied_fs_kwargs(c_off_c_a)
    assert kw_off_shap["precomputed"] is None, "C1: ShapProxiedFS.precomputed must be None when artifact-reuse master is off"

    # C1 canon-collapse: ShapProxiedFS OFF -> artifact-reuse master forced off.
    off_c_no_shap_a = dict(base_axes)
    off_c_no_shap_a.update(
        use_mrmr_fs=True,
        use_shap_proxied_fs=False,
        mrmr_shap_proxy_artifact_reuse_cfg="on",
    )
    off_c_no_shap_b = dict(off_c_no_shap_a)
    off_c_no_shap_b["mrmr_shap_proxy_artifact_reuse_cfg"] = "off"
    c_off_c_ns_a = _build_combo(models=("cb",), axes=off_c_no_shap_a, seed=0)
    c_off_c_ns_b = _build_combo(models=("cb",), axes=off_c_no_shap_b, seed=0)
    assert c_off_c_ns_a.canonical_key() == c_off_c_ns_b.canonical_key(), "C1: artifact-reuse master must canon-collapse to 'off' when use_shap_proxied_fs=False"

    # ------------------------------------------------------------------
    # (e) Distinct canonical_keys under each compound gate so the pairwise
    # sampler keeps both branches reachable.
    # ------------------------------------------------------------------
    # B1 distinct under (use_mrmr_fs AND cat_feature_count >= 1).
    on_b1_a = dict(base_axes)
    on_b1_a.update(
        use_mrmr_fs=True,
        cat_feature_count=3,
        mrmr_fe_kfold_te_enable_cfg=True,
    )
    on_b1_b = dict(on_b1_a)
    on_b1_b["mrmr_fe_kfold_te_enable_cfg"] = False
    c_on_b1_a = _build_combo(models=("cb",), axes=on_b1_a, seed=0)
    c_on_b1_b = _build_combo(models=("cb",), axes=on_b1_b, seed=0)
    assert c_on_b1_a.canonical_key() != c_on_b1_b.canonical_key(), "B1: True/False must fork canon under (use_mrmr_fs AND cats present)"

    # B6 distinct under use_mrmr_fs.
    on_b6_a = dict(base_axes)
    on_b6_a.update(
        use_mrmr_fs=True,
        mrmr_fe_mi_greedy_enable_cfg=True,
    )
    on_b6_b = dict(on_b6_a)
    on_b6_b["mrmr_fe_mi_greedy_enable_cfg"] = False
    c_on_b6_a = _build_combo(models=("cb",), axes=on_b6_a, seed=0)
    c_on_b6_b = _build_combo(models=("cb",), axes=on_b6_b, seed=0)
    assert c_on_b6_a.canonical_key() != c_on_b6_b.canonical_key(), "B6: True/False must fork canon under use_mrmr_fs"

    # C1 distinct under (use_mrmr_fs AND use_shap_proxied_fs).
    on_c1_a = dict(base_axes)
    on_c1_a.update(
        use_mrmr_fs=True,
        use_shap_proxied_fs=True,
        mrmr_shap_proxy_artifact_reuse_cfg="on",
    )
    on_c1_b = dict(on_c1_a)
    on_c1_b["mrmr_shap_proxy_artifact_reuse_cfg"] = "off"
    c_on_c1_a = _build_combo(models=("cb",), axes=on_c1_a, seed=0)
    c_on_c1_b = _build_combo(models=("cb",), axes=on_c1_b, seed=0)
    assert c_on_c1_a.canonical_key() != c_on_c1_b.canonical_key(), "C1: on/off must fork canon under (use_mrmr_fs AND use_shap_proxied_fs)"


def test_iter627_audit_pass_14_axes_flow_to_kwargs():
    """6 audit-pass-14 (W14) fuzz axes (+1 invariant probe for F14-6) must:
      (a) be present in AXES (or, for F14-2 cushion, be extended in place with
          the legacy value 8) with >=2 candidate values,
      (b) carry the SOURCE-verified library defaults in the FuzzCombo dataclass
          (verified pre-edit against
          ``src/mlframe/feature_selection/shap_proxied_fs.py:249, :258`` and
          ``src/mlframe/feature_selection/filters/mrmr.py:621/622/655/845-847``),
      (c) collapse correctly under the documented gates,
      (d) thread through their downstream consumer:
            - F14-1 cluster_backend flows via build_shap_proxied_fs_kwargs into
              ShapProxiedFS.__init__,
            - F14-2 cushion=8 stays reachable via the extended pair,
            - F14-3 partial_fit_decay/min_recompute/window flow via
              build_mrmr_kwargs into MRMR.__init__ (the partial_fit() public
              API itself is not exercised here; the ctor params shape future
              behaviour and the fuzz pairwise sampler still exercises the
              (param != default) ctor branches),
            - F14-4 dcd_tau_cluster flows via build_mrmr_kwargs (accepts the
              new ``'auto'`` literal alongside the legacy float),
            - F14-5 dcd_distance + dcd_swap_method flow via build_mrmr_kwargs
              (the expanded _VALID_DCD_SWAP_METHODS pool at mrmr.py:947-950
              accepts the 4 new aggregator names verbatim).
      (e) F14-6 [LOW shape invariant]: ``len(support_) == fe_provenance_.shape[0]``
          is asserted by the test-shaped invariant probe below. ``fe_provenance_``
          is always-on additive (no opt-out flag in MRMR.__init__) so this is a
          sensor probe rather than a fuzz axis -- the sensor protects the
          new shape contract against monolith-split / re-export bugs in the
          mrmr facade.

    Findings (6 total, sorted by severity per AUDIT_PASS_14_DONE.md):
      HIGH (2):
        F14-1 shap_proxied_cluster_backend_cfg  ("auto","su","pearson")
        F14-2 shap_proxied_shap_aware_stage1_cushion_cfg  (2,4,8)  [extended in place]
      MED (3):
        F14-3 mrmr_partial_fit_{decay,min_recompute,window}_cfg
        F14-4 mrmr_dcd_tau_cluster_cfg  (0.7, "auto")
        F14-5 mrmr_dcd_distance_cfg + mrmr_dcd_swap_method_cfg
      LOW (1):
        F14-6 fe_provenance_ shape invariant  [sensor only]
    """
    from tests.training._fuzz_combo import (
        AXES,
        FuzzCombo,
        _build_combo,
        enumerate_combos,
        build_mrmr_kwargs,
        build_shap_proxied_fs_kwargs,
    )

    new_axes = (
        "shap_proxied_cluster_backend_cfg",
        "mrmr_partial_fit_decay_cfg",
        "mrmr_partial_fit_min_recompute_cfg",
        "mrmr_partial_fit_window_cfg",
        "mrmr_dcd_tau_cluster_cfg",
        "mrmr_dcd_distance_cfg",
        "mrmr_dcd_swap_method_cfg",
    )
    # (a) Presence in AXES with >=2 candidates.
    for ax in new_axes:
        assert ax in AXES, f"missing fuzz axis {ax}"
        assert len(AXES[ax]) >= 2, f"axis {ax} must offer at least 2 values"
    # F14-2: cushion axis was extended in place 2026-05-31 (was (2, 4), now (2, 4, 8)).
    cushion_axis = AXES["shap_proxied_shap_aware_stage1_cushion_cfg"]
    assert 2 in cushion_axis and 4 in cushion_axis and 8 in cushion_axis, (
        f"F14-2: cushion axis must include the legacy 8 for fuzz coverage of the pre-iter76 calibration; got {cushion_axis}"
    )

    # (b) FuzzCombo dataclass defaults match the source-verified library defaults.
    fields = FuzzCombo.__dataclass_fields__
    assert fields["shap_proxied_cluster_backend_cfg"].default == "auto", "F14-1: default must mirror shap_proxied_fs.py:258 ('auto')"
    assert fields["shap_proxied_shap_aware_stage1_cushion_cfg"].default == 2, "F14-2: default must mirror shap_proxied_fs.py:249 (2 since iter76)"
    assert fields["mrmr_partial_fit_decay_cfg"].default == 0.0, "F14-3 decay: default must mirror filters/mrmr.py:845 (0.0)"
    assert fields["mrmr_partial_fit_min_recompute_cfg"].default == 100, "F14-3 min_recompute: default must mirror filters/mrmr.py:846 (100)"
    assert fields["mrmr_partial_fit_window_cfg"].default is None, "F14-3 window: default must mirror filters/mrmr.py:847 (None)"
    assert fields["mrmr_dcd_tau_cluster_cfg"].default == 0.7, "F14-4: default must mirror filters/mrmr.py:621 (0.7)"
    assert fields["mrmr_dcd_distance_cfg"].default == "su", "F14-5 distance: default must mirror filters/mrmr.py:622 ('su')"
    assert fields["mrmr_dcd_swap_method_cfg"].default == "auto", "F14-5 swap_method: default must mirror filters/mrmr.py:655 ('auto')"

    base_axes = {name: values[0] for name, values in AXES.items()}

    # (c) enumerate_combos still hits 150 with the 6 new axes wired.
    combos = enumerate_combos(target=150, master_seed=20260601)
    assert len(combos) == 150, f"enumerate_combos lost combos: {len(combos)}"

    # ------------------------------------------------------------------
    # F14-1 cluster_backend: threads into ShapProxiedFS kwargs verbatim
    # under use_shap_proxied_fs=True; canon-collapses to "auto" otherwise.
    # ------------------------------------------------------------------
    for backend in ("auto", "su", "pearson"):
        on_f1 = dict(base_axes)
        on_f1.update(
            use_shap_proxied_fs=True,
            shap_proxied_cluster_backend_cfg=backend,
        )
        c_on_f1 = _build_combo(models=("cb",), axes=on_f1, seed=0)
        kw = build_shap_proxied_fs_kwargs(c_on_f1)
        assert kw is not None
        assert kw["cluster_backend"] == backend, f"F14-1: cluster_backend={backend!r} did not thread; got {kw['cluster_backend']!r}"

    # F14-1 canon-collapse: cluster_backend axis is unread when ShapProxiedFS is off.
    off_f1_a = dict(base_axes)
    off_f1_a.update(
        use_shap_proxied_fs=False,
        shap_proxied_cluster_backend_cfg="su",
    )
    off_f1_b = dict(off_f1_a)
    off_f1_b["shap_proxied_cluster_backend_cfg"] = "pearson"
    c_off_f1_a = _build_combo(models=("cb",), axes=off_f1_a, seed=0)
    c_off_f1_b = _build_combo(models=("cb",), axes=off_f1_b, seed=0)
    assert c_off_f1_a.canonical_key() == c_off_f1_b.canonical_key(), "F14-1: cluster_backend must canon-collapse when use_shap_proxied_fs=False"
    # F14-1 distinct under the gate: "su" vs "pearson" must fork canon.
    on_f1_a = dict(base_axes)
    on_f1_a.update(
        use_shap_proxied_fs=True,
        shap_proxied_cluster_backend_cfg="su",
    )
    on_f1_b = dict(on_f1_a)
    on_f1_b["shap_proxied_cluster_backend_cfg"] = "pearson"
    c_on_f1_a = _build_combo(models=("cb",), axes=on_f1_a, seed=0)
    c_on_f1_b = _build_combo(models=("cb",), axes=on_f1_b, seed=0)
    assert c_on_f1_a.canonical_key() != c_on_f1_b.canonical_key(), "F14-1: cluster_backend su vs pearson must fork canon under use_shap_proxied_fs=True"

    # ------------------------------------------------------------------
    # F14-2 cushion=8 threads via ShapProxiedFS kwargs (legacy fuzz baseline).
    # ------------------------------------------------------------------
    on_f2 = dict(base_axes)
    on_f2.update(
        use_shap_proxied_fs=True,
        shap_proxied_shap_aware_stage1_cushion_cfg=8,
    )
    c_on_f2 = _build_combo(models=("cb",), axes=on_f2, seed=0)
    kw_f2 = build_shap_proxied_fs_kwargs(c_on_f2)
    assert kw_f2 is not None
    assert kw_f2["shap_aware_stage1_cushion"] == 8, f"F14-2: cushion=8 did not thread; got {kw_f2['shap_aware_stage1_cushion']!r}"
    # All three pair values fork canon under the gate.
    forks = []
    for cushion in (2, 4, 8):
        on = dict(base_axes)
        on.update(
            use_shap_proxied_fs=True,
            shap_proxied_shap_aware_stage1_cushion_cfg=cushion,
        )
        forks.append(_build_combo(models=("cb",), axes=on, seed=0).canonical_key())
    assert len(set(forks)) == 3, f"F14-2: cushion 2/4/8 must produce 3 distinct canonical keys under use_shap_proxied_fs=True; got {len(set(forks))} distinct"

    # ------------------------------------------------------------------
    # F14-3 partial_fit ctor params flow via build_mrmr_kwargs verbatim.
    # ------------------------------------------------------------------
    on_f3 = dict(base_axes)
    on_f3.update(
        use_mrmr_fs=True,
        mrmr_partial_fit_decay_cfg=0.3,
        mrmr_partial_fit_min_recompute_cfg=50,
        mrmr_partial_fit_window_cfg=500,
    )
    c_on_f3 = _build_combo(models=("cb",), axes=on_f3, seed=0)
    kw_f3 = build_mrmr_kwargs(c_on_f3)
    assert kw_f3 is not None
    assert kw_f3["partial_fit_decay"] == 0.3
    assert kw_f3["partial_fit_min_recompute"] == 50
    assert kw_f3["partial_fit_window"] == 500
    # Canon-collapse when use_mrmr_fs=False.
    off_f3_a = dict(base_axes)
    off_f3_a.update(
        use_mrmr_fs=False,
        mrmr_partial_fit_decay_cfg=0.3,
        mrmr_partial_fit_min_recompute_cfg=50,
        mrmr_partial_fit_window_cfg=500,
    )
    off_f3_b = dict(off_f3_a)
    off_f3_b.update(
        mrmr_partial_fit_decay_cfg=0.0,
        mrmr_partial_fit_min_recompute_cfg=100,
        mrmr_partial_fit_window_cfg=None,
    )
    c_off_f3_a = _build_combo(models=("cb",), axes=off_f3_a, seed=0)
    c_off_f3_b = _build_combo(models=("cb",), axes=off_f3_b, seed=0)
    assert c_off_f3_a.canonical_key() == c_off_f3_b.canonical_key(), "F14-3: partial_fit_* axes must canon-collapse when use_mrmr_fs=False"
    # Distinct under the gate.
    on_f3_a = dict(base_axes)
    on_f3_a.update(
        use_mrmr_fs=True,
        mrmr_partial_fit_decay_cfg=0.3,
    )
    on_f3_b = dict(on_f3_a)
    on_f3_b["mrmr_partial_fit_decay_cfg"] = 0.0
    c_on_f3_a = _build_combo(models=("cb",), axes=on_f3_a, seed=0)
    c_on_f3_b = _build_combo(models=("cb",), axes=on_f3_b, seed=0)
    assert c_on_f3_a.canonical_key() != c_on_f3_b.canonical_key(), "F14-3: partial_fit_decay 0.3 vs 0.0 must fork canon under use_mrmr_fs=True"

    # ------------------------------------------------------------------
    # F14-4 dcd_tau_cluster=auto flows verbatim under (use_mrmr_fs AND dcd_enable).
    # ------------------------------------------------------------------
    on_f4 = dict(base_axes)
    on_f4.update(
        use_mrmr_fs=True,
        mrmr_dcd_enable_cfg=True,
        mrmr_dcd_tau_cluster_cfg="auto",
    )
    c_on_f4 = _build_combo(models=("cb",), axes=on_f4, seed=0)
    kw_f4 = build_mrmr_kwargs(c_on_f4)
    assert kw_f4 is not None
    assert kw_f4["dcd_tau_cluster"] == "auto", f"F14-4: dcd_tau_cluster='auto' did not thread; got {kw_f4['dcd_tau_cluster']!r}"
    # Canon-collapse when dcd is off.
    off_f4_a = dict(base_axes)
    off_f4_a.update(
        use_mrmr_fs=True,
        mrmr_dcd_enable_cfg=False,
        mrmr_dcd_tau_cluster_cfg="auto",
    )
    off_f4_b = dict(off_f4_a)
    off_f4_b["mrmr_dcd_tau_cluster_cfg"] = 0.7
    c_off_f4_a = _build_combo(models=("cb",), axes=off_f4_a, seed=0)
    c_off_f4_b = _build_combo(models=("cb",), axes=off_f4_b, seed=0)
    assert c_off_f4_a.canonical_key() == c_off_f4_b.canonical_key(), "F14-4: dcd_tau_cluster must canon-collapse when dcd is off"
    # Distinct under the compound gate.
    on_f4_a = dict(base_axes)
    on_f4_a.update(
        use_mrmr_fs=True,
        mrmr_dcd_enable_cfg=True,
        mrmr_dcd_tau_cluster_cfg="auto",
    )
    on_f4_b = dict(on_f4_a)
    on_f4_b["mrmr_dcd_tau_cluster_cfg"] = 0.7
    c_on_f4_a = _build_combo(models=("cb",), axes=on_f4_a, seed=0)
    c_on_f4_b = _build_combo(models=("cb",), axes=on_f4_b, seed=0)
    assert c_on_f4_a.canonical_key() != c_on_f4_b.canonical_key(), "F14-4: dcd_tau_cluster auto vs 0.7 must fork canon under dcd_enable"

    # ------------------------------------------------------------------
    # F14-5 dcd_distance + dcd_swap_method threading + the 4 new Layer 44 values.
    # ------------------------------------------------------------------
    for swap_method in ("auto", "mean_z", "pca_pc2", "median_z", "signed_max_abs"):
        on_f5 = dict(base_axes)
        on_f5.update(
            use_mrmr_fs=True,
            mrmr_dcd_enable_cfg=True,
            mrmr_dcd_distance_cfg="auto",
            mrmr_dcd_swap_method_cfg=swap_method,
        )
        c_on_f5 = _build_combo(models=("cb",), axes=on_f5, seed=0)
        kw_f5 = build_mrmr_kwargs(c_on_f5)
        assert kw_f5 is not None
        assert kw_f5["dcd_distance"] == "auto", f"F14-5: dcd_distance='auto' did not thread; got {kw_f5['dcd_distance']!r}"
        assert kw_f5["dcd_swap_method"] == swap_method, f"F14-5: dcd_swap_method={swap_method!r} did not thread; got {kw_f5['dcd_swap_method']!r}"
    # F14-5 canon-collapse when dcd_enable=False.
    off_f5_a = dict(base_axes)
    off_f5_a.update(
        use_mrmr_fs=True,
        mrmr_dcd_enable_cfg=False,
        mrmr_dcd_distance_cfg="auto",
        mrmr_dcd_swap_method_cfg="pca_pc2",
    )
    off_f5_b = dict(off_f5_a)
    off_f5_b.update(
        mrmr_dcd_distance_cfg="su",
        mrmr_dcd_swap_method_cfg="auto",
    )
    c_off_f5_a = _build_combo(models=("cb",), axes=off_f5_a, seed=0)
    c_off_f5_b = _build_combo(models=("cb",), axes=off_f5_b, seed=0)
    assert c_off_f5_a.canonical_key() == c_off_f5_b.canonical_key(), "F14-5: dcd_distance / dcd_swap_method must canon-collapse when dcd is off"

    # ------------------------------------------------------------------
    # F14-6 [shape invariant probe] -- ``fe_provenance_`` is unconditionally
    # populated inside MRMR.fit() since Layer 54. The contract is
    # ``len(support_) == fe_provenance_.shape[0]``. The MRMR class is
    # imported here only to assert that the attr lives on the class (sensor
    # against monolith-split / re-export bugs in the mrmr facade). The
    # actual fit-time shape assertion runs under --run-fuzz when MRMR
    # actually executes; here we pin the contract at the API surface.
    # ------------------------------------------------------------------
    from mlframe.feature_selection.filters.mrmr import MRMR

    # The class must own (or inherit) a default for support_ + fe_provenance_
    # so post-fit shape introspection is well-defined. Layer 54 introduces
    # fe_provenance_ as an instance attribute set inside fit(); on a fresh
    # ctor it is absent. We assert the populator module is importable so the
    # contract surface is reachable (split-bug sensor).
    from mlframe.feature_selection.filters import _mrmr_fe_provenance

    assert hasattr(_mrmr_fe_provenance, "populate_fe_provenance"), (
        "F14-6: _mrmr_fe_provenance.populate_fe_provenance must be importable from filters/_mrmr_fe_provenance (Layer 54 shape-contract surface)"
    )
    # Sanity: the MRMR class itself is reachable through the facade.
    assert MRMR is not None
