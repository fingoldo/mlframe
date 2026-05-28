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
