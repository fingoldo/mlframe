"""Fuzz blind-spot sensors for F1 / F2 / F5 / F6 / F7 (per architectural_proposals/fuzz_blind_spots_F1_F2_F5_F6_F7.md).

Each sensor is intentionally narrow: collect-only (validates dataclass / AXES wiring without exercising the full suite) plus 1-2 representative ``FuzzCombo`` constructions on minimal synthetic axes to assert canonicalisation / required-pin behaviour. Full suite execution stays under the ``--run-fuzz`` marker as before.

Findings:

- F1: ``enable_crash_reporting_cfg`` axis. New 2-value axis (False / True) with a Windows-only canon (non-Windows hosts canonicalise to False since crash reporting is a no-op there).
- F2: ``prefer_polarsds=True`` x ``target_type="learning_to_rank"`` required combo pin.
- F5: canonicalisation rule -- ``prep_ext_pysr_enabled_cfg=True`` x ``inject_inf_nan=True`` collapses to PySR-disabled (PySR cannot consume non-finite values).
- F6: ``composite_discovery_enabled_cfg=True`` x ``outlier_detection in {lof, ocsvm}`` reachable through the enumerator (catches the 0-row-val cluster).
- F7: ``dummy_baselines_enabled_cfg=False`` x ``baseline_diagnostics_enabled_cfg=True`` is a real reachable combo (diagnostics path must either auto-enable baselines OR raise; behaviour is asserted live by the existing suite when the combo runs).
"""

from __future__ import annotations

import platform

import pytest

from tests.training._fuzz_combo import AXES, _build_combo

# ---------------------------------------------------------------------------
# Helper: build a FuzzCombo by overlaying axis kwargs onto AXES defaults.
# ---------------------------------------------------------------------------


def _make_combo(**overrides):
    """Make combo."""
    axes = {name: values[0] for name, values in AXES.items()}
    axes.update(overrides)
    return _build_combo(models=("cb",), axes=axes, seed=0)


# ---------------------------------------------------------------------------
# F1: enable_crash_reporting_cfg axis present + Windows-only canonicalisation
# ---------------------------------------------------------------------------


def test_F1_enable_crash_reporting_axis_present_in_AXES():
    """AXES must declare ``enable_crash_reporting_cfg`` with both False and True."""
    assert "enable_crash_reporting_cfg" in AXES, "F1: missing axis entry"
    values = AXES["enable_crash_reporting_cfg"]
    assert False in values and True in values, f"F1: axis must cover both False/True, got {values!r}"


def test_F1_enable_crash_reporting_field_on_FuzzCombo():
    """FuzzCombo dataclass must accept ``enable_crash_reporting_cfg`` (Optional[bool] default False)."""
    combo_default = _make_combo()
    assert hasattr(combo_default, "enable_crash_reporting_cfg")
    combo_on = _make_combo(enable_crash_reporting_cfg=True)
    assert combo_on.enable_crash_reporting_cfg is True
    combo_off = _make_combo(enable_crash_reporting_cfg=False)
    assert combo_off.enable_crash_reporting_cfg is False


def test_F1_enable_crash_reporting_windows_only_canon():
    """Non-Windows hosts must canonicalise ``enable_crash_reporting_cfg=True`` to False so the dedup pass collapses identical-behaviour combos. On Windows the axis stays meaningful."""
    combo_off = _make_combo(enable_crash_reporting_cfg=False)
    combo_on = _make_combo(enable_crash_reporting_cfg=True)
    is_windows = platform.system() == "Windows"
    if is_windows:
        # On Windows the True variant must be distinct from the False variant (the canon does NOT collapse them since the axis is meaningful there).
        assert combo_off.canonical_key() != combo_on.canonical_key(), "F1: on Windows, enable_crash_reporting_cfg=True must NOT canonicalise away"
    else:
        # On non-Windows the True variant must collapse to the False variant.
        assert combo_off.canonical_key() == combo_on.canonical_key(), f"F1: on {platform.system()}, enable_crash_reporting_cfg=True must canonicalise to False"


# ---------------------------------------------------------------------------
# F2: prefer_polarsds=True x target_type=learning_to_rank reachable + distinct
# ---------------------------------------------------------------------------


def test_F2_polarsds_x_ltr_combo_is_reachable_and_distinct():
    """The combo ``prefer_polarsds=True`` x ``target_type="learning_to_rank"`` must construct without error AND must NOT canonicalise away (the canon-table previously didn't pin it)."""
    combo = _make_combo(prefer_polarsds=True, target_type="learning_to_rank")
    assert combo.prefer_polarsds is True
    assert combo.target_type == "learning_to_rank"
    # Sanity: same combo with prefer_polarsds=False is distinct (otherwise the polarsds axis has collapsed and the pin is meaningless).
    combo_no_pld = _make_combo(prefer_polarsds=False, target_type="learning_to_rank")
    assert combo.canonical_key() != combo_no_pld.canonical_key(), "F2: prefer_polarsds True/False should produce distinct canonical keys for LTR"


# ---------------------------------------------------------------------------
# F5: PySR x inject_inf_nan canon
# ---------------------------------------------------------------------------


def test_F5_pysr_x_inf_nan_canonicalises_to_pysr_off():
    """When ``inject_inf_nan=True`` AND ``prep_ext_pysr_enabled_cfg=True``, the canonical key must collapse to the PySR-disabled variant. PySR cannot consume non-finite values, so running it is guaranteed to crash -- the canon prevents the pairwise sampler from wasting budget on a known-bad combo."""
    combo_bad = _make_combo(inject_inf_nan=True, prep_ext_pysr_enabled_cfg=True, target_type="regression")
    combo_canon = _make_combo(inject_inf_nan=True, prep_ext_pysr_enabled_cfg=False, target_type="regression")
    assert combo_bad.canonical_key() == combo_canon.canonical_key(), "F5: prep_ext_pysr_enabled_cfg=True must canonicalise to False when inject_inf_nan=True"


# ---------------------------------------------------------------------------
# F6: composite_discovery x outlier_detection in {lof, ocsvm} reachable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("od", ["lof", "ocsvm"])
def test_F6_composite_discovery_x_outlier_detection_reachable(od):
    """The combo ``composite_discovery_enabled_cfg=True`` x ``outlier_detection=lof/ocsvm`` must construct AND keep both axis values intact through canonicalisation. Regression sensor against the 0-row-val cluster (collected by w8c heartbeat 2026-05-25)."""
    combo = _make_combo(
        composite_discovery_enabled_cfg=True,
        outlier_detection=od,
        target_type="regression",
    )
    assert combo.composite_discovery_enabled_cfg is True
    assert combo.outlier_detection == od


# ---------------------------------------------------------------------------
# F7: dummy_baselines=False x baseline_diagnostics=True reachable
# ---------------------------------------------------------------------------


def test_F7_baselines_off_x_diagnostics_on_combo_reachable():
    """The combo ``dummy_baselines_enabled_cfg=False`` x ``baseline_diagnostics_enabled_cfg=True`` must construct without canon-collapse. The proposal notes this asymmetric path needs explicit fuzz coverage so any silent fallback in production (diagnostics computing dummy stats internally vs raising) surfaces deterministically."""
    combo = _make_combo(
        dummy_baselines_enabled_cfg=False,
        baseline_diagnostics_enabled_cfg=True,
    )
    assert combo.dummy_baselines_enabled_cfg is False
    assert combo.baseline_diagnostics_enabled_cfg is True
    # Distinct from the both-enabled variant.
    combo_both = _make_combo(
        dummy_baselines_enabled_cfg=True,
        baseline_diagnostics_enabled_cfg=True,
    )
    assert combo.canonical_key() != combo_both.canonical_key(), "F7: dummy_baselines_enabled_cfg should NOT canonicalise to True when False"


# ---------------------------------------------------------------------------
# 2026-07-25 audit: 9 previously-unfuzzed axes (6 MRMR fe_*_enable toggles,
# gt_07 fe_budget_learning, gt_03 shap_proxied_prescreen_ranking, gt_05
# composite_gate_kind) -- present in AXES + build kwargs correctly.
# ---------------------------------------------------------------------------

_NEW_MRMR_TOGGLE_AXES = (
    "mrmr_fe_random_fourier_enable_cfg",
    "mrmr_fe_sir_direction_enable_cfg",
    "mrmr_fe_lof_enable_cfg",
    "mrmr_fe_mahalanobis_density_enable_cfg",
    "mrmr_fe_ordinal_pattern_enable_cfg",
    "mrmr_fe_conditional_quantile_rank_enable_cfg",
)


@pytest.mark.parametrize("axis_name", _NEW_MRMR_TOGGLE_AXES)
def test_new_mrmr_fe_toggle_axis_present_and_wired(axis_name):
    """AXES must declare the axis with both False/True, and build_mrmr_kwargs must forward it under its
    bare (non-``_cfg``) MRMR.__init__ name."""
    assert axis_name in AXES, f"missing axis entry: {axis_name}"
    values = AXES[axis_name]
    assert False in values and True in values, f"axis must cover both False/True, got {values!r}"

    from tests.training._fuzz_combo.builders import build_mrmr_kwargs

    bare_name = axis_name[len("mrmr_") : -len("_cfg")]
    combo = _make_combo(use_mrmr_fs=True, **{axis_name: True})
    kw = build_mrmr_kwargs(combo)
    assert kw is not None and kw.get(bare_name) is True, f"{bare_name} not forwarded as True in build_mrmr_kwargs output"


def test_new_mrmr_fe_budget_learning_axis_present_and_wired():
    """mrmr_fe_budget_learning_cfg covers False/True/"auto" and forwards to MRMR.__init__'s fe_budget_learning."""
    assert "mrmr_fe_budget_learning_cfg" in AXES
    values = AXES["mrmr_fe_budget_learning_cfg"]
    assert set(values) == {False, True, "auto"}, f"expected {{False, True, 'auto'}}, got {values!r}"

    from tests.training._fuzz_combo.builders import build_mrmr_kwargs

    combo = _make_combo(use_mrmr_fs=True, mrmr_fe_budget_learning_cfg=True)
    kw = build_mrmr_kwargs(combo)
    assert kw is not None and kw.get("fe_budget_learning") is True


def test_new_shap_proxied_prescreen_ranking_axis_present_and_wired():
    """shap_proxied_prescreen_ranking_cfg forwards to ShapProxiedFS.__init__'s prescreen_ranking."""
    assert "shap_proxied_prescreen_ranking_cfg" in AXES
    values = AXES["shap_proxied_prescreen_ranking_cfg"]
    assert "mean_abs_phi" in values and "banzhaf" in values

    from tests.training._fuzz_combo.builders import build_shap_proxied_fs_kwargs

    combo = _make_combo(use_shap_proxied_fs=True, shap_proxied_prescreen_ranking_cfg="banzhaf")
    kw = build_shap_proxied_fs_kwargs(combo)
    assert kw is not None and kw.get("prescreen_ranking") == "banzhaf"


def test_new_composite_gate_kind_axis_present_and_wired():
    """composite_gate_kind_cfg forwards to CompositeTargetDiscoveryConfig.gate_kind, gated on
    composite_discovery_enabled_cfg AND a regression target (the same gate the sibling composite_*_cfg
    axes use)."""
    assert "composite_gate_kind_cfg" in AXES
    values = AXES["composite_gate_kind_cfg"]
    assert "nnls" in values and "shapley" in values

    from tests.training._fuzz_combo.builders import build_composite_discovery_config_from_flat

    cfg = build_composite_discovery_config_from_flat(enabled=True, gate_kind="shapley")
    assert cfg.gate_kind == "shapley"

    combo = _make_combo(
        composite_discovery_enabled_cfg=True,
        target_type="regression",
        composite_gate_kind_cfg="shapley",
    )
    assert combo.composite_gate_kind_cfg == "shapley"
    combo_off = _make_combo(
        composite_discovery_enabled_cfg=False,
        composite_gate_kind_cfg="shapley",
    )
    assert combo_off.canonical_key() != combo.canonical_key()
