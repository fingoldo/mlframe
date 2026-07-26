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
# 2026-07-25 audit: 6 unfuzzed MRMR fe_*_enable toggles + fe_budget_learning +
# shap_proxied prescreen_ranking + composite gate_kind, all newly wired axes.
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
    """Each new MRMR fe_*_enable toggle must be in AXES and reach FuzzCombo, gated on use_mrmr_fs."""
    assert axis_name in AXES, f"missing axis entry: {axis_name}"
    values = AXES[axis_name]
    assert False in values and True in values, f"axis must cover both False/True, got {values!r}"
    combo_on = _make_combo(use_mrmr_fs=True, **{axis_name: True})
    combo_off = _make_combo(use_mrmr_fs=True, **{axis_name: False})
    assert getattr(combo_on, axis_name) is True
    assert combo_on.canonical_key() != combo_off.canonical_key(), f"{axis_name} True/False must be distinct when use_mrmr_fs=True"
    combo_gated = _make_combo(use_mrmr_fs=False, **{axis_name: True})
    combo_gated_off = _make_combo(use_mrmr_fs=False, **{axis_name: False})
    assert combo_gated.canonical_key() == combo_gated_off.canonical_key(), f"{axis_name} must canonicalise away when use_mrmr_fs=False"


def test_new_mrmr_fe_budget_learning_axis_present_and_wired():
    """``mrmr_fe_budget_learning_cfg`` must be in AXES and reach FuzzCombo, gated on use_mrmr_fs."""
    assert "mrmr_fe_budget_learning_cfg" in AXES
    values = AXES["mrmr_fe_budget_learning_cfg"]
    assert False in values and True in values and "auto" in values
    combo = _make_combo(use_mrmr_fs=True, mrmr_fe_budget_learning_cfg=True)
    assert combo.mrmr_fe_budget_learning_cfg is True
    combo_gated = _make_combo(use_mrmr_fs=False, mrmr_fe_budget_learning_cfg=True)
    assert combo_gated.canonical_key() == _make_combo(use_mrmr_fs=False, mrmr_fe_budget_learning_cfg="auto").canonical_key()


def test_new_shap_proxied_prescreen_ranking_axis_present_and_wired():
    """``shap_proxied_prescreen_ranking_cfg`` must be in AXES and reach FuzzCombo, gated on use_shap_proxied_fs."""
    assert "shap_proxied_prescreen_ranking_cfg" in AXES
    values = AXES["shap_proxied_prescreen_ranking_cfg"]
    assert "mean_abs_phi" in values and "banzhaf" in values
    combo = _make_combo(use_shap_proxied_fs=True, shap_proxied_prescreen_ranking_cfg="banzhaf")
    assert combo.shap_proxied_prescreen_ranking_cfg == "banzhaf"
    combo_gated = _make_combo(use_shap_proxied_fs=False, shap_proxied_prescreen_ranking_cfg="banzhaf")
    assert combo_gated.canonical_key() == _make_combo(use_shap_proxied_fs=False, shap_proxied_prescreen_ranking_cfg="mean_abs_phi").canonical_key()


def test_new_composite_gate_kind_axis_present_and_wired():
    """``composite_gate_kind_cfg`` must be in AXES and reach FuzzCombo, gated on composite_discovery_enabled_cfg x regression."""
    assert "composite_gate_kind_cfg" in AXES
    values = AXES["composite_gate_kind_cfg"]
    assert "nnls" in values and "shapley" in values
    combo = _make_combo(composite_discovery_enabled_cfg=True, target_type="regression", composite_gate_kind_cfg="shapley")
    assert combo.composite_gate_kind_cfg == "shapley"
    combo_gated = _make_combo(composite_discovery_enabled_cfg=False, target_type="regression", composite_gate_kind_cfg="shapley")
    assert combo_gated.canonical_key() == _make_combo(composite_discovery_enabled_cfg=False, target_type="regression", composite_gate_kind_cfg="nnls").canonical_key()
