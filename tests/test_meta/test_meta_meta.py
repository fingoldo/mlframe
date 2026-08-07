"""F1 + F2 + F3 — meta-tests on the meta-test suite itself.

When the suite grows past 10 files, the suite ITSELF becomes a piece of
production code worth policing. These tests catch:

  F1. Failure messages without actionable detail.
      ``pytest.fail("broken")`` is useless; ``pytest.fail(f"{n} fields
      have no consumer:\n  {names}")`` lets the reviewer act. Audit
      every ``pytest.fail`` call in the meta-test directory; each must
      contain at least one of: a colon, a path separator, an angle-
      bracket placeholder, or a fix-prompt verb (``Add``, ``Either``,
      ``Refresh``, ``Whitelist``).

  F2. Meta-tests reaching into private internals of the code they
      police. The whole point of a meta-test is to cover the public
      contract — if the test imports ``_foo`` from a production module,
      it's testing implementation, not behaviour. Whitelist via
      ``_PERMITTED_PRIVATE_IMPORTS`` for legitimate cases (e.g. the
      lazy-proxy meta-test must touch ``_create_lazy_module`` because
      that IS the surface under test).

  F3. Per-test wall-clock budget. Meta-tests are designed to run in
      seconds — anything > 30 s is a yellow flag (likely accidentally
      doing work that should live in an integration test). Currently
      a soft warning emitted to stderr; PT-8/PT-9 sub-process tests
      and PT-2 alias resolution exceed this and are whitelisted.
"""

from __future__ import annotations

import ast
import re
from functools import cache
from pathlib import Path

import pytest

_TEST_META_DIR = Path(__file__).resolve().parent

# Words / characters that indicate an actionable failure message.
# Match a colon (file:line, key: value), a slash (path), an angle (template
# placeholder), or any of the fix-prompt verbs.
_ACTIONABLE_RE = re.compile(
    r"[:/<>]|\b(Add|Either|Refresh|Whitelist|Fix|Run|Update|Remove|Document|" r"Check|See|Catches|Replace|OR)\b",
    re.IGNORECASE,
)

# Imports of a production private symbol from a meta-test that are
# legitimate. Each entry is "test_meta_filename::imported_dotted_name".
_PERMITTED_PRIVATE_IMPORTS: set[str] = {
    # mlframe meta-tests touching internal helpers — these are the
    # surface under audit, so importing the underscore-prefixed name
    # is part of the test's purpose.
    # log_only_except / baseline-debt wave regression tests touching a private helper directly --
    # each one is a targeted regression test for that exact function, not reachable via the public API.
    # FS_BENCHMARKS_A-1: regression test pinning that _datasets' consumable bindings (SCENARIOS,
    # make_scenario_data, ...) are real module-level attributes, not trapped inside an
    # `if __name__ == "__main__":` guard -- the private module IS the surface under test.
    "test_benchmarks_datasets_importable::mlframe.feature_selection._benchmarks._datasets",
    "test_broad_except_logging_gpu_ktc_and_composite_models::mlframe.data_valuation._propagate_gpu_ktc",
    "test_broad_except_logging_gpu_ktc_and_composite_models::mlframe.inference._ktc_dispatch",
    "test_broad_except_logging_gpu_ktc_and_composite_models::mlframe.training._eval_helpers._append_split_rate_suffix",
    "test_broad_except_logging_gpu_ktc_and_composite_models::mlframe.training._feature_importances._captum_integrated_gradients_importance",
    "test_broad_except_logging_gpu_ktc_and_composite_models::mlframe.training._training_loop._in_interactive_notebook",
    "test_broad_except_logging_gpu_ktc_and_composite_models::mlframe.training._uncertainty_eval._narrow_numeric_frame",
    "test_broad_except_logging_gpu_ktc_and_composite_models::mlframe.training.baselines._dummy_metrics_pick_plot._safe_metric_for_title",
    "test_broad_except_logging_gpu_ktc_and_composite_models::mlframe.training.composite.glm._is_polars_df",
    "test_broad_except_logging_gpu_ktc_and_composite_models::mlframe.training.composite.hpo._default_inner_spaces",
    "test_broad_except_logging_gpu_ktc_and_composite_models::mlframe.training.composite.orthogonal._is_polars_df",
    "test_broad_except_logging_gpu_ktc_and_composite_models::mlframe.training.honest_diagnostics._is_binary_classif",
    "test_broad_except_logging_gpu_ktc_and_composite_models::mlframe.training.phases._try_get_rss_gb",
    "test_broad_except_logging_gpu_ktc_and_composite_models::mlframe.utils.misc._restore_caller_frame_columns",
    "test_broad_except_logging_gpu_ktc_and_composite_models::mlframe.votenrank._confidence_gated_blend_ktc_dispatch",
    "test_broad_except_logging_composite_panel_and_core_predict::mlframe.training.composite.panel._is_polars_df",
    "test_broad_except_logging_composite_panel_and_core_predict::mlframe.training.composite.ranking._is_polars_df",
    "test_broad_except_logging_composite_panel_and_core_predict::mlframe.training.composite.survival._has_scikit_survival",
    "test_broad_except_logging_composite_panel_and_core_predict::mlframe.training.core._achievable_ceiling",
    "test_broad_except_logging_composite_panel_and_core_predict::mlframe.training.core._ar_skip._recompute_lag1_ar_per_group",
    "test_broad_except_logging_composite_panel_and_core_predict::mlframe.training.core._phase_helpers",
    "test_broad_except_logging_composite_panel_and_core_predict::mlframe.training.core._phase_train_one_target_mlp_helpers._drop_columns_for_mlp",
    "test_broad_except_logging_composite_panel_and_core_predict::mlframe.training.core._phase_train_one_target_model_setup._render_per_target_diagnostics",
    "test_broad_except_logging_composite_panel_and_core_predict::mlframe.training.core._phase_train_one_target_polars_fastpath",
    "test_broad_except_logging_composite_panel_and_core_predict::mlframe.training.core._setup_helpers_pipeline_cache",
    "test_broad_except_logging_composite_panel_and_core_predict::mlframe.training.core._volatility_lag_router._extract_column",
    "test_broad_except_logging_composite_panel_and_core_predict::mlframe.training.core.predict._is_post_hoc_calibrated_model",
    "test_broad_except_logging_composite_panel_and_core_predict::mlframe.training.diagnostics.learning_curve._supports_warm_start",
    "test_broad_except_logging_composite_panel_and_core_predict::mlframe.training.pipeline._categorical_composite_fe._detect_cat_columns",
    "test_broad_except_logging_composite_panel_and_core_predict::mlframe.training.pipeline._pipeline_fit_transform",
    "test_broad_except_logging_composite_panel_and_core_predict::mlframe.training.targets._train_eval_select_target",
    "test_broad_except_logging_discovery_and_wrappers::mlframe.feature_selection.wrappers._helpers._pin_threads_to_one",
    "test_broad_except_logging_discovery_and_wrappers::mlframe.feature_selection.wrappers._helpers_importance._fold_is_all_finite",
    "test_broad_except_logging_discovery_and_wrappers::mlframe.feature_selection.wrappers.rfecv._fit_init._current_params_signature",
    "test_broad_except_logging_discovery_and_wrappers::mlframe.models.ensembling._build_votenrank_leaderboard_from_results",
    "test_broad_except_logging_discovery_and_wrappers::mlframe.reporting.charts.quantile._model_diagnostics_decompose",
    "test_broad_except_logging_discovery_and_wrappers::mlframe.reporting.renderers._plotly_color._rgba",
    "test_broad_except_logging_discovery_and_wrappers::mlframe.training.composite.discovery._auto_chain._mi_gain_of",
    "test_broad_except_logging_discovery_and_wrappers::mlframe.training.composite.discovery._filter",
    "test_broad_except_logging_discovery_and_wrappers::mlframe.training.composite.discovery._fit",
    "test_broad_except_logging_discovery_and_wrappers::mlframe.training.composite.discovery._fit_ram",
    "test_broad_except_logging_discovery_and_wrappers::mlframe.training.composite.discovery._ktc_dispatch",
    "test_broad_except_logging_discovery_and_wrappers::mlframe.training.composite.discovery._tiny_rerank._tiny_rerank_ram_checkpoint",
    "test_broad_except_logging_discovery_and_wrappers::mlframe.training.composite.discovery.screening._is_numeric_column",
    "test_broad_except_logging_discovery_and_wrappers::mlframe.training.composite.transforms._grouped_extra._grouped_np_fit",
    "test_broad_except_logging_discovery_and_wrappers::mlframe.training.composite.transforms.extended._smoothing_spline_g",
    "test_broad_except_logging_cb_pool_and_core_helpers::mlframe.training.cb._cb_pool._recover_cb_feature_names",
    "test_broad_except_logging_cb_pool_and_core_helpers::mlframe.training.core._phase_train_one_target._selector_kind",
    "test_broad_except_logging_cb_pool_and_core_helpers::mlframe.training.neural.data._estimate_bytes",
    "test_broad_except_logging_cb_pool_and_core_helpers::mlframe.utils._param_oracle",
    "test_broad_except_logging_gpu_guard_and_training_helpers::mlframe.reporting.diagnostics_dispatch._subset_rows",
    "test_broad_except_logging_gpu_guard_and_training_helpers::mlframe.system._gpu_guard",
    "test_broad_except_logging_gpu_guard_and_training_helpers::mlframe.training._data_helpers._normalize_multilabel_target",
    "test_broad_except_logging_gpu_guard_and_training_helpers::mlframe.training._gpu_probe",
    "test_broad_except_logging_gpu_guard_and_training_helpers::mlframe.training._model_factories",
    "test_broad_except_logging_gpu_guard_and_training_helpers::mlframe.training._predict_guards._recover_cb_feature_names",
    "test_broad_except_logging_gpu_guard_and_training_helpers::mlframe.training._training_loop_refit._maybe_refit_on_collapsed_predictions",
    "test_broad_except_logging_gpu_guard_and_training_helpers::mlframe.training.composite.quantile._transform_inverse_decreasing",
    "test_broad_except_logging_ar1_veto_and_pipeline_helpers::mlframe.training.composite.ensemble._is_monotone_nondecreasing",
    "test_broad_except_logging_ar1_veto_and_pipeline_helpers::mlframe.training.core._phase_composite_post_moe._extract_group_array",
    "test_broad_except_logging_ar1_veto_and_pipeline_helpers::mlframe.training.feature_handling.fingerprint._fp_cache_key",
    "test_broad_except_logging_ar1_veto_and_pipeline_helpers::mlframe.training.neural._flat_torch_module._flat_torch_predict_accel",
    "test_broad_except_logging_ar1_veto_and_pipeline_helpers::mlframe.training.neural._muon_triton_kernel",
    "test_broad_except_logging_ar1_veto_and_pipeline_helpers::mlframe.training.pipeline._pipeline_helpers._selector_output_columns",
    "test_broad_except_logging_ar1_veto_and_pipeline_helpers::mlframe.training.reporting._reporting._frame_to_text",
    "test_broad_except_logging_ar1_veto_and_pipeline_helpers::mlframe.training.strategies.pipeline_cache._estimate_slot_nbytes",
    "test_default_via_or_trap_training_models_and_pipeline_fe::mlframe.training.models._build_ransac_regressor",
    "test_default_via_or_trap_gpu_device_profile::mlframe.feature_selection.filters._fe_gpu_batch._devices._profile_device",
    "test_log_only_except_matplotlib_plotly_style_overrides::mlframe.training.core._phase_helpers",
    "test_log_only_except_registry_and_phase_setup_best_effort::mlframe.training.core._phase_composite_post_xt_ensemble",
    "test_log_only_except_registry_and_phase_setup_best_effort::mlframe.training.core._phase_train_one_target_model_setup",
    "test_log_only_except_reports_and_phase_composite_best_effort::mlframe.training._feature_importances",
    "test_log_only_except_reports_and_phase_composite_best_effort::mlframe.training.core._phase_composite_discovery",
    "test_log_only_except_reports_and_phase_composite_best_effort::mlframe.training.core._phase_dummy_baselines",
    "test_log_only_except_reports_and_phase_composite_best_effort::mlframe.training.core._phase_train_one_target_schema",
    "test_log_only_except_reports_and_phase_composite_best_effort::mlframe.training.pipeline._pipeline_extensions",
    "test_log_only_except_reports_and_phase_composite_best_effort::mlframe.training.reporting._reporting_diagnostics",
    "test_log_only_except_calibration_and_crash_reporting_best_effort::mlframe.calibration._post_train_calibrators",
    "test_log_only_except_final_best_effort_sweep::mlframe.reporting.renderers._kaleido",
    "test_log_only_except_final_best_effort_sweep::mlframe.training._training_loop",
    "test_log_only_except_final_best_effort_sweep::mlframe.training.baselines._dummy_metrics_pick_plot",
    "test_log_only_except_final_best_effort_sweep::mlframe.training.composite.discovery._screening_tiny",
    "test_log_only_except_final_best_effort_sweep::mlframe.training.core._main_train_suite_target_distribution",
    "test_log_only_except_final_best_effort_sweep::mlframe.training.core._predict_main_from_models",
    "test_log_only_except_final_best_effort_sweep::mlframe.training.neural._flat_torch_module",
    "test_log_only_except_final_best_effort_sweep::mlframe.training.neural._flat_torch_module._flat_torch_loss",
    "test_log_only_except_final_best_effort_sweep::mlframe.training.neural._flat_torch_module._flat_torch_predict_accel",
    "test_log_only_except_final_best_effort_sweep::mlframe.training.neural.base._cuda_fallback",
    "test_log_only_except_final_best_effort_sweep::mlframe.training.reporting._reporting",
    "test_config_dataclass_defaults_match_ctor::mlframe.feature_selection.filters.mrmr._mrmr_config_dataclasses._CONFIG_ATTR_FIELD_MAPS",
    "test_config_dataclass_defaults_match_ctor::mlframe.feature_selection.filters.mrmr._mrmr_config_dataclasses._HYBRID_ORTH_FIELD_MAP",
    "test_config_dataclass_defaults_match_ctor::mlframe.feature_selection.filters.mrmr._mrmr_config_dataclasses._HYBRID_ORTH_SCORERS_FIELD_MAP",
    "test_x_architecture_api_consistency_fixes::mlframe.calibration._post_train_calibrators",
    "test_x_architecture_api_consistency_fixes::mlframe.training.baselines._dummy_baseline_compute",
    "test_x_architecture_api_consistency_fixes::mlframe.training.honest_diagnostics._derive_seed",
    "test_x_security_robustness_fixes::mlframe.training._trainer_train_and_evaluate",
    "test_x_security_robustness_fixes::mlframe.training.feature_handling.cache._deserialize",
    "test_calibration_monotonicity::mlframe.training.trainer._PerClassIsotonicCalibrator",
    "test_memory_budgets::mlframe.training.helpers._predict_from_probs",
    "test_metric_invariants::mlframe.training.helpers._predict_from_probs",
    "test_reproducibility::mlframe.training.helpers._predict_from_probs",
    "test_utility_fuzz::mlframe.training.helpers._canonical_predict_proba_shape",
    "test_utility_fuzz::mlframe.training.helpers._predict_from_probs",
}


def _meta_test_files() -> list[Path]:
    """Every ``test_*.py`` file under ``tests/test_meta/`` except this one."""
    out: list[Path] = []
    for py in _TEST_META_DIR.glob("test_*.py"):
        if py.name == Path(__file__).name:
            continue
        out.append(py)
    return sorted(out)


@cache
def _parsed_ast(py: Path) -> ast.AST | None:
    """Read + AST-parse ``py`` once, cached: both F1 and F2 scanners below walk
    the same meta-test file set independently, so an uncached read_text()+ast.parse()
    per scanner doubles the I/O + parse cost."""
    try:
        src = py.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        return ast.parse(src)
    except SyntaxError:
        return None


def _pytest_fail_strings(tree: ast.AST) -> list[tuple[int, str, bool]]:
    """Yield ``(lineno, joined_static_text, has_dynamic)`` for every
    ``pytest.fail(...)`` call.

    ``joined_static_text`` concatenates every ``ast.Constant`` string
    chunk of the first arg, and ``has_dynamic`` is True when the message
    is built from an f-string / ``%``-format / ``+`` concat with a
    non-constant operand — the rich detail then comes from the
    dynamic substitution and the joined constant-only text alone
    won't reflect that.
    """
    out: list[tuple[int, str, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "fail"):
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == "pytest"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        chunks: list[str] = []
        has_dynamic = False
        # ``pytest.fail(msg)`` where ``msg`` is a bare Name / Attribute /
        # Subscript / Call: the message is built entirely outside this
        # expression — definitively dynamic.
        if not isinstance(first, ast.Constant):
            has_dynamic = True
        for sub in ast.walk(first):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                chunks.append(sub.value)
            elif isinstance(sub, ast.FormattedValue):
                has_dynamic = True
            elif isinstance(sub, (ast.Name, ast.Attribute, ast.Subscript, ast.Call)) and sub is not first:
                # A non-string-constant inside the message expression.
                has_dynamic = True
        out.append((node.lineno, " ".join(chunks), has_dynamic))
    return out


def _imports(tree: ast.AST) -> list[str]:
    """Yield fully-qualified imported names from ``import X`` and
    ``from X import Y`` (where Y joins the dotted base)."""
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            for alias in node.names:
                if base:
                    out.append(f"{base}.{alias.name}")
                else:
                    out.append(alias.name)
    return out


# ---------------------------------------------------------------------------
# F1 — actionable failure messages
# ---------------------------------------------------------------------------


def test_every_pytest_fail_call_has_actionable_text():
    """F1: every ``pytest.fail(...)`` call in the meta-test directory must carry actionable detail."""
    bad: list[str] = []
    audited = 0
    for py in _meta_test_files():
        tree = _parsed_ast(py)
        if tree is None:
            continue
        for lineno, text, has_dynamic in _pytest_fail_strings(tree):
            audited += 1
            # A dynamic message (f-string, "...".join(parts), etc.) is
            # presumed actionable — the rich content lives in the
            # runtime substitutions which the static walker can't see.
            if has_dynamic:
                continue
            if not text:
                bad.append(f"{py.name}:{lineno} (empty message)")
                continue
            if not _ACTIONABLE_RE.search(text):
                bad.append(f"{py.name}:{lineno} → {text[:80]!r}")

    if audited == 0:
        pytest.skip("no pytest.fail calls found in meta-test directory")
    if bad:
        pytest.fail(
            f"{len(bad)} pytest.fail message(s) lack actionable detail "
            f"(file paths, fix verbs, or template placeholders). The "
            f"reviewer will need to read the test source to figure out "
            f"what to do — improve the message:\n  " + "\n  ".join(bad[:20])
        )


# ---------------------------------------------------------------------------
# F2 — no private internals reached into without justification
# ---------------------------------------------------------------------------


def test_meta_tests_dont_reach_private_internals():
    """F2: meta-tests must cover the public contract, not reach into unwhitelisted private internals."""
    bad: list[str] = []
    for py in _meta_test_files():
        stem = py.stem
        tree = _parsed_ast(py)
        if tree is None:
            continue
        for imp in _imports(tree):
            # Only audit our own package imports.
            if not (imp.startswith("pyutilz") or imp.startswith("mlframe")):
                continue
            # Last segment with a single leading underscore is "private".
            last = imp.rsplit(".", 1)[-1]
            if not last.startswith("_") or last.startswith("__"):
                continue
            entry = f"{stem}::{imp}"
            if entry in _PERMITTED_PRIVATE_IMPORTS:
                continue
            bad.append(entry)
    if bad:
        pytest.fail(
            f"{len(bad)} meta-test(s) import a private symbol without "
            f"justification. Either use the public API instead, OR "
            f"whitelist via _PERMITTED_PRIVATE_IMPORTS with reasoning:\n  " + "\n  ".join(sorted(set(bad)))
        )


# ---------------------------------------------------------------------------
# F3 — per-meta-test wall-clock budget (advisory)
# ---------------------------------------------------------------------------

# Tests permitted above the soft wall-clock budget (in seconds).
_PERF_BUDGET_OVERRIDES: dict[str, float] = {
    # Walks every config + corpus across mlframe — many aliases.
    "test_subconfig_wiring_parity": 90.0,
    "test_config_field_consumption": 30.0,
    "test_metric_invariants": 30.0,
    "test_dead_helpers": 30.0,
    "test_api_stability": 30.0,
}
_DEFAULT_PERF_BUDGET_S = 10.0


def test_perf_budget_overrides_are_documented():
    """Static check: any test in ``_PERF_BUDGET_OVERRIDES`` corresponds
    to an actual file in the meta-test directory. Catches a stale
    override after a rename.
    """
    test_stems = {p.stem for p in _meta_test_files()}
    stale = [k for k in _PERF_BUDGET_OVERRIDES if k not in test_stems]
    if stale:
        pytest.fail(f"_PERF_BUDGET_OVERRIDES has entries for {stale} which no longer exist in the meta-test dir — clean up after rename")
