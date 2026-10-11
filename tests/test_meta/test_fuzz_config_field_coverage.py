"""Every field of the training config surface should have a corresponding fuzz axis.

The fuzz combo system exists to exercise the real training config surface, not just a hand-picked
subset of it. A config field nobody ever randomises silently stays at its library default across the
entire fuzz suite -- any bug that only shows up when that field is flipped (``remove_constant_columns=
False``, ``skip_categorical_encoding=True``, a non-default ``category_encoder``, ...) will never be
caught by fuzzing, no matter how many combos run.

Two coverage mechanisms, checked together:

1. **Named axes** -- a ``FuzzCombo`` field whose exact name appears (whole word) in the fuzz harness's
   own source: ``tests/training/_fuzz_combo/*.py``, ``tests/training/_fuzz_suite_helpers.py``,
   ``tests/training/fuzz/test_fuzz_suite.py``, ``tests/training/run_fuzz_10k.py``,
   ``profiling/profile_one_combo.py``.
2. **Generic randomization** -- :func:`tests.training._fuzz_combo.field_randomizer.randomize_scalar_fields`
   resamples every bool / bounded-numeric / Literal / Enum field a named axis left at its default,
   opt-in via ``MLFRAME_FUZZ_RANDOMIZE_ALL_FIELDS`` (see ``_fuzz_suite_helpers.py`` /
   ``profile_one_combo.py`` / ``builders.py`` for the wiring). A field counts as covered this way only
   when its OWN config class is actually wired through that mechanism (``_RANDOMIZER_WIRED_CLASSES``
   below) -- ``ModelHyperparamsConfig`` additionally round-trips through its own model via
   ``_randomize_hyperparams_dict`` (its builder returns a plain dict, not a model instance).

What's left after both mechanisms (see ``_NOT_GENERICALLY_FUZZABLE`` below) is a short, reviewed,
per-field-reasoned list -- not an accepted debt pile. Closing one of these needs real domain work (a
frame-derived column name, a structured dict payload, a validator-constrained string whose allowed set
isn't visible in the type annotation), not a mechanical re-run of this test. The remaining count may
only shrink; a new entry needs the same kind of reason the existing ones have, not a bare name.
"""

from __future__ import annotations

import re
from pathlib import Path

import mlframe.training.configs as _cfgmod
from tests.training._fuzz_combo.field_randomizer import GENERICALLY_FUZZABLE_SHAPE

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FUZZ_COMBO_DIR = _REPO_ROOT / "tests" / "training" / "_fuzz_combo"
_CORPUS_FILES = (
    "tests/training/_fuzz_suite_helpers.py",
    "tests/training/fuzz/test_fuzz_suite.py",
    "tests/training/run_fuzz_10k.py",
    "profiling/profile_one_combo.py",
)

#: The training config surface the fuzz harness is meant to exercise.
_CONFIG_CLASSES = {
    "TrainingConfig": _cfgmod.TrainingConfig,
    "FeatureSelectionConfig": _cfgmod.FeatureSelectionConfig,
    "FeatureTypesConfig": _cfgmod.FeatureTypesConfig,
    "PreprocessingConfig": _cfgmod.PreprocessingConfig,
    "TrainingSplitConfig": _cfgmod.TrainingSplitConfig,
    "PreprocessingBackendConfig": _cfgmod.PreprocessingBackendConfig,
    "TrainingBehaviorConfig": _cfgmod.TrainingBehaviorConfig,
    "ModelHyperparamsConfig": _cfgmod.ModelHyperparamsConfig,
    "LinearModelConfig": _cfgmod.LinearModelConfig,
    "OutputConfig": _cfgmod.OutputConfig,
    "ReportingConfig": _cfgmod.ReportingConfig,
    "CompositeTargetDiscoveryConfig": _cfgmod.CompositeTargetDiscoveryConfig,
}

#: Classes whose construction site actually passes the result through ``randomize_scalar_fields``
#: (directly, or -- for ``ModelHyperparamsConfig``, whose builder returns a plain dict, not a model --
#: via a round trip through the real model in ``_randomize_hyperparams_dict``). See
#: ``_fuzz_suite_helpers.py`` (``_configs_for_combo``, ``_preprocessing_for_combo``,
#: ``_feature_selection_config_for_combo``, ``_randomize_hyperparams_dict``),
#: ``tests/training/fuzz/test_fuzz_suite.py`` (``output_config=``/``reporting_config=``/
#: ``_build_linear_cfg``) and ``tests/training/_fuzz_combo/builders.py``
#: (``build_composite_discovery_config``). ``TrainingConfig`` is NOT here: it is never built as a single
#: object in the fuzz harness (its fields reach ``train_mlframe_models_suite`` as direct kwargs), so
#: there is no call site to wire the randomizer into.
_RANDOMIZER_WIRED_CLASSES = {
    "PreprocessingBackendConfig",
    "TrainingSplitConfig",
    "FeatureTypesConfig",
    "TrainingBehaviorConfig",
    "CompositeTargetDiscoveryConfig",
    "PreprocessingConfig",
    "FeatureSelectionConfig",
    "OutputConfig",
    "ReportingConfig",
    "LinearModelConfig",
    "ModelHyperparamsConfig",
}

#: Reviewed, reasoned remainder -- every entry needs real domain work, not a mechanical axis.
_NOT_GENERICALLY_FUZZABLE = {
    # Bare callables: there is no value space to sample from.
    "TrainingConfig.metamodel_func",
    "TrainingBehaviorConfig.metamodel_func",
    "ReportingConfig.custom_ice_metric",
    "ReportingConfig.custom_rice_metric",
    # Structured dict/nested-model payloads a blind randomizer would corrupt into an invalid shape
    # (each key has its own contract, e.g. a scoring dict's values must be sklearn scorer names).
    "TrainingConfig.linear_config",
    "TrainingBehaviorConfig.callback_params",
    "TrainingBehaviorConfig.cb_fit_params",
    "TrainingBehaviorConfig.default_classification_scoring",
    "TrainingBehaviorConfig.default_regression_scoring",
    "TrainingBehaviorConfig.isotonic_risk_kwargs",
    "TrainingBehaviorConfig.threshold_optimizer_kwargs",
    "TrainingBehaviorConfig.precomputed_fairness_subgroups",
    "OutputConfig.diagnostics_kwargs",
    "ReportingConfig.decision_costs",
    "ReportingConfig.diagnostic_splits",
    "ReportingConfig.feature_importance_config",
    "ReportingConfig.learning_curve",
    "ModelHyperparamsConfig.catboost_custom_classif_metrics",
    "ModelHyperparamsConfig.catboost_custom_regr_metrics",
    # Column names the real frame must contain -- a random string just trips "column not found", testing
    # nothing about the lever itself. Needs a combo-aware axis that reads the actual built frame's columns.
    "TrainingSplitConfig.id_column",
    "CompositeTargetDiscoveryConfig.group_column",
    "CompositeTargetDiscoveryConfig.per_group_column",
    "CompositeTargetDiscoveryConfig.engineer_causal_group_column",
    "CompositeTargetDiscoveryConfig.dominant_features_hint",
    "CompositeTargetDiscoveryConfig.forbidden_base_patterns",
    # A real file path / split-id file on disk, or an explicit calendar boundary -- needs the actual
    # split machinery's state, not a random string/date.
    "TrainingSplitConfig.split_ids_path",
    "TrainingSplitConfig.reuse_splits",
    "TrainingSplitConfig.test_start",
    "TrainingSplitConfig.test_end",
    "TrainingSplitConfig.val_start",
    "TrainingSplitConfig.val_end",
    # Causal-lag feature-engineering knobs: lags/ops/windows only mean something together, as a coherent
    # recipe, and first_fill only applies when lags is non-empty -- needs a dedicated semantic axis, not
    # independent random values per field.
    "CompositeTargetDiscoveryConfig.engineer_causal_lags",
    "CompositeTargetDiscoveryConfig.engineer_causal_ops",
    "CompositeTargetDiscoveryConfig.engineer_causal_trailing_windows",
    "CompositeTargetDiscoveryConfig.engineer_causal_first_fill",
    # ``str``-typed but constrained to an allowed set enforced by a runtime validator invisible to the
    # annotation itself (not a ``Literal``) -- needs the validator's actual allowed-value list read out
    # field by field, not a shape-based guess.
    "FeatureTypesConfig.datetime_methods",
    "LinearModelConfig.model_type",
    "LinearModelConfig.learning_rate",
    "ModelHyperparamsConfig.def_classif_metric",
    "ModelHyperparamsConfig.def_regr_metric",
    "CompositeTargetDiscoveryConfig.auto_base_null_block_length",
    "CompositeTargetDiscoveryConfig.base_ranking_criterion",
    "CompositeTargetDiscoveryConfig.cross_target_calibration_method",
    "CompositeTargetDiscoveryConfig.fail_on_no_gain",
    "CompositeTargetDiscoveryConfig.tiny_consensus",
    "CompositeTargetDiscoveryConfig.tiny_rerank_backend",
    "TrainingBehaviorConfig.target_temporal_audit_unit",
    "TrainingBehaviorConfig.mlp_drop_per_group_constants_pattern",
    "FeatureSelectionConfig.ace",
    # ``List[str]``/tuple-shaped report-panel selectors: each is a whitelist of named panels/tokens this
    # report family understands; needs that whitelist read out of the renderer, not a random string list.
    "ReportingConfig.binary_panels",
    "ReportingConfig.multilabel_panels",
    "ReportingConfig.regression_panels",
    "ReportingConfig.plot_outputs",
    "ReportingConfig.title_metrics_tokens",
    "ReportingConfig.regression_title_metrics_tokens",
    "ReportingConfig.calibration_colormap",
    "OutputConfig.plot_file",
}


def uncovered_fields(classes: dict, corpus: str, randomizer_wired: frozenset) -> list[str]:
    """``"ClassName.field"`` for every pydantic field of ``classes`` that is neither mentioned (whole word)
    in ``corpus`` nor closed by the generic randomizer (its class is in ``randomizer_wired`` AND its
    annotation shape is one :func:`GENERICALLY_FUZZABLE_SHAPE` recognises)."""
    out = []
    for cls_name, cls in classes.items():
        for field_name, info in cls.model_fields.items():
            if re.search(rf"\b{re.escape(field_name)}\b", corpus):
                continue
            if cls_name in randomizer_wired and GENERICALLY_FUZZABLE_SHAPE(info.annotation):
                continue
            out.append(f"{cls_name}.{field_name}")
    return sorted(out)


def _fuzz_harness_corpus() -> str:
    """Concatenated source of the fuzz harness: the ``_fuzz_combo`` package plus the suite-helper /
    pytest-suite / profiling call sites that build config objects from a combo."""
    parts = [path.read_text(encoding="utf-8", errors="replace") for path in sorted(_FUZZ_COMBO_DIR.glob("*.py"))]
    assert len(parts) >= 5, f"only {len(parts)} modules found under {_FUZZ_COMBO_DIR} -- the fuzz_combo package moved or shrank"
    for rel in _CORPUS_FILES:
        path = _REPO_ROOT / rel
        assert path.is_file(), f"{rel} is part of the fuzz-coverage corpus but no longer exists -- update _CORPUS_FILES"
        parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def test_no_new_uncovered_config_fields():
    """A config field with neither a named axis nor generic-randomizer coverage must already be a reasoned entry in ``_NOT_GENERICALLY_FUZZABLE``."""
    got = set(uncovered_fields(_CONFIG_CLASSES, _fuzz_harness_corpus(), frozenset(_RANDOMIZER_WIRED_CLASSES)))
    new = sorted(got - _NOT_GENERICALLY_FUZZABLE)
    closed = sorted(_NOT_GENERICALLY_FUZZABLE - got)
    assert not new, (
        f"{len(new)} config field(s) have no fuzz axis, no generic-randomizer coverage, and are not yet a "
        f"reasoned entry in _NOT_GENERICALLY_FUZZABLE: {new}"
    )
    assert not closed, f"these fields are now covered -- drop them from _NOT_GENERICALLY_FUZZABLE: {closed}"


def test_the_scan_flags_an_uncovered_field_and_honours_both_coverage_mechanisms():
    """Canary: a field with neither a named mention nor generic-randomizer-eligible class+shape is flagged; one with either is not."""

    class _Wired:
        """Stand-in config class whose construction site is wired through the randomizer."""

        model_fields = {"wired_bool_field": type("F", (), {"annotation": bool})()}

    class _Unwired:
        """Stand-in config class with NO randomizer wiring -- only a named-axis mention can cover its fields."""

        model_fields = {
            "named_field": type("F", (), {"annotation": str})(),
            "totally_uncovered_field": type("F", (), {"annotation": str})(),
        }

    corpus = "cfg = PreprocessingConfig(named_field=combo.named_field_cfg)\n"
    classes = {"Wired": _Wired, "Unwired": _Unwired}
    result = uncovered_fields(classes, corpus, frozenset({"Wired"}))
    assert result == ["Unwired.totally_uncovered_field"]
