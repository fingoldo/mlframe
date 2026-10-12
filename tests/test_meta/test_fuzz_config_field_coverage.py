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

import dataclasses
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

import mlframe.training.configs as _cfgmod
from tests.training._fuzz_combo.field_randomizer import GENERICALLY_FUZZABLE_SHAPE
from tests.training._fuzz_combo.semantic_fields import DERIVED_FIELDS, SEMANTIC_FIELDS

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

#: Classes whose construction site also runs :func:`randomize_semantic_fields` (the same call sites as above, plus ``TrainingConfig``, which the
#: fuzz suite builds only through ``training_config_for_combo`` for the suite's ``training_config=`` carrier).
_SEMANTIC_WIRED_CLASSES = _RANDOMIZER_WIRED_CLASSES | {"TrainingConfig"}

#: Reviewed, reasoned remainder -- fields no mechanism reaches. Empty: every field that used to sit here now has a domain-aware generator in
#: ``semantic_fields.py`` (a real frame column, a replayable split file, a callee-signature callable, a validator's allowed set) or, for
#: ``LinearModelConfig.model_type``, a dedicated ``linear_model_type_cfg`` combo axis. A new entry needs a reason grounded in the consuming
#: code, not a bare name.
_NOT_GENERICALLY_FUZZABLE: frozenset = frozenset()


def uncovered_fields(
    classes: dict,
    corpus: str,
    randomizer_wired: frozenset,
    semantic: frozenset = frozenset(),
    semantic_wired: frozenset = frozenset(),
    derived: Optional[dict] = None,
) -> list[str]:
    """``"ClassName.field"`` for every pydantic field of ``classes`` that no mechanism covers: a whole-word mention in ``corpus``, the generic
    randomizer (class in ``randomizer_wired`` AND a :func:`GENERICALLY_FUZZABLE_SHAPE` annotation), a semantic generator (field in ``semantic``
    AND class in ``semantic_wired``), or derivation from a source field that is itself mentioned (``derived``)."""
    derived = derived or {}
    out = []
    for cls_name, cls in classes.items():
        for field_name, info in cls.model_fields.items():
            key = f"{cls_name}.{field_name}"
            if key in semantic and cls_name in semantic_wired:
                continue
            if re.search(rf"\b{re.escape(field_name)}\b", corpus):
                continue
            if cls_name in randomizer_wired and GENERICALLY_FUZZABLE_SHAPE(info.annotation):
                continue
            src = derived.get(key)
            if src is not None and re.search(rf"\b{re.escape(src.split('.', 1)[1])}\b", corpus):
                continue
            out.append(key)
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
    got = set(
        uncovered_fields(
            _CONFIG_CLASSES,
            _fuzz_harness_corpus(),
            frozenset(_RANDOMIZER_WIRED_CLASSES),
            semantic=SEMANTIC_FIELDS,
            semantic_wired=frozenset(_SEMANTIC_WIRED_CLASSES),
            derived=DERIVED_FIELDS,
        )
    )
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


def test_every_semantic_field_is_a_real_wired_config_field():
    """A semantic generator registered under a misspelled field or class, or for a class whose construction site never runs it, would close a
    gap on paper only."""
    real = {f"{c}.{f}" for c, cls in _CONFIG_CLASSES.items() for f in cls.model_fields}
    assert SEMANTIC_FIELDS <= real, f"semantic generators name non-existent fields: {sorted(SEMANTIC_FIELDS - real)}"
    unwired = sorted(k for k in SEMANTIC_FIELDS if k.split(".", 1)[0] not in _SEMANTIC_WIRED_CLASSES)
    assert not unwired, f"semantic fields whose class is not built through randomize_semantic_fields: {unwired}"


def _semantic_base(cls_name: str) -> Any:
    """A default instance of ``cls_name``, with required fields filled the way its fuzz construction site fills them."""
    if cls_name == "TrainingConfig":
        return _cfgmod.TrainingConfig(target_name="t", model_name="m")
    if cls_name == "OutputConfig":
        return _cfgmod.OutputConfig(data_dir=tempfile.gettempdir())
    return _CONFIG_CLASSES[cls_name]()


def _differs(a: Any, b: Any) -> bool:
    """``a != b`` tolerating values whose ``==`` is not a plain bool (frames or arrays inside payloads)."""
    try:
        return bool(a != b)
    except Exception:
        return a is not b


def test_semantic_generators_move_every_registered_field_off_its_default(monkeypatch):
    """Each semantic field takes a non-default, assignment-validated value on some real combo: a generator that always returns ``{}``, or whose
    value never validates, cannot pass for coverage."""
    from tests.training._fuzz_combo import enumerate_combos
    from tests.training._fuzz_combo.semantic_fields import randomize_semantic_fields

    monkeypatch.setenv("MLFRAME_FUZZ_RANDOMIZE_ALL_FIELDS", "1")
    by_class: dict = {}
    for key in SEMANTIC_FIELDS:
        cls_name, field = key.split(".", 1)
        by_class.setdefault(cls_name, []).append(field)
    hit: set = set()
    combos = [dataclasses.replace(c, n_rows=300) for c in enumerate_combos(target=150, master_seed=7)]
    for i, combo in enumerate(combos):
        for cls_name, fields in by_class.items():
            default = _semantic_base(cls_name)
            sampled = randomize_semantic_fields(
                _semantic_base(cls_name), combo, np.random.default_rng((combo.seed, i)), linear_config=_cfgmod.LinearModelConfig(alpha=0.5)
            )
            hit.update(f"{cls_name}.{f}" for f in fields if _differs(getattr(sampled, f), getattr(default, f)))
        if hit == SEMANTIC_FIELDS:
            break
    assert hit == SEMANTIC_FIELDS, f"never moved off the default across {len(combos)} combos: {sorted(SEMANTIC_FIELDS - hit)}"


def test_derived_fields_follow_their_source():
    """``title_metrics_tokens`` is overwritten from ``title_metrics_template`` at every construction, so varying the template (a named axis) is
    the only way the tokens vary. Pinned so a change making the tokens independently settable re-opens the gap instead of hiding it."""
    for derived, source in DERIVED_FIELDS.items():
        cls_name, field = derived.split(".", 1)
        src_field = source.split(".", 1)[1]
        cfg = _CONFIG_CLASSES[cls_name](**{src_field: "ROC_AUC KS"})
        assert getattr(cfg, field) == ("ROC_AUC", "KS")
        cfg = _CONFIG_CLASSES[cls_name](**{src_field: "ROC_AUC KS", field: ("ICE",)})
        assert getattr(cfg, field) == ("ROC_AUC", "KS"), f"{derived} became independently settable; give it its own generator"
