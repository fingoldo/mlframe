"""Every field of the training config surface should have a corresponding fuzz axis.

The fuzz combo system (``tests/training/_fuzz_combo/``) exists to exercise the real training config
surface, not just a hand-picked subset of it. A config field nobody ever randomises is a field that
silently stays at its library default across the entire fuzz suite -- any bug that only shows up when
that field is flipped (``remove_constant_columns=False``, ``skip_categorical_encoding=True``, a non-default
``category_encoder``, ...) will never be caught by fuzzing, no matter how many combos run.

This is baselined, not gated outright: 12 config classes carry ~490 fields between them (one,
``CompositeTargetDiscoveryConfig``, alone has 179), and closing every gap is a standing project, not a
single commit. The number that matters is whether a NEW gap appears -- the baseline may only shrink.

Heuristic, not semantic: a field counts as "covered" when its exact name appears as a whole word
anywhere in the ``_fuzz_combo`` package's source (axis definitions, ``FuzzCombo`` fields, builder
kwargs). This under-counts gaps on short/generic names that collide with unrelated identifiers
(``n_rows``, ``tail``, ``columns``) and over-counts on names only ever mentioned in a comment -- it is
a tripwire for regressions, not a proof of real behavioral coverage.
"""

from __future__ import annotations

import re
from pathlib import Path

import orjson

import mlframe.training.configs as _cfgmod

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FUZZ_COMBO_DIR = _REPO_ROOT / "tests" / "training" / "_fuzz_combo"
_BASELINE = Path(__file__).resolve().parent / "_fuzz_config_field_coverage_baseline.json"

#: The training config surface the fuzz harness is meant to exercise (see the project's own fuzz-mode
#: convention: every combo canonicalises into one of these objects or a direct kwarg of
#: ``train_mlframe_models_suite``).
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


def uncovered_fields(classes: dict, corpus: str) -> list[str]:
    """``"ClassName.field"`` for every pydantic field of ``classes`` never mentioned (whole word) in ``corpus``."""
    out = []
    for cls_name, cls in classes.items():
        for field_name in cls.model_fields:
            if not re.search(rf"\b{re.escape(field_name)}\b", corpus):
                out.append(f"{cls_name}.{field_name}")
    return sorted(out)


def _fuzz_combo_corpus() -> str:
    """Concatenated source of every ``tests/training/_fuzz_combo/*.py`` module -- the fuzz harness's own axis-definition surface."""
    parts = [path.read_text(encoding="utf-8", errors="replace") for path in sorted(_FUZZ_COMBO_DIR.glob("*.py"))]
    assert len(parts) >= 5, f"only {len(parts)} modules found under {_FUZZ_COMBO_DIR} -- the fuzz_combo package moved or shrank"
    return "\n".join(parts)


def test_no_new_uncovered_config_fields():
    """A config field with no corresponding fuzz axis is baselined; the baseline may only shrink."""
    got = set(uncovered_fields(_CONFIG_CLASSES, _fuzz_combo_corpus()))
    recorded = set(orjson.loads(_BASELINE.read_text(encoding="utf-8")))
    new = sorted(got - recorded)
    gone = sorted(recorded - got)
    assert not new, (
        f"{len(new)} config field(s) have no fuzz axis and are not yet baselined -- add a FuzzCombo axis "
        f"covering them (tests/training/_fuzz_combo/combo.py + axes.py + builders.py), or add them to "
        f"{_BASELINE.name} if genuinely not fuzzable: {new}"
    )
    assert not gone, f"these fields are now covered by a fuzz axis -- drop them from {_BASELINE.name}: {gone}"


def test_the_scan_flags_an_uncovered_field_and_honours_a_covered_one():
    """Canary: a field name absent from the corpus is flagged; one present (even as a longer identifier) is not, but only via a real whole-word boundary."""

    class _Fake:
        """Stand-in for a pydantic config class: only ``model_fields`` is read."""

        model_fields = {"totally_uncovered_field": None, "fillna_value": None, "scaler": None}

    corpus = "combo.fillna_value_cfg = 1.0\npreprocessing_config = PreprocessingConfig(fillna_value=1.0, scaler='standard')\n"
    assert uncovered_fields({"Fake": _Fake}, corpus) == ["Fake.totally_uncovered_field"]
