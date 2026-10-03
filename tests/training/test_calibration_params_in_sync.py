"""The generated calibration parameter models mirror the callables they were generated from."""

from __future__ import annotations

import importlib
import importlib.util
import sys

import pytest

signature_models = pytest.importorskip("pyutilz.dev.signature_models")

from mlframe.training.calibration_params import _generate
from mlframe.training.calibration_params._spec import SPECS


@pytest.mark.parametrize("spec", SPECS, ids=[s.key for s in SPECS])
def test_the_generated_model_has_no_drift_against_the_callable(spec):
    """No parameter added, removed, re-defaulted or retyped since the module was generated."""
    model = getattr(importlib.import_module(f"mlframe.training.calibration_params.{spec.key}"), spec.class_name)
    assert signature_models.signature_drift(model, spec.target(), exclude=spec.exclude, check_annotations=False) == []


def test_the_generator_check_passes():
    """``python -m mlframe.training.calibration_params._generate --check`` finds every committed module equal to a fresh render."""
    assert _generate.main(["--check"]) == 0


@pytest.mark.parametrize("spec", SPECS, ids=[s.key for s in SPECS])
def test_a_fresh_render_defines_the_same_model_as_the_committed_module(spec, tmp_path, monkeypatch):
    """Executing a fresh render gives the committed module's model: same fields, annotations and defaults."""
    committed = getattr(importlib.import_module(f"mlframe.training.calibration_params.{spec.key}"), spec.class_name)
    fresh_path = tmp_path / f"fresh_{spec.key}.py"
    fresh_path.write_text(_generate.render(spec), encoding="utf-8")
    module_spec = importlib.util.spec_from_file_location(f"fresh_{spec.key}", fresh_path)
    fresh_module = importlib.util.module_from_spec(module_spec)
    monkeypatch.setitem(sys.modules, module_spec.name, fresh_module)
    module_spec.loader.exec_module(fresh_module)
    fresh = getattr(fresh_module, spec.class_name)
    fresh.model_rebuild(_types_namespace=vars(fresh_module))
    assert list(fresh.model_fields) == list(committed.model_fields)
    for name, field in committed.model_fields.items():
        assert fresh.model_fields[name].annotation == field.annotation, name
        assert fresh.model_fields[name].default == field.default, name


def test_the_sparse_models_forward_only_what_was_written():
    """``to_kwargs`` carries the written fields, a dict round trip rebuilds an equal model, and an unknown name raises."""
    from mlframe.training.calibration_params.configs import IsotonicRiskConfig, ThresholdOptimizerConfig

    cfg = ThresholdOptimizerConfig(n_thresholds=50, cv=3)
    assert cfg.to_kwargs() == {"n_thresholds": 50, "cv": 3}
    assert ThresholdOptimizerConfig(**cfg.model_dump()) == cfg
    with pytest.raises(ValueError, match="n_threshold"):
        ThresholdOptimizerConfig(n_threshold=50)
    with pytest.raises(ValueError):
        IsotonicRiskConfig(segment_ratio_threshold=0.0)


def test_importing_the_calibration_models_does_not_import_the_calibration_steps():
    """Building a config must not import the calibration modules the models were generated from."""
    import subprocess

    lines = [
        "import sys",
        "import mlframe.training",
        "before = set(sys.modules)",
        "import mlframe.training.calibration_params.configs",
        "new = sorted(set(sys.modules) - before)",
        "heavy = [m for m in new if m.startswith(('mlframe.calibration', 'sklearn.isotonic'))]",
        "assert not heavy, heavy",
    ]
    result = subprocess.run([sys.executable, "-c", chr(10).join(lines)], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
