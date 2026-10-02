"""The generated selector parameter models mirror the selector constructors, and import without them.

``mlframe.training.fs_params.<selector>`` are generated from the constructors' signatures. If a constructor gains, loses or retypes a parameter
and the models are not regenerated, a config would silently accept stale names or reject new ones -- exactly what generating them is meant to
prevent. Regenerate with ``python -m mlframe.training.fs_params._generate``.
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys

import pytest

signature_models = pytest.importorskip("pyutilz.dev.signature_models")

from mlframe.training.fs_params import _generate
from mlframe.training.fs_params._spec import SPECS


@pytest.mark.parametrize("spec", SPECS, ids=[s.key for s in SPECS])
def test_the_generated_model_has_no_drift_against_the_constructor(spec):
    """No parameter added, removed, re-defaulted, made (non-)required or retyped since the module was generated."""
    model = getattr(importlib.import_module(f"mlframe.training.fs_params.{spec.key}"), spec.class_name)
    assert signature_models.signature_drift(model, spec.target(), exclude=spec.exclude, check_annotations=False) == []


@pytest.mark.parametrize("spec", SPECS, ids=[s.key for s in SPECS])
def test_a_fresh_render_defines_the_same_model_as_the_committed_module(spec, tmp_path, monkeypatch):
    """Executing a fresh render gives the model the committed module defines: same fields, annotations and defaults (a hand edit or a stale module fails)."""
    committed = getattr(importlib.import_module(f"mlframe.training.fs_params.{spec.key}"), spec.class_name)
    fresh_path = tmp_path / f"fresh_{spec.key}.py"
    fresh_path.write_text(_generate.render(spec), encoding="utf-8")
    module_spec = importlib.util.spec_from_file_location(f"fresh_{spec.key}", fresh_path)
    fresh_module = importlib.util.module_from_spec(module_spec)
    monkeypatch.setitem(sys.modules, module_spec.name, fresh_module)  # pydantic resolves forward references through the defining module
    module_spec.loader.exec_module(fresh_module)
    fresh = getattr(fresh_module, spec.class_name)
    fresh.model_rebuild(_types_namespace=vars(fresh_module))
    assert list(fresh.model_fields) == list(committed.model_fields)
    for name, field in committed.model_fields.items():
        assert fresh.model_fields[name].annotation == field.annotation, name
        assert fresh.model_fields[name].default == field.default, name


def test_the_enum_parameters_accept_exactly_the_selectors_own_values():
    """A ``Literal`` field is built from the selector's accepted-value tuple, and every such value validates."""
    import typing

    for spec in SPECS:
        model = getattr(importlib.import_module(f"mlframe.training.fs_params.{spec.key}"), spec.class_name)
        for name, values in spec.enums().items():
            annotation = model.model_fields[name].annotation
            accepted = set()
            stack = [annotation]
            while stack:
                tp = stack.pop()
                if typing.get_origin(tp) is typing.Literal:
                    accepted.update(typing.get_args(tp))
                else:
                    stack.extend(typing.get_args(tp))
                    if tp is type(None):
                        accepted.add(None)
            assert accepted == set(values), f"{spec.key}.{name}: {accepted} != {set(values)}"


def test_importing_the_config_models_does_not_import_the_selectors():
    """The point of generating: building a config must not pay for importing MRMR, RFECV or shap."""
    lines = [
        "import sys",
        "import mlframe.training",
        "before = set(sys.modules)",
        "import mlframe.training.fs_params.configs",
        "new = sorted(set(sys.modules) - before)",
        "heavy = [m for m in new if m.startswith(('mlframe.feature_selection', 'shap', 'catboost'))]",
        "assert not heavy, heavy",
    ]
    result = subprocess.run([sys.executable, "-c", chr(10).join(lines)], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
