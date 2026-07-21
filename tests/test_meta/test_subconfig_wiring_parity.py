"""Meta-test — every BaseModel-typed field on a config class (a *sub-config*
field) must demonstrably reach the trainer / pipeline that should consume it.

Catches the failure mode encountered in the audit-2026-04-28 sweep:
``TrainingConfig`` declared ``linear_config: LinearModelConfig`` and
``behavior: TrainingBehaviorConfig`` as sub-configs, but ``train_and_evaluate
_model`` accepted standalone parameters named ``linear_model_config`` and
``behavior_config`` instead — so any TrainingConfig.linear_config / .behavior
the user populated was silently discarded. The orphaned fields were
*structurally* unused even though their CLASS name appeared elsewhere in
the corpus (because the trainer used the SAME class but a different field
name).

The test enumerates every sub-config field on every parent config and asserts
that the literal field name (``cfg.linear_config``, ``cfg.behavior``) appears
in the production corpus — NOT just the underlying class name. A class can be
instantiated freely; only the *field-on-parent-config* path is policed here.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from pydantic import BaseModel

import mlframe
from mlframe.training import configs as configs_module

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent

# Hard whitelist for sub-config fields known to be consumed via routes the
# bare-name grep can't see. Cite consumer location.
_KNOWN_INDIRECT_SUBCONFIGS: dict[str, str] = {}

# User-deferred, mirrors the same list in test_config_field_consumption.py
# but scoped to sub-config fields so the failure message is targeted.
_USER_DEFERRED_SUBCONFIGS: dict[str, str] = {
    "TrainingConfig.linear_config": "shadowed by trainer.py kwarg `linear_model_config`",
    "TrainingConfig.tree_config": "shadowed by trainer.py kwarg `tree_model_config`",
    "TrainingConfig.mlp_config": "shadowed by trainer.py kwarg `mlp_config`",
    "TrainingConfig.ngb_config": "shadowed by trainer.py kwarg `ngb_model_config`",
    "TrainingConfig.behavior": "shadowed by trainer.py kwarg `behavior_config`",
}


def _consumer_corpus() -> str:
    """Returns ``'\n'.join(chunks)`` (after 2 setup steps)."""
    chunks: list[str] = []
    for py in MLFRAME_DIR.rglob("*.py"):
        if py.resolve() == Path(configs_module.__file__).resolve():
            continue
        if "test" in py.parts or "__pycache__" in py.parts:
            continue
        try:
            chunks.append(py.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
    return "\n".join(chunks)


def _is_basemodel_field(cls: type[BaseModel], field_name: str) -> bool:
    """Does this field's annotation resolve to a BaseModel subclass?

    Looks through ``Optional[...]`` / ``Union[...]`` wrappers — many
    sub-config fields are typed as ``Optional[LinearModelConfig]``.
    """
    field_info = cls.model_fields[field_name]
    annotation = field_info.annotation
    # Pydantic v2 stores the annotation post-resolution. Walk Union args.
    args = getattr(annotation, "__args__", ())
    candidates = (annotation, *args) if args else (annotation,)
    for cand in candidates:
        try:
            if inspect.isclass(cand) and issubclass(cand, BaseModel):
                return True
        except TypeError:
            continue
    return False


def _subconfig_fields() -> list[tuple[type[BaseModel], str]]:
    """Returns ``out`` (after 2 setup steps)."""
    out: list[tuple[type[BaseModel], str]] = []
    for _, obj in inspect.getmembers(configs_module, inspect.isclass):
        if not (issubclass(obj, BaseModel) and obj is not BaseModel):
            continue
        if obj.__module__ not in {
            configs_module.__name__,
            f"{configs_module.__package__}._preprocessing_configs",
            f"{configs_module.__package__}._model_configs",
            f"{configs_module.__package__}._training_runtime_configs",
            f"{configs_module.__package__}._composite_target_discovery_config",
            f"{configs_module.__package__}._reporting_configs",
            f"{configs_module.__package__}._configs_base",
            f"{configs_module.__package__}._feature_selection_config",
        }:
            continue
        for field_name in obj.model_fields:
            if _is_basemodel_field(obj, field_name):
                out.append((obj, field_name))
    return out


def _is_threaded(field_name: str, corpus: str) -> bool:
    """A sub-config field is threaded if its bare name appears as an
    attribute access (``cfg.field_name``) OR a kwarg pass-through
    (``field_name=...``) somewhere in production code.
    """
    return f".{field_name}" in corpus or f"{field_name}=" in corpus


def test_every_subconfig_field_is_threaded():
    """Every subconfig field is threaded."""
    corpus = _consumer_corpus()
    fields = _subconfig_fields()
    assert fields, "no BaseModel-typed sub-config fields found"

    orphaned: list[str] = []
    for cls, field_name in fields:
        qualified = f"{cls.__name__}.{field_name}"
        if qualified in _KNOWN_INDIRECT_SUBCONFIGS:
            continue
        if qualified in _USER_DEFERRED_SUBCONFIGS:
            continue
        if not _is_threaded(field_name, corpus):
            orphaned.append(qualified)

    if orphaned:
        pytest.fail(
            f"{len(orphaned)} sub-config field(s) declared on a parent "
            f"config but never read by production code (bare field name "
            f"never appears as ``.{{name}}`` or ``{{name}}=``). The class "
            f"may be used elsewhere — only this PATH (parent.field) is "
            f"orphaned. Either thread the field through the trainer / "
            f"pipeline, OR rename the trainer kwarg to match, OR whitelist "
            f"with reasoning:\n  " + "\n  ".join(orphaned)
        )
