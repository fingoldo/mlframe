"""Meta-test — every Pydantic config must survive a default-instantiation
round-trip via ``model_dump()`` / re-instantiation.

  cfg = Cls()
  rebuilt = Cls(**cfg.model_dump())
  assert rebuilt == cfg

Catches a cluster of bugs that look unrelated:
  * Mutable default factory shared between instances (the classic
    ``def f(x=[]): x.append(...)`` trap, ported into Pydantic via
    ``Field(default=...)`` without ``default_factory``).
  * ``model_dump()`` losing a field set via ``model_extra`` / ``Config.extra="allow"``.
  * Custom ``@model_validator(mode="after")`` that mutates state — round-trip
    surfaces non-determinism.
  * Frozen-field policy preventing reconstruction (rare but happens).

Configs flow through MLflow / pickle / job-queues all the time in
production; a round-trip break manifests as "the loaded config doesn't
behave like the saved one". This test gates that contract per-class.

For configs whose default-construction requires user-supplied arguments
(``TrainingConfig(target_name=..., model_name=...)``), the test injects
sentinel strings so the round-trip can run anyway. Field types we can't
sensibly auto-fill (callables, opaque ``Any``) are exempted via
``_REQUIRES_USER_ARGS``.
"""

from __future__ import annotations

import inspect

import pytest
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from mlframe.training import configs as configs_module

# Configs we can't construct with no arguments (required fields without
# defaults). For each, supply a minimal sentinel kwarg dict that makes
# default-construction succeed. The round-trip then exercises everything
# else.
_REQUIRES_USER_ARGS: dict[str, dict] = {
    "TrainingConfig": {"target_name": "_round_trip_sentinel_target", "model_name": "_round_trip_sentinel_model"},
}

# Fields we know can't be round-tripped because their value is a Python
# callable / opaque type that ``model_dump`` may serialise differently
# than ``model_validate`` accepts back. Each entry lists the field name
# alone (applied across all classes — these are typically generic).
_OPAQUE_FIELDS: set[str] = {
    # Callables and label-encoders are typically lambdas / sklearn objects
    # whose pickle / dump roundtrip behaviour is not under our control.
    "metamodel_func",
    "callback_params",  # often nested objects
    "target_label_encoder",
    "Dist",  # ngboost.distns.* class objects
    "Score",
    "model_kwargs",  # ConfidenceAnalysisConfig accepts arbitrary nested
    # The trainer/eval pipeline mutates these data containers in-place
    # during a run, but the empty default round-trips fine — included
    # here only because their dynamic values can be ndarrays which
    # ``==`` doesn't compare elementwise.
    "df",
    "train_df",
    "val_df",
    "test_df",
    "target",
    "train_target",
    "val_target",
    "test_target",
    "train_idx",
    "val_idx",
    "test_idx",
    "group_ids",
    "sample_weight",
    "train_preds",
    "train_probs",
    "val_preds",
    "val_probs",
    "test_preds",
    "test_probs",
}


def _config_classes() -> list[type[BaseModel]]:
    out = []
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
        out.append(obj)
    return out


def _has_default(field_info) -> bool:
    """A field is constructable without user input if it has either a
    plain default or a default_factory."""
    if field_info.default is not PydanticUndefined and field_info.default is not Ellipsis:
        return True
    if field_info.default_factory is not None:
        return True
    return False


def _can_default_construct(cls: type[BaseModel]) -> tuple[bool, dict]:
    """Returns (constructable, sentinel_kwargs)."""
    extra_kwargs = _REQUIRES_USER_ARGS.get(cls.__name__, {})
    for name, info in cls.model_fields.items():
        if name in extra_kwargs:
            continue
        if not _has_default(info):
            return False, extra_kwargs
    return True, extra_kwargs


def _strip_opaque(d: dict) -> dict:
    """Remove fields known to be opaque under round-trip from a dump."""
    return {k: v for k, v in d.items() if k not in _OPAQUE_FIELDS}


def test_every_config_round_trips_via_model_dump():
    """``Cls(**cls().model_dump()) == cls()`` for every default-
    constructable Pydantic config in mlframe.training.configs.
    """
    failures: list[str] = []
    skipped: list[str] = []
    classes = _config_classes()
    assert classes

    for cls in classes:
        ok, extra = _can_default_construct(cls)
        if not ok:
            skipped.append(f"{cls.__name__} (required fields without sentinels in _REQUIRES_USER_ARGS)")
            continue
        try:
            original = cls(**extra)
        except Exception as e:
            failures.append(f"{cls.__name__} default construct raised: {type(e).__name__}: {e}")
            continue

        try:
            dumped = original.model_dump()
        except Exception as e:
            failures.append(f"{cls.__name__}.model_dump() raised: {type(e).__name__}: {e}")
            continue

        try:
            rebuilt = cls(**dumped)
        except Exception as e:
            failures.append(f"{cls.__name__}(**dumped) raised — round-trip broken: {type(e).__name__}: {e}")
            continue

        # Compare via dump-then-strip-opaque to dodge ndarray ==
        # ambiguity. If a non-opaque field differs the dumps will too.
        a = _strip_opaque(original.model_dump())
        b = _strip_opaque(rebuilt.model_dump())
        if a != b:
            # Find the first differing key for an actionable message.
            keys = set(a) | set(b)
            diffs = [k for k in sorted(keys) if a.get(k) != b.get(k)]
            failures.append(f"{cls.__name__} round-trip mutated state: diffs in {diffs[:5]} (showing up to 5)")

    if failures:
        pytest.fail(
            f"{len(failures)} config class(es) failed model_dump() round-trip:\n  "
            + "\n  ".join(failures)
            + (f"\n  (skipped {len(skipped)} class(es) needing sentinels: {', '.join(skipped)})" if skipped else "")
        )


def test_no_shared_mutable_default_across_instances():
    """Pydantic v2 defaults are deepcopied by default (unlike dataclasses
    in pre-3.11). This test asserts that mutation of one instance's list/
    dict-typed field doesn't reflect in a sibling instance — guards
    against a future refactor that switches to a non-copying default
    strategy by accident.
    """
    failures: list[str] = []
    classes = _config_classes()

    for cls in classes:
        ok, extra = _can_default_construct(cls)
        if not ok:
            continue
        a = cls(**extra)
        b = cls(**extra)
        for name, info in cls.model_fields.items():
            val_a = getattr(a, name, None)
            val_b = getattr(b, name, None)
            if isinstance(val_a, (list, dict, set)) and val_a is val_b:
                failures.append(
                    f"{cls.__name__}.{name}: two fresh instances share the "
                    f"same {type(val_a).__name__} object (id={id(val_a)}) — "
                    f"mutation will leak between callers"
                )

    if failures:
        pytest.fail(
            f"{len(failures)} mutable-default field(s) shared across instances — Pydantic must deep-copy these on construction:\n  " + "\n  ".join(failures)
        )
