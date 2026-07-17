"""Meta-test — every Pydantic config field defined in
``mlframe/training/configs.py`` must be referenced by at least one
production-side consumer (anything in ``mlframe/`` outside ``tests/``).

Catches the failure mode where a Field is added to a config model but never
threaded into the trainer / strategy / pipeline that should consume it.
Symptoms: silent no-op flag, "I configured X=True but nothing changed"
debugging sessions, dead config bloat that obscures the real surface area.

Static check, no fixtures, no DB. Runs in <1 s on the full configs.py.

Ported from glossum-backend's ``tests/test_storage/test_loaded_fields_used.py``
(originally for ``enrichment_loader`` return-dict keys; the ML analogue is
Pydantic field consumption).
"""

from __future__ import annotations

import inspect
import re
from functools import lru_cache
from pathlib import Path

import pytest
from pydantic import BaseModel

import mlframe
from mlframe.training import configs as configs_module

# Derive package root from the imported package itself — the mlframe repo is a
# flat-layout package (files live directly at the repo root, not under a nested
# ``mlframe/mlframe/`` directory), so a path computed via ``parents[N]`` from
# this test file would point at a non-existent dir and silently load an empty
# corpus, making every field look "unused".
MLFRAME_DIR = Path(mlframe.__file__).resolve().parent


# Field names defined on Pydantic's BaseModel itself or on our ``BaseConfig``
# helper that don't represent user-facing config (validators, model_config).
_PYDANTIC_INTERNAL = {
    "model_config",
    "model_fields",
    "model_computed_fields",
    "model_extra",
    "model_fields_set",
}

# Hard whitelist — fields known to be consumed by indirect routes a static
# grep can't see (e.g. via ``getattr(cfg, name)`` in a loop, exposed as a
# CLI flag elsewhere). Each entry MUST cite the consumer location.
# Keep this list small and audited; entries grow technical debt.
_KNOWN_INDIRECT_CONSUMERS: dict[str, str] = {
    # Example shape (uncomment / extend as needed):
    # "TrainingControlConfig.experimental_flag": "consumed by getattr loop in trainer.py:NNN",
}

# Fields the maintainer has surfaced and explicitly chose to defer cleanup on
# (suspected dead but kept around pending decision). Treated like
# ``_KNOWN_INDIRECT_CONSUMERS`` for test-pass purposes; separated only so the
# technical debt stays visible at file scope. Drain to zero over time.
# When wiring or deleting one of these, remove the corresponding line here.
_USER_DEFERRED_DEAD: dict[str, str] = {
    "FairnessConfig.protected_attributes": "duplicate of behavior_config.fairness_features (core.py:1545)",
    "FairnessConfig.fairness_metrics": "name collides with metrics.compute_fairness_metrics function",
    "MultilabelDispatchConfig.cv": "ClassifierChain.cv knob — chain dispatch hardcodes cv=5; wire when chain ensemble path is exercised",
    "TreeModelConfig.hgb_kwargs": "duplicate of ModelHyperparamsConfig.hgb_kwargs (which IS consumed via model_dump splat)",
    "TrainingConfig.linear_config": "shadowed by trainer.py kwarg `linear_model_config`; orphan until TrainingConfig becomes the canonical entrypoint",
    "TrainingConfig.tree_config": "shadowed by trainer.py kwarg `tree_model_config`",
    "TrainingConfig.mlp_config": "shadowed by trainer.py kwarg `mlp_config`",
    "TrainingConfig.ngb_config": "shadowed by trainer.py kwarg `ngb_model_config`",
    "TrainingConfig.behavior": "shadowed by trainer.py kwarg `behavior_config`",
    "NGBConfig.minibatch_frac": "NGBConfig class never instantiated — NGB is configured via ngb_kwargs dict instead",
    "NGBConfig.Dist": "same — class never instantiated in production",
    # 2026-05-15 — new entries surfaced after the src/ migration; same pattern
    # as the shadowed-by-kwarg cluster above, kept as deferred-dead until the
    # responsible subsystem author rewires.
    "AutoMLConfig.automl_show_fi": "shadowed by FeatureSelectionConfig.show_fi; AutoML branch reads the latter",
    "EnsemblingConfig.force_legacy": "legacy-path opt-in; current code unconditionally uses the new ensembling kernel",
    "EnsemblingConfig.accumulator": "accumulator strategy knob — single-strategy hardcoded in current build",
    "FeatureSelectionConfig.rfecv_kwargs": "RFECV kwargs threading from FSConfig not yet wired; users pass via rfecv_models_params direct dict",
    "PreprocessingBackendConfig.fallback_to_sklearn": "auto-fallback already implicit in pipeline.py:_apply_polars_ds; flag never read",
    "QuantileRegressionConfig.point_estimate_alpha": "point estimate currently hardcoded to 0.5 (median) inside quantile dispatch",
    "QuantileRegressionConfig.coverage_pairs": "coverage_pairs validator exists on the config but reporting path uses alphas directly",
    "TreeModelConfig.lgb_kwargs": "duplicate of ModelHyperparamsConfig.lgb_kwargs (which IS consumed via model_dump splat)",
    "TreeModelConfig.xgb_kwargs": "duplicate of ModelHyperparamsConfig.xgb_kwargs (which IS consumed via model_dump splat)",
    "CompositeTargetDiscoveryConfig.force_inject_diff_on_top_ablation_pct": "ablation-only knob for the diff-injection sensitivity study; not consumed by production discovery path (kept for the bench/audit script to vary)",
    "CompositeTargetDiscoveryConfig.structural_fragility_max_amplification_ratio": "deprecated: structural-fragility gate replaced the absolute amplitude-vs-std(y) test (scale-buggy) with the scale-invariant between/total variance ratio; field kept for back-compat with configs that set it, has no effect (see _composite_target_discovery_config_base.py:618-620)",
    "SliceStableESConfig.pareto_risk_quantile": "Pareto-aware best_iter selection knob; the slice-stable ES infrastructure shipped without the Pareto-front consumer (referenced only in roadmap docstring at _slice_pareto_plot.py). Kept for the upcoming Pareto-aware selector wave.",
}


def _all_config_classes() -> list[type[BaseModel]]:
    """Every BaseModel subclass declared in mlframe.training.configs."""
    out = []
    for _, obj in inspect.getmembers(configs_module, inspect.isclass):
        if obj is BaseModel:
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
        if not issubclass(obj, BaseModel):
            continue
        out.append(obj)
    return out


@lru_cache(maxsize=1)
def _consumer_corpus() -> str:
    """Concatenate every .py file under mlframe/ outside test directories.

    Cached: two tests in this file each call it independently, and re-reading/re-concatenating
    ~1300 files from disk (~35-40s) is pure duplicate I/O with no per-call variation.
    """
    chunks: list[str] = []
    for py in MLFRAME_DIR.rglob("*.py"):
        # Skip the configs module itself (a field referenced only inside its
        # own class definition is still "unused").
        if py.resolve() == Path(configs_module.__file__).resolve():
            continue
        if "test" in py.parts or "__pycache__" in py.parts:
            continue
        try:
            chunks.append(py.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
    return "\n".join(chunks)


def _class_to_var_candidates(cls_name: str) -> set[str]:
    """Heuristic snake-case names a Pydantic instance of ``cls_name`` is
    likely to be bound to in calling code.

    Matches the common patterns we observed in mlframe:
      ``TrainingBehaviorConfig`` → ``training_behavior_config`` |
      ``training_behavior`` | ``behavior_config``
      ``ModelHyperparamsConfig`` → ``model_hyperparams_config`` |
      ``model_hyperparams`` | ``hyperparams_config``
      ``LinearModelConfig`` → ``linear_model_config`` | ``linear_model``
      | ``model_config`` (less specific — kept anyway since it's still
      a valid ``.model_dump`` source).
    """
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", cls_name).lower()
    short = snake.replace("_config", "")
    candidates = {snake, short}
    parts = short.split("_")
    if parts:
        candidates.add(parts[-1] + "_config")  # behavior_config / hyperparams_config
    return candidates


def _class_is_dumped(cls: type[BaseModel], corpus: str) -> bool:
    """True if any plausibly-named instance variable of this class has
    ``.model_dump(`` called on it in the corpus.

    When code does ``**cfg.model_dump()`` or
    ``other_dict.update(cfg.model_dump())``, every public field flows
    out as a kwarg / dict key — so a static grep on the bare field name
    would miss the indirect consumer. This detector exempts the whole
    class from the "unused" check in those cases.

    Heuristic only: snake-cased class-name variants must appear immediately
    before ``.model_dump(``. Brittle if callers name their variable in
    creative ways; ``_KNOWN_INDIRECT_CONSUMERS`` is the explicit-whitelist
    escape hatch.

    Plain substring check, not ``\b``-anchored regex: measured 89x faster on the
    ~26MB corpus (75s -> 0.85s dominated this test's wall time) with no behaviour
    loss in the direction that matters -- the only candidates that gain a match
    without the word-boundary anchor make a class MORE likely to be treated as
    dumped (fewer fields flagged unused), the same lenient-by-design direction
    the rest of this heuristic already accepts.
    """
    candidates = _class_to_var_candidates(cls.__name__)
    return any(f"{c}.model_dump(" in corpus for c in candidates)


def _is_referenced(field_name: str, corpus: str) -> bool:
    """Heuristic: any of these patterns counts as a consumer reference."""
    needles = (
        f".{field_name}",  # attribute access: cfg.foo
        f'"{field_name}"',  # dict-style: d["foo"]
        f"'{field_name}'",  # dict-style: d['foo']
        f"[{field_name!r}]",  # bracket form
        f"{field_name}=",  # kwarg passthrough
    )
    return any(needle in corpus for needle in needles)


# ---------------------------------------------------------------------------
# Self-tests for the heuristic itself — these guard against the harness
# silently degrading (e.g. path bug → empty corpus → 0 false negatives, every
# field is "consumed" because no field can be found) and the heuristic-pattern
# set rotting out from under us.
# ---------------------------------------------------------------------------


def test_consumer_corpus_is_substantial():
    """Self-test: ``_consumer_corpus`` must load enough source for the audit
    to be meaningful. Below the floor either the package layout changed
    (path needs updating) or large directories are being silently skipped.
    """
    corpus = _consumer_corpus()
    floor_bytes = 100_000
    assert len(corpus) > floor_bytes, (
        f"corpus too small ({len(corpus):,} bytes < {floor_bytes:,}) — "
        f"MLFRAME_DIR probably mis-resolved (got {MLFRAME_DIR}, "
        f"exists={MLFRAME_DIR.exists()}). The audit would silently pass "
        f"every field if we ignored this — refusing."
    )


def test_known_consumed_fields_actually_grep():
    """Self-test: a field every reader knows is heavily consumed (e.g.
    ``DataConfig.target``) MUST be found by ``_is_referenced``. If it
    isn't, ``_is_referenced`` regressed and the test will gloss over real
    bugs as "consumed" by accident.
    """
    corpus = _consumer_corpus()
    canaries = ("target", "verbose", "random_state")
    missing = [c for c in canaries if not _is_referenced(c, corpus)]
    assert not missing, f"_is_referenced canary check failed for {missing}: heuristic is broken (or the corpus is empty / mis-resolved)."


def test_every_config_field_has_a_consumer():
    """Every declared config field must be referenced somewhere outside its own class."""
    corpus = _consumer_corpus()
    classes = _all_config_classes()
    assert classes, "no BaseConfig subclasses found in mlframe.training.configs"

    # Pre-compute, once per class, whether the class is splatted via
    # ``cfg.model_dump()`` somewhere — every field of such a class flows
    # through implicitly even if its bare name doesn't appear elsewhere.
    splatted_classes = {cls for cls in classes if _class_is_dumped(cls, corpus)}

    unused: list[str] = []
    checked = 0
    for cls in classes:
        is_splatted = cls in splatted_classes
        for field_name in cls.model_fields:
            if field_name in _PYDANTIC_INTERNAL:
                continue
            qualified = f"{cls.__name__}.{field_name}"
            if qualified in _KNOWN_INDIRECT_CONSUMERS:
                continue
            if qualified in _USER_DEFERRED_DEAD:
                continue
            checked += 1
            if is_splatted:
                # Class is dumped to a dict somewhere — fields are
                # auto-consumed via the splat. Don't penalize.
                continue
            if not _is_referenced(field_name, corpus):
                unused.append(qualified)

    assert checked > 50, f"only {checked} config fields audited — class enumeration broken?"
    if unused:
        msg = (
            f"{len(unused)} config fields defined in mlframe.training.configs "
            f"have no production-side consumer. Either thread them into the "
            f"trainer / strategy / pipeline that should read them, OR remove "
            f"the unused field, OR add an entry to _KNOWN_INDIRECT_CONSUMERS "
            f"with the consumer file:line documented, OR (when intentionally "
            f"deferred) add to _USER_DEFERRED_DEAD with reasoning:\n  " + "\n  ".join(unused)
        )
        pytest.fail(msg)
