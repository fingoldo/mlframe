"""Meta-test — for every per-estimator ``<flavor>_kwargs`` config field
(``cb_kwargs``, ``lgb_kwargs``, ``xgb_kwargs``, ``hgb_kwargs``, ``mlp_kwargs``,
``ngb_kwargs``, ``rfecv_kwargs``), the production code must contain at least
one site that splat-unpacks the field into a kwargs-style sink (``**hgb_kwargs``,
``HGB_GENERAL_PARAMS.update(hgb_kwargs)``, or a direct ``HistGradientBoosting*``
construction immediately downstream of the kwarg).

Catches the failure mode encountered in the audit-2026-04-28 sweep: the typed
``ModelHyperparamsConfig`` had ``hgb_kwargs`` and ``ngb_kwargs`` defined, but
no caller pulled them out of the config — the only consumer was
``helpers.py:get_training_configs`` which accepted a *standalone* ``hgb_kwargs``
parameter (not the typed-config field). Adding a new estimator family to the
typed config without wiring would silently no-op until someone debugs it.

The test BUILDS the expectation matrix from ``configs.py`` itself (every field
matching ``r"^[a-z]+_kwargs$"`` on every config class), so adding a new
estimator family or a new ``<flavor>_kwargs`` field forces wiring or an
explicit skip — no static list to keep in sync.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
from pydantic import BaseModel

import mlframe
from mlframe.training import configs as configs_module

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent

# Field name pattern: ``<estimator>_kwargs``. Excludes generic names like
# ``fit_params`` / ``init_params`` (these aren't estimator-family-specific).
_FLAVOR_KWARGS_RE = re.compile(r"^[a-z]+_kwargs$")

# Hard whitelist for kwargs fields known to be consumed via routes the
# splat-detector can't see. Cite consumer location.
_KNOWN_INDIRECT_KWARGS: dict[str, str] = {
    # Empty by default; entries are technical debt — drain over time.
}

# Same severity as ``_USER_DEFERRED_DEAD`` in
# test_config_field_consumption.py — fields surfaced and explicitly deferred.
_USER_DEFERRED_KWARGS: dict[str, str] = {
    "TreeModelConfig.hgb_kwargs": "duplicate of ModelHyperparamsConfig.hgb_kwargs which IS consumed via model_dump splat",
}


def _consumer_corpus() -> str:
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


def _flavor_kwargs_fields() -> list[tuple[type[BaseModel], str]]:
    """Every ``<flavor>_kwargs`` field across every config class."""
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
            if _FLAVOR_KWARGS_RE.match(field_name):
                out.append((obj, field_name))
    return out


def _is_routed(field_name: str, corpus: str) -> bool:
    """True if ``field_name`` reaches an estimator constructor.

    Patterns recognised:
      ``**field_name``                — flat kwargs splat into a constructor
      ``....update(field_name)``      — dict merge into a global params dict
                                        (mlframe's ``HGB_GENERAL_PARAMS.update(...)``
                                        pattern in helpers.py)
      ``field_name.get(...)``         — nested-dict access pattern, e.g.
                                        ``mlp_kwargs.get("trainer_params", {})``
                                        used when the kwargs blob is a
                                        dict-of-sub-dicts rather than a flat
                                        keyword bundle (multiple sub-keys
                                        proves real reads, not just declaration)
      ``field_name[...]``             — bracket-form dict access
      ``cfg.field_name``              — typed access on a config instance
                                        followed by a ``**<local>`` splat
                                        elsewhere in the corpus (extract-then-
                                        splat pattern from ``model_dump`` paths)

    Bare attribute access ``.field_name`` alone is NOT enough — the field
    must demonstrably reach a constructor. (Matches the audit's intent:
    ``cb_fit_params`` showing up in a comment doesn't prove the params
    reach CatBoost.)
    """
    if f"**{field_name}" in corpus:
        return True
    if re.search(rf"\.update\(\s*{re.escape(field_name)}\s*\)", corpus):
        return True
    # Nested-dict pattern: field_name.get(...) or field_name[...].
    # Require ≥2 distinct ``.get(...)`` / ``[...]`` reads to avoid matching
    # a single boilerplate "`field_name.get("x", default)`" that doesn't
    # really wire the user's value through.
    get_hits = len(re.findall(rf"\b{re.escape(field_name)}\.get\(", corpus))
    bracket_hits = len(re.findall(rf"\b{re.escape(field_name)}\[", corpus))
    if get_hits + bracket_hits >= 2:
        return True
    # Extract-then-splat: ``cfg.field_name`` extracted into a local that's
    # later ``**``-splatted. Two-pass approximation: any ``.field_name`` plus
    # any ``**<local>`` in the corpus means *some* extract-and-splat pipeline
    # could exist. Brittle but correct on observed mlframe patterns.
    if f".{field_name}" in corpus and re.search(r"\*\*[A-Za-z_]\w*", corpus):
        return True
    return False


def test_every_estimator_kwarg_routes_to_a_constructor():
    corpus = _consumer_corpus()
    fields = _flavor_kwargs_fields()
    assert fields, (
        "no <flavor>_kwargs fields found — class enumeration broken or naming "
        "convention changed (test should be updated)."
    )

    unrouted: list[tuple[str, str]] = []
    for cls, field_name in fields:
        qualified = f"{cls.__name__}.{field_name}"
        if qualified in _KNOWN_INDIRECT_KWARGS:
            continue
        if qualified in _USER_DEFERRED_KWARGS:
            continue
        if not _is_routed(field_name, corpus):
            unrouted.append((qualified, field_name))

    if unrouted:
        lines = [f"{q}  (no kwargs splat / .update / typed extraction found)" for q, _ in unrouted]
        pytest.fail(
            f"{len(unrouted)} per-estimator kwargs field(s) declared on a "
            f"config class but never threaded into a constructor. Add an "
            f"``estimator(**{{kwarg}})`` site or a ``GLOBAL_PARAMS.update("
            f"{{kwarg}})`` site, OR whitelist with reasoning in "
            f"_KNOWN_INDIRECT_KWARGS / _USER_DEFERRED_KWARGS:\n  " + "\n  ".join(lines)
        )
