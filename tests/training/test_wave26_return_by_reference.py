"""Wave-26 sensor: getter-side defensive-copy leaks (3 sites).

Wave-26 audit (2026-05-20) found 3 sites where a getter returned
internal mutable state by reference; caller mutation silently
corrupted the next-call return value.

#1 neural/base.py:484 -- get_params(deep=True) deepcopied 4 sibling
   param-dicts but NOT trainer_params + tune_params. sklearn's clone()
   then shared them across original + clone; mutating the clone's
   trainer_params (e.g. swapping the logger) poisoned the original
   estimator still training.

#2 composite_discovery.py:1368 -- report() did ``list(...)`` (shallow)
   over a list of dicts. Outer list decoupled, inner per-candidate
   records leaked by reference. ``discovery.report()[0]["score"] = 999``
   mutated internal state visible on the next ``report()`` call.
   Sibling export_specs() at L1338 already used a comprehension that
   built fresh inner dicts -- inconsistent defensive copying.

#3 composite_discovery.py:1398 -- filter_drops() had the same shape
   as report(): shallow outer list over inner dicts.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any


# ---- #1 neural/base get_params -----------------------------------------


def test_neural_get_params_deep_returns_deepcopy_for_trainer_params():
    """Source-level guard: trainer_params + tune_params are now
    deepcopied when deep=True."""
    import pathlib
    import mlframe as _mlframe
    src = (
        pathlib.Path(_mlframe.__file__).resolve().parent
        / "training" / "neural" / "base.py"
    ).read_text(encoding="utf-8")
    # Pre-fix shape MUST be gone:
    assert '"trainer_params": self.trainer_params,' not in src, (
        "Wave 26 P1 regression: trainer_params returned by reference even "
        "with deep=True; sklearn.clone() shares it across original + clone."
    )
    assert '"tune_params": self.tune_params,' not in src, (
        "Wave 26 P1 regression: tune_params returned by reference."
    )
    # Post-fix marker:
    assert '"trainer_params": deepcopy(self.trainer_params) if deep else self.trainer_params' in src
    assert '"tune_params": deepcopy(self.tune_params) if deep and self.tune_params else self.tune_params' in src


# ---- #2 + #3 composite_discovery report() + filter_drops() ----------------


def test_composite_discovery_report_isolates_inner_dicts():
    """Behavioural: build a fake CompositeTargetDiscovery instance with
    a synthetic report_ list, call report(), mutate the returned inner
    dict, call report() again, assert original value preserved."""
    from mlframe.training.composite_discovery import CompositeTargetDiscovery

    # Sidestep the full constructor (which builds a config) by attaching
    # the attribute on a bare instance via __new__.
    disc = CompositeTargetDiscovery.__new__(CompositeTargetDiscovery)
    disc.report_ = [
        {"name": "spec_a", "score": 0.5, "reason": "kept"},
        {"name": "spec_b", "score": 0.3, "reason": "dropped"},
    ]
    snapshot = disc.report()
    # Mutate the returned inner dict.
    snapshot[0]["score"] = 999
    snapshot[1]["reason"] = "MUTATED"
    # Internal state must be unchanged.
    fresh = disc.report()
    assert fresh[0]["score"] == 0.5, (
        f"Wave 26 P1 regression: report() leaked inner dict by reference; "
        f"caller mutation poisoned internal state. Got fresh[0]['score']="
        f"{fresh[0]['score']}, expected 0.5."
    )
    assert fresh[1]["reason"] == "dropped", (
        f"Wave 26 P1 regression: report() leaked second inner dict; "
        f"fresh[1]['reason']={fresh[1]['reason']!r}, expected 'dropped'."
    )


def test_composite_discovery_filter_drops_isolates_inner_dicts():
    """Same shape for filter_drops()."""
    from mlframe.training.composite_discovery import CompositeTargetDiscovery

    disc = CompositeTargetDiscovery.__new__(CompositeTargetDiscovery)
    disc._filter_drops = [
        {"column": "col_a", "reason": "low_corr", "value": 0.001},
        {"column": "col_b", "reason": "all_nan", "value": 0.0},
    ]
    snapshot = disc.filter_drops()
    snapshot[0]["reason"] = "MUTATED"
    snapshot[1]["value"] = 999.0
    fresh = disc.filter_drops()
    assert fresh[0]["reason"] == "low_corr", (
        f"Wave 26 P1 regression: filter_drops() leaked inner dict; "
        f"fresh[0]['reason']={fresh[0]['reason']!r}."
    )
    assert fresh[1]["value"] == 0.0, (
        f"Wave 26 P1 regression: filter_drops() leaked second inner dict."
    )


def test_composite_discovery_report_source_marker():
    """Source-level guard: the inner-dict comprehension shape must be
    present at both call sites."""
    import pathlib
    import mlframe as _mlframe
    src = (
        pathlib.Path(_mlframe.__file__).resolve().parent
        / "training" / "composite_discovery.py"
    ).read_text(encoding="utf-8")
    # Pre-fix shapes MUST be gone:
    assert "return list(getattr(self, \"report_\", []))" not in src
    assert "return list(getattr(self, \"_filter_drops\", []))" not in src
    # Post-fix markers:
    assert "return [dict(r) for r in getattr(self, \"report_\", [])]" in src
    assert "return [dict(d) for d in getattr(self, \"_filter_drops\", [])]" in src
