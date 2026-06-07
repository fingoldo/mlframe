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


# ---- #1 neural/base get_params -----------------------------------------


def test_neural_get_params_deep_returns_deepcopy_for_trainer_params():
    """get_params(deep=True) must return deepcopied trainer_params + tune_params: mutating the
    returned dicts must NOT poison the original estimator (sklearn.clone shares the get_params
    output). deep=False keeps the by-reference fast path."""
    from mlframe.training.neural.base import PytorchLightningEstimator

    est = PytorchLightningEstimator(
        model_class=object, model_params={}, network_params={},
        datamodule_class=object, datamodule_params={},
        trainer_params={"logger": "orig", "max_epochs": 5},
        tune_params={"n_trials": 10},
    )
    deep = est.get_params(deep=True)
    deep["trainer_params"]["logger"] = "MUTATED"
    deep["tune_params"]["n_trials"] = 999
    assert est.trainer_params["logger"] == "orig", "deep=True must isolate trainer_params"
    assert est.tune_params["n_trials"] == 10, "deep=True must isolate tune_params"

    shallow = est.get_params(deep=False)
    assert shallow["trainer_params"] is est.trainer_params, "deep=False keeps the by-reference path"


# ---- #2 + #3 composite_discovery report() + filter_drops() ----------------


def test_composite_discovery_report_isolates_inner_dicts():
    """Behavioural: build a fake CompositeTargetDiscovery instance with
    a synthetic report_ list, call report(), mutate the returned inner
    dict, call report() again, assert original value preserved."""
    from mlframe.training.composite.discovery import CompositeTargetDiscovery

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
    from mlframe.training.composite.discovery import CompositeTargetDiscovery

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


def test_composite_discovery_report_and_filter_drops_return_fresh_outer_list():
    """report()/filter_drops() must also decouple the OUTER list (not just inner dicts): mutating
    the returned list (append/clear) must not change the next call's result."""
    from mlframe.training.composite.discovery import CompositeTargetDiscovery

    disc = CompositeTargetDiscovery.__new__(CompositeTargetDiscovery)
    disc.report_ = [{"name": "spec_a", "score": 0.5}]
    disc._filter_drops = [{"column": "col_a", "reason": "low_corr"}]

    r = disc.report()
    r.append({"name": "INJECTED", "score": 0.0})
    assert len(disc.report()) == 1, "report() must return a fresh outer list, not the internal one"

    f = disc.filter_drops()
    f.clear()
    assert len(disc.filter_drops()) == 1, "filter_drops() must return a fresh outer list"
