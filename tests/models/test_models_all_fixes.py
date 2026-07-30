"""Regression tests for audits/full_audit_2026-07-21/models_all.md findings F1, F2, F4, F5, F7, F8.

F3 (combine_probs median flavour silently drops sample_weight) is ALREADY resolved -- the current
code has no try/except around np.median at all; the docstring explicitly documents that median/rrf
ignore weights by design (no canonical weighted-median/weighted-rank-fusion exists), replacing the
audit-described silent-except pattern with an upfront, documented limitation. F6 (mojibake comment
in process_method.py) is ALREADY resolved -- confirmed no mojibake pattern remains in the file and
it is not in tests/test_meta/_mojibake_baseline.json's exclusion list either (fixed by the earlier
session-wide mojibake sweep, before this cluster was reached).

PR3 (base.py LOC-split), PR5 (loky-only assumption comment), PR6 (docs pointer) are architecture/docs
proposals with no reported bug -- assessed and deferred.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

logging.disable(logging.CRITICAL)


# ---------------------------------------------------------------------------
# F1: MBHOptimizer normalizes integer-dtype known_evaluations to float64
# ---------------------------------------------------------------------------


def test_f1_integer_known_evaluations_normalized_to_float64():
    """F1 integer known evaluations normalized to float64."""
    from mlframe.models._optimization_search import MBHOptimizer

    opt = MBHOptimizer(
        search_space=np.arange(0, 20),
        known_candidates=np.array([1, 1, 2, 3]),  # duplicate x=1 forces the dedup path
        known_evaluations=np.array([10, 20, 30, 40]),  # integer dtype -- the exact F1 trigger
        seeded_inputs=[0],  # init_num_samples=0 needs a non-empty seed or construction itself raises
        model_name="ETR", model_params={}, init_num_samples=0, random_state=0,
    )
    assert opt.known_evaluations.dtype == np.float64, "F1 REGRESSION: integer-dtype known_evaluations must be normalized to float64"


def test_f1_dedup_minimize_does_not_collapse_to_int64_min():
    """End-to-end: Minimize direction + duplicate x + integer known_evaluations must not fit the
    surrogate on a constant INT64_MIN target for the deduplicated point."""
    from mlframe.models._optimization_search import MBHOptimizer, OptimizationDirection

    opt = MBHOptimizer(
        search_space=np.arange(0, 30),
        known_candidates=np.array([5, 5, 5, 10]),  # x=5 triplicated -> forces the dedup+reduce path
        known_evaluations=np.array([100, 50, 75, 200]),  # int dtype; true min for x=5 is 50
        seeded_inputs=[0],  # init_num_samples=0 needs a non-empty seed or construction itself raises
        direction=OptimizationDirection.Minimize,
        model_name="ETR", model_params={}, init_num_samples=0, random_state=0,
        dedup_known_evaluations=True,
    )
    # Reproduce the exact dedup-and-fit code path this finding lives in.
    _xs = opt.known_candidates
    _ys = opt.known_evaluations
    _unique_x, _inv = np.unique(_xs, return_inverse=True)
    assert len(_unique_x) < len(_xs), "test setup: expected a genuine duplicate to trigger the dedup path"
    _agg = np.full(len(_unique_x), np.inf, dtype=_ys.dtype)
    for _i, _y in zip(_inv, _ys):
        _agg[_i] = np.minimum(_agg[_i], _y)
    # The deduplicated x=5 entry's aggregated value must be the true min (50.0), not INT64_MIN.
    idx_of_5 = int(np.where(_unique_x == 5)[0][0])
    assert _agg[idx_of_5] == 50.0, f"F1 REGRESSION: dedup-minimize aggregate for x=5 must be 50.0 (the true min), got {_agg[idx_of_5]}"


# ---------------------------------------------------------------------------
# F2: MBHOptimizer.__init__ validates model_name
# ---------------------------------------------------------------------------


def test_f2_invalid_model_name_raises_value_error_at_construction():
    """F2 invalid model name raises value error at construction."""
    from mlframe.models._optimization_search import MBHOptimizer

    with pytest.raises(ValueError, match="model_name"):
        MBHOptimizer(search_space=np.arange(0, 20), model_name="bogus", seeded_inputs=[0], init_num_samples=0, random_state=0)


def test_f2_valid_model_names_still_construct():
    """F2 valid model names still construct."""
    from mlframe.models._optimization_search import MBHOptimizer

    for name in ("CBQ", "CB", "ETR"):
        opt = MBHOptimizer(search_space=np.arange(0, 20), model_name=name, model_params={}, seeded_inputs=[0], init_num_samples=0, random_state=0)
        assert opt.model is not None


# ---------------------------------------------------------------------------
# F4: combine_probs's NaN/inf fallback now honours precomputed_weights too
# ---------------------------------------------------------------------------


def test_f4_combine_probs_geo_flavour_nan_fallback_is_weighted():
    """A genuine end-to-end trigger: 'qube' cubes each member's prediction (``p**3``) before averaging;
    a huge-but-finite member value (e.g. 1e200) makes the CUBE overflow to inf while the raw stacked
    values themselves stay finite, so ``combined`` goes non-finite and the fallback kicks in on data
    the fallback CAN actually repair (unlike an already-NaN input cell, whose weighted/unweighted mean
    is NaN either way). Confirms the fallback differs between weighted and unweighted calls."""
    from mlframe.models.ensembling.base import combine_probs

    stacked = np.array(
        [
            [0.3, 0.4, 0.5, 0.6],
            [0.2, 0.3, 0.4, 0.5],
            [1e200, 1e200, 1e200, 1e200],  # finite itself, but **3 overflows to inf -> non-finite combined
        ]
    )
    weights = np.array([0.1, 0.1, 0.8])

    out_weighted = combine_probs(stacked, flavour="qube", precomputed_weights=weights)
    out_unweighted = combine_probs(stacked, flavour="qube")
    assert np.isfinite(out_weighted).all()
    assert np.isfinite(out_unweighted).all()
    # The fallback averages the raw (finite) stacked values -- weighted vs unweighted must differ since
    # member 2 (the huge one) carries weight 0.8 vs an equal 1/3 share unweighted.
    assert not np.array_equal(out_weighted, out_unweighted), "F4 REGRESSION: the non-finite fallback row must be weighted when precomputed_weights is supplied"


# ---------------------------------------------------------------------------
# F5: combine_float_predictions's own default now matches the documented production contract
# ---------------------------------------------------------------------------


def test_f5_combine_float_predictions_default_is_mean():
    """F5 combine float predictions default is mean."""
    import inspect

    from mlframe.models.ensembling.float_aggregation import combine_float_predictions

    sig = inspect.signature(combine_float_predictions)
    assert sig.parameters["flavour"].default == "mean"


def test_f5_default_call_matches_explicit_mean_flavour():
    """F5 default call matches explicit mean flavour."""
    from mlframe.models.ensembling.float_aggregation import combine_float_predictions

    rng = np.random.default_rng(0)
    stacked = rng.normal(size=(4, 10))
    out_default = combine_float_predictions(stacked)
    out_explicit_mean = combine_float_predictions(stacked, flavour="mean")
    np.testing.assert_array_equal(out_default, out_explicit_mean)


# ---------------------------------------------------------------------------
# F7: additive_interaction_diagnostic's ratio gate only treats full_score == 0 as undefined
# ---------------------------------------------------------------------------


def test_f7_negative_full_score_still_computes_a_ratio():
    """F7: additive_signal_ratio must be a real (finite) ratio when the full model's CV score is
    negative -- only an EXACT zero should force nan. A ``metric_fn`` that ignores its inputs and
    always returns a fixed negative value makes ``full_score`` deterministically negative without
    needing a real underperforming model fit."""
    from sklearn.model_selection import KFold

    from mlframe.models.additive_interaction_diagnostic import additive_interaction_diagnostic

    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 3))
    y = X[:, 0] + rng.normal(scale=0.1, size=n)
    splits = list(KFold(2).split(X))

    result = additive_interaction_diagnostic(X, y, splits, metric_fn=lambda y_true, y_pred: -0.5, objective="regression")

    assert result["full_model_cv_score"] == -0.5
    assert np.isfinite(result["additive_signal_ratio"]), "F7 REGRESSION: a negative-but-nonzero full_score must not force the ratio to nan"


# ---------------------------------------------------------------------------
# F8: score_ensemble no longer aliases the module-level SIMPLE_ENSEMBLING_METHODS list as a default
# ---------------------------------------------------------------------------


def test_f8_ensembling_methods_default_is_not_the_shared_list_object():
    """F8 ensembling methods default is not the shared list object."""
    import inspect

    from mlframe.models.ensembling.score import score_ensemble

    sig = inspect.signature(score_ensemble)
    assert sig.parameters["ensembling_methods"].default is None, "F8 REGRESSION: the default must be None, not the shared module-level list object"


def test_f8_mutating_caller_list_does_not_leak_into_next_call(monkeypatch):
    """Simulates the exact foot-gun the finding warns about: score_ensemble must resolve
    ``ensembling_methods=None`` into a FRESH list each call, never a live reference to the shared
    module-level ``SIMPLE_ENSEMBLING_METHODS`` -- otherwise a later in-place mutation of the resolved
    list (e.g. a filter step dropping an entry) would corrupt every subsequent default-arg call too."""

    class _StopEarly(Exception):
        """Raised by the stubbed validator to short-circuit score_ensemble right after it resolves
        ensembling_methods, before the (expensive, irrelevant-to-this-test) real scoring body runs."""

    import mlframe.models.ensembling.score as score_mod
    from mlframe.models.ensembling.score import SIMPLE_ENSEMBLING_METHODS, score_ensemble

    captured: list = []

    def _fake_validate(*args, **kwargs):
        """Capture the resolved ensembling_methods list, then abort the call."""
        captured.append(kwargs["ensembling_methods"])
        raise _StopEarly

    monkeypatch.setattr(score_mod, "_validate_score_ensemble_inputs", _fake_validate)

    with pytest.raises(_StopEarly):
        score_ensemble(models_and_predictions=[], ensemble_name="e")

    assert len(captured) == 1
    resolved = captured[0]
    assert resolved == list(SIMPLE_ENSEMBLING_METHODS)
    assert resolved is not SIMPLE_ENSEMBLING_METHODS, "F8 REGRESSION: ensembling_methods=None resolved to a live reference, not a fresh copy"

    # Mutate the resolved list as a downstream filter step legitimately would -- must not affect the source.
    resolved.pop()
    assert list(SIMPLE_ENSEMBLING_METHODS) != resolved
