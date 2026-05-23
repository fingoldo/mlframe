"""Tests for Fix 3: val is ONLY for early-stopping; outlier gate + flavour selection move to OOF.

Pre-fix, val_preds were reused four ways across the suite: ES detector, outlier-member gate, flavour selection via
compare_ensembles, and the level-1 stacking aggregator. That stacking re-uses val as a "pick a model" surface --
the very surface ES already optimised against -- so the suite double-dips val. Fix 3 cleanly separates:

- val: early-stopping ONLY (no other consumer).
- oof: outlier-member quality gate, compare_ensembles flavour selection (default sort_metric).
- test: untouched holdout.

Tests below cover the two split-routing paths the directive flagged: the gate-source ordering inside
``score_ensemble`` and the warning when ``compare_ensembles`` is called with a ``val.*`` sort metric.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_member_quality_gate_routes_to_oof_not_val():
    """Outlier-member gate must pick OOF as the source split before val/test/train.

    Construct three members. Their val_preds are bland (median-like, no outliers) but their oof_preds carry an
    obvious outlier on member 2. If the gate were still routing to val_preds the outlier wouldn't be detected;
    routing to oof_preds catches it. We monkeypatch ``compute_member_quality_gate`` to record the source array
    it sees and assert the call came from the OOF slice.
    """
    import mlframe.models.ensembling as ens_mod
    from mlframe.models.ensembling import score_ensemble

    rng = np.random.default_rng(0)
    n = 100
    # OOF arrays: member 2 carries a massive offset (the gate's job to catch).
    oof_a = rng.normal(loc=0.0, scale=1.0, size=n)
    oof_b = rng.normal(loc=0.05, scale=1.0, size=n)
    oof_c = rng.normal(loc=50.0, scale=1.0, size=n)  # outlier on OOF
    # Val arrays: all three look identical (gate would NOT catch the outlier here).
    val_a = rng.normal(loc=0.0, scale=1.0, size=n)
    val_b = rng.normal(loc=0.05, scale=1.0, size=n)
    val_c = rng.normal(loc=0.05, scale=1.0, size=n)

    member_a = SimpleNamespace(
        model=None, val_preds=val_a, val_probs=None, test_preds=val_a.copy(), test_probs=None,
        train_preds=val_a.copy(), train_probs=None, oof_preds=oof_a, oof_probs=None,
    )
    member_b = SimpleNamespace(
        model=None, val_preds=val_b, val_probs=None, test_preds=val_b.copy(), test_probs=None,
        train_preds=val_b.copy(), train_probs=None, oof_preds=oof_b, oof_probs=None,
    )
    member_c = SimpleNamespace(
        model=None, val_preds=val_c, val_probs=None, test_preds=val_c.copy(), test_probs=None,
        train_preds=val_c.copy(), train_probs=None, oof_preds=oof_c, oof_probs=None,
    )

    recorded = {}
    # ``score_ensemble`` lives in the ``_ensembling_score`` sibling after the
    # monolith split and imports ``compute_member_quality_gate`` at its OWN
    # module top. Patching the re-exporter on ``mlframe.models.ensembling``
    # has no effect on that binding - we must patch the score module's
    # local reference. (Same pattern as the rrf_njit dispatch fix.)
    import mlframe.models._ensembling_score as ens_score_mod
    real_gate = ens_score_mod.compute_member_quality_gate

    def _spy_gate(preds_list, **kw):
        # Compare against oof_c (outlier marker) to confirm we got the OOF arrays, not val.
        recorded["arrays"] = [np.asarray(p).copy() for p in preds_list]
        return real_gate(preds_list, **kw)

    ens_score_mod.compute_member_quality_gate = _spy_gate
    try:
        # max_ensembling_level=1 keeps the call cheap; only the gate path matters here.
        # train_target supplied so the regression path doesn't crash on missing y.
        # k2_catastrophic_mae_ratio=inf DISABLES the K>2 catastrophic-dropout
        # pre-filter (which would otherwise drop member_c on OOF outlier-MAE
        # BEFORE the quality gate ever runs, leaving K=2 and skipping the gate
        # entirely on the ``len(_gate_preds_for_check) > 2`` guard at L580).
        # The catastrophic-drop path already routes to OOF correctly; this
        # test specifically validates the OOF routing of the FINER gate.
        score_ensemble(
            models_and_predictions=[member_a, member_b, member_c],
            ensemble_name="test",
            train_target=rng.normal(size=n),
            val_target=rng.normal(size=n),
            test_target=rng.normal(size=n),
            max_ensembling_level=1,
            verbose=False,
            ensembling_methods=("arithm",),  # one flavor minimises downstream noise
            uncertainty_quantile=0,  # skip the conf-ensemble branch
            k2_catastrophic_mae_ratio=float("inf"),
        )
    finally:
        ens_score_mod.compute_member_quality_gate = real_gate

    assert "arrays" in recorded, "compute_member_quality_gate was never invoked"
    # The third recorded array must equal oof_c (the outlier-bearing OOF series), not val_c (bland).
    np.testing.assert_array_equal(recorded["arrays"][2], oof_c)
    assert not np.array_equal(recorded["arrays"][2], val_c), (
        "Gate routed to val_preds (val_c) when it should have routed to oof_preds (oof_c)"
    )


def test_compare_ensembles_val_sort_metric_emits_userwarning():
    """``sort_metric='val.*'`` must trigger a warnings.warn(UserWarning); logger-only would be invisible to scripts."""
    from mlframe.models.ensembling import compare_ensembles
    import warnings

    ensembles = {
        "A": SimpleNamespace(
            metrics={
                "oof": {"1": {"integral_error": 0.10}},
                "val": {"1": {"integral_error": 0.10}},
                "test": {"1": {"integral_error": 0.10}},
            }
        ),
    }

    with pytest.warns(UserWarning, match="val is already burned for early"):
        compare_ensembles(ensembles, sort_metric="val.1.integral_error", show_plot=False)

    # And: the default sort path (oof.*) must NOT fire the warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        compare_ensembles(ensembles, show_plot=False)
