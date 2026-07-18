"""biz_value: dispatched importance aggregation measurably beats legacy on its target structures.

Two quantitative wins are pinned:
  1. TREE variance: when one true feature has steady gain and a noise feature has equal MEAN gain but high
     fold-to-fold variance, dispatched ranks the steady true feature above the noisy one; legacy (raw mean)
     ties / mis-ranks. Pinned: dispatched score ratio (steady/noisy) >= 1.8.
  2. LINEAR sign-flip: a sign-flipping decoy is demoted below a weaker but sign-consistent true feature under
     dispatched, but NOT under the legacy abs-mean vote. Pinned: under dispatched the consistent feature
     outranks the flipper; the legacy abs-mean would rank the flipper >= the consistent feature.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.wrappers._enums import VotesAggregation
from mlframe.feature_selection.wrappers._helpers_importance_agg import (
    aggregate_importances_dispatched,
    aggregate_linear,
    aggregate_tree,
)
from mlframe.feature_selection.wrappers._helpers_importance import get_actual_features_ranking


def test_biz_tree_variance_downweight_beats_raw_mean():
    # steady true feature: gain 1.0 every fold. noisy decoy: same mean 1.0 but swings 0<->2.
    """Biz tree variance downweight beats raw mean."""
    fi = {f"r{k}": {"true": 1.0, "noisy": (2.0 if k % 2 == 0 else 0.0)} for k in range(6)}
    disp = aggregate_tree(fi, k_cv=1.0)
    ratio = disp["true"] / max(disp["noisy"], 1e-9)
    assert ratio >= 1.8, f"dispatched must rank steady >> noisy (ratio {ratio:.2f}); measured ~2.0"
    # Legacy raw mean would TIE them (both mean 1.0) -> dispatched is strictly better separation.
    legacy_mean_true = np.mean([fi[r]["true"] for r in fi])
    legacy_mean_noisy = np.mean([fi[r]["noisy"] for r in fi])
    assert abs(legacy_mean_true - legacy_mean_noisy) < 1e-9, "legacy mean cannot separate them"


def test_biz_linear_sign_harmony_beats_abs_mean_vote():
    # consistent true feature: coef +0.8 every fold. flipper decoy: |coef|=1.0 but sign 3+/3-.
    """Biz linear sign harmony beats abs mean vote."""
    {
        "true": {f"r{k}": 0.8 for k in range(6)},
    }
    # build per-run dicts in the run-keyed shape the aggregator expects.
    sg = {f"r{k}": {"true": 0.8, "flip": (1.0 if k < 3 else -1.0)} for k in range(6)}
    abs_fi = {f"r{k}": {"true": 0.8, "flip": 1.0} for k in range(6)}

    disp = aggregate_linear(sg)
    assert (
        disp["true"] > disp["flip"]
    ), f"sign-harmony must demote the sign-flipper below the weaker consistent feature: true={disp['true']:.3f} flip={disp['flip']:.3f}"
    # The flipper's signed mean is ~0 -> score ~0; the consistent feature keeps ~0.8.
    assert disp["true"] >= 0.7
    assert disp["flip"] <= 0.1

    # Legacy abs-mean vote would rank the flipper (|coef| 1.0) AT or ABOVE the consistent feature (0.8).
    legacy_ranks = get_actual_features_ranking(abs_fi, votes_aggregation_method=VotesAggregation.AM)
    assert legacy_ranks[0] == "flip", "legacy abs-mean wrongly prefers the higher-magnitude sign-flipper"

    # Dispatcher (linear) flips that verdict to the correct one.
    disp_ranks = aggregate_importances_dispatched(
        abs_fi,
        family="linear",
        votes_aggregation_method=VotesAggregation.AM,
        signed_importances=sg,
    )
    assert disp_ranks[0] == "true", "dispatched correctly prefers the sign-consistent feature"


def test_biz_kernel_no_regression_vs_legacy():
    # kernel family must produce EXACTLY the legacy ranking (defers) -> zero regression risk.
    """Biz kernel no regression vs legacy."""
    fi = {"r0": {"a": 0.9, "b": 0.3, "c": 0.1}, "r1": {"a": 0.8, "b": 0.4, "c": 0.05}}
    legacy = get_actual_features_ranking(fi, votes_aggregation_method=VotesAggregation.Borda)
    disp = aggregate_importances_dispatched(fi, family="kernel", votes_aggregation_method=VotesAggregation.Borda)
    assert disp == legacy
