"""An outside selector in the roster, so "mlframe wins" is a falsifiable claim rather than a tautology.

A benchmark whose every arm comes from the package being benchmarked cannot produce a losing result for
that package: whichever arm wins, the headline is the same. These tests pin that the outside arms are
really in the roster, really distinct from each other, and really charged for their cost -- the last one
because CatBoost's search runs in native code, where the in-process fit counter sees nothing at all.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection._benchmarks.fs_hybrid._arms import FITS_SOURCE_DECLARED, build_arm_roster
from mlframe.feature_selection._benchmarks.fs_hybrid._arms_external import CATBOOST_ALGORITHMS, CatBoostSelectArm, catboost_available

pytestmark = pytest.mark.skipif(not catboost_available(), reason="catboost is an optional dependency and is not installed here")

ROWS = 600


def _bed(seed: int = 0, n_probes: int = 9) -> Any:
    """Return `(X, y)` with three informative columns and `n_probes` probes."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({f"s{i}": rng.normal(size=ROWS) for i in range(3)})
    for index in range(n_probes):
        frame[f"n{index:02d}"] = rng.normal(size=ROWS)
    score = 1.7 * frame["s0"] + 1.2 * frame["s1"] + 0.9 * frame["s2"]
    labels = (rng.random(ROWS) < 1.0 / (1.0 + np.exp(-score))).astype(np.int64)
    return frame, labels


def test_an_unknown_elimination_criterion_is_refused() -> None:
    """A typo falling back to the default would report one criterion under another's name."""
    with pytest.raises(ValueError, match="unknown catboost selection algorithm"):
        CatBoostSelectArm(algorithm="recursive-by-vibes")


@pytest.mark.slow
@pytest.mark.parametrize("algorithm", sorted(CATBOOST_ALGORITHMS))
def test_each_criterion_selects_its_budget_and_ranks_every_column(algorithm: str) -> None:
    """A matched-K comparison needs the budget honoured, and a ranking needs full coverage of the columns."""
    frame, labels = _bed()

    result = CatBoostSelectArm(algorithm=algorithm, k=3, iterations=60, steps=2, random_state=0).run(frame, labels)

    assert int(np.asarray(result.support, dtype=bool).sum()) == 3
    assert result.ranked_prefix is not None
    assert len(set(result.ranked_prefix)) == frame.shape[1], "the elimination order does not cover every column exactly once"


@pytest.mark.slow
def test_the_declared_score_kind_matches_what_the_arm_returns() -> None:
    """Declaring `continuous` on an elimination ORDER would let an AP read a distance that is not there."""
    frame, labels = _bed()

    result = CatBoostSelectArm(algorithm="shap", k=3, iterations=60, steps=2, random_state=0).run(frame, labels)

    assert result.score_kind == "selection_order"
    assert result.score is None, "a selection_order arm carrying a score vector would be scored by a different statistic"


@pytest.mark.slow
def test_the_arm_is_charged_for_work_the_fit_counter_cannot_see() -> None:
    """CatBoost's search runs in native code, so no Python `fit` is ever called and the counter tallies zero.

    Publishing that zero would put the most expensive arms in the roster at the top of the cost table,
    which is the one direction a cost axis must never be wrong in.
    """
    frame, labels = _bed()

    result = CatBoostSelectArm(algorithm="loss", k=3, iterations=60, steps=2, random_state=0).run(frame, labels)

    assert result.n_model_fits is not None and result.n_model_fits > 0
    assert result.provenance["n_model_fits_source"] == FITS_SOURCE_DECLARED
    assert "cannot instrument" in result.provenance.get("n_model_fits_caveat", "")


@pytest.mark.slow
def test_the_three_criteria_are_distinct_procedures() -> None:
    """A family whose members produce identical output is one arm reported three times.

    Checked on the full elimination ORDER, not on the top-k selection. On a bed with three obvious signals
    every criterion finds those three, and asserting otherwise would be demanding that the weakest of them
    fail an easy bed. Where they must differ is in the order they gave up the columns that carry nothing,
    since that order IS the criterion.

    And it has to be a WIDE bed. Measured: at twelve columns the three criteria produce a bit-identical
    order, because two elimination steps over nine probes is too coarse to separate them; at forty they
    produce three different ones. A narrow bed here would have read as "the algorithm argument does
    nothing", which is a much more alarming conclusion than the one the data supports.
    """
    frame, labels = _bed(n_probes=37)

    orders = {}
    for algorithm in sorted(CATBOOST_ALGORITHMS):
        result = CatBoostSelectArm(algorithm=algorithm, k=4, iterations=60, steps=2, random_state=0).run(frame, labels)
        orders[algorithm] = tuple(result.ranked_prefix or ())

    assert len(set(orders.values())) > 1, f"all three criteria produced the identical ranking, so they are one arm under three names: {orders}"


def test_the_outside_arms_are_in_the_roster_beside_the_repository_s_own() -> None:
    """A roster of only first-party arms cannot produce a losing result for the package it benchmarks."""
    roster = build_arm_roster(20, k=5, random_state=0)

    assert {f"catboost-{name}" for name in CATBOOST_ALGORITHMS} <= set(roster)
    assert {"mrmr", "rfecv", "boruta"} <= set(roster), "the first-party arms the outside ones exist to be compared against are missing"


def test_availability_is_checked_rather_than_assumed() -> None:
    """An arm that raised on import would give itself a reliability of zero for a reason unrelated to selection."""
    assert catboost_available() is True
