"""Keeping the models a feature search already paid for, and being honest about what that buys.

The arm's claim is narrow and the tests below hold it to exactly that: the fits are already paid for by
any wrapper doing this search, so the only new cost is storing predictions. It does NOT claim to make the
search cheaper, and it is not a feature selector -- it reports which columns the ensemble leaned on, which
is a different question from which columns matter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection._benchmarks.fs_hybrid._arms import build_arm_roster
from mlframe.feature_selection._benchmarks.fs_hybrid._arms_byproduct import SUBSET_SIZES, ByProductEnsembleArm, subset_predictions

ROWS = 800


def _bed(seed: int = 0) -> Any:
    """Return `(X, y)` with three informative columns and twelve probes."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({f"s{i}": rng.normal(size=ROWS) for i in range(3)})
    for index in range(12):
        frame[f"n{index:02d}"] = rng.normal(size=ROWS)
    score = 1.6 * frame["s0"] + 1.1 * frame["s1"] + 0.8 * frame["s2"]
    labels = (rng.random(ROWS) < 1.0 / (1.0 + np.exp(-score))).astype(np.int64)
    return frame, labels


def test_subset_predictions_are_out_of_fold() -> None:
    """In-sample predictions would make a memorising member look like the best one and the weights chase it.

    Checked by the property an out-of-fold vector has and an in-sample one does not: its AUC is bounded
    well away from perfect on a bed where a perfect in-sample fit is easy.
    """
    frame, labels = _bed()
    from sklearn.metrics import roc_auc_score

    predictions = subset_predictions(frame, labels, [list(frame.columns)], random_state=0)

    assert len(predictions) == 1
    assert predictions[0].shape == (ROWS,)
    assert roc_auc_score(labels, predictions[0]) < 0.99, "the predictions look in-sample, so the ensemble weights would chase memorisation"


def test_one_prediction_vector_per_subset() -> None:
    """A missing vector would silently drop a member from the ensemble it was supposed to be chosen among."""
    frame, labels = _bed()
    subsets = [["s0"], ["s0", "s1"], ["s0", "s1", "s2"]]

    predictions = subset_predictions(frame, labels, subsets, random_state=0)

    assert len(predictions) == len(subsets)
    assert all(vector.shape == (ROWS,) for vector in predictions)


def test_the_subset_sizes_are_spread_rather_than_adjacent() -> None:
    """Prefixes differing by one column give near-identical models, and an ensemble of those is one model."""
    ratios = [later / earlier for earlier, later in zip(SUBSET_SIZES, SUBSET_SIZES[1:])]

    assert min(ratios) > 1.2, f"adjacent subset sizes are too close to produce distinct members: {SUBSET_SIZES}"


def test_the_arm_scores_every_column_and_declares_that_correctly() -> None:
    """A column in no chosen subset scores zero, which is a statement rather than a gap."""
    frame, labels = _bed()

    result = ByProductEnsembleArm(k=4, random_state=0).run(frame, labels)

    assert result.score_kind == "continuous"
    assert result.score is not None and result.score.shape == (frame.shape[1],)
    assert float(result.score.min()) >= 0.0
    assert float(result.score.max()) > 0.0, "no column carries any ensemble weight, so the ensemble chose nothing"


def test_the_arm_keeps_its_budget() -> None:
    """A matched-K comparison is only matched if the budget is honoured."""
    frame, labels = _bed()

    result = ByProductEnsembleArm(k=4, random_state=0).run(frame, labels)

    assert int(np.asarray(result.support, dtype=bool).sum()) == 4


def test_the_arm_is_charged_for_every_fit_the_search_makes() -> None:
    """The argument for this arm is that the fits are already paid for; under-reporting them assumes the conclusion."""
    frame, labels = _bed()
    arm = ByProductEnsembleArm(k=4, random_state=0, n_splits=3)

    result = arm.run(frame, labels)

    n_subsets = result.provenance["n_subsets"]
    assert result.n_model_fits == n_subsets * 3, f"reported {result.n_model_fits} fits for {n_subsets} subsets over 3 folds"


def test_the_ensemble_reports_which_members_it_actually_used() -> None:
    """A with-replacement hill climb leans on a few members; which ones is the arm's real output."""
    frame, labels = _bed()

    provenance = ByProductEnsembleArm(k=4, random_state=0).run(frame, labels).provenance

    assert provenance["n_members_used"] >= 1
    assert provenance["n_members_used"] <= provenance["n_subsets"]
    assert len(provenance["ensemble_weights"]) == provenance["n_subsets"]


def test_the_internal_optimum_is_recorded_with_its_metric() -> None:
    """A selection score with no metric name cannot be differenced against a holdout score honestly."""
    frame, labels = _bed()

    result = ByProductEnsembleArm(k=4, random_state=0).run(frame, labels)

    assert result.selection_metric == "roc_auc"
    assert result.selection_score is not None and 0.0 <= result.selection_score <= 1.0


def test_the_arm_finds_the_informative_columns_on_a_bed_its_ranking_can_solve() -> None:
    """A floor, not a claim of superiority: the prefixes come from an F-test, so this bed is within reach."""
    frame, labels = _bed()
    names = [str(column) for column in frame.columns]

    result = ByProductEnsembleArm(k=3, random_state=0).run(frame, labels)
    selected = {names[index] for index, keep in enumerate(np.asarray(result.support, dtype=bool)) if keep}

    assert len(selected & {"s0", "s1", "s2"}) >= 2, f"the ensemble leaned on {sorted(selected)} on a bed with three obvious signals"


@pytest.mark.parametrize("seed", [0, 1])
def test_the_arm_is_deterministic_for_one_seed(seed: int) -> None:
    """Two runs of one cell must agree, or the paired difference every statistic computes carries noise."""
    frame, labels = _bed()

    first = ByProductEnsembleArm(k=4, random_state=seed).run(frame, labels)
    second = ByProductEnsembleArm(k=4, random_state=seed).run(frame, labels)

    np.testing.assert_allclose(np.asarray(first.score), np.asarray(second.score))


def test_the_arm_is_in_the_roster() -> None:
    """An arm outside the roster is never run, and the claim it exists to test stays untested."""
    assert "byproduct-ensemble" in build_arm_roster(20, k=5, random_state=0)
