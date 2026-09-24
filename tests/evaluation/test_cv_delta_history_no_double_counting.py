"""An unchanged baseline must not add degrees of freedom every time it is compared against."""

import numpy as np

from mlframe.evaluation.cv_delta_triage import CVDeltaHistory, triage_cv_delta


def test_the_same_baseline_is_pooled_once():
    history = CVDeltaHistory()
    baseline = np.array([0.80, 0.82, 0.79, 0.81, 0.83])
    for _ in range(40):
        history.update(baseline)
    assert history.pooled_dof == 4, "40 comparisons against one 5-fold baseline carry 4 degrees of freedom, not 160"


def test_new_score_vectors_still_accumulate():
    history = CVDeltaHistory()
    rng = np.random.default_rng(0)
    for _ in range(5):
        history.update(rng.normal(0.8, 0.01, size=5))
    assert history.pooled_dof == 20


def test_a_selection_loop_does_not_narrow_its_own_band():
    """The band is the acceptance gate; an artificially narrowed one accepts noise."""
    history = CVDeltaHistory()
    baseline = np.array([0.80, 0.82, 0.79, 0.81, 0.83])
    rng = np.random.default_rng(1)
    bands = [
        triage_cv_delta(baseline, baseline + rng.normal(0, 0.005, 5), "feature_engineering", history=history, min_history_dof=1)["band"]
        for _ in range(40)
    ]
    assert bands[-1] == bands[0], "the band moved although no new information about the noise arrived"
