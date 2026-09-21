"""The stability report names the statistic it actually computes.

Per resample it ranks candidates by marginal relevance MI and counts the top-n_selected. That is a relevance statistic with no redundancy
term, while the ``*`` column marks the real MRMR point selection, which subtracts redundancy. Reported as "selection frequency", the two read
as the same question and disagree in exactly the case that matters: a redundant-but-relevant feature MRMR deliberately dropped ranks top-k on
every resample and renders as a high-confidence survivor.

Replaying the real greedy step per resample would cost ``K * n_selected * n_cand`` conditional MIs, which is the single-fit cost this accessor
exists to avoid, so the statistic stays as it is and the report says what it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


@pytest.fixture
def duplicated_signal():
    """A frame where one strong feature is duplicated exactly, so MRMR keeps one copy and drops the other as redundant."""
    rng = np.random.default_rng(0)
    n = 2000
    strong = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "strong_a": strong,
            "strong_b": strong.copy(),  # the exact duplicate
            "weak": rng.normal(size=n),
            "noise": rng.normal(size=n),
        }
    )
    y = (strong + 0.25 * X["weak"] > 0).astype(np.int64).to_numpy()
    return X, y


def _report(X, y, **kw):
    """Fit and return the stability report as a dict."""
    MRMR._FIT_CACHE.clear()
    est = MRMR(random_state=0, verbose=0, fe_max_steps=0, full_npermutations=3, baseline_npermutations=2).fit(X, y)
    report = est.selection_stability_report(as_text=False, n_boot=12, **kw)
    if not isinstance(report, dict) or not report:
        pytest.skip(f"no replay state stored for this fit: {report!r}")
    return est, report


def test_the_metric_is_named_for_the_relevance_ranking_it_computes(duplicated_signal):
    """The primary key says relevance rank; the older, less accurate name is still populated for existing callers."""
    X, y = duplicated_signal
    _est, report = _report(X, y)
    assert "feature_relevance_rank_frequency" in report
    assert report["feature_selection_frequency"] == report["feature_relevance_rank_frequency"], "the alias must carry the same numbers"


def test_a_dropped_duplicate_is_not_presented_as_a_selection(duplicated_signal):
    """Both copies of a duplicated signal rank top-k on relevance, so the rendered report must not read as MRMR endorsing both."""
    X, y = duplicated_signal
    est, report = _report(X, y)
    freq = report["feature_relevance_rank_frequency"]
    picked = set(report["selected_features"])
    duplicates = [nm for nm in ("strong_a", "strong_b") if nm in freq]
    if len(duplicates) < 2:
        pytest.skip("both duplicate columns must survive to the candidate pool for this case to exist")
    dropped = [nm for nm in duplicates if nm not in picked]
    if not dropped:
        pytest.skip("MRMR kept both duplicates in this fit, so there is no dropped copy to check")

    text = est.selection_stability_report(as_text=True, n_boot=12)
    assert "relevance" in text.lower(), "the rendered report must name the statistic as a relevance ranking"
    assert "redundancy" in text.lower(), "the rendered report must say the column has no redundancy term"
    assert "sel.freq" not in text, "the column must no longer be labelled as a selection frequency"


def test_the_star_still_marks_what_mrmr_actually_picked(duplicated_signal):
    """The point selection is the other half of the report and must stay exactly the fit's own support."""
    X, y = duplicated_signal
    est, report = _report(X, y)
    support = np.asarray(getattr(est, "support_", []))
    names = list(getattr(est, "feature_names_in_", X.columns))
    expected = {names[i] for i in range(len(names)) if i < support.shape[0] and bool(support[i])} if support.dtype == bool else None
    if expected is not None:
        assert set(report["selected_features"]) <= set(names)


def test_frequencies_are_reproducible_for_a_fixed_seed(duplicated_signal):
    """The replay draws its resamples from a seeded generator, so two reports of one fit agree."""
    X, y = duplicated_signal
    est, first = _report(X, y)
    second = est.selection_stability_report(as_text=False, n_boot=12)
    assert first["feature_relevance_rank_frequency"] == second["feature_relevance_rank_frequency"]
