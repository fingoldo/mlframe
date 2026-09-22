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
    # Asserted: a fit that stores no replay state produces no report, and every caller below would then be examining nothing.
    assert isinstance(report, dict) and report, f"the fit stored no replay state, so there is no stability report: {report!r}"
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
    # Both preconditions are asserted: the fixture is seeded and builds an exact duplicate precisely so that one copy ranks top-k on
    # relevance and is then dropped for redundancy. If either stops holding, the rendering claims below are about nothing.
    duplicates = [nm for nm in ("strong_a", "strong_b") if nm in freq]
    assert len(duplicates) == 2, f"both duplicate columns must reach the candidate pool for this case to exist, got {duplicates}"
    dropped = [nm for nm in duplicates if nm not in picked]
    assert dropped, f"MRMR kept both duplicates ({picked}), so the redundancy drop this test describes never happened"

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
    # ``support_`` is an integer index array here, so the earlier boolean-mask branch never ran and this test asserted nothing at all.
    # Both forms are handled now, and the check is equality: the starred set IS the fit's support, not merely a subset of the columns.
    if support.dtype == bool:
        expected = {names[i] for i in range(min(len(names), support.shape[0])) if bool(support[i])}
    else:
        expected = {names[int(i)] for i in support.tolist()}
    assert expected, "the fit selected nothing, so the starred set carries no claim"
    assert set(report["selected_features"]) == expected, f"the report stars {sorted(report['selected_features'])} but the fit's support is {sorted(expected)}"


def test_frequencies_are_reproducible_for_a_fixed_seed(duplicated_signal):
    """The replay draws its resamples from a seeded generator, so two reports of one fit agree."""
    X, y = duplicated_signal
    est, first = _report(X, y)
    second = est.selection_stability_report(as_text=False, n_boot=12)
    assert first["feature_relevance_rank_frequency"] == second["feature_relevance_rank_frequency"]
