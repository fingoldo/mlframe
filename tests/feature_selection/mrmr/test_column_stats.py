"""``MRMR.column_stats_`` / ``fit_meta_``: what the fit already held per column is returned after ``fit`` (training data of the MBHO meta-learner); nothing is recomputed."""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR

EXPECTED_COLUMNS = ["column", "position", "is_engineered", "nbins", "mi_with_target", "mi_confident", "mi_confidence", "entropy", "su", "added_by_emit_both", "origin", "in_output", "support_rank", "mrmr_gain", "dropped_by"]


@pytest.fixture(scope="module")
def fitted():
    """One small fit with the signal in x2 (marginal) and x0*x1 (interaction)."""
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame(rng.standard_normal((n, 8)), columns=[f"x{i}" for i in range(8)])
    y = ((X.x0 * X.x1 + 0.7 * X.x2 + 0.3 * rng.standard_normal(n)) > 0).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MRMR(verbose=0, random_seed=0, fe_max_steps=1).fit(X, y), X


def test_one_row_per_column_with_the_documented_fields(fitted):
    """The table has the documented columns, one row per column of the final code matrix, raw columns first and flagged as not engineered."""
    m, X = fitted
    cs = m.column_stats_
    assert list(cs.columns) == EXPECTED_COLUMNS
    assert cs["position"].tolist() == list(range(len(cs)))
    assert cs["column"].tolist()[: X.shape[1]] == list(X.columns)
    assert not cs["is_engineered"].iloc[: X.shape[1]].any() and cs["is_engineered"].iloc[X.shape[1] :].all()
    assert (cs["nbins"] >= 1).all()


def test_values_are_the_ones_the_screen_held(fitted):
    """The signal column carries the largest raw marginal MI; a confirmed gain, when present, has a confidence in [0, 1]; the selection fate is joined by name."""
    m, X = fitted
    cs = m.column_stats_
    raw = cs.iloc[: X.shape[1]].set_index("column")
    assert raw["mi_with_target"].idxmax() == "x2"
    done = cs.dropna(subset=["mi_confident"])
    assert len(done) > 0 and done["mi_confidence"].between(0.0, 1.0).all()
    assert cs["in_output"].dtype == bool
    assert bool(cs.loc[cs["column"] == "x2", "in_output"].iloc[0])


def test_fit_meta_describes_the_target_and_the_gate(fitted):
    """The fit-level context: rows, classes, the entropy of the target (ln 2 for a balanced binary one), the strategy and the relevance gate."""
    m, X = fitted
    meta = m.fit_meta_
    assert meta["n_rows"] == len(X) and meta["n_classes_y"] == 2 and meta["n_columns"] == len(m.column_stats_)
    assert meta["target_entropy"] == pytest.approx(np.log(2.0), abs=1e-2)
    assert meta["nbins_strategy"] == "sturges" and meta["effective_min_relevance_gain"] > 0.0
    assert sum(meta["class_frequencies"]) == pytest.approx(1.0)


def test_survives_pickle(fitted):
    """The table and the context are plain data and come back unchanged from a pickle round trip."""
    m, _ = fitted
    back = pickle.loads(pickle.dumps(m))  # nosec B301 - the bytes come from pickle.dumps of this test's own object
    pd.testing.assert_frame_equal(back.column_stats_, m.column_stats_)
    assert back.fit_meta_ == m.fit_meta_
