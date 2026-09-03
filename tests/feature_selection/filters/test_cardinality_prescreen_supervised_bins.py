"""The cardinality pre-screen judged supervised bin counts as if they were raw level counts.

``cardinality_prescreen`` drops columns with ``nbins_x > 2*sqrt(n)`` -- a guard against user_id-style categoricals
whose thousand levels make the plug-in MI hopelessly biased. But MRMR discretises numerics with MDLP by default,
and MDLP awards a column MORE bins the better it explains the target. So the guard read signal strength as
cardinality and removed the single most informative feature, silently: the log sat behind ``verbose >= 1``, which
the training suite never enables, and the run reported ``fallback_used_=False`` -- a confident wrong answer.

Measured on the fixture below (n=500, ceiling 2*sqrt(500)=44.7): MDLP gave the driving feature 50 bins against
2-5 for the pure-noise columns, so it alone tripped the ceiling. sklearn's mutual_info_regression puts it at
2.676 against 0.290 for the runner-up.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._screen_predictors_prescreen import cardinality_prescreen
from mlframe.feature_selection.filters.discretization import numeric_column_names

N = 500
CEILING = 2.0 * np.sqrt(N)


@pytest.fixture(scope="module")
def two_subpopulation_frame():
    """A drives the target on 80% of rows, B on the remaining 20%; C and D are pure noise."""
    rng = np.random.default_rng(20260611)
    n_a, n_b = 400, 100
    n = n_a + n_b
    a, b, c, d = (rng.standard_normal(n) for _ in range(4))
    y = np.empty(n)
    noise = 0.05 * rng.standard_normal(n)
    y[:n_a] = 3.0 * a[:n_a] + noise[:n_a]
    y[n_a:] = 3.0 * b[n_a:] + noise[n_a:]
    return pd.DataFrame({"A": a, "B": b, "C": c, "D": d}), pd.Series(y)


def _prescreen(nbins, names, raw_cardinality_cols, verbose=0):
    """Run the pre-screen over a dummy matrix with the given per-column bin counts; last column is the target."""
    data = np.zeros((N, len(nbins)), dtype=np.int16)
    return cardinality_prescreen(data, np.asarray(nbins), names, set(range(len(nbins) - 1)), [len(nbins) - 1], verbose,
                                 raw_cardinality_cols=raw_cardinality_cols)


class TestWhatTheCeilingMayJudge:
    """Bin counts mean different things for categorical and supervised-binned numeric columns."""

    def test_a_supervised_binned_numeric_over_the_ceiling_survives(self):
        """The bug in one assertion: 50 MDLP bins meant A explains the target, not that A is a nuisance column."""
        x, refused = _prescreen([50, 4, 5, 2, 5], ["A", "B", "C", "D", "targ"], raw_cardinality_cols=set())
        assert refused == set()
        assert sorted(x) == [0, 1, 2, 3]

    def test_a_raw_categorical_over_the_ceiling_is_still_refused(self):
        """The guard's real purpose has to keep working -- 1200 levels genuinely cannot be scored honestly."""
        x, refused = _prescreen([1200, 4, 5, 2, 5], ["user_id", "B", "C", "D", "targ"], raw_cardinality_cols={"user_id"})
        assert refused == {0}
        assert 0 not in x

    def test_a_raw_categorical_under_the_ceiling_survives(self):
        """Being categorical is not itself disqualifying; only the level count is."""
        _, refused = _prescreen([12, 4, 5, 2, 5], ["region", "B", "C", "D", "targ"], raw_cardinality_cols={"region"})
        assert refused == set()

    def test_none_keeps_the_legacy_all_columns_behaviour(self):
        """Callers with no supervised strategy have unsupervised bin counts everywhere, as before."""
        _, refused = _prescreen([50, 4, 5, 2, 5], ["A", "B", "C", "D", "targ"], raw_cardinality_cols=None)
        assert refused == {0}

    def test_the_target_column_is_never_refused(self):
        """It is not a candidate to begin with; refusing it would be meaningless bookkeeping."""
        _, refused = _prescreen([4, 4, 5, 2, 900], ["A", "B", "C", "D", "targ"], raw_cardinality_cols={"targ"})
        assert refused == set()

    def test_single_bin_columns_are_left_to_the_relevance_gates(self):
        """A constant column carries no signal, but the CEILING is not what should be dropping it."""
        _, refused = _prescreen([1, 4, 5, 2, 5], ["const", "B", "C", "D", "targ"], raw_cardinality_cols={"const"})
        assert refused == set()


class TestTheRefusalIsAudible:
    """Removing a column from selection must never be silent."""

    def test_a_refusal_warns_even_at_verbose_zero(self, caplog):
        """The suite runs the selector at verbose=0, so an INFO line behind verbose>=1 reached nobody."""
        with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.filters._screen_predictors_prescreen"):
            _prescreen([1200, 4, 5, 2, 5], ["user_id", "B", "C", "D", "targ"], raw_cardinality_cols={"user_id"}, verbose=0)
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "user_id" in text
        assert "cardinality_bias_correction" in text

    def test_no_refusal_stays_quiet(self, caplog):
        """A clean pool must not emit a warning about dropped columns."""
        with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.filters._screen_predictors_prescreen"):
            _prescreen([50, 4, 5, 2, 5], ["A", "B", "C", "D", "targ"], raw_cardinality_cols=set())
        assert caplog.records == []


class TestWhichColumnsAreSupervisedBinned:
    """The numeric/categorical split is what decides eligibility, so it gets its own pin."""

    def test_numeric_columns_are_reported_for_pandas(self):
        """Category, string and bool columns are level codes, not measurements."""
        df = pd.DataFrame({"num": [1.0, 2.0], "cat": pd.Categorical(["a", "b"]), "txt": ["x", "y"], "flag": [True, False]})
        assert numeric_column_names(df) == {"num"}

    def test_numeric_columns_are_reported_for_polars(self):
        """The same split has to hold for the polars schema path."""
        pl = pytest.importorskip("polars")
        df = pl.DataFrame({"num": [1.0, 2.0], "txt": ["x", "y"], "flag": [True, False]})
        assert numeric_column_names(df) == {"num"}


@pytest.mark.slow
class TestTheSelectorEndToEnd:
    """The behaviour the unit pins exist to protect."""

    def _fit(self, X, y):
        """A minimal MRMR fit on the raw columns -- no FE, deterministic seed."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        m = MRMR(verbose=0, use_simple_mode=True, quantization_nbins=5, random_state=20260610, n_workers=1, fe_max_steps=0, max_runtime_mins=1)
        m.fit(X, y)
        return m

    def test_the_driving_feature_is_selected(self, two_subpopulation_frame):
        """Pre-fix this returned ['B'] -- the noise-dominant runner-up -- with fallback_used_=False."""
        X, y = two_subpopulation_frame
        assert "A" in set(self._fit(X, y).get_feature_names_out())

    def test_the_driving_feature_is_scored_at_all(self, two_subpopulation_frame):
        """Pre-fix A had no entry in cached_MIs: it was dropped before any candidate was enumerated."""
        X, y = two_subpopulation_frame
        m = self._fit(X, y)
        assert (0,) in (m.cached_MIs or {})
        assert m.cached_MIs[(0,)] > m.cached_MIs.get((1,), 0.0)

    def test_the_ranking_agrees_with_an_independent_estimator(self, two_subpopulation_frame):
        """sklearn is the outside check that A really is the stronger column, not just a different answer."""
        from sklearn.feature_selection import mutual_info_regression

        X, y = two_subpopulation_frame
        mi = mutual_info_regression(X.values, y.values, random_state=0)
        assert mi[0] > mi[1] > mi[2:].max()


# Measured pre-fix on the gaussian fixture: only ``mdlp`` -- the DEFAULT -- dropped the driving column (selection
# came back as ['B']); mdlp_validated, quantile, sturges, knuth and freedman_diaconis all kept it, because their
# bin counts for that column stay under the ceiling. The passing strategies are parametrised anyway so a future
# change to any of their bin-count formulas cannot re-introduce the defect unnoticed.
BINNING_STRATEGIES = ("mdlp", "mdlp_validated", "quantile", "sturges", "knuth", "freedman_diaconis")


def _make_frame(kind: str, seed: int = 20260611):
    """The two-subpopulation fixture over a given marginal distribution for the features."""
    rng = np.random.default_rng(seed)
    n_a, n_b = 400, 100
    n = n_a + n_b
    if kind == "gaussian":
        cols = [rng.standard_normal(n) for _ in range(4)]
    elif kind == "student_t":  # heavy tails: the regime where plug-in MI bias is worst
        cols = [rng.standard_t(df=3, size=n) for _ in range(4)]
    elif kind == "lognormal":  # strong right skew, so quantile edges bunch up
        cols = [rng.lognormal(sigma=1.0, size=n) for _ in range(4)]
    elif kind == "uniform":
        cols = [rng.uniform(-3.0, 3.0, size=n) for _ in range(4)]
    else:
        raise ValueError(f"unknown distribution {kind!r}")
    a, b, c, d = cols
    y = np.empty(n)
    noise = 0.05 * rng.standard_normal(n)
    y[:n_a] = 3.0 * a[:n_a] + noise[:n_a]
    y[n_a:] = 3.0 * b[n_a:] + noise[n_a:]
    return pd.DataFrame({"A": a, "B": b, "C": c, "D": d}), pd.Series(y)


def _bin_counts(X, y, strategy):
    """Per-column bin counts ``categorize_dataset`` assigns under one strategy."""
    from mlframe.feature_selection.filters.discretization import categorize_dataset

    _, cols, nbins = categorize_dataset(df=X, n_bins=5, nbins_strategy=strategy, y_for_strategy=y.values)
    return dict(zip(cols, (int(v) for v in nbins)))


@pytest.mark.slow
class TestAcrossBinningStrategies:
    """The defect is "bin count used as a cardinality proxy", so it has to be pinned per strategy, not once."""

    def _selected(self, X, y, strategy):
        """Feature names one MRMR fit keeps under ``strategy``."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        m = MRMR(verbose=0, use_simple_mode=True, quantization_nbins=5, random_state=20260610, n_workers=1,
                 fe_max_steps=0, max_runtime_mins=1, nbins_strategy=strategy)
        m.fit(X, y)
        return set(m.get_feature_names_out())

    @pytest.mark.parametrize("strategy", BINNING_STRATEGIES)
    def test_the_driving_feature_survives_every_strategy(self, strategy, two_subpopulation_frame):
        """Whatever a strategy does to bin counts, the strongest column must still reach the selector."""
        X, y = two_subpopulation_frame
        assert "A" in self._selected(X, y, strategy)

    @pytest.mark.parametrize("strategy", BINNING_STRATEGIES)
    def test_the_pool_is_never_emptied(self, strategy, two_subpopulation_frame):
        """A ceiling that can refuse every numeric column at once would leave the selector nothing to rank."""
        X, y = two_subpopulation_frame
        assert self._selected(X, y, strategy), f"{strategy}: pre-screen left an empty candidate pool"

    def test_at_least_one_strategy_actually_trips_the_ceiling(self, two_subpopulation_frame):
        """Guards the guard: if no strategy exceeded the ceiling any more, the tests above would pass vacuously."""
        X, y = two_subpopulation_frame
        tripping = {s for s in BINNING_STRATEGIES if max(_bin_counts(X, y, s).values()) > CEILING}
        assert tripping, f"no strategy exceeds the {CEILING:.1f}-bin ceiling; these tests no longer exercise the bug"

    def test_mdlp_gives_the_driving_column_the_most_bins(self, two_subpopulation_frame):
        """The mechanism in one assertion: supervised binning rewards explanatory power with bins."""
        X, y = two_subpopulation_frame
        counts = _bin_counts(X, y, "mdlp")
        assert counts["A"] == max(counts.values())
        assert counts["A"] > CEILING


@pytest.mark.slow
class TestAcrossDistributions:
    """A bin-count threshold interacts with the marginal shape, so one Gaussian fixture is not enough evidence."""

    @pytest.mark.parametrize("kind", ["gaussian", "student_t", "lognormal", "uniform"])
    def test_the_driving_feature_survives_every_distribution(self, kind):
        """Pre-fix ALL FOUR dropped the driving column under the default strategy -- student_t even kept a pure-
        noise column instead -- so the defect is systematic, not an artefact of one marginal shape."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _make_frame(kind)
        m = MRMR(verbose=0, use_simple_mode=True, quantization_nbins=5, random_state=20260610, n_workers=1, fe_max_steps=0, max_runtime_mins=1)
        m.fit(X, y)
        assert "A" in set(m.get_feature_names_out()), f"{kind}: driving feature dropped"


class TestWhenColumnsCannotBeNamed:
    """A caller that supplies no column names must not silently disable the guard."""

    def test_unnamed_columns_keep_the_legacy_bin_count_judgement(self):
        """Without names nothing can be classified, so the ceiling has to stay in force rather than go inert."""
        data = np.zeros((N, 5), dtype=np.int16)
        _, refused = cardinality_prescreen(data, np.asarray([1200, 4, 5, 2, 5]), None, set(range(4)), [4], 0, raw_cardinality_cols={"user_id"})
        assert refused == {0}
