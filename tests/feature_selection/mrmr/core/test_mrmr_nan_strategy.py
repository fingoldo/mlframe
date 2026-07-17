"""Coverage of ``MRMR.nan_strategy`` across all supported values.

Strategies under test:
    - ``"separate_bin"``  (default, post-2026-05-15): NaN gets a dedicated
      max+1 bin per column; transform preserves NaN in output X.
    - ``"ffill_bfill"``: legacy time-series fill (forward then backward);
      kept for reproducibility of pre-2026-05-15 runs.
    - ``"fillna_zero"``: legacy NaN->0.0; mixes NaN rows into bin-0 with
      true-zero values (biases MI; kept only for reproducibility).
"""

import re

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import MRMR
from mlframe.feature_selection.filters.discretization import (
    _handle_missing,
    categorize_dataset,
)


def _selected_names(m: MRMR) -> list[str]:
    """Map MRMR's int-index support_ back to feature names."""
    names = list(m.feature_names_in_)
    return [names[i] for i in m.support_]


_IDENT = re.compile(r"[A-Za-z_]\w*")


def _output_names(m: MRMR) -> list[str]:
    """transform()-order selected feature names (raw survivors + engineered),
    the dedup-aware view that survives full-mode redundancy folding."""
    return [str(n) for n in m.get_feature_names_out()]


def _col_referenced(col: str, out_names: list[str]) -> bool:
    """True iff input column ``col`` is used by ANY selected feature -- either
    as a raw survivor or folded into an engineered feature whose name embeds it
    (e.g. ``add(signal_a,signal_b)`` references both ``signal_a`` and
    ``signal_b``). Under the full-mode default a linear signal pair is routinely
    de-duplicated into one engineered column, so raw ``support_`` membership
    undercounts; token-reference membership is the faithful recovery view."""
    for nm in out_names:
        if col in set(_IDENT.findall(nm)):
            return True
    return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_with_random_nan():
    """1000 rows x 6 cols, MCAR-style NaN (30% / 50% in 2 of 6 cols) with a
    real signal: y depends only on cols ``signal_a`` + ``signal_b`` (no NaN).
    Other cols are pure noise; the noise-with-NaN cols should NOT outrank the
    true signal cols under any reasonable MI estimator."""
    rng = np.random.default_rng(7)
    n = 1000
    sig_a = rng.normal(size=n)
    sig_b = rng.normal(size=n)
    noise = rng.normal(size=n)
    extra = rng.normal(size=n)
    nan_col_1 = rng.normal(size=n)
    nan_col_2 = rng.normal(size=n)
    nan_col_1[rng.random(n) < 0.3] = np.nan
    nan_col_2[rng.random(n) < 0.5] = np.nan
    y = 2.0 * sig_a + 1.5 * sig_b + 0.1 * rng.normal(size=n)
    return pd.DataFrame(
        {
            "signal_a": sig_a,
            "signal_b": sig_b,
            "noise": noise,
            "extra_noise": extra,
            "noise_with_nan_30pct": nan_col_1,
            "noise_with_nan_50pct": nan_col_2,
        }
    ), pd.Series(y, name="y")


@pytest.fixture
def df_informative_missingness():
    """y depends ONLY on whether ``measured_value`` is missing -- the
    presence/absence pattern carries 100% of the signal. A correct MI
    estimator with NaN-as-separate-bin should detect this column as
    informative."""
    rng = np.random.default_rng(11)
    n = 1500
    is_missing = rng.random(n) < 0.4
    y = is_missing.astype(np.float64) + 0.05 * rng.normal(size=n)
    val = rng.normal(size=n)
    val[is_missing] = np.nan
    distractor_a = rng.normal(size=n)
    distractor_b = rng.normal(size=n)
    return pd.DataFrame(
        {
            "measured_value_with_meaningful_nan": val,
            "distractor_a": distractor_a,
            "distractor_b": distractor_b,
        }
    ), pd.Series(y, name="y")


# ---------------------------------------------------------------------------
# 1. Every strategy fits + transforms without raising
# ---------------------------------------------------------------------------


class TestEveryStrategyRunsCleanly:
    @pytest.mark.parametrize("strategy", ["separate_bin", "ffill_bfill", "fillna_zero"])
    def test_fit_does_not_raise(self, df_with_random_nan, strategy):
        X, y = df_with_random_nan
        m = MRMR(quantization_nbins=4, nan_strategy=strategy, verbose=0)
        m.fit(X, y)
        assert m.support_ is not None
        assert len(m.support_) > 0

    @pytest.mark.parametrize("strategy", ["separate_bin", "ffill_bfill", "fillna_zero"])
    def test_transform_keeps_row_count(self, df_with_random_nan, strategy):
        X, y = df_with_random_nan
        m = MRMR(quantization_nbins=4, nan_strategy=strategy, verbose=0)
        m.fit(X, y)
        Xt = m.transform(X)
        # transform may drop columns (selection) but rows must match input
        assert Xt.shape[0] == X.shape[0]
        # selected col count matches support
        assert Xt.shape[1] >= 1


# ---------------------------------------------------------------------------
# 2. separate_bin: signal cols rank above pure-noise-with-NaN cols
# ---------------------------------------------------------------------------


class TestSeparateBinDoesNotPickPureNanNoiseOverSignal:
    """Under MCAR NaN that's uncorrelated with y, MRMR with separate_bin must
    rank the actual signal columns ABOVE the noise-with-NaN columns. The OLD
    fillna_zero path used to inflate MI on noise-with-NaN columns because the
    NaN rows landed in bin-0 alongside true-zero rows."""

    def test_signal_cols_are_selected(self, df_with_random_nan):
        X, y = df_with_random_nan
        m = MRMR(quantization_nbins=4, nan_strategy="separate_bin", verbose=0)
        m.fit(X, y)
        # Re-baselined for full-mode default (use_simple_mode=False): on the
        # linear target y=2*signal_a+1.5*signal_b full mode de-duplicates the
        # two correlated-to-y signal columns into a SINGLE engineered feature
        # (e.g. add(signal_a,signal_b)) that lives in get_feature_names_out()
        # with an EMPTY/partial raw ``support_``, so the old "both signal cols
        # in support_ names" check fails despite a perfect recovery. Credit a
        # signal column when it is referenced by ANY selected feature (raw or
        # engineered). Still falsifiable: a separate_bin regression that
        # inflated MI on the noise-with-NaN columns would surface those columns
        # and drop the signal references.
        out_names = _output_names(m)
        assert _col_referenced("signal_a", out_names), f"signal_a not used by any selected feature; out={out_names}"
        assert _col_referenced("signal_b", out_names), f"signal_b not used by any selected feature; out={out_names}"
        # The NaN-handling contract this test guards: noise-with-NaN columns
        # must NOT be selected ahead of the signal. Reframed for dedup -- the
        # first selected output feature that references signal must precede any
        # selected noise-with-NaN raw column (a separate_bin bug would surface
        # the NaN-noise columns first).
        signal_positions = [i for i, nm in enumerate(out_names) if _col_referenced("signal_a", [nm]) or _col_referenced("signal_b", [nm])]
        nan_noise_positions = [i for i, nm in enumerate(out_names) if nm in ("noise_with_nan_30pct", "noise_with_nan_50pct")]
        if nan_noise_positions and signal_positions:
            assert min(signal_positions) < min(nan_noise_positions), f"separate_bin must rank signal above noise-with-NaN; got out={out_names}"


# ---------------------------------------------------------------------------
# 3. separate_bin honestly surfaces informative-missingness pattern
# ---------------------------------------------------------------------------


class TestSeparateBinSurfacesInformativeNan:
    """If the missingness pattern carries the signal (y == is_missing), the
    dedicated NaN bin gives that column high MI(X_disc, y) and MRMR picks it."""

    def test_pick_the_informative_nan_column(self, df_informative_missingness):
        X, y = df_informative_missingness
        m = MRMR(quantization_nbins=4, nan_strategy="separate_bin", verbose=0)
        m.fit(X, y)
        names = list(m.feature_names_in_)
        order = [names[i] for i in m.support_]
        assert order[0] == "measured_value_with_meaningful_nan", f"Informative NaN pattern should be picked first; got order={order}"


# ---------------------------------------------------------------------------
# 4. transform() preserves NaN for downstream NaN-aware models
# ---------------------------------------------------------------------------


class TestTransformPreservesNan:
    def test_separate_bin_output_keeps_nan_rows(self, df_with_random_nan):
        X, y = df_with_random_nan
        m = MRMR(quantization_nbins=4, nan_strategy="separate_bin", verbose=0)
        m.fit(X, y)
        Xt = m.transform(X)
        selected = _selected_names(m)
        nan_cols_picked = [c for c in selected if c.startswith("noise_with_nan")]
        if nan_cols_picked:
            for col in nan_cols_picked:
                if isinstance(Xt, pd.DataFrame):
                    assert Xt[col].isna().any(), f"col {col} lost its NaN values in transform output"
                else:
                    idx = selected.index(col)
                    assert np.isnan(Xt[:, idx]).any()


# ---------------------------------------------------------------------------
# 5. Default strategy is separate_bin
# ---------------------------------------------------------------------------


class TestDefaultStrategy:
    def test_default_is_separate_bin(self):
        m = MRMR()
        assert m.nan_strategy == "separate_bin"


# ---------------------------------------------------------------------------
# 6. Invalid strategy at _handle_missing level raises clearly
# ---------------------------------------------------------------------------


class TestUnknownStrategyRaises:
    def test_unknown_strategy_raises_value_error(self):
        arr = np.array([[1.0, np.nan], [2.0, 3.0]])
        with pytest.raises(ValueError, match="unknown missing-value strategy"):
            _handle_missing(arr, strategy="totally_invented")


# ---------------------------------------------------------------------------
# 7. Direct categorize_dataset: NaN gets a dedicated max+1 bin
# ---------------------------------------------------------------------------


class TestCategorizeDatasetSeparateBinDirect:
    def test_nan_rows_get_max_bin_index(self):
        df = pd.DataFrame(
            {
                "col_a": [1.0, 2.0, np.nan, 4.0, np.nan, 6.0, np.nan, 8.0],
                "col_b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            }
        )
        n_bins = 4
        data, cols, nbins = categorize_dataset(
            df=df,
            method="quantile",
            n_bins=n_bins,
            dtype=np.int32,
            missing_strategy="separate_bin",
        )
        col_a_idx = cols.index("col_a")
        nan_positions = [2, 4, 6]
        # All NaN rows in col_a must land in bin == n_bins (the dedicated bin).
        assert (data[nan_positions, col_a_idx] == n_bins).all(), f"NaN rows should be at bin={n_bins}; got {data[nan_positions, col_a_idx]}"
        non_nan_positions = [0, 1, 3, 5, 7]
        assert (data[non_nan_positions, col_a_idx] < n_bins).all()
        assert (data[non_nan_positions, col_a_idx] >= 0).all()
        # nbins for col_a includes the NaN bin
        assert nbins[col_a_idx] == n_bins + 1
        # col_b has no NaN -> stays at n_bins
        col_b_idx = cols.index("col_b")
        assert nbins[col_b_idx] == n_bins

    def test_fillna_zero_mixes_nan_with_zero_into_bin_zero(self):
        df = pd.DataFrame(
            {
                "col_a": [0.0, 0.0, np.nan, np.nan, 5.0, 6.0, 7.0, 8.0],
            }
        )
        data, cols, nbins = categorize_dataset(
            df=df,
            method="uniform",
            n_bins=4,
            dtype=np.int32,
            missing_strategy="fillna_zero",
        )
        # Rows 0/1 (true zeros) and rows 2/3 (NaN-->0) all end up in same bin.
        assert (data[:4, 0] == data[0, 0]).all(), f"fillna_zero mixes NaN with true zeros; got {data[:4, 0]}"

    def test_propagate_smoke(self):
        df = pd.DataFrame({"col_a": [1.0, 2.0, np.nan, 4.0]})
        data, cols, nbins = categorize_dataset(
            df=df,
            method="quantile",
            n_bins=4,
            dtype=np.int32,
            missing_strategy="propagate",
        )
        assert data.shape == (4, 1)


# ---------------------------------------------------------------------------
# 8. Edge case: column that's 100% NaN under separate_bin
# ---------------------------------------------------------------------------


class TestAllNanColumn:
    def test_all_nan_column_under_separate_bin(self):
        """An all-NaN column has no finite values for percentile edges;
        _handle_missing must fall back to a sentinel so discretize doesn't
        crash, then the dedicated NaN bin captures all rows."""
        import warnings

        df = pd.DataFrame(
            {
                "all_nan": [np.nan] * 100,
                "real": np.linspace(0.0, 1.0, 100),
            }
        )
        with warnings.catch_warnings():
            # nanmedian on all-NaN col emits "All-NaN slice" RuntimeWarning;
            # we handle that case and the warning is expected.
            warnings.simplefilter("ignore", RuntimeWarning)
            data, cols, nbins = categorize_dataset(
                df=df,
                method="quantile",
                n_bins=4,
                dtype=np.int32,
                missing_strategy="separate_bin",
            )
        col_idx = cols.index("all_nan")
        assert (data[:, col_idx] == 4).all()
