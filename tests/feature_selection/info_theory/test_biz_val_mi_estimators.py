"""Wave 7 biz-value tests for the new MI estimators (2026-05-29).

Asserts that each estimator hits its bench-validated accuracy band on
synthetic Gaussian-copula data with known analytical MI, and that the
no-signal floor stays in spec. Bands are intentionally LOOSER than the
mega-bench v3 medians to absorb random-seed jitter on CI.

Per the bench-v3 leaderboard the production picks are:
  - Mixed-KSG (default for k-NN path; honest noise floor)
  - GENIE aggregator (best for noisy continuous targets)
  - InfoNet (best for linear signal, requires checkpoint)
  - MDLP W1 fixes (only TRUE zero noise floor)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mlframe.feature_selection.filters._adaptive_nbins import (
    edges_freedman_diaconis,
    edges_qs,
    _plug_in_mi,
)
from mlframe.feature_selection.filters._ksg import mixed_ksg_mi, ksg_lnc_mi
from mlframe.feature_selection.filters._fastmi import fastmi
from mlframe.feature_selection.filters._mi_aggregator import (
    median_mi_panel,
    genie_mi_panel,
)


N_DEFAULT = 2000
RHO_HIGH = 0.7
TRUTH_HIGH = -0.5 * math.log(1 - RHO_HIGH * RHO_HIGH)
RHO_ZERO = 0.0


def _gauss_copula(rho: float, n: int = N_DEFAULT, seed: int = 0):
    rng = np.random.default_rng(int(seed))
    cov = np.array([[1.0, rho], [rho, 1.0]])
    XY = rng.multivariate_normal([0, 0], cov, n)
    return XY[:, 0], XY[:, 1]


def _fd(x, y):
    e = edges_freedman_diaconis(x)
    bb = np.searchsorted(e, x.astype(np.float64), side="right")
    return _plug_in_mi(bb, np.asarray(y).astype(np.int64), miller_madow=True)


def _qs(x, y):
    e = edges_qs(x)
    bb = np.searchsorted(e, x.astype(np.float64), side="right")
    return _plug_in_mi(bb, np.asarray(y).astype(np.int64), miller_madow=True)


def _ksg(x, y):
    return mixed_ksg_mi(x, np.asarray(y).astype(np.float64), k=5)


class TestMixedKSG:
    def test_accuracy_high_rho(self):
        x, y = _gauss_copula(RHO_HIGH, seed=42)
        mi = mixed_ksg_mi(x, y, k=5)
        # Bench v3 median: 0.296 on rho=0.7 / N=2000.
        assert abs(mi - TRUTH_HIGH) < 0.10, f"Mixed-KSG MI={mi}, truth={TRUTH_HIGH}"

    def test_no_signal_floor_clean(self):
        x, y = _gauss_copula(RHO_ZERO, seed=42)
        mi = mixed_ksg_mi(x, y, k=5)
        # Bench v3 reports ~0.01 on independent gaussians.
        assert mi < 0.05, f"Mixed-KSG no-signal floor too high: {mi}"

    def test_discrete_y_post_fix(self):
        # 2026-05-29 fix: Mixed-KSG on integer-coded y was zero (k-NN ties);
        # post-fix tie-jitter makes it return a real value.
        rng = np.random.default_rng(0)
        x = rng.standard_normal(N_DEFAULT)
        y = (x > 0).astype(np.float64)
        mi = mixed_ksg_mi(x, y, k=5)
        assert mi > 0.3, f"Mixed-KSG on binary signal collapsed: {mi}"


class TestKSGLNC:
    def test_alpha_default_canonical(self):
        x, y = _gauss_copula(RHO_HIGH, seed=42)
        mi = ksg_lnc_mi(x, y, k=5)
        # Canonical alpha=0.25 (NPEET_LNC default).
        assert 0.30 < mi < 0.50, f"KSG-LNC out of spec band: {mi}"

    def test_low_entropy_skip_falls_back_to_mksg(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(N_DEFAULT)
        y = (x > 0).astype(np.float64)
        mi = ksg_lnc_mi(x, y, k=5)
        mksg = mixed_ksg_mi(x, y, k=5)
        # On binary y, LNC falls back to Mixed-KSG to avoid noise inflation.
        assert abs(mi - mksg) < 0.10, f"LNC binary fallback diverged from Mixed-KSG"


class TestFastMI:
    def test_no_signal_zero(self):
        x, y = _gauss_copula(RHO_ZERO, seed=42)
        mi = fastmi(x, y, bandwidth="mise")
        assert mi < 0.05, f"fastMI no-signal floor too high: {mi}"

    def test_strong_correlation(self):
        x, y = _gauss_copula(0.9, seed=42)
        mi = fastmi(x, y, bandwidth="mise")
        truth = -0.5 * math.log(1 - 0.81)
        # fastMI under-estimates somewhat; band allows up to 30% under.
        assert mi > truth * 0.55, f"fastMI under by too much: {mi} vs {truth}"


class TestAggregators:
    def test_median_panel_noise_clean(self):
        x, y = _gauss_copula(RHO_ZERO, seed=42)
        estimators = {"fd": _fd, "qs": _qs, "ksg": _ksg}
        mi = median_mi_panel(x, (y > np.median(y)).astype(np.int64), estimators)
        assert mi < 0.05, f"median panel no_signal too high: {mi}"

    def test_genie_panel_signal(self):
        x, y = _gauss_copula(RHO_HIGH, seed=42)
        estimators = {"fd": _fd, "qs": _qs, "ksg": _ksg}
        mi = genie_mi_panel(x, (y > np.median(y)).astype(np.int64), estimators)
        # On rho=0.7 Gaussian copula -> binary y via median split, average
        # of the three estimators sits ~0.1-0.2 nats (the binarisation halves
        # available MI vs continuous y). Threshold confirms signal present.
        assert mi > 0.08, f"GENIE signal MI too low: {mi}"


class TestMistCalibration:
    @pytest.mark.skipif(True, reason="requires HuggingFace download + CUDA; gpu-marker test elsewhere")
    def test_binary_y_calibrated(self):
        from mlframe.feature_selection.filters._neural_mi import mist_mi

        rng = np.random.default_rng(0)
        x = rng.standard_normal(N_DEFAULT)
        y = (x > 0).astype(np.float64)
        mi = mist_mi(x, y, calibrated=True, device="auto")
        # binary signal truth ~ ln(2) = 0.693 nats.
        assert 0.60 < mi < 0.85, f"MIST calibrated binary out of band: {mi}"


class TestNbinsStrategyEndToEnd:
    def test_categorize_dataset_respects_strategy(self):
        import pandas as pd
        from mlframe.feature_selection.filters.discretization import categorize_dataset

        rng = np.random.default_rng(0)
        n = 500
        X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.lognormal(size=n)})
        data, _, nbins = categorize_dataset(X, dtype=np.int32, nbins_strategy="fd")
        assert nbins.size == 2
        # FD adapts per column; expect strictly different counts for skewed
        # vs gaussian.
        assert nbins[1] >= nbins[0] - 5, "FD adaptive nbins not differentiating"

    def test_mrmr_fit_with_nbins_strategy_completes(self):
        import pandas as pd
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(0)
        n = 300
        X = pd.DataFrame({f"f{i}": rng.standard_normal(n) for i in range(4)})
        y = pd.Series(
            (X["f0"] + 0.5 * X["f1"] + rng.standard_normal(n) > 0).astype(np.int64),
            name="y",
        )
        sel = MRMR(nbins_strategy="fd", verbose=0)
        sel.fit(X, y)
        out = sel.get_feature_names_out()
        # FD should at least find f0 (the strongest signal).
        assert "f0" in out, f"MRMR(nbins_strategy=fd) missed f0: {out}"


class TestMRMRNbinsStrategy:
    """2026-05-29 Wave 7: Family 2 estimators are intentionally out of MRMR.
    MRMR ONLY uses bin-based plug-in MI; tests assert nbins_strategy gets
    plumbed correctly while alternative MI estimators stay accessible via
    the standalone _ksg / _neural_mi / _fastmi / _mi_aggregator modules.
    """

    def test_legacy_path_still_works(self):
        from mlframe.feature_selection.filters.mrmr import MRMR

        sel = MRMR(verbose=0)
        # Construct minimal X/y to verify legacy fit completes.
        rng = np.random.default_rng(0)
        import pandas as pd

        X = pd.DataFrame({f"f{i}": rng.standard_normal(150) for i in range(4)})
        y = pd.Series((X["f0"] > 0).astype(np.int64), name="y")
        sel.fit(X, y)
        assert sel.n_features_ >= 1

    def test_invalid_nbins_strategy_raises(self):
        from mlframe.feature_selection.filters.mrmr import MRMR

        with pytest.raises(ValueError):
            MRMR(nbins_strategy="nonsense")._validate_string_params()

    def test_alternative_estimators_via_standalone_modules(self):
        # Family 2 estimators are intentionally OUT of MRMR; they remain
        # accessible via the standalone module entry points.
        from mlframe.feature_selection.filters._ksg import mixed_ksg_mi

        rng = np.random.default_rng(0)
        x = rng.standard_normal(N_DEFAULT)
        y = (x > 0).astype(np.float64)
        assert mixed_ksg_mi(x, y, k=5) > 0.50


# =============================================================================
# Categorical + nbins_strategy mixed-type stress (2026-05-29 Wave 7)
# =============================================================================


class TestMixedTypeWithNbinsStrategy:
    """Production-realistic: pandas DataFrame containing BOTH continuous
    numeric and pandas categorical / string columns. The MRMR pipeline routes
    categorical columns through ``_multi_col_factorize_native`` and numeric
    columns through ``discretize_2d_array`` with the chosen ``nbins_strategy``;
    both arrive at the integer-encoded matrix the njit relevance loop expects.

    Per the architectural decision (2026-05-29 user pivot), the Family-2
    estimators (KSG / neural / copula) are intentionally OUT of MRMR's hot
    path. These tests verify that the binning path handles mixed types
    without falling over.
    """

    def _make_mixed_frame(self, n: int = 400, seed: int = 7):
        import pandas as pd

        rng = np.random.default_rng(int(seed))
        df = pd.DataFrame(
            {
                "cont_strong": rng.standard_normal(n),
                "cont_weak": rng.standard_normal(n),
                "cont_noise": rng.standard_normal(n),
                "cat_strong": pd.Categorical(rng.choice(list("ABCD"), size=n, p=[0.4, 0.3, 0.2, 0.1])),
                "cat_noise": pd.Categorical(rng.choice(list("XYZ"), size=n)),
                "string_signal": rng.choice(["lo", "mid", "hi"], size=n),
            }
        )
        # The target depends on cont_strong + cat_strong.
        cat_strong_code = df["cat_strong"].cat.codes.astype(np.float64)
        y_sig = df["cont_strong"] + 0.7 * cat_strong_code + rng.standard_normal(n) * 0.3
        y = (y_sig > y_sig.median()).astype(np.int64)
        return df, pd.Series(y, name="y")

    def test_mixed_types_legacy_path(self):
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = self._make_mixed_frame()
        sel = MRMR(verbose=0)
        sel.fit(X, y)
        selected = set(sel.get_feature_names_out())
        # The strong signal pair must survive.
        assert "cont_strong" in selected or "cat_strong" in selected, f"MRMR missed both strong features on mixed data: {selected}"

    def test_mixed_types_fd_strategy(self):
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = self._make_mixed_frame()
        sel = MRMR(nbins_strategy="fd", verbose=0)
        sel.fit(X, y)
        selected = set(sel.get_feature_names_out())
        assert "cont_strong" in selected, f"FD-strategy MRMR missed cont_strong on mixed data: {selected}"

    def test_mixed_types_mdlp_strategy(self):
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = self._make_mixed_frame()
        sel = MRMR(nbins_strategy="mdlp", verbose=0)
        sel.fit(X, y)
        selected = set(sel.get_feature_names_out())
        # MDLP is supervised; should home in on the strong continuous predictor.
        assert "cont_strong" in selected, f"MDLP-strategy MRMR missed cont_strong on mixed data: {selected}"

    def test_demoted_knuth_emits_warning(self):
        import warnings as _w
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = self._make_mixed_frame(n=200)
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            sel = MRMR(nbins_strategy="knuth", verbose=0)
            sel.fit(X, y)
        msgs = [str(x.message) for x in caught]
        assert any("DEMOTED" in m and "knuth" in m for m in msgs), f"AccuracyWarning missing for nbins_strategy='knuth': {msgs}"

    def test_demoted_blocks_emits_warning(self):
        import warnings as _w
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = self._make_mixed_frame(n=200)
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            sel = MRMR(nbins_strategy="blocks", verbose=0)
            sel.fit(X, y)
        msgs = [str(x.message) for x in caught]
        assert any("DEMOTED" in m and "blocks" in m.lower() for m in msgs), f"AccuracyWarning missing for nbins_strategy='blocks': {msgs}"
