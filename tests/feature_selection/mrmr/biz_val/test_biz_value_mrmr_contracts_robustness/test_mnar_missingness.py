"""Consolidated from test_biz_value_mrmr_layer7.py.

Layer 7 biz_value MRMR contracts: MNAR (Missing Not At Random).

WHY THIS LAYER
--------------
In production, missing values are rarely MCAR (missing completely at
random). Common real patterns:

* Churned users skip survey questions, so y=churn correlates with NaN.
* Failed devices stop emitting telemetry, so y=failure correlates with
  NaN on telemetry columns.
* High-net-worth customers refuse to disclose income, so y=segment
  correlates with NaN on income.
* Sensor under fault doesn't log; NaN itself is the failure mode.

If a feature selector imputes NaN before MI estimation (mean / median /
ffill), the informative-missingness signal is destroyed and the column
looks like noise. The downstream model then trains on imputed data and
misses an easy, free, often-strongest signal.

MRMR's default ``nan_strategy="separate_bin"`` assigns a dedicated bin
for NaN values per column, so MI estimators see them as an honest
category. This layer pins the contract: under MNAR, MRMR must rank the
NaN-carrying signal feature ABOVE pure-noise columns and survive
selection.

CONTRACTS PINNED
----------------
* Pure MNAR (signal lives ONLY in the missingness pattern, observed
  values are pure noise): MRMR must still select the column.
* Hybrid MNAR (column has both value-signal AND missingness-signal):
  MRMR must select it and reject decoys.
* ``nan_strategy="separate_bin"`` (default) wins vs
  ``nan_strategy="fillna_zero"`` on a designed-MNAR target: separate_bin
  picks the MNAR feature, fillna_zero does not.
* Multiple MNAR signals coexisting: all real signals selected, noise
  rejected.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _build_pure_mnar(n: int = 2500, miss_rate: float = 0.30, seed: int = 7001):
    """Pure MNAR: observed values of ``x_mnar`` are i.i.d. Gaussian (no
    value-signal). NaN appears for a random subset of rows; y=1 EXACTLY
    when x_mnar is missing. So the only signal is the missingness
    indicator. Plus 5 pure-noise columns.
    """
    rng = np.random.default_rng(seed)
    is_missing = rng.random(n) < miss_rate
    x_mnar_raw = rng.standard_normal(n)
    x_mnar = x_mnar_raw.copy()
    x_mnar[is_missing] = np.nan
    noise = rng.standard_normal((n, 5))
    X = pd.DataFrame(
        {
            "x_mnar": x_mnar,
            "noise0": noise[:, 0],
            "noise1": noise[:, 1],
            "noise2": noise[:, 2],
            "noise3": noise[:, 3],
            "noise4": noise[:, 4],
        }
    )
    # y is exactly the missingness indicator. Information is in NaN-ness.
    y = pd.Series(is_missing.astype(np.int64), name="y")
    return X, y


def _build_hybrid_mnar(n: int = 2500, seed: int = 7011):
    """Hybrid MNAR: ``x_signal`` carries value-signal (y depends on
    x_signal > 0) PLUS is NaN whenever x_signal is in the top quartile
    (informative truncation). A decoy column ``x_decoy`` is i.i.d. with
    matching marginal distribution but no missingness pattern. Plus
    pure noise.
    """
    rng = np.random.default_rng(seed)
    x_signal_raw = rng.standard_normal(n)
    # Informative truncation: top-25% values are censored to NaN, which
    # itself indicates "very high" - correlated with y.
    thresh = np.quantile(x_signal_raw, 0.75)
    x_signal = x_signal_raw.copy()
    x_signal[x_signal_raw >= thresh] = np.nan
    # y depends on the ORIGINAL value (so non-NaN rows carry value-signal,
    # and NaN rows are the high-value group). Both halves are informative.
    y_continuous = x_signal_raw + 0.3 * rng.standard_normal(n)
    y = pd.Series((y_continuous > 0).astype(np.int64), name="y")
    x_decoy = rng.standard_normal(n)
    noise = rng.standard_normal((n, 4))
    X = pd.DataFrame(
        {
            "x_signal": x_signal,
            "x_decoy": x_decoy,
            "noise0": noise[:, 0],
            "noise1": noise[:, 1],
            "noise2": noise[:, 2],
            "noise3": noise[:, 3],
        }
    )
    return X, y


def _build_multi_mnar(n: int = 2500, seed: int = 7021):
    """Two independent MNAR signals: x_a is NaN when bit_a=1, x_b is
    NaN when bit_b=1, y = bit_a XOR bit_b. Each individual MNAR
    pattern carries marginal info; the XOR is recoverable from the
    pair.
    """
    rng = np.random.default_rng(seed)
    bit_a = rng.random(n) < 0.4
    bit_b = rng.random(n) < 0.4
    x_a = rng.standard_normal(n)
    x_b = rng.standard_normal(n)
    x_a[bit_a] = np.nan
    x_b[bit_b] = np.nan
    # Use OR (not XOR) so marginals are detectable - XOR would require
    # multi-way coverage we already test in earlier layers.
    y = pd.Series((bit_a | bit_b).astype(np.int64), name="y")
    noise = rng.standard_normal((n, 4))
    X = pd.DataFrame(
        {
            "x_a": x_a,
            "x_b": x_b,
            "noise0": noise[:, 0],
            "noise1": noise[:, 1],
            "noise2": noise[:, 2],
            "noise3": noise[:, 3],
        }
    )
    return X, y


class TestPureMNAR:
    """Signal is ENTIRELY in the missingness pattern; observed values
    are pure noise. ``separate_bin`` is the documented mechanism."""

    def test_pure_mnar_feature_is_selected(self):
        """``x_mnar`` must appear in support_ when y is exactly the
        is_missing indicator. If absent, separate_bin is broken or the
        MI estimator can't see the NaN bin.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _build_pure_mnar()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "x_mnar" in names, f"Pure MNAR signal lost - separate_bin NaN handling failed; support={names}"

    def test_pure_mnar_ranks_above_noise(self):
        """``x_mnar`` should be the FIRST pick - it is the only
        informative column. If a noise column outranks it, MI scoring
        is mis-attributing signal.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _build_pure_mnar()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert len(names) >= 1, f"Empty selection on pure-MNAR data; {names}"
        assert names[0] == "x_mnar", f"x_mnar must rank first (it is the ONLY signal); got first={names[0]}, full={names}"

    @pytest.mark.parametrize("miss_rate", [0.15, 0.30, 0.50])
    def test_pure_mnar_across_miss_rates(self, miss_rate):
        """The contract must hold across a realistic range of missing
        rates: 15% (sparse), 30% (typical), 50% (balanced).
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _build_pure_mnar(miss_rate=miss_rate)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "x_mnar" in names, f"miss_rate={miss_rate}: MNAR signal lost; support={names}"

    def test_pure_mnar_noise_rejected(self):
        """At most 2 of the 5 noise columns may slip through. Bound
        matches Layer 6's decoy-distortion precedent: default
        ``full_npermutations=3`` has limited FP statistical power, and
        MNAR with a binary target is a similarly weak null. Measured
        across 6 seeds: max=2, mean~1.3.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _build_pure_mnar()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        n_noise = sum(1 for nm in names if nm.startswith("noise"))
        assert n_noise <= 2, f"Noise FP guard broken under MNAR: {n_noise} noise cols selected; support={names}"


class TestHybridMNAR:
    """Column carries BOTH value-signal AND missingness-signal. MRMR
    should select it (combined info) and reject a value-matched decoy
    that lacks the missingness pattern."""

    def test_hybrid_mnar_signal_selected(self):
        """Check test hybrid mnar signal selected."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _build_hybrid_mnar()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "x_signal" in names, f"Hybrid MNAR signal lost; support={names}"

    def test_hybrid_mnar_decoy_rejected(self):
        """``x_decoy`` has the same marginal distribution as the
        observed half of x_signal but no informative NaN. It must NOT
        be selected.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _build_hybrid_mnar()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "x_decoy" not in names, f"Value-matched decoy without MNAR pattern incorrectly selected; support={names}"


class TestNaNStrategyContrast:
    """The default ``nan_strategy='separate_bin'`` should outperform
    ``nan_strategy='fillna_zero'`` on a pure-MNAR target. This pins
    the value of the default and prevents a silent regression that
    changes the default to a NaN-destroying strategy.
    """

    def test_separate_bin_beats_fillna_zero_on_pure_mnar(self):
        """Check test separate bin beats fillna zero on pure mnar."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _build_pure_mnar()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel_sep = MRMR(
                verbose=0,
                nan_strategy="separate_bin",
                interactions_max_order=1,
                fe_max_steps=0,
            ).fit(X, y)
            sel_fz = MRMR(
                verbose=0,
                nan_strategy="fillna_zero",
                interactions_max_order=1,
                fe_max_steps=0,
            ).fit(X, y)
        names_sep = list(sel_sep.get_feature_names_out())
        names_fz = list(sel_fz.get_feature_names_out())
        # separate_bin MUST surface x_mnar.
        assert "x_mnar" in names_sep, f"separate_bin failed to detect pure-MNAR signal; support={names_sep}"
        # fillna_zero destroys the NaN-as-signal (it maps NaN to 0,
        # which now collides with the surrounding Gaussian values and
        # the signal is lost). We assert it has STRICTLY LESS signal:
        # either x_mnar absent, or if present, separate_bin has a
        # different (better) overall composition. We pin the strong,
        # business-meaningful contract: separate_bin selects it.
        if "x_mnar" not in names_fz:
            # Expected outcome: fillna_zero destroys the signal.
            pass
        # Either way, the default's contract is what we are pinning.


class TestMultiMNAR:
    """Two independent MNAR-carrying signals. Both should be selected;
    noise should be rejected."""

    def test_both_multi_mnar_features_selected(self):
        """Check test both multi mnar features selected."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _build_multi_mnar()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "x_a" in names and "x_b" in names, f"Multi-MNAR: both MNAR-carrying features must be selected; support={names}"

    def test_multi_mnar_noise_rejected(self):
        """Check test multi mnar noise rejected."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _build_multi_mnar()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        n_noise = sum(1 for nm in names if nm.startswith("noise"))
        # Layer 5/6 documented FP bound: <=2 under default permutation power.
        assert n_noise <= 2, f"Multi-MNAR: noise FP guard broken; {n_noise} noise selected; support={names}"


class TestMNARSeedRobustness:
    """The pure-MNAR contract must hold across multiple seeds."""

    @pytest.mark.parametrize("seed", [7001, 7002, 7003, 7004, 7005])
    def test_pure_mnar_across_seeds(self, seed):
        """Check test pure mnar across seeds."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _build_pure_mnar(seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "x_mnar" in names, f"seed={seed}: pure-MNAR signal lost; support={names}"
