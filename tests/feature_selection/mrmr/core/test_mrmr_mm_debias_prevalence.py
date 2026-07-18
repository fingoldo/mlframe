"""Unit + biz_value tests for the Miller-Madow debias of the FE joint-prevalence
ratio gate + the IRON-RULE maxT-floor co-update (2026-06-09, backlog #1 + #4).

The mechanism is benchmark-REJECTED as a default (it admits cross-mix noise on the
weak F2 -- see ``MRMR.fe_mm_debias_prevalence``), but the closed-form estimator, the
occupied-K refinement (#4), the denominator-positivity guard, and the consistent
maxT-floor debias are all retained as an OPT-IN. These tests pin:

  * the closed-form MM MI correction ``I - (k_x-1)(k_y-1)/2n`` (and the ``k<=1`` passthrough);
  * occupied-K counting (#4) == non-empty bins;
  * the prevalence-ratio debias raises the ratio for a real 1-D summary of an
    over-binned 2-D joint, with the denominator-positivity guard falling back to raw;
  * the maxT floor MM co-update keeps genuine synergy above the floor AND noise at/below
    it (the floor is NOT weakened by the ratio relaxation -- IRON RULE);
  * the DEFAULT is OFF => selection is byte-stable vs the raw-plug-in path.
"""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Unit: the closed-form Miller-Madow MI correction.
# ---------------------------------------------------------------------------
class TestMillerMadowMICorrection:
    """Groups tests covering TestMillerMadowMICorrection."""
    def test_closed_form_bias_subtraction(self):
        """Closed form bias subtraction."""
        from mlframe.feature_selection.filters.info_theory import mi_miller_madow_correct

        # I - (k_x-1)(k_y-1)/2n
        assert mi_miller_madow_correct(0.5, 10, 5, 1000) == pytest.approx(0.5 - 9 * 4 / 2000.0)
        assert mi_miller_madow_correct(1.18, 100, 10, 500) == pytest.approx(1.18 - 99 * 9 / 1000.0)

    def test_degenerate_cardinality_passthrough(self):
        """Degenerate cardinality passthrough."""
        from mlframe.feature_selection.filters.info_theory import mi_miller_madow_correct

        # k_x <= 1 or k_y <= 1 -> the bias term is 0 / negative; pass the plug-in through.
        assert mi_miller_madow_correct(0.5, 1, 5, 1000) == 0.5
        assert mi_miller_madow_correct(0.5, 10, 1, 1000) == 0.5
        assert mi_miller_madow_correct(0.5, 0, 5, 1000) == 0.5

    def test_vanishes_at_large_n(self):
        """Vanishes at large n."""
        from mlframe.feature_selection.filters.info_theory import mi_miller_madow_correct

        # ->0 as n->inf : large-n selection byte-untouched.
        big = mi_miller_madow_correct(0.5, 100, 10, 10_000_000)
        assert abs(big - 0.5) < 1e-4


class TestOccupiedK:
    """Groups tests covering TestOccupiedK."""
    def test_counts_nonempty_bins(self):
        """Counts nonempty bins."""
        from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_gates import _occupied_k

        # codes use only 3 of 10 nominal bins -> occupied-K = 3 (#4 collapse).
        codes = np.array([0, 0, 4, 4, 9, 9, 9], dtype=np.int64)
        assert _occupied_k(codes) == 3
        assert _occupied_k(np.array([], dtype=np.int64)) == 0


# ---------------------------------------------------------------------------
# Unit: the debiased prevalence ratio + denominator-positivity guard.
# ---------------------------------------------------------------------------
class TestMMDebiasedPrevalenceRatio:
    """Groups tests covering TestMMDebiasedPrevalenceRatio."""
    def test_raises_ratio_for_overbinned_joint(self):
        """Raises ratio for overbinned joint."""
        from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_gates import (
            mm_debiased_prevalence_ratio,
        )

        # He2(a)*b @ n=500: raw 0.65/1.18 = 0.554 (< 0.90 bar) -> MM (occ K_joint=100) > 0.90.
        raw = 0.65 / 1.18
        mm = mm_debiased_prevalence_ratio(0.65, 1.18, k_eng=10, k_joint=100, k_y=10, n=500)
        assert raw < 0.90
        assert mm > raw
        assert mm > 0.90

    def test_denominator_guard_falls_back_to_raw(self):
        """Denominator guard falls back to raw."""
        from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_gates import (
            mm_debiased_prevalence_ratio,
        )

        # Tiny joint MI vs a huge MM term -> corrected denom <= 0; must NOT explode/sign-flip,
        # must fall back to the raw ratio (the existing gate behaviour).
        raw = 0.1 / 0.05
        mm = mm_debiased_prevalence_ratio(0.1, 0.05, k_eng=10, k_joint=100, k_y=10, n=200)
        assert mm == pytest.approx(raw)

    def test_zero_or_degenerate_inputs(self):
        """Zero or degenerate inputs."""
        from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_gates import (
            mm_debiased_prevalence_ratio,
        )

        assert mm_debiased_prevalence_ratio(0.3, 0.0, k_eng=10, k_joint=100, k_y=10, n=500) == 0.0
        # single-class target -> no correction, raw ratio.
        assert mm_debiased_prevalence_ratio(0.3, 0.6, k_eng=10, k_joint=100, k_y=1, n=500) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Unit: the IRON-RULE maxT-floor co-update keeps noise rejected.
# ---------------------------------------------------------------------------
class TestMaxTFloorMMCoUpdate:
    """Groups tests covering TestMaxTFloorMMCoUpdate."""
    def _wide_synergy(self, n=1500, n_noise=40, seed=20260603):
        """Wide synergy."""
        rng = np.random.default_rng(seed)
        x = {f"x{i}": rng.normal(size=n) for i in range(1, 7)}
        lin = 1.5 * np.sign(x["x1"] * x["x2"]) + 1.2 * np.sign(x["x3"] * x["x4"]) + 1.0 * np.sign(x["x5"] * x["x6"] + 0.3 * x["x5"])
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-lin))).astype(int)
        d = dict(x)
        for j in range(n_noise):
            d[f"noise_{j}"] = rng.normal(size=n)
        d["y"] = y
        return pd.DataFrame(d)

    def test_mm_floor_keeps_genuine_rejects_noise(self):
        """Mm floor keeps genuine rejects noise."""
        from mlframe.feature_selection.filters.discretization import categorize_dataset
        from mlframe.feature_selection.filters.info_theory import merge_vars, batch_pair_mi_prange
        from mlframe.feature_selection.filters._permutation_null import (
            pooled_pair_permutation_null_joint_mi_floor,
            pairwise_mm_joint_bias,
        )

        df = self._wide_synergy()
        cols = list(df.columns)
        data, _c, nb = categorize_dataset(df=df, method="quantile", n_bins=8, dtype=np.int16)
        yi = cols.index("y")
        cy, fy, _ = merge_vars(factors_data=data, vars_indices=[yi], var_is_nominal=None, factors_nbins=nb, dtype=np.int16)
        feat = [cols.index(c) for c in cols if c != "y"]
        pairs = list(combinations(feat, 2))
        pa = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
        pb = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs))
        mis = batch_pair_mi_prange(data, pa, pb, np.ascontiguousarray(nb), cy, fy)
        floor_mm = pooled_pair_permutation_null_joint_mi_floor(
            factors_data=data,
            nbins=nb,
            pair_a=pa,
            pair_b=pb,
            classes_y=cy,
            freqs_y=fy,
            n_permutations=25,
            quantile=0.95,
            random_seed=42,
            mm_debias=True,
        )
        bias = pairwise_mm_joint_bias(data, pa, pb, nb, int(fy.shape[0]))
        gk = {tuple(sorted((cols.index(a), cols.index(b)))) for a, b in [("x1", "x2"), ("x3", "x4"), ("x5", "x6")]}
        gen_clear, noise_below = [], []
        for k, (a, b) in enumerate(pairs):
            cmp = mis[k] - bias[k]
            key = tuple(sorted((a, b)))
            if key in gk:
                gen_clear.append(cmp >= floor_mm)
            elif cols[a].startswith("noise_") and cols[b].startswith("noise_"):
                noise_below.append(cmp <= floor_mm)
        # Genuine synergy clears the MM-debiased floor; the floor is NOT weakened.
        assert all(gen_clear) and len(gen_clear) == 3, f"genuine synergy pairs did not all clear the MM floor {floor_mm}"
        # The overwhelming majority of noise pairs sit at/below the MM floor (outer guard intact).
        assert float(np.mean(noise_below)) >= 0.95, f"only {np.mean(noise_below):.2%} of noise pairs at/below the MM floor {floor_mm}"

    def test_raw_vs_mm_floor_consistent_scale(self):
        """The MM floor and the per-pair-debiased pair_mi are on the SAME scale, so the
        raw floor minus the max per-pair bias is a sanity lower bound (floor shifts DOWN
        with the values, never relatively tightening)."""
        from mlframe.feature_selection.filters.discretization import categorize_dataset
        from mlframe.feature_selection.filters.info_theory import merge_vars
        from mlframe.feature_selection.filters._permutation_null import (
            pooled_pair_permutation_null_joint_mi_floor,
        )

        df = self._wide_synergy()
        cols = list(df.columns)
        data, _c, nb = categorize_dataset(df=df, method="quantile", n_bins=8, dtype=np.int16)
        yi = cols.index("y")
        cy, fy, _ = merge_vars(factors_data=data, vars_indices=[yi], var_is_nominal=None, factors_nbins=nb, dtype=np.int16)
        feat = [cols.index(c) for c in cols if c != "y"]
        pairs = list(combinations(feat, 2))
        pa = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
        pb = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs))
        kw = dict(factors_data=data, nbins=nb, pair_a=pa, pair_b=pb, classes_y=cy, freqs_y=fy, n_permutations=25, quantile=0.95, random_seed=42)
        floor_raw = pooled_pair_permutation_null_joint_mi_floor(mm_debias=False, **kw)
        floor_mm = pooled_pair_permutation_null_joint_mi_floor(mm_debias=True, **kw)
        # MM shifts the floor DOWN (per-pair bias subtracted) but it stays finite + below raw.
        assert floor_mm <= floor_raw
        assert np.isfinite(floor_mm)


# ---------------------------------------------------------------------------
# biz_value / regression: the DEFAULT is OFF => byte-stable selection.
# ---------------------------------------------------------------------------
class TestMMDebiasDefaultOff:
    """Groups tests covering TestMMDebiasDefaultOff."""
    @pytest.mark.timeout(300)
    def test_default_is_off_and_byte_stable(self):
        """The shipped DEFAULT is ``fe_mm_debias_prevalence=False`` (bench-rejected as a
        default). Fitting with the explicit default vs the explicit False must produce an
        IDENTICAL selection (the flag does not silently change the shipped behaviour)."""
        import inspect
        from mlframe.feature_selection.filters.mrmr import MRMR

        default = inspect.signature(MRMR.__init__).parameters["fe_mm_debias_prevalence"].default
        assert default is False, "MM-debias must ship default-OFF (bench-rejected as default)"

        rng = np.random.default_rng(7)
        n = 2000
        a = rng.random(n) + 0.1
        b = rng.random(n) + 0.1
        c = rng.random(n) + 0.1
        d = rng.random(n) * 2 * np.pi
        y = 0.2 * a**2 / b + np.log(c * 2.0) * np.sin(d / 3.0)
        df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d})
        ys = pd.Series(y, name="y")

        m_default = MRMR(verbose=0, random_seed=1).fit(df, ys)
        m_explicit_off = MRMR(verbose=0, random_seed=1, fe_mm_debias_prevalence=False).fit(df, ys)
        assert list(m_default.get_feature_names_out()) == list(m_explicit_off.get_feature_names_out()), (
            "explicit-default and explicit-False selections differ -- default is not OFF/byte-stable"
        )
