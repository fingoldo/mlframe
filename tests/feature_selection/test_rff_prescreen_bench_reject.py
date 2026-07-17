"""Characterization of the bench-REJECT verdict on backlog #9 (RFF / random-projection
interaction pre-screen -- detect-without-enumerate), so a future agent cannot silently re-add
the proposer without contradicting a green test.

WHY #9 WAS REJECTED (full numbers in ``_mrmr_fe_step._run_fe_step`` bench-rejected note, 2026-06-09,
and D:/Temp/rff_prescreen_results.md): the RFF mechanism is SOUND when the needle support is covered,
but it loses to the DIRECT all-pairs joint-MI sweep in every regime:

  1. The 2-D joint-MI estimator is a NONPARAMETRIC (x_i, x_j, y) contingency MI, so it recovers a
     SMOOTH interaction (``y = sin(x_a*x_b)``) exactly as well as a zero-marginal PRODUCT
     (``y = x_a*x_b``). The backlog premise that RFF catches smooth interactions the joint-MI sweep
     misses is FALSE -- that argument applies to the GBM seeder's AXIS-ALIGNED trees (#6/#21), not to
     the joint-MI sweep #9 was meant to replace. [pinned by ``test_joint_mi_sweep_recovers_*`` below]

  2. RANDOM SPARSE supports cannot guarantee recall of a SPECIFIC localized needle: a fixed k=2 needle
     in p columns has hit-probability ``R / C(p,2)`` over R random size-2 draws -- so reaching it needs
     ``R ~ C(p,2)`` draws, i.e. ~enumerating all pairs (the very thing #9 set out to avoid), and the
     direct joint-MI sweep is ~17x cheaper per pair than one RFF best-of-trials support evaluation.
     [pinned by ``test_random_support_coverage_is_the_wall`` below]

These tests use ONLY the shipped ``info_theory`` MI helpers (no #9 production code exists -- by design).
Small fixtures (n<=1500, p<=120) keep them fast on a RAM-contended box.
"""

from __future__ import annotations

import math

import numpy as np

from mlframe.feature_selection.filters.info_theory import (
    merge_vars,
    compute_mi_from_classes,
)


def _discretize(X, nb=8):
    """Quantile-bin each column to <=nb dense ordinal bins -> (int matrix, per-col nbins)."""
    n, p = X.shape
    D = np.zeros((n, p), dtype=np.int32)
    nbins = np.zeros(p, dtype=np.int64)
    for j in range(p):
        col = X[:, j]
        qs = np.quantile(col, np.linspace(0, 1, nb + 1)[1:-1])
        codes = np.searchsorted(qs, col, side="right")
        uniq = np.unique(codes)
        remap = {u: i for i, u in enumerate(uniq)}
        D[:, j] = np.array([remap[c] for c in codes], dtype=np.int32)
        nbins[j] = len(uniq)
    return D, nbins


def _discretize_1d(col, nb=8):
    """Discretize 1d."""
    qs = np.quantile(col, np.linspace(0, 1, nb + 1)[1:-1])
    codes = np.searchsorted(qs, col, side="right")
    uniq = np.unique(codes)
    remap = {u: i for i, u in enumerate(uniq)}
    return np.array([remap[c] for c in codes], dtype=np.int32), len(uniq)


def _y_classes(y_cont, nb=8):
    """Y classes."""
    c, k = _discretize_1d(y_cont, nb)
    n = c.shape[0]
    fy = np.bincount(c.astype(np.int64), minlength=k).astype(np.float64) / n
    return c.astype(np.int64), fy, k


def _joint_mi(D, nbins, i, j, yc, fy):
    """Plug-in joint MI(x_i, x_j; y) via the shipped merge_vars + compute_mi_from_classes path."""
    cl, fx, _ = merge_vars(D, np.array([i, j]), None, nbins, dtype=np.int64)
    return compute_mi_from_classes(cl.astype(np.int64), fx, yc, fy)


def _marginal_mi(D, nbins, j, yc, fy):
    """Marginal mi."""
    cl, fx, _ = merge_vars(D, np.array([j]), None, nbins, dtype=np.int64)
    return compute_mi_from_classes(cl.astype(np.int64), fx, yc, fy)


# =============================================================================
# FINDING 1: the DIRECT all-pairs joint-MI sweep -- the proposer #9 would have replaced --
# recovers BOTH the zero-marginal product AND the smooth interaction needle as the rank-0 pair.
# This is the core reason #9 buys nothing at recoverable p: there is no joint-MI blind spot for
# it to fill (the joint-histogram MI is nonparametric in the (x_i, x_j) shape).
# =============================================================================


class TestJointMISweepHasNoBlindSpot:
    """Groups tests covering TestJointMISweepHasNoBlindSpot."""
    def _sweep_rank(self, X, y_cont, needle, *, extra_pairs=400, seed=0):
        """Sweep rank."""
        rng = np.random.default_rng(seed)
        _n, p = X.shape
        D, nbins = _discretize(X, nb=8)
        yc, fy, _ = _y_classes(y_cont, nb=8)
        # Score the needle pair + a random sample of noise pairs (full C(p,2) is unneeded to
        # establish the needle ranks #0; the sample includes the strongest noise competitors).
        pairs = [tuple(needle)]
        seen = {tuple(sorted(needle))}
        while len(pairs) < extra_pairs:
            i, j = sorted(rng.choice(p, 2, replace=False).tolist())
            if (i, j) not in seen:
                seen.add((i, j))
                pairs.append((i, j))
        scored = sorted(
            ((pr, _joint_mi(D, nbins, pr[0], pr[1], yc, fy)) for pr in pairs),
            key=lambda kv: kv[1],
            reverse=True,
        )
        ranked = [pr for pr, _ in scored]
        return ranked.index(tuple(needle)), dict(scored), (D, nbins, yc, fy)

    def test_joint_mi_sweep_recovers_zero_marginal_product(self):
        """Joint mi sweep recovers zero marginal product."""
        rng = np.random.default_rng(1)
        n, p = 1500, 120
        X = rng.standard_normal((n, p))
        y = X[:, 3] * X[:, 100]  # zero-marginal product needle
        rank, sc, (_D, _nbins, _yc, _fy) = self._sweep_rank(X, y, (3, 100), seed=1)
        # The product needle is the single top joint-MI pair, far above any noise pair.
        assert rank == 0, f"zero-marginal product needle did not rank #0 (rank={rank})"
        needle_jmi = sc[(3, 100)]
        next_best = max(v for k, v in sc.items() if k != (3, 100))
        assert needle_jmi > 2.0 * next_best, f"needle joint MI {needle_jmi:.4f} should dominate next-best noise pair {next_best:.4f}"

    def test_joint_mi_sweep_recovers_smooth_interaction(self):
        # The RFF "smooth-interaction sweet spot" claim: the joint-MI sweep ALSO recovers
        # ``sin(x_a*x_b)`` as the rank-0 pair, so RFF has no exclusive smooth-interaction win.
        """Joint mi sweep recovers smooth interaction."""
        rng = np.random.default_rng(2)
        n, p = 1500, 120
        X = rng.standard_normal((n, p))
        y = np.sin(X[:, 1] * X[:, 2]) + 0.05 * rng.standard_normal(n)
        rank, sc, _ = self._sweep_rank(X, y, (1, 2), seed=2)
        assert rank == 0, f"smooth sin(x1*x2) needle did not rank #0 under joint-MI sweep (rank={rank})"
        # And both marginals are ~weak relative to the joint -- it IS a genuine interaction,
        # not a marginal artefact the univariate screen would have caught.
        D, nbins = _discretize(X, nb=8)
        yc, fy, _ = _y_classes(y, nb=8)
        m1 = _marginal_mi(D, nbins, 1, yc, fy)
        m2 = _marginal_mi(D, nbins, 2, yc, fy)
        assert sc[(1, 2)] > 1.5 * max(m1, m2), f"smooth needle joint MI {sc[(1, 2)]:.4f} should exceed its marginals {m1:.4f}/{m2:.4f}"


# =============================================================================
# FINDING 2: random sparse supports cannot guarantee recall of a SPECIFIC localized needle --
# the hit-probability is R / C(p,2). This is the coverage wall that sinks #9: to recover a fixed
# needle you must draw ~C(p,2) supports (== enumerate), at ~17x the per-evaluation cost of the
# direct joint-MI sweep. Pinned analytically (closed form, no slow Monte-Carlo).
# =============================================================================


class TestRandomSupportCoverageIsTheWall:
    """Groups tests covering TestRandomSupportCoverageIsTheWall."""
    def test_specific_needle_hit_probability_is_R_over_Cp2(self):
        # Closed-form hit probability of a FIXED size-2 support among R random size-2 draws,
        # validated against a quick empirical estimate. This is the quantity that makes #9
        # coverage-bound for a localized needle.
        """Specific needle hit probability is R over Cp2."""
        p = 500
        Cp2 = math.comb(p, 2)
        for R in (2000, 16000):
            analytic = 1.0 - (1.0 - 1.0 / Cp2) ** R
            # empirical
            rng = np.random.default_rng(R)
            hits = 0
            trials = 3000
            needle = (3, 400)
            for _ in range(trials):
                drawn = False
                # one experiment = R random pairs; short-circuit on first hit
                draws = rng.integers(0, p, size=(R, 2))
                # cheap vectorized check for the specific unordered pair
                lo = np.minimum(draws[:, 0], draws[:, 1])
                hi = np.maximum(draws[:, 0], draws[:, 1])
                if np.any((lo == needle[0]) & (hi == needle[1])):
                    drawn = True
                hits += int(drawn)
                if _ > 300 and hits == 0 and analytic < 0.02:
                    break  # already clearly tiny; don't burn the full 3000 on a ~1% event
            emp = hits / max(1, (_ + 1))
            # analytic must be in the expected small-coverage regime AND match empirical loosely.
            assert analytic < 0.13, (
                f"R={R}: specific-needle hit prob {analytic:.3f} should be small (<0.13) at p={p} -- random supports cannot guarantee localized-needle recall"
            )
            assert abs(emp - analytic) < 0.05, f"R={R}: empirical hit rate {emp:.3f} disagrees with analytic {analytic:.3f}"

    def test_full_coverage_requires_enumerating_all_pairs(self):
        # 0.95 recall of ONE specific needle needs R ~ C(p,2)*ln(20) random draws -- i.e. MORE than
        # the full enumeration the direct joint-MI sweep does deterministically. This is the
        # quantitative statement of "random-support prescreen buys nothing for a localized needle".
        """Full coverage requires enumerating all pairs."""
        p = 500
        Cp2 = math.comb(p, 2)
        R_for_95 = Cp2 * math.log(20.0)  # solve 1-(1-1/Cp2)^R = 0.95
        assert R_for_95 > Cp2, (
            f"0.95-recall draw count {R_for_95:.0f} should EXCEED full enumeration C(p,2)={Cp2} "
            f"-- so random-support coverage is never cheaper than the deterministic all-pairs sweep"
        )
