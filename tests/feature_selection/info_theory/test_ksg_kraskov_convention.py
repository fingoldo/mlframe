"""Regression test for the KSG-1 neighbour-count / digamma convention (filters/_ksg.py).

SA13: ``_kraskov1_aggregate`` (the LNC base) used ``psi(n_x)`` while ``n_x`` is counted by ``_count_within_eps`` in the
self-EXCLUDING strict-``<`` convention, the same count the mixed-KSG aggregate pairs with ``psi(n_x + 1)``. That open/closed
ball mismatch systematically biases the estimate upward. The fix aligns the Kraskov-1 site to ``psi(n_x + 1)``.

The test pins both halves of the contract: (1) on an INDEPENDENT (X, Y) pair the fixed estimator centers near 0 and is
markedly less biased than the pre-fix ``psi(n_x)`` convention; (2) on a known-MI Gaussian it lands within tolerance of
the analytic value.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mlframe.feature_selection.filters._ksg import ksg_lnc_mi


def _independent_mi_estimates(n=800, n_seeds=8):
    """Independent mi estimates."""
    out = []
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        out.append(ksg_lnc_mi(x, y, k=5, low_entropy_skip=False, seed=s))
    return np.array(out)


def test_independent_pair_centers_near_zero():
    """On X ⊥ Y the KSG-LNC estimate must center near 0; the aligned psi(n_x+1) convention keeps the mean small."""
    est = _independent_mi_estimates()
    assert est.mean() < 0.025, f"independent-pair MI should be ~0 with the aligned convention, got mean {est.mean():.4f}"


def test_pre_fix_psi_nx_convention_is_more_biased():
    """Demonstrates the bug: the pre-fix ``psi(n_x)`` (no +1) Kraskov-1 base is systematically MORE biased on X ⊥ Y.

    This replicates the pre-fix aggregate inline so the regression is self-contained and FAILS to match the fixed
    estimator's centering. The fixed ``psi(n_x+1)`` mean must be at least ~2x closer to 0.
    """
    from sklearn.neighbors import KDTree
    from mlframe.feature_selection.filters._ksg import (
        _digamma_scalar,
        _count_within_eps,
        _lnc_correction_v2,
    )

    def _ksg_lnc_prefix(x, y, k=5, alpha=0.25, seed=0):
        """Ksg lnc prefix."""
        rng = np.random.default_rng(seed)
        x = np.asarray(x, float).ravel() + 1e-10 * rng.standard_normal(len(x))
        y = np.asarray(y, float).ravel() + 1e-10 * rng.standard_normal(len(y))
        n = x.size
        xy = np.column_stack([x, y])
        tree = KDTree(xy, metric="chebyshev")
        _, idx_k = tree.query(xy, k=k + 1)
        dvx = np.zeros(n)
        dvy = np.zeros(n)
        for i in range(n):
            for j in range(k + 1):
                jj = idx_k[i, j]
                dvx[i] = max(dvx[i], abs(xy[jj, 0] - xy[i, 0]))
                dvy[i] = max(dvy[i], abs(xy[jj, 1] - xy[i, 1]))
        dvx = np.maximum(dvx, 1e-15)
        dvy = np.maximum(dvy, 1e-15)
        nx = _count_within_eps(x, dvx)
        ny = _count_within_eps(y, dvy)
        psi_k = _digamma_scalar(float(k))
        psi_n = _digamma_scalar(float(n))
        # PRE-FIX: psi(n_x) WITHOUT the +1 -> open/closed-ball mismatch vs the self-excluding count.
        s = np.mean([_digamma_scalar(float(nx[i])) + _digamma_scalar(float(ny[i])) for i in range(n)])
        classical = psi_k - 1.0 / k + psi_n - s
        lnc = 0.0
        for i in range(n):
            nb = xy[idx_k[i]] - xy[i]
            lnc += _lnc_correction_v2(nb, math.log(dvx[i]), math.log(dvy[i]), alpha) / n
        return max(0.0, classical + lnc)

    prefix = np.array(
        [_ksg_lnc_prefix(np.random.default_rng(s).standard_normal(800), np.random.default_rng(s + 1000).standard_normal(800), seed=s) for s in range(8)]
    )
    fixed = _independent_mi_estimates()
    assert prefix.mean() > 2.0 * fixed.mean(), (
        f"pre-fix psi(n_x) convention should be markedly more biased on X⊥Y: prefix mean {prefix.mean():.4f} vs fixed {fixed.mean():.4f}"
    )


def test_known_mi_gaussian_within_tolerance():
    """On a bivariate Gaussian with rho=0.6 the true MI is -0.5*ln(1-rho^2); the fixed estimator must be within tolerance."""
    rho = 0.6
    true_mi = -0.5 * math.log(1.0 - rho**2)
    est = []
    for s in range(8):
        rng = np.random.default_rng(100 + s)
        x = rng.standard_normal(800)
        e = rng.standard_normal(800)
        y = rho * x + math.sqrt(1.0 - rho**2) * e
        est.append(ksg_lnc_mi(x, y, k=5, low_entropy_skip=False, seed=s))
    mean_est = float(np.mean(est))
    assert mean_est == pytest.approx(true_mi, abs=0.05), f"Gaussian rho=0.6 MI estimate {mean_est:.4f} should be within 0.05 of true {true_mi:.4f}"
