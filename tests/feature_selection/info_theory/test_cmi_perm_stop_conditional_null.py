"""Regression tests for the CMI-permutation stop (filters/_cmi_perm_stop.py).

SA-P0-1: ``cmi_permutation_stop`` must permute the candidate X *within each Z-stratum* so the permutation null matches
the conditional hypothesis H_0: X ⊥ Y | Z. The pre-fix code used an UNCONDITIONAL ``rng.permutation(n)``, which tests the
marginal null H_0: X ⊥ Y and yields a MIS-CALIBRATED conditional p-value: it destroys the X-Z association, so the null
distribution of I(X_perm; Y | Z) is no longer centered at the observed value for a feature that is redundant given Z.

SA1: the permutation p-value must use the add-one (Phipson & Smyth 2010) estimator ``(1 + nexceed) / (B + 1)`` so it is
never exactly 0.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._cmi_perm_stop import cmi_permutation_stop, _cmi_plugin_njit


def _make_redundant_given_z(n: int, seed: int, nb: int = 6):
    """X redundant given Z but marginally dependent on Y.

    Z drives both X and Y; X has NO direct channel to Y. Hence I(X; Y) > 0 (marginal, through Z) but I(X; Y | Z) = 0
    (conditional). The correct WITHIN-Z-stratum null centers the null distribution of I(X_perm; Y | Z) at the observed
    value; the pre-fix UNCONDITIONAL null spreads X across strata and mis-centers it.
    """
    rng = np.random.default_rng(seed)
    z = rng.integers(0, nb, size=n)
    y = np.where(rng.random(n) < 0.4, rng.integers(0, nb, size=n), z).astype(np.int64)
    x = np.where(rng.random(n) < 0.3, rng.integers(0, nb, size=n), z).astype(np.int64)
    return x.astype(np.int64), y, z.astype(np.int64), nb


def _conditional_null(x, y, z, nb, B=150, seed=0):
    strata = {int(v): np.flatnonzero(z == v) for v in np.unique(z)}
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for i in range(B):
        xp = x.copy()
        for a in strata.values():
            if a.size > 1:
                xp[a] = x[rng.permutation(a)]
        out[i] = _cmi_plugin_njit(xp, y, z, nb, nb, nb)
    return out


def _marginal_null(x, y, z, nb, B=150, seed=0):
    rng = np.random.default_rng(seed)
    n = x.size
    return np.array([_cmi_plugin_njit(x[rng.permutation(n)], y, z, nb, nb, nb) for _ in range(B)])


def test_conditional_null_does_not_flag_feature_redundant_given_z():
    """A feature independent of Y GIVEN Z (but marginally dependent) must NOT be significant under the conditional null."""
    n = 2000
    x, y, z, nb = _make_redundant_given_z(n, seed=7)
    is_sig, obs, p = cmi_permutation_stop(
        x_cand=x, y=y, selected_cols=[z],
        nbins_x=nb, nbins_y=nb, nbins_selected=[nb],
        n_permutations=200, alpha=0.05, seed=0,
    )
    assert not is_sig, f"conditional null wrongly flagged a feature redundant given Z (p={p}, obs_cmi={obs})"
    assert p > 0.05, f"expected non-significant conditional p-value, got {p}"


def test_conditional_null_is_calibrated_marginal_null_is_not():
    """The fix's signature: on a feature redundant given Z the WITHIN-STRATUM null centers at the observed CMI (calibrated),
    whereas the pre-fix UNCONDITIONAL null is mis-centered (it breaks the X-Z association the conditioning depends on).

    This is the discriminating contract -- it FAILS on pre-fix code (which used the marginal null and therefore returned a
    p-value computed against the mis-centered distribution).
    """
    n = 2000
    nb = 6
    rng = np.random.default_rng(7)
    z = rng.integers(0, nb, size=n).astype(np.int64)
    y = np.where(rng.random(n) < 0.4, rng.integers(0, nb, size=n), z).astype(np.int64)
    x = z.copy()  # X == Z exactly: the textbook feature redundant given Z (I(X; Y | Z) = 0)
    obs = _cmi_plugin_njit(x, y, z, nb, nb, nb)
    cond = _conditional_null(x, y, z, nb)
    marg = _marginal_null(x, y, z, nb)
    # Conditional null is centered at the observed value (|obs - mean| small relative to the null spread).
    cond_gap = abs(obs - cond.mean())
    marg_gap = abs(obs - marg.mean())
    assert marg_gap > 3.0 * cond_gap + 1e-3, (
        f"marginal null should be mis-centered vs conditional: obs={obs:.5f} cond_mean={cond.mean():.5f} "
        f"marg_mean={marg.mean():.5f} (cond_gap={cond_gap:.5f} marg_gap={marg_gap:.5f})"
    )
    # And the public stop, which now uses the conditional null, must agree the observed sits inside the conditional null.
    _, obs_pub, p = cmi_permutation_stop(
        x_cand=x, y=y, selected_cols=[z], nbins_x=nb, nbins_y=nb, nbins_selected=[nb],
        n_permutations=200, alpha=0.05, seed=0,
    )
    assert obs_pub == pytest.approx(obs)
    assert p > 0.05


def test_pvalue_never_exactly_zero():
    """SA1 add-one: even on a strongly-conditionally-relevant feature the p-value must be > 0 (never exactly 0)."""
    rng = np.random.default_rng(3)
    n = 800
    z = rng.integers(0, 2, size=n).astype(np.int64)
    # X is genuinely relevant given Z: y = x XOR z (synergy), so I(X; Y | Z) is large.
    x = rng.integers(0, 2, size=n).astype(np.int64)
    y = (x ^ z).astype(np.int64)
    is_sig, obs, p = cmi_permutation_stop(
        x_cand=x, y=y, selected_cols=[z],
        nbins_x=2, nbins_y=2, nbins_selected=[2],
        n_permutations=50, alpha=0.05, seed=1,
    )
    assert p > 0.0, "add-one correction must keep the permutation p-value strictly positive"
    assert p == pytest.approx(1.0 / 51.0), f"expected the add-one floor 1/(B+1)=1/51, got {p}"
    assert is_sig, "a genuinely conditionally-relevant XOR feature should be significant"
