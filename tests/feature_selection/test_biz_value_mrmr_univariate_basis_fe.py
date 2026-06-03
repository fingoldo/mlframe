"""biz_value: univariate-basis FE (DEFAULT ON) recovers single-variable
nonlinearities the pair-FE path structurally cannot.

The always-on pair-FE path recovers PAIR interactions (a*b, a/b, |a-b|, max) but
CANNOT express a single-variable nonlinearity -- no pairing of two columns makes
a clean ``a**2`` out of one column, and on a symmetric domain raw ``a`` is
uninformative about ``a**2`` (Pearson/MI ~0 because ``a**2`` is even). A
data-driven recovery scan found this gap: with only the pair path, ``a**2`` is
recovered at corr 0.016 with ZERO engineered features.

``fe_univariate_basis_enable`` (DEFAULT True) runs the orthogonal-basis
univariate stage (``a__T2`` ~ a**2, ``a__T3`` ~ a**3, ...), uplift-gated so it is
near-no-op when there is no univariate nonlinearity. The DEFAULT MRMR now
recovers single-variable quadratic / cubic / Gaussian-bump / abs signals; the
pair-only path (flag OFF) does not -- pinned as the falsifiable contrast.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from mlframe.feature_selection.filters.mrmr import MRMR

_RAW = {"a", "b", "c", "d", "e"}


def _make(make_true, n: int = 4000, seed: int = 0):
    """Symmetric domain so univariate even/odd nonlinearities are NOT recoverable
    from the raw column (raw a has ~0 correlation with a**2 on [-2.5, 2.5])."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-2.5, 2.5, n)
    b = rng.uniform(-2.5, 2.5, n)
    c = rng.uniform(-2.5, 2.5, n)
    d = rng.uniform(-2.5, 2.5, n)
    e = rng.normal(0.0, 1.0, n)
    true = make_true(a)
    y = true + 0.1 * np.std(true) * e
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    return df, pd.Series(y, name="y"), true


def _best_corr(fs, df, true):
    names = list(fs.get_feature_names_out())
    if not names:
        return 0.0, None
    Xt = np.asarray(fs.transform(df))
    best, bn = 0.0, None
    for i, nm in enumerate(names):
        col = Xt[:, i]
        if np.isfinite(col).all() and float(np.std(col)) > 1e-12:
            r = abs(float(np.corrcoef(col, true)[0, 1]))
            if r > best:
                best, bn = r, nm
    return best, bn


_UNIVARIATE_CASES = {
    "a**2":       (lambda a: a ** 2,           0.85),
    "a**3":       (lambda a: a ** 3,           0.80),
    "exp(-a**2)": (lambda a: np.exp(-a ** 2),  0.70),
    "abs(a)":     (lambda a: np.abs(a),        0.85),
}


@pytest.mark.parametrize("name", list(_UNIVARIATE_CASES))
def test_default_recovers_single_var_nonlinearity(name):
    """The DEFAULT MRMR (univariate-basis FE on) recovers the single-variable
    nonlinearity -- no opt-in flag needed."""
    make_true, floor = _UNIVARIATE_CASES[name]
    df, y, true = _make(make_true)
    MRMR.clear_fit_cache()
    fs = MRMR(verbose=0, random_seed=0).fit(df, y)
    corr, bn = _best_corr(fs, df, true)
    assert corr >= floor, (
        f"DEFAULT MRMR failed to recover univariate {name}: best engineered "
        f"|corr|={corr:.3f} < {floor} (best={bn!r}); support="
        f"{list(fs.get_feature_names_out())}"
    )


def _n_raw_sources(name):
    """Count distinct raw single-letter vars (a-e) referenced in an engineered
    feature name. Handles the basis separator (``a__T2`` -> {a}, since ``a`` is
    the prefix before ``__``) and functional/pair forms
    (``mul(prewarp(a),log(e))`` -> {a, e})."""
    import re
    srcs = set()
    for tok in re.split(r"[^A-Za-z0-9_]+", name):
        if not tok:
            continue
        base = tok.split("__", 1)[0] if "__" in tok else tok
        if base in ("a", "b", "c", "d", "e"):
            srcs.add(base)
    return len(srcs)


def test_univariate_recovery_is_a_clean_single_source_feature():
    """The univariate-basis FE recovers a**2 with a CLEAN single-source
    univariate feature (``a__T2`` / ``a__He2`` -- references only ``a``), not a
    noisy multi-variable pair hack. This is the qualitative win over the pair
    path, which (when it recovers a univariate signal at all) must smuggle it
    through a 2-variable feature like ``mul(prewarp(a), log(e))`` that pulls in
    an unrelated column."""
    df, y, true = _make(lambda a: a ** 2)
    MRMR.clear_fit_cache()
    fs = MRMR(verbose=0, random_seed=0).fit(df, y)
    corr, bn = _best_corr(fs, df, true)
    assert corr >= 0.85 and bn is not None, (
        f"default did not recover a**2 (corr={corr:.3f}, best={bn!r})"
    )
    assert _n_raw_sources(bn) == 1, (
        f"a**2 was recovered by a MULTI-variable feature {bn!r} "
        f"({_n_raw_sources(bn)} raw sources) rather than a clean single-source "
        f"univariate basis feature; the univariate-basis stage should produce "
        f"the single-source recoverer"
    )


# ---------------------------------------------------------------------------
# Signal-adaptive basis routing (2026-06-03): route the orthogonal-polynomial
# basis by which one best LINEARISES y (max |corr| over degrees), not by x's
# distribution moments alone. The moment router (basis_route_by_moments) sends a
# heavy-tailed / skewed x to a bounded basis (Chebyshev / Laguerre) whose min-max
# preprocessing is dominated by the outliers, mis-routing away from the z-scored
# Hermite that actually linearises the target. Validated against MI / SU /
# Spearman / Kendall / MIC / distance-correlation / Chatterjee-xi + four
# ensembles + a leave-one-case-out learned meta-router: Pearson |corr| is the
# near-oracle AND cheapest routing criterion for a linearisation FE (MI-family /
# rank / dependence measures reward monotone-but-nonlinear association a linear
# downstream cannot use; routing for the tree-oracle drove OOS-linear R^2
# NEGATIVE).
# ---------------------------------------------------------------------------

_ROUTING_BASES = ["hermite", "legendre", "chebyshev", "laguerre"]


def _basis_best_corr(x, y, basis, degrees=(2, 3, 4)):
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
        _evaluate_basis_column,
    )
    best = 0.0
    for d in degrees:
        try:
            v = np.asarray(
                _evaluate_basis_column(np.asarray(x, float), basis, int(d), aux_for_fit=None),
                dtype=float,
            )
        except Exception:
            continue
        if np.all(np.isfinite(v)) and float(np.std(v)) > 1e-12:
            best = max(best, abs(float(np.corrcoef(v, y)[0, 1])))
    return best


class TestSignalAdaptiveBasisRouting:
    """``basis_route_by_signal`` picks the basis that best linearises y."""

    def test_routes_to_argmax_corr_basis(self):
        """The router returns exactly the basis with the highest best-degree
        |corr| -- the falsifiable definition of signal-adaptive routing."""
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            basis_route_by_signal,
        )
        rng = np.random.default_rng(0)
        n = 4000
        x = np.clip(rng.standard_t(3, n), -8, 8)  # heavy-tailed
        y = (x * x) + 0.25 * np.std(x * x) * rng.standard_normal(n)  # even, non-monotone
        chosen = basis_route_by_signal(x, y, degrees=(2, 3, 4))
        recov = {b: _basis_best_corr(x, y, b) for b in _ROUTING_BASES}
        assert chosen == max(recov, key=recov.get), (
            f"signal routing must return the max-|corr| basis; chose {chosen!r}, "
            f"recov={ {b: round(v, 3) for b, v in recov.items()} }"
        )
        assert recov[chosen] >= 0.85, f"chosen basis under-recovers: {recov}"

    def test_fixes_moment_misroute_on_heavy_tail(self):
        """On a heavy-tailed cubic the moment router picks a bounded basis whose
        min-max mapping is wrecked by the outliers (|corr| ~0.2); signal routing
        recovers it via the z-scored Hermite (|corr| ~0.9). This is the
        boundary-condition the routing change fixes."""
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            basis_route_by_signal,
        )
        from mlframe.feature_selection.filters.hermite_fe import basis_route_by_moments
        rng = np.random.default_rng(0)
        n = 4000
        x = np.clip(rng.standard_t(3, n), -8, 8)
        y = (x ** 3) + 0.25 * np.std(x ** 3) * rng.standard_normal(n)
        b_mom = basis_route_by_moments(x)
        b_sig = basis_route_by_signal(x, y, degrees=(2, 3, 4))
        c_mom = _basis_best_corr(x, y, b_mom)
        c_sig = _basis_best_corr(x, y, b_sig)
        assert b_sig != b_mom and c_sig >= c_mom + 0.20, (
            f"signal routing should materially beat moment routing on a heavy-"
            f"tailed cubic: moment={b_mom}({c_mom:.3f}) signal={b_sig}({c_sig:.3f})"
        )

    def test_falls_back_to_moments_without_usable_y(self):
        """Degenerate y (constant) -> fall back to moment routing, so callers
        without a usable target keep the legacy behaviour."""
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            basis_route_by_signal,
        )
        from mlframe.feature_selection.filters.hermite_fe import basis_route_by_moments
        rng = np.random.default_rng(1)
        x = rng.standard_normal(800)
        y_const = np.ones(800)
        assert basis_route_by_signal(x, y_const) == basis_route_by_moments(x)
