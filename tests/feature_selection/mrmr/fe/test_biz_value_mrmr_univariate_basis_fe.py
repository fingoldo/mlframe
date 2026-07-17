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


def _make(make_true, n: int = 4000, seed: int = 0, noise: float = 0.1):
    """Symmetric domain so univariate even/odd nonlinearities are NOT recoverable
    from the raw column (raw a has ~0 correlation with a**2 on [-2.5, 2.5]).

    ``noise`` scales the additive ``e`` term on the target (``y = true + noise*std(true)*e``); pass ``noise=0.0`` for a clean
    target where the single-source univariate basis is the genuinely-optimal recoverer (a multi-source composite then has no
    noise component to capture, so it cannot out-score the clean single-source feature)."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-2.5, 2.5, n)
    b = rng.uniform(-2.5, 2.5, n)
    c = rng.uniform(-2.5, 2.5, n)
    d = rng.uniform(-2.5, 2.5, n)
    e = rng.normal(0.0, 1.0, n)
    true = make_true(a)
    y = true + noise * np.std(true) * e
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
    "a**2": (lambda a: a**2, 0.85),
    "a**3": (lambda a: a**3, 0.80),
    "exp(-a**2)": (lambda a: np.exp(-(a**2)), 0.70),
    "abs(a)": (lambda a: np.abs(a), 0.85),
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
    # Noise-free target so the single-source univariate basis (a__T2 ~ a**2) is the genuinely-optimal recoverer. On the default noisy y the
    # step-2 multi-step composite add(prewarp(e), a__T2) legitimately outscores it by ALSO capturing the +0.1*std*e term -- that composite is
    # a real improvement (composite discovery has its own tests), NOT the "unrelated-column 2-var pair hack" this test guards against. With no
    # noise to capture, the clean single-source univariate feature wins, which is exactly the univariate-basis quality this test asserts.
    df, y, true = _make(lambda a: a**2, noise=0.0)
    MRMR.clear_fit_cache()
    fs = MRMR(verbose=0, random_seed=0).fit(df, y)
    corr, bn = _best_corr(fs, df, true)
    assert corr >= 0.85 and bn is not None, f"default did not recover a**2 (corr={corr:.3f}, best={bn!r})"
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
        except Exception:  # nosec B112 -- best-effort skip of one iteration on a non-fatal error; the test's own assertions are unaffected
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
            f"signal routing must return the max-|corr| basis; chose {chosen!r}, recov={ {b: round(v, 3) for b, v in recov.items()} }"
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
        y = (x**3) + 0.25 * np.std(x**3) * rng.standard_normal(n)
        b_mom = basis_route_by_moments(x)
        b_sig = basis_route_by_signal(x, y, degrees=(2, 3, 4))
        c_mom = _basis_best_corr(x, y, b_mom)
        c_sig = _basis_best_corr(x, y, b_sig)
        assert b_sig != b_mom and c_sig >= c_mom + 0.20, (
            f"signal routing should materially beat moment routing on a heavy-tailed cubic: moment={b_mom}({c_mom:.3f}) signal={b_sig}({c_sig:.3f})"
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


class TestIntAsCatBasisSkip:
    """Regression sensor: the orthogonal-polynomial AND adaptive-Fourier univariate bases must NOT fire on an integer-valued
    low-cardinality categorical group key. ``T_n`` / sin/cos of an arbitrary label code (region 0..9) is spurious periodicity that
    floods the candidate pool and displaces the genuinely-useful grouped aggregates of that key (the fe_auto grouped_agg failure)."""

    def test_is_int_as_cat_axis_classifies_correctly(self):
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            _is_int_as_cat_axis,
        )

        rng = np.random.default_rng(0)
        assert _is_int_as_cat_axis(np.tile(np.arange(10), 60).astype(float)) is True
        assert _is_int_as_cat_axis(rng.standard_normal(600)) is False  # continuous
        assert _is_int_as_cat_axis(np.arange(600).astype(float)) is False  # high-card int (counts)
        assert _is_int_as_cat_axis(np.array([0.0, 1.0] * 200)) is False  # binary card 2 < 3
        assert _is_int_as_cat_axis(np.array([1.0, 2.0, 3.0])) is False  # n < 8

    def test_basis_expansion_skips_int_as_cat_column(self):
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            generate_univariate_basis_features,
        )

        rng = np.random.default_rng(1)
        n = 2000
        region = rng.integers(0, 10, n).astype(np.int64)  # int-as-cat group key
        x = rng.standard_normal(n)  # continuous
        X = pd.DataFrame({"region": region, "x": x})
        out = generate_univariate_basis_features(X, degrees=(2, 3), basis="chebyshev")
        assert not any(c.startswith("region__") for c in out.columns), f"basis expansion fired on the int-as-cat key 'region': {list(out.columns)}"
        assert any(c.startswith("x__") for c in out.columns), f"basis expansion must still fire on the continuous 'x': {list(out.columns)}"

    def test_adaptive_fourier_skips_int_as_cat_column(self):
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            generate_extra_basis_features,
        )

        rng = np.random.default_rng(2)
        n = 2000
        region = rng.integers(0, 10, n).astype(np.int64)
        x = rng.standard_normal(n)
        # y correlates with both so the adaptive detector is tempted to fire.
        y = (region % 2 == 0).astype(float) + 0.3 * x
        X = pd.DataFrame({"region": region, "x": x})
        out, _meta = generate_extra_basis_features(
            X,
            cols=["region", "x"],
            extra_bases=("fourier",),
            y=y,
            fourier_adaptive=True,
            fourier_chirp=True,
        )
        assert not any(str(c).startswith("region__") for c in out.columns), f"adaptive Fourier fired on the int-as-cat key 'region': {list(out.columns)}"
