"""Coverage-focused tests for ``mlframe.feature_selection.filters.hermite_fe``.

The companion ``test_biz_val_filters_hermite_fe.py`` checks quantitative wins on
synthetics; this file walks the surface of the module so that every basis,
optimizer, dispatch ladder branch, and helper path is exercised at least once.
Tests are deliberately small (n <= 300, n_trials in 10..40 where possible) so
the file completes in well under a minute on CPU.
"""

from __future__ import annotations

import math
import os
import pickle
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

try:
    from tests.conftest import fast_subset
except ImportError:  # pragma: no cover

    def fast_subset(values, **_):
        return list(values)


# --fast collapses the basis sweep to a single representative per axis. ``hermite``
# is the historical default; ``fourier`` is the smallest extra-bases family.
_POLY_BASES_FAST = fast_subset(["hermite", "legendre", "chebyshev", "laguerre"], representative="hermite")
_EXTRA_BASES_FAST = fast_subset(["fourier", "rbf", "sigmoid", "pade"], representative="fourier")
_DEGREES_FAST = fast_subset([1, 2, 3, 4], representative=2)


from mlframe.feature_selection.filters import hermite_fe as hfe
from mlframe.feature_selection.filters.hermite_fe import (
    HermiteResult,
    _atan2,
    _baseline_mi_pair,
    _canonical_seeds,
    _chebval_njit,
    _chebval_njit_parallel,
    _eval_coef_pair,
    _hermeval_njit,
    _hermeval_njit_parallel,
    _l2_normalize_pair,
    _lagval_njit,
    _lagval_njit_parallel,
    _legval_njit,
    _legval_njit_parallel,
    _log_abs_signed,
    _make_dispatch,
    _plugin_mi_classif_batch_njit,
    _plugin_mi_classif_njit,
    _plugin_mi_regression_batch_njit,
    _plugin_mi_regression_njit,
    _quantile_bin_njit,
    _run_cma_search,
    _safe_div,
    _select_diverse_topm,
    basis_route_by_moments,
    detect_pair_symmetry,
    optimise_hermite_pair,
    optimise_pair_multimode,
    polyeval_dispatch,
    _NJIT_FUNCS,
    _NJIT_PAR_FUNCS,
    _POLY_BASES,
    _PAR_THRESHOLD,
    _CUDA_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Tiny shared synthetics. Keep n small (optimise_hermite_pair is O(n*n_trials)).
# ---------------------------------------------------------------------------


def _xor_pair(n=200, seed=42):
    rng = np.random.default_rng(seed)
    x_a = rng.normal(size=n)
    x_b = rng.normal(size=n)
    y = (np.sign(x_a * x_b) > 0).astype(np.int64)
    return x_a, x_b, y


def _additive_pair(n=200, seed=42):
    rng = np.random.default_rng(seed)
    x_a = rng.normal(size=n)
    x_b = rng.normal(size=n)
    y = (x_a + 0.7 * x_b > 0).astype(np.int64)
    return x_a, x_b, y


def _uniform_pair(n=200, seed=7):
    rng = np.random.default_rng(seed)
    x_a = rng.uniform(-1, 1, size=n)
    x_b = rng.uniform(-1, 1, size=n)
    y = (x_a**2 - x_b**2 > 0).astype(np.int64)
    return x_a, x_b, y


def _positive_pair(n=200, seed=11):
    rng = np.random.default_rng(seed)
    x_a = rng.exponential(scale=1.0, size=n)
    x_b = rng.exponential(scale=1.0, size=n)
    y = (x_a > x_b).astype(np.int64)
    return x_a, x_b, y


def _periodic_pair(n=200, seed=21):
    rng = np.random.default_rng(seed)
    x_a = rng.uniform(-1, 1, size=n)
    x_b = rng.normal(size=n)
    y = (np.sin(2 * np.pi * x_a) > 0).astype(np.int64)
    return x_a, x_b, y


def _regression_pair(n=200, seed=33):
    rng = np.random.default_rng(seed)
    x_a = rng.normal(size=n)
    x_b = rng.normal(size=n)
    y = (x_a * x_b + 0.1 * rng.normal(size=n)).astype(np.float64)
    return x_a, x_b, y


# ---------------------------------------------------------------------------
# 1. Low-level njit kernels: every polynomial basis, both single + parallel.
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_quantile_bin_njit_returns_valid_bin_indices():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    bins = _quantile_bin_njit(x, 10)
    assert bins.dtype == np.int32
    assert bins.min() >= 0 and bins.max() < 10
    # equi-frequency: each bin within +-1 of n / n_bins
    counts = np.bincount(bins, minlength=10)
    assert (counts.max() - counts.min()) <= 1


def test_plugin_mi_classif_njit_increases_with_signal():
    rng = np.random.default_rng(0)
    x_noise = rng.normal(size=500)
    y = (rng.normal(size=500) > 0).astype(np.int64)
    x_signal = (y + 0.2 * rng.normal(size=500)).astype(np.float64)
    mi_noise = _plugin_mi_classif_njit(x_noise, y, 20)
    mi_signal = _plugin_mi_classif_njit(x_signal, y, 20)
    assert mi_signal > mi_noise
    assert mi_noise >= 0.0


def test_plugin_mi_regression_njit_increases_with_signal():
    rng = np.random.default_rng(1)
    n = 500
    x = rng.normal(size=n)
    y_signal = (x + 0.1 * rng.normal(size=n)).astype(np.float64)
    y_noise = rng.normal(size=n).astype(np.float64)
    mi_signal = _plugin_mi_regression_njit(x, y_signal, 20)
    mi_noise = _plugin_mi_regression_njit(x, y_noise, 20)
    assert mi_signal > mi_noise


def test_plugin_mi_batch_classif_matches_loop():
    rng = np.random.default_rng(2)
    n = 400
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + 0.2 * rng.normal(size=n) > 0).astype(np.int64)
    batch = _plugin_mi_classif_batch_njit(np.ascontiguousarray(X), y, 20)
    loop = np.array([_plugin_mi_classif_njit(np.ascontiguousarray(X[:, j]), y, 20) for j in range(3)])
    np.testing.assert_allclose(batch, loop, rtol=1e-10, atol=1e-12)


def test_plugin_mi_batch_regression_matches_loop():
    rng = np.random.default_rng(3)
    n = 400
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.1 * rng.normal(size=n)).astype(np.float64)
    batch = _plugin_mi_regression_batch_njit(np.ascontiguousarray(X), y, 20)
    loop = np.array([_plugin_mi_regression_njit(np.ascontiguousarray(X[:, j]), y, 20) for j in range(2)])
    np.testing.assert_allclose(batch, loop, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    "njit_fn,par_fn,reference",
    [
        (_hermeval_njit, _hermeval_njit_parallel, lambda x, c: np.polynomial.hermite_e.hermeval(x, c)),
        (_legval_njit, _legval_njit_parallel, lambda x, c: np.polynomial.legendre.legval(x, c)),
        (_chebval_njit, _chebval_njit_parallel, lambda x, c: np.polynomial.chebyshev.chebval(x, c)),
        (_lagval_njit, _lagval_njit_parallel, lambda x, c: np.polynomial.laguerre.lagval(x, c)),
    ],
)
def test_poly_njit_kernels_match_numpy(njit_fn, par_fn, reference):
    rng = np.random.default_rng(4)
    x = rng.uniform(-0.9, 0.9, size=200).astype(np.float64)
    c = np.array([0.1, -0.3, 0.5, -0.2, 0.4], dtype=np.float64)
    out_njit = njit_fn(np.ascontiguousarray(x), c)
    out_par = par_fn(np.ascontiguousarray(x), c)
    expected = reference(x, c)
    np.testing.assert_allclose(out_njit, expected, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(out_par, expected, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize(
    "njit_fn",
    [
        _hermeval_njit,
        _legval_njit,
        _chebval_njit,
        _lagval_njit,
        _hermeval_njit_parallel,
        _legval_njit_parallel,
        _chebval_njit_parallel,
        _lagval_njit_parallel,
    ],
)
def test_poly_njit_handles_zero_and_single_coef(njit_fn):
    x = np.linspace(-0.5, 0.5, 64).astype(np.float64)
    out_empty = njit_fn(np.ascontiguousarray(x), np.zeros(0, dtype=np.float64))
    assert out_empty.shape == x.shape
    assert np.all(out_empty == 0.0)
    out_const = njit_fn(np.ascontiguousarray(x), np.array([0.7], dtype=np.float64))
    assert np.allclose(out_const, 0.7)


# ---------------------------------------------------------------------------
# 2. polyeval_dispatch + env-var overrides exercise the size-aware ladder.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("basis", _POLY_BASES_FAST)
def test_polyeval_dispatch_matches_njit_for_small_n(basis):
    rng = np.random.default_rng(5)
    x = rng.uniform(-0.9, 0.9, size=300).astype(np.float64)
    c = np.array([0.0, 1.0, -0.5, 0.3], dtype=np.float64)
    # small n < _PAR_THRESHOLD -> njit branch
    out = polyeval_dispatch(basis, x, c)
    ref = _NJIT_FUNCS[basis](np.ascontiguousarray(x), c)
    np.testing.assert_allclose(out, ref, rtol=1e-10)


def test_polyeval_dispatch_force_njit_par_env(monkeypatch):
    monkeypatch.setenv("MLFRAME_POLYEVAL_BACKEND", "njit_par")
    rng = np.random.default_rng(6)
    x = rng.uniform(-0.5, 0.5, size=128).astype(np.float64)
    c = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    out = polyeval_dispatch("chebyshev", x, c)
    ref = _NJIT_PAR_FUNCS["chebyshev"](np.ascontiguousarray(x), c)
    np.testing.assert_allclose(out, ref, rtol=1e-10)


def test_polyeval_dispatch_force_njit_env(monkeypatch):
    monkeypatch.setenv("MLFRAME_POLYEVAL_BACKEND", "njit")
    rng = np.random.default_rng(7)
    x = rng.uniform(-0.5, 0.5, size=300).astype(np.float64)
    c = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    out = polyeval_dispatch("hermite", x, c)
    # He_1(x) = x with c = [0, 1, 0, 0]
    np.testing.assert_allclose(out, x, rtol=1e-10)


def test_polyeval_dispatch_cuda_silent_fallback(monkeypatch):
    """When user forces cuda but cupy isn't installed, dispatch silently falls back."""
    monkeypatch.setenv("MLFRAME_POLYEVAL_BACKEND", "cuda")
    x = np.linspace(-0.5, 0.5, 50).astype(np.float64)
    c = np.array([0.0, 1.0], dtype=np.float64)
    out = polyeval_dispatch("legendre", x, c)
    # Either CUDA worked or fallback returned something finite.
    assert np.all(np.isfinite(out))


def test_polyeval_dispatch_par_threshold_routes_par(monkeypatch):
    """n >= _PAR_THRESHOLD routes via the parallel ladder (default backend, no cuda).

    Direct ``polyeval_dispatch`` call with a large array; we monkeypatch the
    module-level threshold (no reload, which would re-bind classes and break
    pickling/identity for other tests in the same process).
    """
    monkeypatch.setenv("MLFRAME_POLYEVAL_BACKEND", "")
    monkeypatch.setattr(hfe, "_PAR_THRESHOLD", 50)
    monkeypatch.setattr(hfe, "_CUDA_THRESHOLD", 10_000_000)
    x = np.linspace(-0.5, 0.5, 200).astype(np.float64)
    c = np.array([0.0, 1.0, 0.2], dtype=np.float64)
    out = hfe.polyeval_dispatch("legendre", x, c)
    ref = hfe._NJIT_PAR_FUNCS["legendre"](np.ascontiguousarray(x), c)
    np.testing.assert_allclose(out, ref, rtol=1e-10)


def test_make_dispatch_returns_callable_with_name():
    fn = _make_dispatch("chebyshev")
    assert callable(fn)
    assert "chebyshev" in fn.__name__
    out = fn(np.array([0.1, 0.2], dtype=np.float64), np.array([0.0, 1.0], dtype=np.float64))
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# 3. Preprocess helpers / bin funcs / basis-route heuristic.
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_basis_route_by_moments_branches():
    rng = np.random.default_rng(8)
    # Too-small array -> chebyshev fallback
    assert basis_route_by_moments(rng.normal(size=5)) == "chebyshev"
    # Heavy-tailed positive -> laguerre
    skewed = rng.exponential(scale=1.0, size=500)
    assert basis_route_by_moments(skewed) == "laguerre"
    # Bounded uniform -> chebyshev (spread_ratio small)
    bounded = rng.uniform(-0.5, 0.5, size=500)
    assert basis_route_by_moments(bounded) == "chebyshev"
    # Near-Gaussian wide -> hermite
    gaussian = rng.normal(scale=3.0, size=2000)
    pick = basis_route_by_moments(gaussian)
    assert pick in {"hermite", "chebyshev"}


def test_safe_div_handles_zero_denominator():
    a = np.array([1.0, -1.0, 2.0])
    b = np.array([0.0, 0.0, 1.0])
    out = _safe_div(a, b)
    assert np.all(np.isfinite(out))


def test_atan2_and_log_abs_signed_finite():
    a = np.array([1.0, -2.0, 0.0, 3.0])
    b = np.array([-1.0, 2.0, 0.0, 1.0])
    assert np.all(np.isfinite(_atan2(a, b)))
    assert np.all(np.isfinite(_log_abs_signed(a, b)))


# ---------------------------------------------------------------------------
# 4. _canonical_seeds for every supported degree.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("degree", _DEGREES_FAST)
def test_canonical_seeds_shape_and_count(degree):
    seeds = _canonical_seeds("hermite", degree)
    # At least the identity P_1 seed for degree >= 1.
    assert len(seeds) >= 1
    for s in seeds:
        assert s.shape == (degree + 1,)
        assert s.dtype == np.float64


def test_canonical_seeds_degree_zero_empty():
    assert _canonical_seeds("hermite", 0) == []


# ---------------------------------------------------------------------------
# 5. _l2_normalize_pair: normal + degenerate (zero-norm) path.
# ---------------------------------------------------------------------------


def test_l2_normalize_pair_unit_norm():
    a = np.array([1.0, 2.0])
    b = np.array([-1.0, 0.5])
    na, nb = _l2_normalize_pair(a, b, target_norm=1.0)
    total = float(np.sqrt(np.sum(na**2) + np.sum(nb**2)))
    assert abs(total - 1.0) < 1e-9


def test_l2_normalize_pair_zero_passes_through():
    a = np.zeros(3)
    b = np.zeros(3)
    na, nb = _l2_normalize_pair(a, b)
    assert np.allclose(na, 0)
    assert np.allclose(nb, 0)


# ---------------------------------------------------------------------------
# 6. detect_pair_symmetry.
# ---------------------------------------------------------------------------


def test_detect_pair_symmetry_symmetric_vs_asymmetric():
    rng = np.random.default_rng(101)
    n = 300
    x_a = rng.normal(size=n)
    x_b = rng.normal(size=n)
    # symmetric: y = sign(a^2 + b^2 - r^2)
    r2 = np.median(x_a**2 + x_b**2)
    y_sym = ((x_a**2 + x_b**2) > r2).astype(np.int64)
    sym_score = detect_pair_symmetry(x_a, x_b, y_sym, discrete_target=True)
    # asymmetric: y depends only on a
    y_asym = (x_a > 0).astype(np.int64)
    asym_score = detect_pair_symmetry(x_a, x_b, y_asym, discrete_target=True)
    assert 0.0 <= sym_score <= 1.0
    assert 0.0 <= asym_score <= 1.0


# ---------------------------------------------------------------------------
# 7. _baseline_mi_pair: plugin + ksg branches, classif + regression.
# ---------------------------------------------------------------------------


def test_baseline_mi_pair_plugin_classif():
    x_a, x_b, y = _xor_pair(n=300)
    mi = _baseline_mi_pair(x_a, x_b, y, discrete_target=True, mi_estimator="plugin", plugin_n_bins=15)
    assert math.isfinite(mi) and mi >= 0.0


def test_baseline_mi_pair_plugin_regression():
    x_a, x_b, y = _regression_pair(n=300)
    mi = _baseline_mi_pair(x_a, x_b, y, discrete_target=False, mi_estimator="plugin")
    assert math.isfinite(mi) and mi >= 0.0


def test_baseline_mi_pair_ksg_classif():
    x_a, x_b, y = _xor_pair(n=200)
    mi = _baseline_mi_pair(x_a, x_b, y, discrete_target=True, mi_estimator="ksg", n_neighbors=5)
    assert math.isfinite(mi) and mi >= 0.0


def test_baseline_mi_pair_ksg_regression():
    x_a, x_b, y = _regression_pair(n=200)
    mi = _baseline_mi_pair(x_a, x_b, y, discrete_target=False, mi_estimator="ksg", n_neighbors=5)
    assert math.isfinite(mi) and mi >= 0.0


# ---------------------------------------------------------------------------
# 8. _eval_coef_pair direct: covers normal + direction_only + eval_func_b.
# ---------------------------------------------------------------------------


def _make_eval_kwargs(x_a, x_b, y, *, discrete=True):
    basis_info = _POLY_BASES["chebyshev"]
    z_a, _ = basis_info["fit"](x_a)
    z_b, _ = basis_info["fit"](x_b)
    z_a = np.ascontiguousarray(z_a, dtype=np.float64)
    z_b = np.ascontiguousarray(z_b, dtype=np.float64)
    y_njit = np.asarray(y, dtype=np.int64) if discrete else np.asarray(y, dtype=np.float64)
    bf_names = ["add", "mul", "sub"]
    bf_callables = [np.add, np.multiply, np.subtract]
    return dict(
        z_a=z_a,
        z_b=z_b,
        eval_func=_chebval_njit,
        bf_callables=bf_callables,
        bf_names=bf_names,
        y=y,
        y_njit=y_njit,
        mi_estimator="plugin",
        plugin_n_bins=15,
        n_neighbors=5,
        discrete_target=discrete,
        l2_penalty=0.05,
    )


def test_eval_coef_pair_returns_finite_score():
    x_a, x_b, y = _xor_pair(n=300)
    kw = _make_eval_kwargs(x_a, x_b, y)
    coef_a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    coef_b = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    score, raw_mi, bf_idx = _eval_coef_pair(coef_a, coef_b, **kw)
    assert math.isfinite(score)
    assert raw_mi >= 0
    assert 0 <= bf_idx < 3


def test_eval_coef_pair_direction_only():
    x_a, x_b, y = _xor_pair(n=300)
    kw = _make_eval_kwargs(x_a, x_b, y)
    coef_a = np.array([0.0, 5.0, 0.0], dtype=np.float64)
    coef_b = np.array([0.0, 5.0, 0.0], dtype=np.float64)
    score, raw_mi, bf_idx = _eval_coef_pair(
        coef_a,
        coef_b,
        direction_only=True,
        **kw,
    )
    assert math.isfinite(score)
    assert bf_idx >= 0


def test_eval_coef_pair_with_eval_func_b():
    """Cover the eval_func_b parameter introduced for factory bases."""
    x_a, x_b, y = _xor_pair(n=300)
    kw = _make_eval_kwargs(x_a, x_b, y)
    coef_a = np.array([0.0, 1.0], dtype=np.float64)
    coef_b = np.array([0.0, 1.0], dtype=np.float64)
    score, raw_mi, bf_idx = _eval_coef_pair(
        coef_a,
        coef_b,
        eval_func_b=_chebval_njit,
        **kw,
    )
    assert math.isfinite(score)


def test_eval_coef_pair_handles_non_finite_eval():
    """If eval_func returns nan/inf, function returns (-inf, 0, -1)."""
    x_a, x_b, y = _xor_pair(n=200)
    kw = _make_eval_kwargs(x_a, x_b, y)

    def bad_eval(z, c):
        out = np.empty_like(z)
        out[:] = np.nan
        return out

    kw["eval_func"] = bad_eval
    coef_a = np.array([0.0, 1.0], dtype=np.float64)
    coef_b = np.array([0.0, 1.0], dtype=np.float64)
    score, raw_mi, bf_idx = _eval_coef_pair(coef_a, coef_b, **kw)
    assert score == -np.inf
    assert bf_idx == -1


def test_eval_coef_pair_all_bin_funcs_fail():
    """If every bin_func errors out (here divides by zero from a constant array),
    eval_coef_pair returns (-inf, 0, -1)."""
    x_a, x_b, y = _xor_pair(n=200)
    kw = _make_eval_kwargs(x_a, x_b, y)

    def raises_bf(a, b):
        raise RuntimeError("forced failure")

    kw["bf_callables"] = [raises_bf, raises_bf]
    kw["bf_names"] = ["e1", "e2"]
    coef_a = np.array([0.0, 1.0], dtype=np.float64)
    coef_b = np.array([0.0, 1.0], dtype=np.float64)
    score, _, bf_idx = _eval_coef_pair(coef_a, coef_b, **kw)
    assert score == -np.inf
    assert bf_idx == -1


def test_eval_coef_pair_ksg_estimator_path():
    """Cover the KSG branch of _eval_coef_pair."""
    x_a, x_b, y = _xor_pair(n=180)
    kw = _make_eval_kwargs(x_a, x_b, y)
    kw["mi_estimator"] = "ksg"
    kw["y_njit"] = None
    coef_a = np.array([0.0, 1.0], dtype=np.float64)
    coef_b = np.array([0.0, 1.0], dtype=np.float64)
    score, raw_mi, bf_idx = _eval_coef_pair(coef_a, coef_b, **kw)
    assert math.isfinite(score)


def test_eval_coef_pair_ksg_regression_path():
    x_a, x_b, y = _regression_pair(n=180)
    kw = _make_eval_kwargs(x_a, x_b, y, discrete=False)
    kw["mi_estimator"] = "ksg"
    kw["y_njit"] = None
    coef_a = np.array([0.0, 1.0], dtype=np.float64)
    coef_b = np.array([0.0, 1.0], dtype=np.float64)
    score, raw_mi, bf_idx = _eval_coef_pair(coef_a, coef_b, **kw)
    assert math.isfinite(score)


# ---------------------------------------------------------------------------
# 9. _select_diverse_topm: empty + happy + cross-degree padding.
# ---------------------------------------------------------------------------


def test_select_diverse_topm_empty():
    assert _select_diverse_topm([], top_m=3) == []


def test_select_diverse_topm_keeps_diverse_entries():
    rng = np.random.default_rng(99)
    history = []
    for _ in range(8):
        ca = rng.normal(size=3)
        cb = rng.normal(size=3)
        history.append((float(rng.uniform()), 0.1, 0, ca, cb))
    kept = _select_diverse_topm(history, top_m=4, min_l2_distance=0.2)
    assert 1 <= len(kept) <= 4
    # Highest score should always be kept first.
    assert kept[0] == max(history, key=lambda r: r[0])


def test_select_diverse_topm_cross_degree_padding():
    """Coef vectors of differing lengths must be zero-padded for comparison."""
    history = [
        (1.0, 0.2, 0, np.array([1.0, 0.0]), np.array([0.0, 1.0])),
        (0.9, 0.1, 0, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        (0.5, 0.05, 0, np.array([0.0, -1.0]), np.array([-1.0, 0.0])),
    ]
    kept = _select_diverse_topm(history, top_m=3, min_l2_distance=0.1)
    assert len(kept) >= 1


# ---------------------------------------------------------------------------
# 10. _run_cma_search direct (small budget) + warm_start branch.
# ---------------------------------------------------------------------------


def test_run_cma_search_with_warm_seeds():
    pytest.importorskip("cma")
    x_a, x_b, y = _xor_pair(n=200)
    kw = _make_eval_kwargs(x_a, x_b, y)
    seeds = [np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])]  # ca + cb concatenated
    res = _run_cma_search(
        ca_size=3,
        cb_size=3,
        coef_range=(-2.0, 2.0),
        n_trials=15,
        seed=42,
        direction_only=False,
        warm_start_seeds=seeds,
        eval_kwargs=kw,
        popsize=8,
    )
    assert res is not None
    coef_a, coef_b, bf_idx, raw_mi, n_evals = res
    assert coef_a.shape == (3,)
    assert coef_b.shape == (3,)
    assert n_evals > 0
    assert bf_idx >= 0


def test_run_cma_search_without_warm_seeds():
    pytest.importorskip("cma")
    x_a, x_b, y = _xor_pair(n=200)
    kw = _make_eval_kwargs(x_a, x_b, y)
    res = _run_cma_search(
        ca_size=3,
        cb_size=3,
        coef_range=(-2.0, 2.0),
        n_trials=12,
        seed=7,
        direction_only=False,
        warm_start_seeds=[],
        eval_kwargs=kw,
        popsize=6,
    )
    assert res is None or len(res) == 5


def test_run_cma_search_track_history():
    pytest.importorskip("cma")
    x_a, x_b, y = _xor_pair(n=200)
    kw = _make_eval_kwargs(x_a, x_b, y)
    res = _run_cma_search(
        ca_size=3,
        cb_size=3,
        coef_range=(-2.0, 2.0),
        n_trials=10,
        seed=42,
        direction_only=False,
        warm_start_seeds=[np.zeros(6)],
        eval_kwargs=kw,
        popsize=6,
        track_history=True,
    )
    assert res is not None
    coef_a, coef_b, bf_idx, raw_mi, n_evals, history = res
    assert isinstance(history, list)


def test_run_cma_search_direction_only():
    pytest.importorskip("cma")
    x_a, x_b, y = _xor_pair(n=200)
    kw = _make_eval_kwargs(x_a, x_b, y)
    res = _run_cma_search(
        ca_size=2,
        cb_size=2,
        coef_range=(-2.0, 2.0),
        n_trials=10,
        seed=42,
        direction_only=True,
        warm_start_seeds=[np.array([0.0, 1.0, 0.0, 1.0])],
        eval_kwargs=kw,
        popsize=6,
    )
    assert res is not None


# ---------------------------------------------------------------------------
# 11. optimise_hermite_pair: each polynomial basis with tiny budget.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("basis", _POLY_BASES_FAST)
def test_optimise_hermite_pair_each_polynomial_basis(basis):
    x_a, x_b, y = _xor_pair(n=200) if basis == "hermite" else _uniform_pair(n=200)
    if basis == "laguerre":
        x_a, x_b, y = _positive_pair(n=200)
    res = optimise_hermite_pair(
        x_a,
        x_b,
        y,
        discrete_target=True,
        min_degree=2,
        max_degree=2,
        n_trials=12,
        basis=basis,
        optimizer="cma",
        sweep_degrees=False,
        warm_start=True,
        multi_fidelity=False,
        use_trivial_baseline=False,
        baseline_uplift_threshold=0.0,  # accept any improvement
    )
    # Either a valid result or None when search didn't beat baseline -- both legitimate paths.
    assert res is None or isinstance(res, HermiteResult)
    if res is not None:
        assert res.basis == basis
        assert res.coef_a.shape == (3,)


# ---------------------------------------------------------------------------
# 12. Non-polynomial bases (fourier, rbf, sigmoid, pade).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("basis", _EXTRA_BASES_FAST)
def test_optimise_hermite_pair_extra_bases(basis):
    """Each extra basis triggers the eval_njit (fourier/pade) or eval_njit_factory (rbf/sigmoid) path."""
    if basis == "fourier":
        x_a, x_b, y = _periodic_pair(n=200)
    elif basis == "pade":
        x_a, x_b, y = _xor_pair(n=200)
    else:
        x_a, x_b, y = _xor_pair(n=200)
    res = optimise_hermite_pair(
        x_a,
        x_b,
        y,
        discrete_target=True,
        min_degree=2,
        max_degree=2,
        n_trials=10,
        basis=basis,
        optimizer="cma",
        sweep_degrees=False,
        warm_start=True,
        multi_fidelity=False,
        use_trivial_baseline=False,
        baseline_uplift_threshold=0.0,
    )
    assert res is None or isinstance(res, HermiteResult)
    # RBF/Sigmoid are factory-bases where eval_dispatch is built lazily; transform() requires non-None dispatch.
    if res is not None and _POLY_BASES[basis].get("eval_dispatch") is not None:
        out = res.transform(x_a, x_b)
        assert out.shape == x_a.shape


# ---------------------------------------------------------------------------
# 13. Optimizer + estimator dispatch branches.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_optimise_hermite_pair_optuna_path():
    x_a, x_b, y = _xor_pair(n=200)
    res = optimise_hermite_pair(
        x_a,
        x_b,
        y,
        discrete_target=True,
        min_degree=2,
        max_degree=2,
        n_trials=12,
        basis="chebyshev",
        optimizer="optuna",
        sweep_degrees=False,
        warm_start=True,
        multi_fidelity=False,
        use_trivial_baseline=False,
        baseline_uplift_threshold=0.0,
        early_stop_no_improve=8,
    )
    assert res is None or isinstance(res, HermiteResult)


@pytest.mark.slow
def test_optimise_hermite_pair_optuna_no_early_stop():
    """Cover the else-branch (no early_stop) of optimise_hermite_pair."""
    x_a, x_b, y = _xor_pair(n=200)
    res = optimise_hermite_pair(
        x_a,
        x_b,
        y,
        discrete_target=True,
        min_degree=2,
        max_degree=2,
        n_trials=10,
        basis="chebyshev",
        optimizer="optuna",
        sweep_degrees=False,
        warm_start=False,
        multi_fidelity=False,
        use_trivial_baseline=False,
        baseline_uplift_threshold=0.0,
        early_stop_no_improve=0,
    )
    assert res is None or isinstance(res, HermiteResult)


def test_optimise_hermite_pair_cma_with_sweep_degrees():
    x_a, x_b, y = _xor_pair(n=200)
    res = optimise_hermite_pair(
        x_a,
        x_b,
        y,
        discrete_target=True,
        min_degree=2,
        max_degree=4,
        n_trials=10,
        basis="chebyshev",
        optimizer="cma",
        sweep_degrees=True,
        warm_start=True,
        multi_fidelity=False,
        use_trivial_baseline=False,
        baseline_uplift_threshold=0.0,
    )
    if res is not None:
        assert 2 <= res.degree_a <= 4


def test_optimise_hermite_pair_direction_only_arg():
    x_a, x_b, y = _xor_pair(n=200)
    res = optimise_hermite_pair(
        x_a,
        x_b,
        y,
        discrete_target=True,
        min_degree=2,
        max_degree=2,
        n_trials=10,
        basis="chebyshev",
        optimizer="cma",
        sweep_degrees=False,
        warm_start=True,
        direction_only=True,
        multi_fidelity=False,
        use_trivial_baseline=False,
        baseline_uplift_threshold=0.0,
    )
    assert res is None or isinstance(res, HermiteResult)


def test_optimise_hermite_pair_warm_start_false():
    x_a, x_b, y = _xor_pair(n=200)
    res = optimise_hermite_pair(
        x_a,
        x_b,
        y,
        discrete_target=True,
        min_degree=2,
        max_degree=2,
        n_trials=10,
        basis="chebyshev",
        optimizer="cma",
        sweep_degrees=False,
        warm_start=False,
        multi_fidelity=False,
        use_trivial_baseline=False,
        baseline_uplift_threshold=0.0,
    )
    assert res is None or isinstance(res, HermiteResult)


@pytest.mark.slow
def test_optimise_hermite_pair_multi_fidelity_path():
    """Need n >= 4000 to trigger the multi-fidelity subsample branch."""
    rng = np.random.default_rng(13)
    n = 4200
    x_a = rng.normal(size=n)
    x_b = rng.normal(size=n)
    y = (np.sign(x_a * x_b) > 0).astype(np.int64)
    res = optimise_hermite_pair(
        x_a,
        x_b,
        y,
        discrete_target=True,
        min_degree=2,
        max_degree=2,
        n_trials=10,
        basis="chebyshev",
        optimizer="cma",
        sweep_degrees=False,
        warm_start=True,
        multi_fidelity=True,
        use_trivial_baseline=False,
        baseline_uplift_threshold=0.0,
    )
    assert res is None or isinstance(res, HermiteResult)


def test_optimise_hermite_pair_trivial_baseline_branch():
    """use_trivial_baseline=True hits the best_trivial_pair branch."""
    x_a, x_b, y = _xor_pair(n=200)
    res = optimise_hermite_pair(
        x_a,
        x_b,
        y,
        discrete_target=True,
        min_degree=2,
        max_degree=2,
        n_trials=10,
        basis="chebyshev",
        optimizer="cma",
        sweep_degrees=False,
        warm_start=True,
        multi_fidelity=False,
        use_trivial_baseline=True,
        baseline_uplift_threshold=0.0,
    )
    assert res is None or isinstance(res, HermiteResult)


def test_optimise_hermite_pair_regression_target():
    """discrete_target=False exercises the regression code paths in baselines & MI estimator."""
    x_a, x_b, y = _regression_pair(n=200)
    res = optimise_hermite_pair(
        x_a,
        x_b,
        y,
        discrete_target=False,
        min_degree=2,
        max_degree=2,
        n_trials=10,
        basis="chebyshev",
        optimizer="cma",
        sweep_degrees=False,
        warm_start=True,
        multi_fidelity=False,
        use_trivial_baseline=False,
        baseline_uplift_threshold=0.0,
    )
    assert res is None or isinstance(res, HermiteResult)


def test_optimise_hermite_pair_ksg_estimator():
    x_a, x_b, y = _xor_pair(n=180)
    res = optimise_hermite_pair(
        x_a,
        x_b,
        y,
        discrete_target=True,
        min_degree=2,
        max_degree=2,
        n_trials=8,
        basis="chebyshev",
        optimizer="cma",
        mi_estimator="ksg",
        n_neighbors=5,
        sweep_degrees=False,
        warm_start=True,
        multi_fidelity=False,
        use_trivial_baseline=False,
        baseline_uplift_threshold=0.0,
    )
    assert res is None or isinstance(res, HermiteResult)


def test_optimise_hermite_pair_neighbors_auto_pick_small():
    """n < 1000 -> n_neighbors=7 branch."""
    x_a, x_b, y = _xor_pair(n=300)
    _ = optimise_hermite_pair(
        x_a,
        x_b,
        y,
        discrete_target=True,
        n_neighbors=None,  # exercise auto
        min_degree=2,
        max_degree=2,
        n_trials=6,
        basis="chebyshev",
        optimizer="cma",
        sweep_degrees=False,
        warm_start=False,
        multi_fidelity=False,
        use_trivial_baseline=False,
        baseline_uplift_threshold=0.0,
    )


# ---------------------------------------------------------------------------
# 14. Invalid args -> ValueError.
# ---------------------------------------------------------------------------


def test_optimise_hermite_pair_invalid_estimator():
    x_a, x_b, y = _xor_pair(n=100)
    with pytest.raises(ValueError, match="mi_estimator"):
        optimise_hermite_pair(x_a, x_b, y, mi_estimator="bogus")


def test_optimise_hermite_pair_invalid_optimizer():
    x_a, x_b, y = _xor_pair(n=100)
    with pytest.raises(ValueError, match="optimizer"):
        optimise_hermite_pair(x_a, x_b, y, optimizer="bogus")


def test_optimise_hermite_pair_invalid_basis():
    x_a, x_b, y = _xor_pair(n=100)
    with pytest.raises(ValueError, match="basis"):
        optimise_hermite_pair(x_a, x_b, y, basis="quaternion")


def test_optimise_pair_multimode_invalid_basis():
    x_a, x_b, y = _xor_pair(n=100)
    with pytest.raises(ValueError, match="basis"):
        optimise_pair_multimode(x_a, x_b, y, basis="quaternion")


# ---------------------------------------------------------------------------
# 15. optimise_pair_multimode.
# ---------------------------------------------------------------------------


def test_optimise_pair_multimode_basic():
    x_a, x_b, y = _xor_pair(n=200)
    results = optimise_pair_multimode(
        x_a,
        x_b,
        y,
        top_m=3,
        min_l2_distance=0.2,
        min_degree=2,
        max_degree=2,
        n_trials=12,
        basis="chebyshev",
        warm_start=True,
        sweep_degrees=False,
        baseline_uplift_threshold=0.0,
    )
    assert isinstance(results, list)
    for r in results:
        assert isinstance(r, HermiteResult)


def test_optimise_pair_multimode_factory_basis_rbf():
    """Factory basis path (RBF) in multimode: covers the per-feature eval_func_b branch."""
    x_a, x_b, y = _xor_pair(n=200)
    results = optimise_pair_multimode(
        x_a,
        x_b,
        y,
        top_m=2,
        min_l2_distance=0.2,
        min_degree=2,
        max_degree=2,
        n_trials=10,
        basis="rbf",
        warm_start=True,
        sweep_degrees=False,
        baseline_uplift_threshold=0.0,
    )
    assert isinstance(results, list)


def test_optimise_pair_multimode_regression():
    x_a, x_b, y = _regression_pair(n=200)
    results = optimise_pair_multimode(
        x_a,
        x_b,
        y,
        top_m=2,
        min_degree=2,
        max_degree=2,
        n_trials=10,
        basis="chebyshev",
        discrete_target=False,
        sweep_degrees=False,
        baseline_uplift_threshold=0.0,
    )
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# 16. HermiteResult dataclass + transform + pickle roundtrip.
# ---------------------------------------------------------------------------


def test_hermite_result_construct_and_transform():
    rng = np.random.default_rng(55)
    n = 300
    x_a = rng.uniform(-1, 1, size=n)
    x_b = rng.uniform(-1, 1, size=n)
    basis_info = _POLY_BASES["chebyshev"]
    _, prep_a = basis_info["fit"](x_a)
    _, prep_b = basis_info["fit"](x_b)
    res = HermiteResult(
        coef_a=np.array([0.0, 1.0, 0.0]),
        coef_b=np.array([0.0, 1.0, 0.0]),
        bin_func_name="mul",
        bin_func=np.multiply,
        mi=0.5,
        baseline_mi=0.1,
        uplift=5.0,
        degree_a=2,
        degree_b=2,
        basis="chebyshev",
        preprocess_a=prep_a,
        preprocess_b=prep_b,
    )
    transformed = res.transform(x_a, x_b)
    assert transformed.shape == x_a.shape
    assert np.all(np.isfinite(transformed))


def test_hermite_result_pickle_roundtrip():
    res = HermiteResult(
        coef_a=np.array([1.0, 0.0]),
        coef_b=np.array([0.0, 1.0]),
        bin_func_name="add",
        bin_func=np.add,
        mi=0.2,
        baseline_mi=0.1,
        uplift=2.0,
        degree_a=1,
        degree_b=1,
        basis="chebyshev",
        preprocess_a=dict(lo=-1.0, hi=1.0),
        preprocess_b=dict(lo=-1.0, hi=1.0),
    )
    blob = pickle.dumps(res)
    loaded = pickle.loads(blob)
    np.testing.assert_array_equal(loaded.coef_a, res.coef_a)
    np.testing.assert_array_equal(loaded.coef_b, res.coef_b)
    assert loaded.bin_func_name == "add"
    assert loaded.basis == "chebyshev"


# ---------------------------------------------------------------------------
# 17. Preprocess helpers (z-score / minmax / shift).
# ---------------------------------------------------------------------------


def test_preprocess_zscore_then_apply_roundtrip():
    rng = np.random.default_rng(91)
    x = rng.normal(size=200)
    z, params = hfe._preprocess_zscore(x)
    z2 = hfe._apply_zscore(x, params)
    np.testing.assert_allclose(z, z2, rtol=1e-10)


def test_preprocess_minmax_then_apply_roundtrip():
    rng = np.random.default_rng(92)
    x = rng.uniform(-3, 7, size=200)
    z, params = hfe._preprocess_minmax_neg1_1(x)
    z2 = hfe._apply_minmax(x, params)
    np.testing.assert_allclose(z, z2, rtol=1e-10)
    assert z.min() >= -1.0001 and z.max() <= 1.0001


def test_preprocess_shift_then_apply_roundtrip():
    rng = np.random.default_rng(93)
    x = rng.normal(loc=2.0, scale=1.0, size=200)
    z, params = hfe._preprocess_shift_nonneg(x)
    z2 = hfe._apply_shift(x, params)
    np.testing.assert_allclose(z, z2, rtol=1e-10)
    assert z.min() >= 0.0


# ---------------------------------------------------------------------------
# 18. Constant-input edge cases that exercise epsilon-guarded preprocess paths.
# ---------------------------------------------------------------------------


def test_preprocess_constant_input_safe():
    """Constant array must not crash any preprocess (epsilon-guarded)."""
    x = np.ones(100)
    z1, _ = hfe._preprocess_zscore(x)
    z2, _ = hfe._preprocess_minmax_neg1_1(x)
    z3, _ = hfe._preprocess_shift_nonneg(x)
    for arr in (z1, z2, z3):
        assert np.all(np.isfinite(arr))


# ---------------------------------------------------------------------------
# 19. polyeval_dispatch over every basis name (incl. extras).
# ---------------------------------------------------------------------------


def test_poly_bases_registry_contains_all_expected():
    expected = {"hermite", "legendre", "chebyshev", "laguerre", "fourier", "rbf", "sigmoid", "pade"}
    assert expected.issubset(set(_POLY_BASES.keys()))
    for name in ("hermite", "legendre", "chebyshev", "laguerre"):
        info = _POLY_BASES[name]
        assert info["kind"] == "polynomial"
        assert callable(info["fit"]) and callable(info["apply"])
        assert callable(info["eval_dispatch"])


def test_par_threshold_and_cuda_threshold_are_ints():
    assert isinstance(_PAR_THRESHOLD, int) and _PAR_THRESHOLD > 0
    assert isinstance(_CUDA_THRESHOLD, int) and _CUDA_THRESHOLD > 0
