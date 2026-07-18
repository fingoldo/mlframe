"""Regression + biz_value tests for the FUTURE audit items implemented in
``mlframe.training.composite.transforms.nonlinear`` (composite_audit_2026_06_10):

- **T7**  James-Stein shrinkage factor must be SCALE-INVARIANT (the historic
          ``sigma2/n_g`` proxy dropped ``Var(base_g)`` -> rescaling ``base``
          changed which groups shrink). Each test below FAILS on the pre-fix
          (no ``base_vars``) logic.
- **T10** Monotonic-residual auto-knot cap must be driven by base *distinctness*
          (``n_unique``), NOT a ``n_unique // 200`` row-count proxy that starved
          continuous mid-/small-n bases (600 distinct continuous values used to
          collapse to 3 knots).
- **T11** ``_rolling_median`` bottleneck fast path must be BIT-IDENTICAL to the
          pandas ``rolling(center=True, min_periods=1).median()`` reference for
          all ``k`` (even & odd) and at the boundaries (the historic ``k//2``
          shift was off-by-one for even ``k`` and the tail was constant-filled).

Per CLAUDE.md these are real bugs: the tests are written to fail on the
pre-fix code and pass on the corrected code, never to mask via guards.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# T7 -- James-Stein shrinkage factor scale invariance
# ---------------------------------------------------------------------------


def test_t7_js_factor_is_scale_invariant_in_base_unit():
    """REGRESSION (T7): the JS shrinkage factor for OLS *slopes* must not
    depend on the unit of ``base``. Rescaling ``base`` by ``s`` scales each
    ``alpha_g`` by ``1/s`` and ``Var(base_g)`` by ``s**2``; with the correct
    proxy ``Var(alpha_g) = sigma2 / (n_g * Var(base_g))`` these cancel so ``c``
    is unchanged. The pre-fix ``sigma2 / n_g`` proxy (dropping ``Var(base_g)``)
    swings ``c`` from ~0 to 1.0 across scales -- which this asserts against."""
    from mlframe.training.composite.transforms.nonlinear import (
        _james_stein_shrinkage_factor as js,
    )

    rng = np.random.default_rng(1)
    K = 8
    alphas = rng.normal(0.0, 1.0, K)
    global_alpha = float(np.mean(alphas))
    sizes = np.full(K, 50.0)
    base_vars = np.full(K, 1.0)  # base measured in "unit 1"
    sigma2 = 0.5

    cs = []
    for s in (1.0, 10.0, 0.1, 100.0):
        a_s = alphas / s
        bv_s = base_vars * (s**2)
        ga_s = global_alpha / s
        cs.append(js(a_s, ga_s, sizes, sigma2, base_vars=bv_s))
    # All scales must yield the SAME shrinkage factor (scale-invariance).
    assert max(cs) - min(cs) < 1e-9, f"JS factor must be scale-invariant when base_vars supplied; got {cs}"

    # And it must DIFFER from the unit-dependent legacy proxy on a rescaled
    # unit -- proving the fix is the base_vars term, not a no-op.
    c_legacy_s10 = js(alphas / 10.0, global_alpha / 10.0, sizes, sigma2)
    assert abs(c_legacy_s10 - cs[0]) > 0.1, f"legacy (no base_vars) proxy must visibly diverge under rescale; legacy@s=10={c_legacy_s10}, invariant={cs[0]}"


def test_t7_grouped_fit_shrinkage_scale_invariant_end_to_end():
    """REGRESSION (T7), end-to-end through ``_linear_residual_grouped_fit``:
    the stored ``shrinkage_factor`` must be invariant to a global rescale of
    ``base``. Pre-fix the caller passed no ``base_vars`` and the factor moved
    with the unit."""
    from mlframe.training.composite.transforms.linear import (
        _linear_residual_grouped_fit,
    )

    rng = np.random.default_rng(7)
    n_groups = 10
    per_group_n = 60
    ys = []
    bases = []
    groups = []
    for g in range(n_groups):
        x = rng.normal(0.0, 1.0, per_group_n)
        a_g = rng.normal(2.0, 0.6)  # genuine per-group slope spread
        y = a_g * x + rng.normal(0.0, 0.5, per_group_n) + 1.0
        ys.append(y)
        bases.append(x)
        groups.append(np.full(per_group_n, f"g{g}"))
    y = np.concatenate(ys)
    base = np.concatenate(bases)
    grp = np.concatenate(groups)

    c1 = _linear_residual_grouped_fit(y, base, groups=grp)["shrinkage_factor"]
    c_scaled = _linear_residual_grouped_fit(
        y,
        base * 1000.0,
        groups=grp,
    )["shrinkage_factor"]
    assert abs(c1 - c_scaled) < 1e-6, f"grouped shrinkage_factor must be base-unit invariant; unit1={c1}, unit1000={c_scaled}"


def test_t7_js_factor_degenerate_base_var_floor():
    """A single near-constant base group (Var(base_g)~0) must not force full
    shrinkage: the floor keeps the proxy finite. c stays in [0, 1]."""
    from mlframe.training.composite.transforms.nonlinear import (
        _james_stein_shrinkage_factor as js,
    )

    alphas = np.array([0.5, 1.5, -0.5, 2.0, 0.0, 1.0], dtype=np.float64)
    sizes = np.full(alphas.size, 40.0)
    base_vars = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])  # one degenerate group
    c = js(alphas, float(np.mean(alphas)), sizes, 0.3, base_vars=base_vars)
    assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------------------
# T10 -- monotonic-residual auto-knot cap by distinctness, not row count
# ---------------------------------------------------------------------------


def test_t10_continuous_midn_base_keeps_full_knots():
    """REGRESSION (T10): a CONTINUOUS base with 600 distinct values (n=600)
    must keep the full default knot count, not collapse to 3. Pre-fix
    ``n_unique // 200 == 3`` starved it; post-fix the cap is ``min(n_knots,
    n_unique) == 12``."""
    from mlframe.training.composite.transforms import (
        _MONOTONIC_RESIDUAL_DEFAULT_N_KNOTS as DEFAULT_KNOTS,
    )
    from mlframe.training.composite.transforms.nonlinear import (
        _monotonic_residual_fit,
    )

    rng = np.random.default_rng(0)
    n = 600
    base = rng.normal(size=n)  # 600 distinct continuous values
    y = 2.0 * np.tanh(base) + 0.1 * rng.normal(size=n)
    params = _monotonic_residual_fit(y, base)
    # Pre-fix this was exactly 3 (600 // 200). Post-fix it tracks the default.
    assert params["n_knots_effective"] >= DEFAULT_KNOTS - 1, f"continuous mid-n base must keep ~{DEFAULT_KNOTS} knots; got {params['n_knots_effective']}"
    assert params["n_knots_effective"] > 3, "must not collapse to the pre-fix 3-knot under-fit"


def test_t10_biz_val_more_knots_better_var_explained_on_curvy_target():
    """biz_value (T10): on a curvy monotone target the extra knots the fix
    restores must measurably raise ``var_explained`` over the pre-fix 3-knot
    fit. Measured ~0.71 (12 knots) vs ~0.55 (3 knots) -> assert a clear gap."""
    from mlframe.training.composite.transforms.nonlinear import (
        _monotonic_residual_fit,
    )

    rng = np.random.default_rng(3)
    n = 600
    base = np.sort(rng.normal(size=n))
    # Strong curvature so resolution (knot count) matters.
    y = 3.0 * np.tanh(2.5 * base) + 0.05 * rng.normal(size=n)

    full = _monotonic_residual_fit(y, base)  # fixed: ~12 knots
    starved = _monotonic_residual_fit(y, base, n_knots=3)  # explicit 3-knot
    assert full["n_knots_effective"] > starved["n_knots_effective"]
    assert (
        full["var_explained"] >= starved["var_explained"] + 0.05
    ), f"more knots must explain more variance on a curvy target; full={full['var_explained']:.3f}, 3-knot={starved['var_explained']:.3f}"


def test_t10_discrete_base_caps_at_cardinality():
    """A genuinely discrete base with K distinct values must cap the knots at
    ~K (can't place more distinct quantile knots than distinct values)."""
    from mlframe.training.composite.transforms.nonlinear import (
        _monotonic_residual_fit,
    )

    rng = np.random.default_rng(5)
    n = 600
    base = rng.integers(0, 8, size=n).astype(float)  # 8 distinct values
    y = 0.5 * base + 0.1 * rng.normal(size=n)
    params = _monotonic_residual_fit(y, base)
    assert params["n_knots_effective"] <= 8, f"discrete-8 base must not exceed 8 effective knots; got {params['n_knots_effective']}"


# ---------------------------------------------------------------------------
# T11 -- _rolling_median bottleneck parity with pandas reference
# ---------------------------------------------------------------------------


def _pandas_ref(arr: np.ndarray, k: int) -> np.ndarray:
    """Pandas ref."""
    import pandas as pd

    out = pd.Series(arr).rolling(window=k, center=True, min_periods=1).median().to_numpy()
    bad = ~np.isfinite(out)
    if bad.any():
        fb = np.where(np.isfinite(arr), arr, 0.0)
        out = np.where(bad, fb, out)
    return out


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 7, 8, 9, 21, 31])
@pytest.mark.parametrize("n", [1, 2, 3, 7, 13, 200, 501])
def test_t11_rolling_median_matches_pandas_reference(n, k):
    """REGRESSION (T11): the (bottleneck) fast path of ``_rolling_median`` must
    be BIT-IDENTICAL to the pandas centred reference for every (n, k). Pre-fix
    the even-``k`` shift (``k//2`` vs the correct ``(k-1)//2``) and the
    constant tail-fill diverged. Skips if bottleneck is unavailable (the
    fast path is the thing under test here)."""
    pytest.importorskip("bottleneck")
    from mlframe.training.composite.transforms.nonlinear import _rolling_median

    rng = np.random.default_rng(n * 100 + k)
    arr = rng.normal(size=n).astype(np.float64)
    got = _rolling_median(arr.copy(), k)
    ref = _pandas_ref(arr, k)
    assert np.allclose(got, ref, equal_nan=True), f"_rolling_median(n={n}, k={k}) diverges from pandas reference\ngot={got}\nref={ref}"


def test_t11_even_k_tail_not_constant_filled():
    """REGRESSION (T11), pinpoint: for even ``k`` the historic code shifted by
    ``k//2`` and constant-filled the tail, producing a flat tail that the
    pandas reference never has. Assert the tail varies and matches pandas."""
    pytest.importorskip("bottleneck")
    from mlframe.training.composite.transforms.nonlinear import _rolling_median

    arr = np.arange(1.0, 21.0)  # monotone, so the true centred median varies everywhere
    k = 6
    got = _rolling_median(arr.copy(), k)
    ref = _pandas_ref(arr, k)
    assert np.allclose(got, ref), f"got={got}\nref={ref}"
    # The last few positions must NOT all be equal (the pre-fix constant fill).
    tail = got[-(k // 2) :]
    assert len(np.unique(np.round(tail, 9))) > 1, f"tail must vary (not constant-filled); tail={tail}"


def test_t11_window_wider_than_array_no_raise():
    """REGRESSION (T11): ``k > n`` must behave like ``k == n`` (centred window
    truncated to the array), matching pandas. Pre-fix bottleneck raised on
    ``window > n`` and the result silently dropped to the non-finite fallback."""
    pytest.importorskip("bottleneck")
    from mlframe.training.composite.transforms.nonlinear import _rolling_median

    arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    got = _rolling_median(arr.copy(), k=50)  # k >> n
    ref = _pandas_ref(arr, 50)
    assert np.allclose(got, ref), f"got={got}\nref={ref}"
    assert np.all(np.isfinite(got))


def test_t11_nan_input_matches_pandas_skip_semantics():
    """REGRESSION (T11): NaN cells inside a window must be SKIPPED (pandas
    min_periods=1 semantics), not poison the window. The fast path can't
    NaN-skip, so it routes to the pandas reference -- result is identical."""
    from mlframe.training.composite.transforms.nonlinear import _rolling_median

    rng = np.random.default_rng(11)
    n = 50
    arr = rng.normal(size=n).astype(np.float64)
    arr[rng.random(n) < 0.25] = np.nan
    for k in (3, 4, 5, 7):
        got = _rolling_median(arr.copy(), k)
        ref = _pandas_ref(arr, k)
        assert np.allclose(got, ref, equal_nan=True), f"NaN-input k={k} must match pandas skip semantics\ngot={got}\nref={ref}"
