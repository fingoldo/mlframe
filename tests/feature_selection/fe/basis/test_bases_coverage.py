"""Coverage tests for mlframe.feature_selection.filters.bases.

Pushes line coverage from ~41% to >=80% by exercising degenerate inputs, numerical-stability branches, kernel boundary
conditions, every coef-size mapping, every canonical-seeds shape, and end-to-end fit -> apply -> eval round-trips for
all four basis families (Fourier / RBF / Sigmoid / Pade).

Does NOT duplicate the public-API smoke + biz_value tests already in test_bases.py.
"""

from __future__ import annotations

# Pre-import astropy so its logger init runs before coverage/pytest patches warnings.showwarning - the package init chain
# of mlframe.feature_selection.filters pulls in astropy via discretization.py, and a late init under coverage.py raises
# astropy.logger.LoggingError. Importing here first sidesteps that ordering.
try:
    import astropy  # noqa: F401
except Exception:
    pass

import math

import numpy as np
import pytest

from mlframe.feature_selection.filters.bases import (
    EXTRA_BASES,
    _fourier_eval_njit,
    _pade_eval_njit,
    _rbf_eval_kernel_njit,
    _sigmoid_eval_kernel_njit,
)

try:
    from tests.conftest import fast_subset
except ImportError:  # pragma: no cover

    def fast_subset(values, **_):
        return list(values)


# In --fast mode, the basis sweep collapses to one canonical family per test.
# ``fourier`` is the smallest, most-stable kernel and is the historical default.
_FAMILIES_FAST = fast_subset(["fourier", "rbf", "sigmoid", "pade"], representative="fourier")
_DEGREES_FAST = fast_subset([1, 2, 3, 5, 9], representative=3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eval(basis: dict, z: np.ndarray, c: np.ndarray, params: dict) -> np.ndarray:
    """Family-agnostic eval: prefer direct ``eval_njit`` over the factory closure used by RBF / Sigmoid."""
    if "eval_njit" in basis:
        return basis["eval_njit"](z, c)
    fn = basis["eval_njit_factory"](params)
    return fn(z, c)


# ---------------------------------------------------------------------------
# Degenerate inputs (constant arrays, single elements, empty coefs)
# ---------------------------------------------------------------------------


class TestDegenerateInputs:
    def test_fourier_fit_constant_array_maps_degenerate_to_zero(self):
        # hi == lo: a constant column is degenerate. Rather than a 1e-12 span floor (which amplifies a single
        # outlier's rounding noise into a high-frequency sin(2*pi*k*z) garbage feature), fit maps it to z=0 and
        # flags degenerate; apply must honour the flag instead of dividing by the 0.0 span.
        b = EXTRA_BASES["fourier"]
        x = np.full(20, 3.14, dtype=np.float64)
        z, params = b["fit"](x)
        assert z.shape == x.shape
        assert params["degenerate"] is True
        assert params["span"] == 0.0
        assert np.all(z == 0.0)
        # Re-apply must honour the degenerate flag and return the same all-zero z (no division by span=0.0).
        z2 = b["apply"](x, params)
        np.testing.assert_allclose(z, z2)
        assert np.all(np.isfinite(z2))

    def test_fourier_fit_single_element(self):
        b = EXTRA_BASES["fourier"]
        x = np.array([2.0], dtype=np.float64)
        z, params = b["fit"](x)
        assert z.shape == (1,)
        assert math.isfinite(params["span"])

    def test_fourier_eval_empty_z(self):
        # n == 0 branch: must return empty without index error.
        out = _fourier_eval_njit(np.zeros(0, dtype=np.float64), np.array([1.0, 0.0]))
        assert out.shape == (0,)

    def test_rbf_fit_constant_array(self):
        # std == 0 path: bandwidth gets the 1e-12 floor; centres collapse to the constant value.
        b = EXTRA_BASES["rbf"]
        x = np.full(50, 7.0, dtype=np.float64)
        _, params = b["fit"](x)
        assert params["bandwidth"] > 0.0
        assert np.all(params["centres"] == 7.0)

    def test_sigmoid_fit_constant_array(self):
        # iqr == 0 path: floor at 1e-12 keeps slope finite (but huge); no NaN / inf.
        b = EXTRA_BASES["sigmoid"]
        x = np.full(50, -5.0, dtype=np.float64)
        _, params = b["fit"](x)
        assert math.isfinite(params["slope"])
        assert params["slope"] > 0.0
        assert np.all(params["thresholds"] == -5.0)

    def test_pade_fit_constant_array(self):
        # std == 0: a constant column is degenerate. An additive 1e-12 std floor would blow a single outlier to
        # z~8 and feed the rational Horner a garbage feature, so fit maps it to z=0 and flags degenerate instead.
        b = EXTRA_BASES["pade"]
        x = np.full(30, 1.5, dtype=np.float64)
        z, params = b["fit"](x)
        assert np.all(np.isfinite(z))
        assert np.all(z == 0.0)
        assert params["degenerate"] is True
        assert params["std"] == 0.0

    def test_pade_apply_degenerate_returns_zeros(self):
        # apply honours the degenerate flag set by fit (std==0.0 always arrives with degenerate=True from fit),
        # returning all-zeros without dividing by the 0.0 std.
        b = EXTRA_BASES["pade"]
        x = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        z = b["apply"](x, dict(mean=1.0, std=0.0, degenerate=True))
        assert np.all(np.isfinite(z))
        assert np.all(z == 0.0)

    def test_rbf_eval_empty_coefs_returns_zeros(self):
        # nc == 0 guard in kernel.
        z = np.linspace(-1, 1, 30, dtype=np.float64)
        centres = np.linspace(-1, 1, 9, dtype=np.float64)
        out = _rbf_eval_kernel_njit(z, np.array([], dtype=np.float64), centres, 0.5)
        np.testing.assert_allclose(out, 0.0)

    def test_sigmoid_eval_empty_coefs_returns_zeros(self):
        # nc == 0 guard in kernel.
        z = np.linspace(-1, 1, 30, dtype=np.float64)
        thr = np.linspace(-1, 1, 9, dtype=np.float64)
        out = _sigmoid_eval_kernel_njit(z, np.array([], dtype=np.float64), thr, 2.0)
        np.testing.assert_allclose(out, 0.0)

    def test_pade_eval_empty_returns_zeros(self):
        # nc < 2 short-circuit.
        z = np.linspace(-1, 1, 20, dtype=np.float64)
        out = _pade_eval_njit(z, np.array([], dtype=np.float64))
        np.testing.assert_allclose(out, 0.0)
        out2 = _pade_eval_njit(z, np.array([1.0]))
        np.testing.assert_allclose(out2, 0.0)


# ---------------------------------------------------------------------------
# Numerical-stability branches in the sigmoid kernel
# ---------------------------------------------------------------------------


class TestSigmoidStability:
    def test_extreme_positive_arg_branch(self):
        # arg >> 0 -> arg >= 0 branch: sigmoid -> 1.0. Use extreme slope * (z - tau).
        z = np.array([10.0], dtype=np.float64)
        thr = np.array([0.0], dtype=np.float64)
        c = np.array([1.0], dtype=np.float64)
        out = _sigmoid_eval_kernel_njit(z, c, thr, slope=100.0)
        # sigma(1000) should saturate to ~1.
        assert math.isfinite(out[0])
        assert abs(out[0] - 1.0) < 1e-9

    def test_extreme_negative_arg_branch(self):
        # arg << 0 -> arg < 0 branch: sigmoid -> 0.0.
        z = np.array([-10.0], dtype=np.float64)
        thr = np.array([0.0], dtype=np.float64)
        c = np.array([1.0], dtype=np.float64)
        out = _sigmoid_eval_kernel_njit(z, c, thr, slope=100.0)
        assert math.isfinite(out[0])
        assert abs(out[0] - 0.0) < 1e-9

    def test_zero_arg_at_threshold(self):
        # arg == 0 -> arg >= 0 branch: sigma(0) = 0.5.
        z = np.array([0.0], dtype=np.float64)
        thr = np.array([0.0], dtype=np.float64)
        c = np.array([1.0], dtype=np.float64)
        out = _sigmoid_eval_kernel_njit(z, c, thr, slope=4.0)
        assert abs(out[0] - 0.5) < 1e-12

    def test_mixed_branches_in_one_call(self):
        # Both arg >= 0 and arg < 0 exercised across a single z vector for one row at a time.
        z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float64)
        thr = np.array([0.0], dtype=np.float64)
        c = np.array([1.0], dtype=np.float64)
        out = _sigmoid_eval_kernel_njit(z, c, thr, slope=2.0)
        # Strictly increasing for positive coef and positive slope.
        assert np.all(np.diff(out) > 0.0)
        # Bounded 0..1.
        assert np.all(out >= 0.0) and np.all(out <= 1.0)


# ---------------------------------------------------------------------------
# Pade denominator clamp + Horner branches
# ---------------------------------------------------------------------------


class TestPadeClamp:
    def test_clamp_triggers_at_pole(self):
        # Denominator = 1 + b_1 * z with b_1 = -1 hits zero at z = 1.0 - sample exactly at that pole.
        # c = [a_0, a_1, b_1] = [0, 1, -1] -> num = z, den = 1 - z. At z = 1: |den| < 1e-3 -> out = 0.
        z = np.array([1.0], dtype=np.float64)
        c = np.array([0.0, 1.0, -1.0], dtype=np.float64)
        out = _pade_eval_njit(z, c)
        assert out[0] == 0.0

    def test_clamp_near_pole_close_but_inside(self):
        # Just inside the |den| < 1e-3 clamp window.
        z = np.array([0.9995], dtype=np.float64)
        c = np.array([0.0, 1.0, -1.0], dtype=np.float64)
        out = _pade_eval_njit(z, c)
        # den = 1 - 0.9995 = 5e-4 < 1e-3 -> clamp.
        assert out[0] == 0.0

    def test_no_clamp_outside_window(self):
        # |den| = 0.5 > 1e-3 -> num / den computed normally.
        z = np.array([0.5], dtype=np.float64)
        c = np.array([0.0, 1.0, -1.0], dtype=np.float64)
        out = _pade_eval_njit(z, c)
        # num = 0.5, den = 0.5 -> 1.0.
        assert abs(out[0] - 1.0) < 1e-12

    def test_degree_two_horner_paths(self):
        # nc = 5, p = 2: exercises the inner Horner loops for both numerator (range(p-1,-1,-1)) and denominator (range(2p-1,p,-1)).
        # c = [a_0, a_1, a_2, b_1, b_2] = [1, 0, 0, 0, 0] -> num = 1, den = 1, out = 1 everywhere.
        z = np.linspace(-2.0, 2.0, 20, dtype=np.float64)
        c = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        out = _pade_eval_njit(z, c)
        np.testing.assert_allclose(out, 1.0)

    def test_higher_degree_finite(self):
        # nc = 7 (deg = 3) - exercise longer Horner loops. Use a stable rational so the clamp does NOT fire.
        z = np.linspace(-1.5, 1.5, 50, dtype=np.float64)
        # num = 1 + 0.1 z + 0.01 z^2 + 0.001 z^3; den = 1 (all b_k = 0).
        c = np.array([1.0, 0.1, 0.01, 0.001, 0.0, 0.0, 0.0], dtype=np.float64)
        out = _pade_eval_njit(z, c)
        assert np.all(np.isfinite(out))
        # Mostly close to the numerator polynomial value.
        expected = 1.0 + 0.1 * z + 0.01 * z**2 + 0.001 * z**3
        np.testing.assert_allclose(out, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# coef_size_func across all degrees 1..9 + saturation for RBF / Sigmoid
# ---------------------------------------------------------------------------


class TestCoefSize:
    def test_fourier_2k(self):
        b = EXTRA_BASES["fourier"]
        for d in range(1, 10):
            assert b["coef_size_func"](d) == 2 * d
        # Degree 0 clamps to max(1, 0) = 1 -> 2.
        assert b["coef_size_func"](0) == 2

    def test_rbf_saturates_at_9(self):
        b = EXTRA_BASES["rbf"]
        # degree + 1 until saturation at 9.
        assert b["coef_size_func"](1) == 2
        assert b["coef_size_func"](2) == 3
        assert b["coef_size_func"](8) == 9
        assert b["coef_size_func"](9) == 9
        assert b["coef_size_func"](50) == 9
        # Degree 0 -> max(1, 1) -> min(1, 9) -> 1 (single centre).
        assert b["coef_size_func"](0) == 1

    def test_sigmoid_saturates_at_9(self):
        b = EXTRA_BASES["sigmoid"]
        for d in range(1, 9):
            assert b["coef_size_func"](d) == d + 1
        assert b["coef_size_func"](9) == 9
        assert b["coef_size_func"](20) == 9
        assert b["coef_size_func"](0) == 1

    def test_pade_two_d_plus_one(self):
        b = EXTRA_BASES["pade"]
        for d in range(1, 6):
            assert b["coef_size_func"](d) == 2 * d + 1
        assert b["coef_size_func"](0) == 3  # max(1, 0) = 1 -> 2 * 1 + 1.


# ---------------------------------------------------------------------------
# canonical_seeds_func: every degree 1..9 must produce arrays sized to coef_size_func(d)
# ---------------------------------------------------------------------------


class TestCanonicalSeedsShape:
    @pytest.mark.parametrize("family", _FAMILIES_FAST)
    @pytest.mark.parametrize("degree", _DEGREES_FAST)
    def test_seeds_match_coef_size(self, family: str, degree: int):
        b = EXTRA_BASES[family]
        K = b["coef_size_func"](degree)
        seeds = b["canonical_seeds_func"](degree)
        assert len(seeds) >= 1
        for s in seeds:
            assert s.shape == (K,)
            assert s.dtype == np.float64
            assert np.all(np.isfinite(s))

    def test_fourier_degree1_minimal_seeds(self):
        # degree=1 -> K=2; exactly two seeds: pure sin (s[0]=1) and pure cos (s[1]=1).
        seeds = EXTRA_BASES["fourier"]["canonical_seeds_func"](1)
        assert len(seeds) == 2
        np.testing.assert_array_equal(seeds[0], np.array([1.0, 0.0]))
        np.testing.assert_array_equal(seeds[1], np.array([0.0, 1.0]))

    def test_rbf_seeds_include_uniform_offset(self):
        # Last seed is the uniform-weight mean centre seed (all 1/K).
        seeds = EXTRA_BASES["rbf"]["canonical_seeds_func"](5)
        K = EXTRA_BASES["rbf"]["coef_size_func"](5)
        assert len(seeds) == K + 1  # K one-hot + 1 uniform offset
        np.testing.assert_allclose(seeds[-1], np.ones(K) / K)

    def test_sigmoid_seeds_include_cumulative_ramp(self):
        # Last seed is the cumulative monotone ramp.
        seeds = EXTRA_BASES["sigmoid"]["canonical_seeds_func"](5)
        K = EXTRA_BASES["sigmoid"]["coef_size_func"](5)
        assert len(seeds) == K + 1
        np.testing.assert_allclose(seeds[-1], np.linspace(0.0, 1.0, K))

    def test_pade_degree1_has_two_seeds_only(self):
        # Pade canonical seeds: pure z (if p>=1) always; pure z^2 only if p>=2; reciprocal always.
        # p = 1 -> 2 seeds: pure z + reciprocal.
        seeds = EXTRA_BASES["pade"]["canonical_seeds_func"](1)
        assert len(seeds) == 2
        np.testing.assert_array_equal(seeds[0], np.array([0.0, 1.0, 0.0]))
        np.testing.assert_array_equal(seeds[1], np.array([1.0, 0.0, 1.0]))

    def test_pade_degree2_includes_pure_zsq(self):
        # p = 2 -> 3 seeds: pure z, pure z^2, reciprocal.
        seeds = EXTRA_BASES["pade"]["canonical_seeds_func"](2)
        assert len(seeds) == 3
        np.testing.assert_array_equal(seeds[1], np.array([0.0, 0.0, 1.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# RBF kernel boundary conditions
# ---------------------------------------------------------------------------


class TestRBFKernel:
    def test_single_centre(self):
        # K = 1: kernel reduces to one Gaussian bump.
        z = np.linspace(-2.0, 2.0, 50, dtype=np.float64)
        centres = np.array([0.0], dtype=np.float64)
        c = np.array([1.0], dtype=np.float64)
        out = _rbf_eval_kernel_njit(z, c, centres, bandwidth=1.0)
        # Symmetric around 0, peak at z == 0.
        peak_idx = int(np.argmax(out))
        assert abs(z[peak_idx]) < 0.1
        # Tail decay positive and finite.
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0.0)

    def test_zero_weights_with_centres(self):
        # c = zeros but length matches centres -> output is zero.
        z = np.linspace(-1, 1, 20, dtype=np.float64)
        centres = np.linspace(-1, 1, 9, dtype=np.float64)
        c = np.zeros(9, dtype=np.float64)
        out = _rbf_eval_kernel_njit(z, c, centres, bandwidth=0.5)
        np.testing.assert_allclose(out, 0.0)

    def test_c_shorter_than_centres(self):
        # nc < K branch: inner loop iterates min(K, nc) times - only first nc centres are weighted.
        z = np.array([0.0], dtype=np.float64)
        centres = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        c = np.array([1.0], dtype=np.float64)  # only weights centre 0
        out = _rbf_eval_kernel_njit(z, c, centres, bandwidth=1.0)
        # Only the centre-0 contribution: exp(0) = 1.
        assert abs(out[0] - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Sigmoid kernel: c shorter than thresholds + branch coverage
# ---------------------------------------------------------------------------


class TestSigmoidKernel:
    def test_c_shorter_than_thresholds(self):
        # min(K, nc) loop short-circuit when nc < K.
        z = np.array([0.0], dtype=np.float64)
        thr = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        c = np.array([1.0], dtype=np.float64)
        out = _sigmoid_eval_kernel_njit(z, c, thr, slope=4.0)
        # arg = 4 * (0 - (-1)) = 4 -> sigma(4) -> ~0.982.
        assert 0.9 < out[0] < 1.0


# ---------------------------------------------------------------------------
# Fourier kernel: explicit K and 2K coefficient layout
# ---------------------------------------------------------------------------


class TestFourierKernel:
    def test_pure_cos_seed_matches_numpy(self):
        # Second canonical seed is pure cos(2 * pi * k=1 * z).
        z = np.linspace(0.0, 1.0, 200, dtype=np.float64)
        c = np.array([0.0, 1.0], dtype=np.float64)  # K=1, [a_1=0, b_1=1]
        out = _fourier_eval_njit(z, c)
        np.testing.assert_allclose(out, np.cos(2 * math.pi * z), atol=1e-12)

    def test_multi_harmonic_superposition(self):
        # K=3, weights pick sin(2*pi*1*z) + cos(2*pi*2*z) + sin(2*pi*3*z) summed.
        z = np.linspace(0.0, 1.0, 200, dtype=np.float64)
        c = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float64)
        out = _fourier_eval_njit(z, c)
        expected = np.sin(2 * math.pi * 1 * z) + np.cos(2 * math.pi * 2 * z) + np.sin(2 * math.pi * 3 * z)
        np.testing.assert_allclose(out, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Roundtrip: apply(x, params) == fit(x)[0] for the same x
# ---------------------------------------------------------------------------


class TestApplyMatchesFit:
    @pytest.mark.fast
    @pytest.mark.parametrize("family", _FAMILIES_FAST)
    def test_apply_matches_fit_z(self, family: str):
        b = EXTRA_BASES[family]
        rng = np.random.default_rng(seed=42)
        x = rng.normal(size=200).astype(np.float64)
        z_fit, params = b["fit"](x)
        z_apply = b["apply"](x, params)
        # For RBF / Sigmoid the fit returns x unchanged (just float64 cast); apply returns ascontiguousarray.
        np.testing.assert_allclose(z_fit, z_apply, atol=1e-12)


# ---------------------------------------------------------------------------
# Composition: fit -> apply -> eval round-trip on a shared 100-row fixture
# ---------------------------------------------------------------------------


class TestComposition:
    @pytest.fixture
    def x100(self) -> np.ndarray:
        return np.linspace(-2.0, 2.0, 100, dtype=np.float64)

    def test_fourier_compose(self, x100):
        b = EXTRA_BASES["fourier"]
        z, params = b["fit"](x100)
        seeds = b["canonical_seeds_func"](degree=2)
        for c in seeds:
            out = b["eval_njit"](z, c)
            assert out.shape == x100.shape
            assert np.all(np.isfinite(out))
        # apply == fit z so eval on apply(x) gives same outputs.
        z2 = b["apply"](x100, params)
        for c in seeds:
            np.testing.assert_allclose(b["eval_njit"](z, c), b["eval_njit"](z2, c), atol=1e-12)

    def test_rbf_compose(self, x100):
        b = EXTRA_BASES["rbf"]
        z, params = b["fit"](x100)
        seeds = b["canonical_seeds_func"](degree=3)
        evalfn = b["eval_njit_factory"](params)
        for c in seeds:
            out = evalfn(z, c)
            assert out.shape == x100.shape
            assert np.all(np.isfinite(out))
        # apply ascontiguousarray must keep the eval result identical.
        z2 = b["apply"](x100, params)
        for c in seeds:
            np.testing.assert_allclose(evalfn(z, c), evalfn(z2, c), atol=1e-12)

    def test_sigmoid_compose(self, x100):
        b = EXTRA_BASES["sigmoid"]
        z, params = b["fit"](x100)
        seeds = b["canonical_seeds_func"](degree=3)
        evalfn = b["eval_njit_factory"](params)
        for c in seeds:
            out = evalfn(z, c)
            assert out.shape == x100.shape
            assert np.all(np.isfinite(out))

    def test_pade_compose(self, x100):
        b = EXTRA_BASES["pade"]
        z, params = b["fit"](x100)
        seeds = b["canonical_seeds_func"](degree=2)
        for c in seeds:
            out = b["eval_njit"](z, c)
            assert out.shape == x100.shape
            # Even clamp-zeroed outputs are finite.
            assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Registry-level introspection (every entry exposes its declared API)
# ---------------------------------------------------------------------------


class TestRegistryContract:
    @pytest.mark.parametrize("family", _FAMILIES_FAST)
    def test_required_keys(self, family: str):
        b = EXTRA_BASES[family]
        # Common keys.
        for k in ("fit", "apply", "coef_size_func", "canonical_seeds_func", "dist_note", "kind"):
            assert k in b, f"{family} missing key {k}"
        assert b["kind"] == "non-polynomial"
        assert isinstance(b["dist_note"], str) and len(b["dist_note"]) > 0
        # Eval path: either direct ``eval_njit`` (Fourier / Pade) or factory closure (RBF / Sigmoid).
        assert ("eval_njit" in b) ^ ("eval_njit_factory" in b)
