"""A genuinely CONSTANT combo passed the rejection gate at large scale and was binned as a degenerate column.

The gate computed `var = ss / n - mean * mean` from raw power sums and compared it to `1e-18`. That formula's
cancellation noise floor is roughly `eps * mean^2`, far above the gate. Measured on a column of 4000 identical
values: at scale 1e12 the "variance" comes out as 1.34e8, and at 1e15 as 1.41e14 -- both hugely positive, so the
gate passed and a column with literally one distinct value went on to be quantile-binned into a single bin.

Five copies carried it: the serial njit kernel, the parallel one, both unary-table twins, and the cupy GPU twin.

The audit that found this also predicted the converse -- an informative combo whose true variance sits under the
noise floor being silently rejected as the `-1.0` sentinel. That half does NOT reproduce here, and the reason is
worth recording: `_apply_binary` scrubs every value through `np.float32` before the variance is computed. The
float32 ulp at scale `m` is about `1.2e-7 * m`, while the float64 cancellation floor is `2.2e-16 * m^2 / m^2`
relative -- so any combo that survives the scrub as non-constant differs by at least one ulp and therefore has a
variance roughly 16x above the floor. Measured across scales 1.7e9 / 1e12 / 1e15 at 1, 2, 8 and 64 ulps of
spread, the old formula agreed with the true variance to within 8% every time and never once landed negative.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._usability_njit_pool import (
    _pair_combo_mi_njit,
    _pair_combo_mi_njit_parallel,
    _pair_combo_mi_njit_table,
    _pair_combo_mi_njit_table_parallel,
)

KERNELS = [_pair_combo_mi_njit, _pair_combo_mi_njit_parallel]
N = 4000
NBINS = 8
REJECTED = -1.0


def _y_terms(y_codes):
    """The (h_y, k_y) pair the kernels take alongside the target codes."""
    k_y = int(y_codes.max()) + 1
    counts = np.bincount(y_codes, minlength=k_y).astype(np.float64)
    pr = counts[counts > 0] / counts.sum()
    return float(-(pr * np.log(pr)).sum()), k_y


def _call(kernel, x1, x2, y_codes, *, unary=0, binary=1, nu_tab=None):
    """One combo through `kernel`, returning its MI or the rejection sentinel. `binary=1` is add."""
    h_y, k_y = _y_terms(y_codes)
    qs = np.linspace(0.0, 1.0, NBINS + 1)[1:-1].astype(np.float64)
    args = (
        x1,
        x2,
        y_codes,
        h_y,
        k_y,
        qs,
        np.array([unary], dtype=np.int64),
        np.array([unary], dtype=np.int64),
        np.array([binary], dtype=np.int64),
        float(x1.min()),
        float(x2.min()),
    )
    return float(kernel(*args, nu_tab)[0] if nu_tab is not None else kernel(*args)[0])


@pytest.fixture
def y():
    """A balanced binary target."""
    return np.random.default_rng(0).integers(0, 2, N).astype(np.int64)


class TestAConstantComboIsRejectedAtEveryScale:
    """The half that reproduces, at the scales where it does."""

    @pytest.mark.parametrize("kernel", KERNELS)
    @pytest.mark.parametrize("scale", [1.0, 1.7e9, 1e12, 1e15])
    def test_a_constant_combo_gets_the_sentinel(self, kernel, scale, y):
        """At 1e12 the old formula reported a variance of 1.34e8 for a column with one distinct value."""
        const = np.full(N, scale / 2.0)
        assert _call(kernel, const, const, y) == REJECTED, f"a constant combo at scale {scale:g} passed the rejection gate"

    @pytest.mark.parametrize("kernel", [_pair_combo_mi_njit_table, _pair_combo_mi_njit_table_parallel])
    @pytest.mark.parametrize("scale", [1e12, 1e15])
    def test_the_unary_table_twins_reject_it_too(self, kernel, scale, y):
        """The same gate was copied into both table variants."""
        const = np.full(N, scale / 2.0)
        assert _call(kernel, const, const, y, nu_tab=1) == REJECTED

    def test_the_old_formula_really_did_pass_it(self):
        """The measurement the fix rests on, so the claim above is not taken on trust."""
        v = np.full(N, np.float32(1e12), dtype=np.float64)
        old = float((v**2).sum() / N - (v.sum() / N) ** 2)
        assert old > 1e-18, "the fixture no longer exercises the cancellation error; the tests above would prove nothing"


class TestInformativeCombosAreStillScored:
    """The gate must reject only what is genuinely constant."""

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_an_ordinary_informative_combo_is_scored(self, kernel, y):
        """The common, unit-scale case."""
        rng = np.random.default_rng(1)
        x = np.where(y == 1, 1.0, -1.0) + rng.normal(0, 0.3, N)
        assert _call(kernel, x, x, y) > 0.1

    @pytest.mark.parametrize("kernel", KERNELS)
    @pytest.mark.parametrize("scale", [1.0, 1e6, 1.7e9])
    def test_a_signal_survives_at_every_scale_it_survives_float32_at(self, kernel, scale, y):
        """`_apply_binary` scrubs to float32, so the signal is sized relative to that resolution."""
        rng = np.random.default_rng(2)
        step = max(float(np.spacing(np.float32(scale))) * 64.0, 0.5)
        x = scale / 2.0 + np.where(y == 1, step, -step) + rng.normal(0, step / 4.0, N)
        assert _call(kernel, x, x, y) != REJECTED

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_a_pure_noise_combo_scores_near_zero(self, kernel, y):
        """The gate must not have become a blanket accept."""
        x = np.random.default_rng(3).normal(0, 1, N)
        assert _call(kernel, x, x, y) < 0.05

    def test_the_serial_and_parallel_kernels_agree(self, y):
        """Both were changed; a divergence between them would be a new bug."""
        rng = np.random.default_rng(4)
        x = np.where(y == 1, 1.0, -1.0) + rng.normal(0, 0.3, N)
        assert _call(_pair_combo_mi_njit, x, x, y) == pytest.approx(_call(_pair_combo_mi_njit_parallel, x, x, y))

    def test_the_centred_variance_matches_numpy(self):
        """The replacement formula itself, checked against a reference rather than assumed."""
        v = np.float32(1e12) + np.random.default_rng(5).integers(0, 9, N).astype(np.float32) * np.spacing(np.float32(1e12))
        v = v.astype(np.float64)
        mean = v.sum() / N
        assert ((v - mean) ** 2).sum() / N == pytest.approx(float(np.var(v)), rel=1e-12)
