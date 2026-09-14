"""The FE collinearity-dedup correlation must survive a large-offset column (mrmr_audit_2026-09-14 NUM-1).

All three dedup backends compute ``var = Sxx - Sx*Sx/n`` from raw power sums (the k=2 form of the
cancellation bug class this repo has hit repeatedly). On a price-like column -- offset far larger than
spread -- the variance degrades to numerical garbage, trips this file's own ``var <= 1e-24`` guard, and the
entry becomes ``nan``. ``nan`` means "not a duplicate" to the caller, so genuinely collinear engineered
columns survive dedup. The dispatcher now row-centres first (Pearson r is translation-invariant, so the
shift is exact); these tests pin that.
"""

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_dedup import (
    _pairwise_complete_abs_corr,
    _pc_corr_numpy,
    _row_center_finite,
)


def _collinear_pair(offset: float, spread: float, n: int = 4000, seed: int = 0):
    """An exactly collinear (|r| == 1) row pair whose values sit at ``offset`` with tiny ``spread``."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, spread, n) + offset
    return np.vstack([base]), np.vstack([base * 2.0 + 1.0])


@pytest.mark.parametrize("offset,spread", [(8.5e3, 0.05), (1e6, 1e-2), (1e8, 1e-3), (1e10, 1e-2)])
def test_exactly_collinear_columns_are_still_detected_under_a_large_offset(offset, spread):
    """A perfectly collinear pair must read as |r| == 1 regardless of the columns' offset."""
    Q, R = _collinear_pair(offset, spread)
    got = _pairwise_complete_abs_corr(Q, R)[0, 0]
    assert np.isfinite(got), f"offset={offset:g} produced {got!r} -- a nan here means 'not a duplicate'"
    assert got == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("seed", range(4))
def test_the_uncentred_power_sum_form_is_what_actually_breaks(seed):
    """Teeth for the fix: the raw backend (bypassing the dispatcher's centring) really does fail here.

    Without this the parametrised test above could pass for a reason unrelated to the centring. The regime
    is chosen so the cancellation is guaranteed by the arithmetic rather than by a lucky draw: at offset
    1e12 / spread 1e-4 the variance signal is ~1e-32 of ``Sxx``'s magnitude, far past float64's ~1e-16
    relative precision, so every seed degrades (verified 8/8 while writing this).
    """
    Q, R = _collinear_pair(1e12, 1e-4, seed=seed)
    raw = _pc_corr_numpy(Q, R)[0, 0]
    assert np.isnan(raw) or raw < 0.92 or raw > 1.0, f"expected the uncentred form to degrade, got {raw!r}"
    centred = _pc_corr_numpy(_row_center_finite(Q), _row_center_finite(R))[0, 0]
    assert centred == pytest.approx(1.0, abs=1e-6)


def test_a_correlation_above_one_is_never_returned():
    """|Pearson r| > 1 is mathematically impossible; the uncentred form returned 2.12 at this offset."""
    Q, R = _collinear_pair(1e6, 1e-2)
    got = _pairwise_complete_abs_corr(Q, R)[0, 0]
    assert got <= 1.0 + 1e-9, f"got {got!r}, which cannot be a correlation coefficient"


def test_centring_leaves_an_ordinary_uncentred_pair_unchanged():
    """No-offset data must be unaffected -- the shift is exact, not an approximation."""
    rng = np.random.default_rng(7)
    a = rng.normal(0.0, 1.0, 3000)
    Q, R = np.vstack([a]), np.vstack([a * 3.0 - 7.0])
    assert _pairwise_complete_abs_corr(Q, R)[0, 0] == pytest.approx(1.0, abs=1e-9)


def test_row_centring_preserves_nans_and_dtype():
    """All-NaN and partial-NaN rows keep their mask; the caller's dtype passthrough is unchanged."""
    A = np.array([[1.0, 2.0, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    out = _row_center_finite(A)
    assert out.dtype == np.float32
    assert np.isnan(out[0, 2]) and np.all(np.isnan(out[1]))
    assert out[0, 0] == pytest.approx(-0.5) and out[0, 1] == pytest.approx(0.5)


def test_an_uncorrelated_large_offset_pair_is_not_promoted_to_a_duplicate():
    """Negative control: centring must not manufacture correlation where there is none."""
    rng = np.random.default_rng(11)
    n = 4000
    Q = np.vstack([rng.normal(0.0, 1e-3, n) + 1e8])
    R = np.vstack([rng.normal(0.0, 1e-3, n) + 1e8])
    got = _pairwise_complete_abs_corr(Q, R)[0, 0]
    assert abs(got) < 0.2, f"independent columns read as |r|={got!r}"
