"""TRAINING_FEATURE_HANDLING_TARGETS-1 regression test: ordered_target_encode_batch must support
noise_count_halflife, matching its sibling ordered_target_encode.

The bug (fixed): ordered_target_encode_batch lacked the noise_count_halflife parameter entirely --
noise_std applied a constant relative noise magnitude to every row regardless of how many prior
observations that row's category had accumulated, even though the single-column function already
supports a count-decayed schedule (heavy noise on low-count rows, tapering as running count grows).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.feature_handling.ordered_target_encoder import ordered_target_encode_batch

pytestmark = pytest.mark.fast


def test_batch_accepts_noise_count_halflife_kwarg():
    """ordered_target_encode_batch must accept noise_count_halflife without raising TypeError."""
    rng = np.random.default_rng(0)
    n = 200
    cols = {"c0": rng.integers(0, 20, n), "c1": rng.integers(0, 20, n)}
    y = rng.normal(size=n)
    order = np.arange(n)

    result = ordered_target_encode_batch(cols, y, order=order, smoothing=1.0, noise_std=0.5, noise_count_halflife=3.0, random_state=1)
    assert set(result.keys()) == set(cols.keys())
    for arr in result.values():
        assert arr.shape == (n,)
        assert np.isfinite(arr).all()


def test_batch_noise_count_halflife_none_is_bit_identical_to_omitting_param():
    """noise_count_halflife=None is bit-identical to leaving the kwarg out entirely, per-column."""
    rng = np.random.default_rng(2)
    n = 300
    cols = {"c0": rng.integers(0, 30, n), "c1": rng.integers(0, 30, n)}
    y = rng.normal(size=n)
    order = np.arange(n)

    omitted = ordered_target_encode_batch(cols, y, order=order, smoothing=1.0, noise_std=0.4, random_state=11)
    explicit_none = ordered_target_encode_batch(
        cols, y, order=order, smoothing=1.0, noise_std=0.4, noise_count_halflife=None, random_state=11,
    )
    for name in cols:
        np.testing.assert_array_equal(omitted[name], explicit_none[name])


def test_batch_noise_count_halflife_decays_toward_zero_noise_for_high_count_rows():
    """Effective per-column noise decays to ~0 at high running counts, matching ordered_target_encode's
    single-column schedule -- proves the per-column running_count is used, not a shared/global one."""
    n = 500
    cols = {"only": np.zeros(n, dtype=int)}
    order = np.arange(n)
    y = np.full(n, 5.0)

    enc_a = ordered_target_encode_batch(cols, y, order=order, smoothing=1.0, noise_std=1.0, noise_count_halflife=2.0, random_state=1)["only"]
    enc_b = ordered_target_encode_batch(cols, y, order=order, smoothing=1.0, noise_std=1.0, noise_count_halflife=2.0, random_state=2)["only"]

    tail_a, tail_b = enc_a[-50:], enc_b[-50:]
    np.testing.assert_allclose(tail_a, np.full_like(tail_a, 5.0), atol=1e-6)
    np.testing.assert_allclose(tail_b, np.full_like(tail_b, 5.0), atol=1e-6)

    early_a, early_b = enc_a[1:6], enc_b[1:6]
    assert np.abs(early_a - early_b).max() > 0.05
