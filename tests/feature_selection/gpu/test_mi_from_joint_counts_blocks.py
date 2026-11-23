"""The device MI finalisation goes through the pairs in blocks and gives the sums of the one-pass formula."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._batch_pair_mi_cuda_kernels import _mi_from_joint_counts_cupy


def _reference(counts: np.ndarray, fx: np.ndarray, freqs_y: np.ndarray, n: int) -> np.ndarray:
    """The one-pass cupy-free formula over the whole accumulator."""
    inv_n = 1.0 / n
    jf = counts.astype(np.float64) * inv_n
    prob_x = fx.astype(np.float64) * inv_n
    denom = prob_x[:, :, None] * freqs_y[None, None, :]
    valid = (counts != 0) & (fx[:, :, None] != 0) & (freqs_y[None, None, :] > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(valid, jf * np.log(jf / denom), 0.0)
    return terms.sum(axis=(1, 2))


@pytest.mark.parametrize("n_pairs", [1, 3, 8, 21])
def test_blocked_finalisation_equals_one_pass(n_pairs):
    """1, 3, 8 and 21 pairs (fewer than, equal to and more than the 8 blocks) give the one-pass MI of every pair."""
    rng = np.random.default_rng(n_pairs)
    n, joint, ny = 5000, 12, 4
    counts = rng.multinomial(n, np.full(joint * ny, 1.0 / (joint * ny)), size=n_pairs).reshape(n_pairs, joint, ny).astype(np.int64)
    fx = counts.sum(axis=2)
    freqs_y = counts[0].sum(axis=0) / n
    got = cp.asnumpy(_mi_from_joint_counts_cupy(cp.asarray(counts), cp.asarray(fx), cp.asarray(freqs_y), n))
    np.testing.assert_allclose(got, _reference(counts, fx, freqs_y, n), rtol=1e-12, atol=1e-14)
