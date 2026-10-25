"""The batched MI + noise gate over a pair's candidate codes equals the per-candidate ``mi_direct`` loop, value for value.

The FE pair search scores a pair's candidates either in one batched call (shared buffer present) or one ``mi_direct`` call per candidate (recompute fallback
when the buffer does not fit the free-RAM budget). The fallback is only allowed to differ in speed, so the two must agree exactly.
"""

import numpy as np
import pytest

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_dispatch import _dispatch_batch_mi_with_noise_gate
from mlframe.feature_selection.filters.info_theory import batch_mi_with_noise_gate, use_su_normalization
from mlframe.feature_selection.filters.permutation import mi_direct


def _problem(n, k, nbins, ny, seed):
    """Candidate codes (n, k), target codes with a few informative candidates, and the y helpers the scorers take."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, ny, size=n).astype(np.int32)
    codes = rng.integers(0, nbins, size=(n, k)).astype(np.int8)
    for j in range(0, k, 7):  # some candidates carry signal about y, some are noise
        flip = rng.random(n) < 0.35
        codes[:, j] = np.where(flip, codes[:, j], (y * nbins // ny + rng.integers(0, 2, size=n)).clip(0, nbins - 1)).astype(np.int8)
    freqs_y = np.bincount(y, minlength=ny).astype(np.float64) / n
    return codes, y, np.ascontiguousarray(y.copy()), freqs_y


@pytest.mark.parametrize("n,npermutations", [(1500, 20), (4000, 10)])
def test_batched_candidate_mi_equals_the_per_candidate_loop(n, npermutations):
    """Same candidates, same target, same permutation count: identical ``fe_mi`` for every column."""
    nbins, ny, k = 10, 3, 42
    codes, y, y_safe, freqs_y = _problem(n, k, nbins, ny, seed=5)
    per_candidate = np.array(
        [
            mi_direct(
                codes[:, j].reshape(-1, 1),
                x=np.array([0], dtype=np.int64),
                y=None,
                factors_nbins=np.array([nbins], dtype=np.int64),
                classes_y=y,
                classes_y_safe=y_safe.copy(),
                freqs_y=freqs_y,
                min_nonzero_confidence=0.95,
                npermutations=npermutations,
            )[0]
            for j in range(k)
        ]
    )
    batched = np.asarray(
        _dispatch_batch_mi_with_noise_gate(
            disc_2d=np.ascontiguousarray(codes),
            quantization_nbins=nbins,
            classes_y=y,
            classes_y_safe=y_safe.copy(),
            freqs_y=freqs_y,
            npermutations=npermutations,
            min_nonzero_confidence=0.95,
            use_su=use_su_normalization(),
            batch_mi_kernel=batch_mi_with_noise_gate,
        )
    )
    np.testing.assert_array_equal(batched, per_candidate)
