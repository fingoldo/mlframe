"""The noise gate evaluated on the distinct code columns returns, for every column, the value the full evaluation returns."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_gate_dedupe as dd
from mlframe.feature_selection.filters.info_theory._batch_kernels_v3 import batch_mi_with_noise_gate_v3


def _inputs(n: int = 3000, k: int = 600, seed: int = 0):
    """A code matrix where about half of the columns repeat an earlier column byte for byte, a target with real signal in a few columns, and the gate's arguments."""
    rng = np.random.default_rng(seed)
    classes_y = rng.integers(0, 4, n).astype(np.int64)
    base = rng.integers(0, 8, (n, k // 2)).astype(np.int8)
    base[:, :20] = (classes_y[:, None] * 2 + rng.integers(0, 3, (n, 20))).astype(np.int8) % 8  # informative columns
    order = rng.permutation(k)
    pool = np.concatenate([base, base[:, rng.integers(0, base.shape[1], k - base.shape[1])]], axis=1)[:, order]
    freqs_y = np.bincount(classes_y, minlength=4) / n
    kwargs = dict(
        classes_y=classes_y, classes_y_safe=classes_y, freqs_y=freqs_y, npermutations=99, base_seed=np.uint64(0), min_nonzero_confidence=0.99, use_su=False,
        dtype=np.int32, classes_dtype=np.int8,
    )
    return np.ascontiguousarray(pool), np.full(k, 8, dtype=np.int64), kwargs


def test_distinct_columns_find_the_copies_and_map_every_column_to_its_first_copy():
    """``keep`` holds the first column of each group, ``inverse`` maps every column to the row of its group, and the columns of a group are byte-identical."""
    codes, _nb, _kw = _inputs()
    keep, inverse = dd.distinct_columns(codes)
    assert keep.size < codes.shape[1] and np.all(np.diff(keep) > 0)
    checked = 0
    for j in range(codes.shape[1]):
        assert np.array_equal(codes[:, j], codes[:, keep[inverse[j]]])
        assert keep[inverse[j]] <= j
        checked += 1
    assert checked == codes.shape[1], "every column must be mapped to its group's first copy"


def test_gate_on_distinct_columns_equals_the_full_evaluation_bit_for_bit():
    """The gated MI of every column equals the kernel run over all columns, informative and noise columns alike."""
    codes, nbins, kw = _inputs()
    full = np.asarray(batch_mi_with_noise_gate_v3(disc_2d=codes, factors_nbins=nbins, **kw))
    deduped = dd.gate_on_distinct_columns(batch_mi_with_noise_gate_v3, codes, nbins, **kw)
    np.testing.assert_array_equal(deduped, full)
    assert np.any(full > 0) and np.any(full == 0), "the fixture must contain columns the gate keeps and columns it zeroes"


def test_a_fingerprint_collision_cannot_merge_different_columns(monkeypatch):
    """With every fingerprint forced equal the confirmation on the codes keeps the distinct columns apart and the result is unchanged."""
    codes, nbins, kw = _inputs(k=300)
    monkeypatch.setattr(dd, "_column_fingerprints", lambda c, a, b: np.zeros((c.shape[1], 2), dtype=np.uint64))
    full = np.asarray(batch_mi_with_noise_gate_v3(disc_2d=codes, factors_nbins=nbins, **kw))
    np.testing.assert_array_equal(dd.gate_on_distinct_columns(batch_mi_with_noise_gate_v3, codes, nbins, **kw), full)


def test_a_small_batch_and_a_batch_without_copies_use_the_kernel_directly():
    """Under the minimum width, or with no repeated column, the kernel is called on the matrix as given."""
    codes, nbins, kw = _inputs(k=600)
    seen = []

    def kernel(**kwargs):
        """Records the width the kernel received."""
        seen.append(kwargs["disc_2d"].shape[1])
        return np.zeros(kwargs["disc_2d"].shape[1])

    dd.gate_on_distinct_columns(kernel, codes[:, :100], nbins[:100], **kw)
    rng = np.random.default_rng(1)
    unique = rng.integers(0, 8, (500, 300)).astype(np.int8)
    dd.gate_on_distinct_columns(kernel, unique, np.full(300, 8), **kw)
    assert seen == [100, 300]
    out = dd.gate_on_distinct_columns(kernel, codes, nbins, **kw)
    assert out.shape == (600,) and seen[-1] < 600
