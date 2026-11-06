"""The family gate reads the observed MI the noise gate already computed for the same resident chunk, and the numbers are the ones it would compute itself."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._batch_mi_noise_gate_bits import bits_gate_mi, labels_key, recall_observed
from mlframe.feature_selection.filters._family_maxt_gate_device import _FamilyOnDevice, _observed_from_noise_gate
from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_chunk_codes import ChunkCodes, LazyCodeColumn
from mlframe.feature_selection.filters._lazy_host_codes import LazyHostCodes

N, K, K_Y, NBINS, PERMS = 3000, 40, 3, 8, 20


def _setup(seed: int = 0):
    """Labels, codes (one column carries the target), their device copies, the label-permutation matrix (row 0 the original labels) and the bin counts."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K_Y, N).astype(np.int64)
    codes = rng.integers(0, NBINS, size=(N, K)).astype(np.int8)
    codes[:, 0] = (y * 2 + rng.integers(0, 2, N)).astype(np.int8)  # one column that carries the target
    d_codes = cp.asarray(np.ascontiguousarray(codes))
    perms = np.stack([y] + [rng.permutation(y) for _ in range(PERMS)]).astype(np.int32)
    return y, codes, d_codes, cp.asarray(perms), np.full(K, NBINS, dtype=np.int64)


def _chunk(d_codes):
    """A chunk of pair-search codes over the resident array, as the pair search holds it."""
    host = np.empty(d_codes.shape, dtype=np.int8)
    return ChunkCodes(host, d_codes, [False])


def test_the_family_gate_gets_the_noise_gates_observed_mi_and_it_equals_its_own():
    """After the gated call the family gate's observed MI of the chunk's columns is the recalled vector, bit for bit what ``_FamilyOnDevice.observed`` computes."""
    y, _codes, d_codes, d_y, nbins = _setup()
    chunk = _chunk(d_codes)
    key = labels_key(y)
    gated = bits_gate_mi(d_codes, nbins, d_y, K_Y, (0.99,), key)
    assert gated is not None
    cols = [LazyCodeColumn(chunk, j) for j in (0, 3, 7, 11, 39)]
    got = _observed_from_noise_gate(cols, y)
    assert got is not None
    fam = _FamilyOnDevice(cp, [c.dev for c in cols], y, K_Y)
    assert np.array_equal(got, fam.observed())


def test_the_gated_vector_is_unchanged_by_keeping_the_ungated_one():
    """The host rebuilds the gated vector from the ungated one and the mask: identical to the call that keeps nothing."""
    y, _codes, d_codes, d_y, nbins = _setup(1)
    plain = bits_gate_mi(d_codes, nbins, d_y, K_Y, (0.99,))
    keeping = bits_gate_mi(d_codes, nbins, d_y, K_Y, (0.99,), labels_key(y))
    assert np.array_equal(plain, keeping)
    assert (plain == 0).any() and (plain > 0).any()  # the fixture has both zeroed and surviving columns


def test_other_labels_or_another_array_do_not_match():
    """The recalled vector belongs to one array and one label vector; anything else computes its own."""
    y, _codes, d_codes, d_y, nbins = _setup(2)
    bits_gate_mi(d_codes, nbins, d_y, K_Y, (0.99,), labels_key(y))
    other_y = np.roll(y, 1)
    assert recall_observed(d_codes, labels_key(y)) is not None
    assert recall_observed(d_codes, labels_key(other_y)) is None
    assert recall_observed(d_codes.copy(), labels_key(y)) is None
    chunk = _chunk(d_codes)
    assert _observed_from_noise_gate([LazyCodeColumn(chunk, 1)], other_y) is None


def test_a_batch_that_is_not_a_lazy_chunk_column_is_not_touched():
    """A ``LazyHostCodes`` batch (a plain device view) gets no recalled vector and is never copied to the host by the lookup."""
    y, _codes, d_codes, d_y, nbins = _setup(3)
    bits_gate_mi(d_codes, nbins, d_y, K_Y, (0.99,), labels_key(y))
    batch = LazyHostCodes(d_codes[:, :1], np.int8)
    assert _observed_from_noise_gate([batch], y) is None
    assert batch._host is None, "the lookup must not materialise the batch on the host"
