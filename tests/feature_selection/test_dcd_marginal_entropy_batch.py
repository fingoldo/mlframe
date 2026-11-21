"""The batched marginal-entropy warm-up of the DCD pair-SU path gives exactly the entropies of the serial merge_vars + entropy loop, and leaves a column with out-of-range codes to it."""

import types

import numpy as np

from mlframe.feature_selection.filters import _dcd_pair_su_batch as M
from mlframe.feature_selection.filters.info_theory import entropy, merge_vars


def _state():
    """A stand-in for the DCD state with the fields the warm-up reads and writes."""
    return types.SimpleNamespace(column_entropy_cache={}, _fn_arr_cached=None, _pair_idx_buf=None)


def _data(n=4000, p=40, seed=0):
    """Codes of mixed cardinality (2..12 bins), one constant column, one with an empty class."""
    rng = np.random.default_rng(seed)
    nb = rng.integers(2, 13, size=p).astype(np.int64)
    fd = np.stack([rng.integers(0, nb[j], size=n) for j in range(p)], axis=1).astype(np.int8)
    fd[:, 3] = 0
    nb[3] = 5
    fd[:, 7] = np.where(fd[:, 7] == 2, 3, fd[:, 7])
    nb[7] = max(int(nb[7]), 4)
    return fd, nb


def _serial(fd, nb, cols):
    """The entropies the loop of the serial path writes."""
    out = {}
    buf = np.empty(2, dtype=np.int64)
    for c in cols:
        buf[0] = c
        _, freqs, _ = merge_vars(fd, buf[:1], None, nb, np.int32, 1, None, False)
        out[c] = float(entropy(freqs, 0.0))
    return out


def test_batched_marginals_equal_the_serial_loop_bit_for_bit():
    """Every column of a 40-column frame: the same float from the batch and from the loop."""
    fd, nb = _data()
    cols = list(range(fd.shape[1]))
    ec: dict = {}
    left = M._warm_marginals_in_blocks(ec, cols, fd, nb, entropy)
    assert left == []
    assert ec == _serial(fd, nb, cols)


def test_small_blocks_and_out_of_range_codes(monkeypatch):
    """A block budget of a few columns gives the same values; a column with a code beyond its bin count is handed back, not guessed."""
    fd, nb = _data(seed=1)
    monkeypatch.setattr(M, "_BATCH_MARGINALS_BLOCK_BYTES", 5 * fd.shape[0])
    fd[10, 9] = nb[9] + 1
    ec: dict = {}
    left = M._warm_marginals_in_blocks(ec, list(range(fd.shape[1])), fd, nb, entropy)
    assert left == [9] and 9 not in ec
    ref = _serial(fd, nb, [c for c in range(fd.shape[1]) if c != 9])
    assert all(ec[c] == ref[c] for c in ref)


def test_warm_marginal_entropies_uses_the_batch_and_fills_the_cache():
    """Through the public warm-up: the cache holds every column of the pairs after one call, equal to the serial values."""
    fd, nb = _data(seed=2)
    pairs = [(a, b) for a in range(0, 20) for b in range(20, 30)]
    st = _state()
    M._warm_marginal_entropies(st, pairs, fd, nb, np.int32)
    needed = sorted({c for ab in pairs for c in ab})
    assert sorted(st.column_entropy_cache) == needed
    ref = _serial(fd, nb, needed)
    assert all(st.column_entropy_cache[c] == ref[c] for c in needed)
