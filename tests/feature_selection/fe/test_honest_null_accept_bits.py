"""The sorted-space bitset acceptance of the honest null accepts exactly the pairs the exact full scan accepts."""

import numpy as np
import pytest


def _setup(n, kc, n_perm, p1, ties, planted, seed):
    """Sorted orders, tie flags, permuted targets (a subset planted to depend on a column, same number of ones) and the exact k4 reference output."""
    cp = pytest.importorskip("cupy")
    try:  # audit-exempt: except_skip_masks_call_under_test -- the guarded call only asks the runtime for a CUDA device; nothing under test runs in the try
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    from mlframe.feature_selection.filters._family_maxt_null_v2_njit import xlogx_table
    from mlframe.feature_selection.filters._honest_null_device import _entropy, _pack_bits, _tie_flags
    from mlframe.feature_selection.filters._honest_null_k4 import get_kernel_k4
    from mlframe.feature_selection.filters._mdlp_honest_njit import _presort_columns

    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, kc))
    if ties:
        raw[:, ::3] = np.round(raw[:, ::3] * 15)
        raw[:, 1] = 2.0  # a constant column: every cut is a tie
    order = _presort_columns(raw)
    tie = _tie_flags(raw, order)
    y = (rng.random(n) < p1).astype(np.int8)
    yperm = np.stack([rng.permutation(y) for _ in range(n_perm)])
    m = int(y.sum())
    for b in range(0, n_perm, planted):
        score = raw[:, b % kc] * (0.15 + 0.05 * (b % 5)) + rng.normal(size=n)
        lab = np.zeros(n, dtype=np.int8)
        lab[np.argsort(-score)[:m]] = 1
        yperm[b] = lab
    counts = np.bincount(yperm[0].astype(np.int64), minlength=2)
    h_full = _entropy(counts, n)
    tab = cp.asarray(xlogx_table(n))
    yb = _pack_bits(yperm)
    g, c, i = cp.empty(n_perm * kc), cp.empty(n_perm * kc, dtype=cp.int32), cp.empty(n_perm * kc, dtype=cp.int32)
    get_kernel_k4(cp)(((n_perm * kc + 127) // 128,), (128,), (cp.asarray(order), cp.asarray(tie), cp.asarray(np.ascontiguousarray(yb.T).view(np.uint8)), tab, cp.asarray(counts.astype(np.int64)), np.int32(n), np.int32(n_perm), np.int32(kc), np.int32(60), np.float64(h_full), np.int32(yb.shape[1]), g, c, i))
    return cp, order, tie, yperm, h_full, tab, (cp.asnumpy(g), cp.asnumpy(c), cp.asnumpy(i))


@pytest.mark.parametrize(
    "n,kc,n_perm,p1,ties,planted",
    [(20000, 24, 120, 0.4, False, 7), (20011, 24, 120, 0.4, True, 7), (30011, 16, 90, 0.07, True, 5), (5003, 13, 70, 0.5, False, 6), (9999, 10, 60, 0.95, False, 4), (64000, 8, 50, 0.3, True, 3)],
)
def test_acceptance_equals_the_full_scan(n, kc, n_perm, p1, ties, planted):
    """The accept booleans are identical to the exact scan's, and the (gain, count, index) of every accepted pair, and the candidate counts of all pairs, are equal."""
    from mlframe.feature_selection.filters._honest_null_accept_bits import root_split_accept_gpu
    from mlframe.feature_selection.filters._honest_null_device import _accepted

    alpha = 0.05
    cp, order, tie, yperm, h_full, tab, (g_ref, c_ref, i_ref) = _setup(n, kc, n_perm, p1, ties, planted, seed=n + kc)
    g, c, i = cp.empty(n_perm * kc), cp.empty(n_perm * kc, dtype=cp.int32), cp.empty(n_perm * kc, dtype=cp.int32)
    assert root_split_accept_gpu(cp, order, tie, yperm, n, n_perm, kc, 60, alpha, h_full, tab, g, c, i)
    g, c, i = cp.asnumpy(g), cp.asnumpy(c), cp.asnumpy(i)

    def accepted(gain, ncand, idx):
        """The host acceptance rule on (permutations, columns) arrays."""
        return _accepted(gain.reshape(kc, n_perm).T.copy(), ncand.reshape(kc, n_perm).T.copy(), idx.reshape(kc, n_perm).T.copy(), n, 2, alpha)

    ref_ok, got_ok = accepted(g_ref, c_ref, i_ref), accepted(g, c, i)
    assert ref_ok.sum() > 0, "the planted signal must make some pairs accepted, else the comparison is empty"
    np.testing.assert_array_equal(ref_ok, got_ok)
    sel = np.flatnonzero(ref_ok.T.reshape(-1))
    np.testing.assert_array_equal(g[sel], g_ref[sel])
    np.testing.assert_array_equal(i[sel], i_ref[sel])
    np.testing.assert_array_equal(c, c_ref)


def test_declines_when_the_counts_differ_or_the_bitset_is_too_large():
    """Permutations with different numbers of ones, and a column too long for the shared bitset, are left to the per-step kernel."""
    from mlframe.feature_selection.filters._honest_null_accept_bits import accept_fits

    y = np.zeros((3, 1000), dtype=np.int8)
    y[:, :300] = 1
    assert accept_fits(1000, y)
    y[1, 5] = 0
    assert not accept_fits(1000, y)
    big = np.zeros((2, 10), dtype=np.int8)
    assert not accept_fits(10_000_000, big)


def test_precomputed_sequences_serve_rows_beyond_the_shared_budget():
    """At 250011 rows the shared bitset does not fit: the precomputed-sequence variant takes over and still accepts exactly what the full scan accepts."""
    from mlframe.feature_selection.filters._honest_null_accept_bits import accept_fits, root_split_accept_gpu
    from mlframe.feature_selection.filters._honest_null_device import _accepted

    n, kc, n_perm, alpha = 250011, 3, 24, 0.05
    cp, order, tie, yperm, h_full, tab, (g_ref, c_ref, i_ref) = _setup(n, kc, n_perm, 0.4, True, 5, seed=11)
    assert not accept_fits(n, yperm)
    g, c, i = cp.empty(n_perm * kc), cp.empty(n_perm * kc, dtype=cp.int32), cp.empty(n_perm * kc, dtype=cp.int32)
    assert root_split_accept_gpu(cp, order, tie, yperm, n, n_perm, kc, 60, alpha, h_full, tab, g, c, i)
    g, c, i = cp.asnumpy(g), cp.asnumpy(c), cp.asnumpy(i)

    def accepted(gain, ncand, idx):
        """The host acceptance rule on (permutations, columns) arrays."""
        return _accepted(gain.reshape(kc, n_perm).T.copy(), ncand.reshape(kc, n_perm).T.copy(), idx.reshape(kc, n_perm).T.copy(), n, 2, alpha)

    ref_ok, got_ok = accepted(g_ref, c_ref, i_ref), accepted(g, c, i)
    assert ref_ok.sum() > 0
    np.testing.assert_array_equal(ref_ok, got_ok)
    np.testing.assert_array_equal(c, c_ref)


@pytest.mark.parametrize("mode", ["global", "pre"])
def test_every_variant_equals_the_full_scan_at_a_shared_size(monkeypatch, mode):
    """The global-table and precomputed-sequence variants, forced at a size the shared one handles, accept what the full scan accepts (permutation chunking included: 140 permutations)."""
    from mlframe.feature_selection.filters._honest_null_accept_bits import root_split_accept_gpu
    from mlframe.feature_selection.filters._honest_null_device import _accepted

    monkeypatch.setenv("MLFRAME_HONEST_ACCEPT_MODE", mode)
    n, kc, n_perm, alpha = 20011, 5, 140, 0.05
    cp, order, tie, yperm, h_full, tab, (g_ref, c_ref, i_ref) = _setup(n, kc, n_perm, 0.4, True, 7, seed=3)
    g, c, i = cp.empty(n_perm * kc), cp.empty(n_perm * kc, dtype=cp.int32), cp.empty(n_perm * kc, dtype=cp.int32)
    assert root_split_accept_gpu(cp, order, tie, yperm, n, n_perm, kc, 60, alpha, h_full, tab, g, c, i)
    g, c, i = cp.asnumpy(g), cp.asnumpy(c), cp.asnumpy(i)

    def accepted(gain, ncand, idx):
        """The host acceptance rule on (permutations, columns) arrays."""
        return _accepted(gain.reshape(kc, n_perm).T.copy(), ncand.reshape(kc, n_perm).T.copy(), idx.reshape(kc, n_perm).T.copy(), n, 2, alpha)

    ref_ok, got_ok = accepted(g_ref, c_ref, i_ref), accepted(g, c, i)
    assert ref_ok.sum() > 0
    np.testing.assert_array_equal(ref_ok, got_ok)
    np.testing.assert_array_equal(c, c_ref)


@pytest.mark.parametrize("variant", ["noqueue", "tiny_queue"])
def test_queue_variant_equals_the_thread_serial_and_overflow_paths(variant, monkeypatch):
    """The default kernel (surviving words shared out over the warps of a block) gives the same (gain, count, index) as the thread-serial version and as a build whose queue holds two words (every pair overflows into the thread-serial path)."""
    import mlframe.feature_selection.filters._honest_null_accept_bits as A
    from mlframe.feature_selection.filters._raw_module_cache import RawModuleCache

    n, kc, n_perm = 30011, 12, 80
    cp, order, tie, yperm, h_full, tab, _ref = _setup(n, kc, n_perm, 0.4, True, 5, seed=3)

    def run():
        """One launch of whichever module is installed."""
        g, c, i = cp.empty(n_perm * kc), cp.empty(n_perm * kc, dtype=cp.int32), cp.empty(n_perm * kc, dtype=cp.int32)
        assert A.root_split_accept_gpu(cp, order, tie, yperm, n, n_perm, kc, 60, 0.05, h_full, tab, g, c, i)
        return cp.asnumpy(g), cp.asnumpy(c), cp.asnumpy(i)

    want = run()
    if variant == "noqueue":
        monkeypatch.setattr(A, "_MODULE", A._MODULE_NOQUEUE)
    else:
        monkeypatch.setattr(A, "_MODULE", RawModuleCache(A._SRC.replace("#define QCAP 512", "#define QCAP 2")))
    got = run()
    for w, g in zip(want, got):
        np.testing.assert_array_equal(g, w)
