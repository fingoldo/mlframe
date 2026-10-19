"""The content memo answers a repeated pure request from the first result, never shares storage with a caller, and the pair-MI table and the order-2 floor use it on large requests."""

import numpy as np

from mlframe.feature_selection.filters._content_memo import ContentMemo, copy_value


def test_key_separates_content_dtype_shape_and_extras():
    """Arrays that differ in one value, dtype or shape, and extras that differ, give different keys; equal content gives the same key."""
    memo = ContentMemo(4)
    a = np.arange(12, dtype=np.int64).reshape(3, 4)
    assert memo.key([a], 1) == memo.key([a.copy()], 1)
    b = a.copy()
    b[1, 2] += 1
    assert memo.key([a], 1) != memo.key([b], 1)
    assert memo.key([a], 1) != memo.key([a.astype(np.int32)], 1)
    assert memo.key([a], 1) != memo.key([a.reshape(4, 3)], 1)
    assert memo.key([a], 1) != memo.key([a], 2)


def test_stored_value_is_independent_of_what_the_caller_holds():
    """Editing the object that was put, or an object that was got, does not change what a later get returns."""
    memo = ContentMemo(4)
    key = memo.key([np.arange(5)])
    value = (np.arange(4.0), {"cuda": 2}, "name")
    memo.put(key, value)
    value[0][:] = -1.0
    value[1]["cuda"] = 99
    first = memo.get(key)
    np.testing.assert_array_equal(first[0], np.arange(4.0))
    assert first[1] == {"cuda": 2} and first[2] == "name"
    first[0][:] = -2.0
    np.testing.assert_array_equal(memo.get(key)[0], np.arange(4.0))


def test_oldest_entry_is_evicted_first_and_a_hit_refreshes_an_entry():
    """With room for two entries, the least recently used one is dropped when a third arrives."""
    memo = ContentMemo(2)
    keys = [memo.key([np.full(3, i)]) for i in range(3)]
    memo.put(keys[0], 0.0)
    memo.put(keys[1], 1.0)
    assert memo.get(keys[0]) == 0.0  # refresh the first
    memo.put(keys[2], 2.0)
    assert memo.get(keys[1]) is None and memo.get(keys[0]) == 0.0 and memo.get(keys[2]) == 2.0 and len(memo) == 2


def test_copy_value_handles_nested_structures():
    """Arrays, tuples, lists and dicts are copied recursively; scalars pass through."""
    arr = np.arange(3)
    src = ([arr, (arr, 1.5)], {"a": arr})
    out = copy_value(src)
    out[0][0][0] = 9
    out[1]["a"][0] = 9
    assert arr[0] == 0 and out[0][1][1] == 1.5


def _pair_inputs(seed=0, n=3000, p=12, bins=6, classes=4):
    """Integer codes, a target and the all-pairs index arrays of a small pool."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, bins, (n, p)).astype(np.int8)
    y = rng.integers(0, classes, n).astype(np.int64)
    freqs = np.bincount(y, minlength=classes).astype(np.float64) / n
    pa, pb = np.triu_indices(p, 1)
    return data, np.full(p, bins, dtype=np.int64), pa.astype(np.int64), pb.astype(np.int64), y, freqs


def test_order2_floor_repeat_skips_the_shuffle_kernels_and_matches(monkeypatch):
    """A repeated floor request is answered from the stored per-shuffle maxima, equals the computed value, and a different seed recomputes."""
    from mlframe.feature_selection.filters import _permutation_null as pn
    from mlframe.feature_selection.filters import info_theory

    calls = []
    real = info_theory.batch_pair_mi_perm_batched

    def counting(*args, **kwargs):
        """Count the CPU shuffle-kernel calls and run the real kernel."""
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(info_theory, "batch_pair_mi_perm_batched", counting)
    monkeypatch.setattr("mlframe.feature_selection.filters._pair_mi_perm_cuda.floor_backend_choice", lambda n, n_pairs: "cpu")
    monkeypatch.setattr(pn, "_PAIR_FLOOR_MEMO_MIN_WORK", 1)
    pn._PAIR_FLOOR_MEMO.clear()
    data, nb, pa, pb, y, freqs = _pair_inputs()
    kw = dict(n_permutations=8, random_seed=5)
    first = pn.pooled_pair_permutation_null_joint_mi_floor(data, nb, pa, pb, y, freqs, **kw)
    again = pn.pooled_pair_permutation_null_joint_mi_floor(data.copy(), nb, pa, pb, y, freqs, **kw)
    assert len(calls) == 1 and again == first
    other_quantile = pn.pooled_pair_permutation_null_joint_mi_floor(data, nb, pa, pb, y, freqs, quantile=0.5, **kw)
    assert len(calls) == 1 and other_quantile <= first
    pn.pooled_pair_permutation_null_joint_mi_floor(data, nb, pa, pb, y, freqs, n_permutations=8, random_seed=6)
    assert len(calls) == 2
    monkeypatch.setattr(pn, "_PAIR_FLOOR_MEMO_MIN_WORK", 10**15)
    assert pn.pooled_pair_permutation_null_joint_mi_floor(data, nb, pa, pb, y, freqs, **kw) == first
    pn._PAIR_FLOOR_MEMO.clear()


def test_chunked_pair_mi_repeat_returns_an_equal_table_without_recomputing(monkeypatch):
    """The all-pairs MI table of an identical request is served from the memo as equal, independent arrays."""
    from mlframe.feature_selection.filters import batch_pair_mi_gpu as mod

    calls = []
    real = mod._dispatch_batch_pair_mi_chunked_uncached

    def counting(*args, **kwargs):
        """Count the real computations."""
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, "_dispatch_batch_pair_mi_chunked_uncached", counting)
    monkeypatch.setattr(mod, "_CHUNKED_MI_MEMO_MIN_WORK", 1)
    mod._CHUNKED_MI_MEMO.clear()
    data, nb, _pa, _pb, y, freqs = _pair_inputs()
    ids = np.arange(data.shape[1], dtype=np.int64)
    a1, b1, mi1, counts1 = mod.dispatch_batch_pair_mi_chunked(data, ids, nb, y, freqs, force_backend="cpu")
    a2, b2, mi2, counts2 = mod.dispatch_batch_pair_mi_chunked(data.copy(), ids, nb, y, freqs, force_backend="cpu")
    assert len(calls) == 1 and counts1 == counts2
    for first, second in ((a1, a2), (b1, b2), (mi1, mi2)):
        np.testing.assert_array_equal(first, second)
    mi2[:] = -1.0
    _a3, _b3, mi3, _c3 = mod.dispatch_batch_pair_mi_chunked(data, ids, nb, y, freqs, force_backend="cpu")
    np.testing.assert_array_equal(mi3, mi1)
    y_changed = y.copy()
    y_changed[0] = (y_changed[0] + 1) % 4
    mod.dispatch_batch_pair_mi_chunked(data, ids, nb, y_changed, freqs, force_backend="cpu")
    assert len(calls) == 2
    mod._CHUNKED_MI_MEMO.clear()
