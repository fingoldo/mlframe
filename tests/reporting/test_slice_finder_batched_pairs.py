"""The parallel batched pair kernel (_batched_pair_sum_count) must be bit-identical to the serial per-pair
_aggregate_combo path -- it only changes HOW the thousands of independent pair reductions are scheduled (one
prange pass across cores instead of a single-core python loop), never the numerics."""
import numpy as np
import pytest

from mlframe.reporting.charts import slice_finder as sf


@pytest.mark.skipif(not sf._HAS_NUMBA_SLICE, reason="numba unavailable")
def test_batched_pairs_bit_identical_to_serial_aggregate():
    rng = np.random.default_rng(0)
    n, p = 8000, 14
    mat = rng.standard_normal((n, p))
    err = np.ascontiguousarray(rng.standard_normal(n))
    codes, _edges = sf._bin_matrix(mat, 4)
    nbins_per = [4] * p

    pairs = [(0, 1), (2, 5), (3, 13), (7, 9), (10, 11)]
    f0 = np.array([c[0] for c in pairs], dtype=np.int64)
    f1 = np.array([c[1] for c in pairs], dtype=np.int64)
    s0 = np.array([nbins_per[c[1]] for c in pairs], dtype=np.int64)
    ncells = np.array([nbins_per[c[0]] * nbins_per[c[1]] for c in pairs], dtype=np.int64)
    codes_t = np.ascontiguousarray(codes.T)
    bsums, bcounts = sf._batched_pair_sum_count(codes_t, err, f0, f1, s0, int(ncells.max()))

    for k, c in enumerate(pairs):
        s_ser, c_ser, _strides = sf._aggregate_combo(codes, err, c, nbins_per)
        nc = int(ncells[k])
        assert np.array_equal(bsums[k, :nc], s_ser), f"sums diverged for pair {c}"
        assert np.array_equal(bcounts[k, :nc], c_ser), f"counts diverged for pair {c}"


@pytest.mark.skipif(not sf._HAS_NUMBA_SLICE, reason="numba unavailable")
def test_find_weak_slices_table_unchanged_by_batching(monkeypatch):
    rng = np.random.default_rng(3)
    n, p = 6000, 10
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    yhat = y + rng.standard_normal(n) * 0.3
    yhat[X[:, 0] > 1.0] += 3.0  # a real weak region on feature 0
    names = [f"f{i}" for i in range(p)]

    res_batched = sf.find_weak_slices(X, y, yhat, feature_names=names, seed=1)
    # Force the serial path (no batching) and compare the ranked table.
    monkeypatch.setattr(sf, "_HAS_NUMBA_SLICE", False)
    res_serial = sf.find_weak_slices(X, y, yhat, feature_names=names, seed=1)

    tb, ts = res_batched.table, res_serial.table
    assert list(tb["features"]) == list(ts["features"])
    assert np.allclose(tb["mean_error"].to_numpy(), ts["mean_error"].to_numpy())
    assert list(tb["support"]) == list(ts["support"])


def test_find_weak_slices_defers_labels_to_displayed_top_k(monkeypatch):
    """The human-readable ``bounds`` labels (``_bin_label`` + f-string + join) are built only for the
    displayed top_k rows, not for every candidate cell across all enumerated combos. Regression sensor:
    pre-deferral the loop called ``_bin_label`` once per feature of every valid cell (thousands of calls,
    all but top_k discarded); the deferral caps that at ``top_k * max_arity``. Also pins that the surfaced
    bounds stay correct so the deferral did not corrupt the labels it now builds lazily."""
    rng = np.random.default_rng(7)
    n, p = 4000, 12
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    yhat = y + rng.standard_normal(n) * 0.4
    names = [f"f{i}" for i in range(p)]

    calls = {"n": 0}
    real_bin_label = sf._bin_label

    def _counting_bin_label(edges, b):
        calls["n"] += 1
        return real_bin_label(edges, b)

    monkeypatch.setattr(sf, "_bin_label", _counting_bin_label)

    top_k, max_arity = 7, 2
    res = sf.find_weak_slices(X, y, yhat, feature_names=names, top_k=top_k, max_arity=max_arity, seed=1)

    # Deferral contract: labels are built only for displayed rows, so _bin_label fires at most
    # top_k * max_arity times (one per feature of each of the <=top_k surfaced slices).
    assert calls["n"] <= top_k * max_arity, (
        f"_bin_label called {calls['n']} times; deferral should cap at {top_k * max_arity} "
        f"(one per feature of the displayed top_k rows)"
    )
    # And the lazily-built bounds are still correct: each 2-feature slice label names both features
    # with a bracketed range and the ' & ' separator.
    for bounds, feats in zip(res.table["bounds"], res.table["features"]):
        for fname in feats:
            assert fname in bounds
        assert bounds.count("[") == len(feats)
