"""The group-sorted chance-gate kernels must reproduce the legacy global-key-sort kernels bit-for-bit (same MI, same edge decisions)."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_cluster_su_chance as new
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster_su_joint import dense_relabel_codes

_LEGACY = pathlib.Path(new.__file__).parent / "_benchmarks" / "su_chance_legacy.py"


@pytest.fixture(scope="module")
def legacy():
    spec = importlib.util.spec_from_file_location("su_chance_legacy", _LEGACY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # numba's cache=True resolves the defining module through sys.modules
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop(spec.name, None)


def _scenario(seed: int):
    rng = np.random.default_rng(seed)
    n = int(rng.choice([300, 2_000, 8_000]))
    cols = []
    for _ in range(int(rng.integers(3, 7))):
        kind = rng.integers(0, 4)
        if kind == 0:
            cols.append(rng.integers(0, max(2, n // int(rng.integers(2, 8))), n))  # ID-like
        elif kind == 1:
            cols.append(rng.integers(0, int(rng.integers(2, 12)), n))  # low-card, tied
        elif kind == 2:
            cols.append(rng.zipf(1.6, n) % max(2, n // 3))  # heavy ties + long tail
        else:
            base = cols[-1] if cols else rng.integers(0, 50, n)
            cols.append(np.where(rng.random(n) < 0.1, rng.integers(0, 50, n), base))  # noisy copy
    cols.append(rng.permutation(cols[0].max() + 1)[cols[0]])  # recoded duplicate
    return [np.asarray(c, dtype=np.int64) for c in cols]


@pytest.mark.parametrize("seed", range(40))
def test_edge_decisions_identical_to_legacy_kernel(legacy, seed):
    cols = _scenario(seed)
    ii, jj = np.triu_indices(len(cols), 1)
    ei, ej = ii.astype(np.int64), jj.astype(np.int64)
    for thr in (0.3, 0.5):
        for scope in (0.1, 0.0):  # scope 0.0 forces rescoring of every non-near-unique edge, incl. tied low-card ones
            old = legacy.veto_chance_edges(cols, ei, ej, thr, scope_ratio=scope, seed=seed)
            got = new.veto_chance_edges(cols, ei, ej, thr, scope_ratio=scope, seed=seed)
            assert np.array_equal(old[0], got[0]) and np.array_equal(old[1], got[1])


def test_scored_su_and_null_bit_identical_to_legacy(legacy):
    cols = _scenario(3)
    n = cols[0].shape[0]
    f = len(cols)
    dense = np.empty((f, n), dtype=np.int32)
    nbins = np.array([dense_relabel_codes(c, dense[k]) for k, c in enumerate(cols)], dtype=np.int64)
    cnts = [np.bincount(dense[k], minlength=int(nbins[k])).astype(np.int64) for k in range(f)]
    ent = np.array([new._entropy(c, n) for c in cnts])
    offs = np.concatenate([[0], np.cumsum(nbins)[:-1]]).astype(np.int64)
    counts = np.concatenate(cnts)
    ii, jj = np.triu_indices(f, 1)
    ea, eb = ii.astype(np.int64), jj.astype(np.int64)
    ref = legacy._score_edges(dense, nbins, counts, offs, ent, ea, eb, 2, 7)
    major = np.unique(ea)
    order_row = np.full(f, -1, dtype=np.int64)
    order_row[major] = np.arange(major.size)
    orders = np.empty((major.size, n), dtype=np.int32)
    new._group_orders(dense, nbins, counts, offs, major, orders)
    got = new._score_edges(dense, nbins, counts, offs, ent, orders, order_row, ea, eb, 2, 7)
    assert np.array_equal(ref, got)


def test_group_orders_is_stable_grouping_permutation():
    rng = np.random.default_rng(0)
    n = 1_000
    cols = [rng.integers(0, 40, n), rng.integers(0, 7, n)]
    dense = np.empty((2, n), dtype=np.int32)
    nbins = np.array([dense_relabel_codes(c, dense[k]) for k, c in enumerate(cols)], dtype=np.int64)
    cnts = [np.bincount(dense[k], minlength=int(nbins[k])).astype(np.int64) for k in range(2)]
    offs = np.array([0, nbins[0]], dtype=np.int64)
    orders = np.empty((2, n), dtype=np.int32)
    new._group_orders(dense, nbins, np.concatenate(cnts), offs, np.array([0, 1], dtype=np.int64), orders)
    for r in range(2):
        assert np.array_equal(orders[r], np.argsort(dense[r], kind="stable"))
