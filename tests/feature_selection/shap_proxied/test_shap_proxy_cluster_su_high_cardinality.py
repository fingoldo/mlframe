"""SU clustering must stay memory-bounded and correct when a column has a huge code space.

The pairwise kernel used to size one ``max_nb x max_nb`` int64 joint buffer per outer row, so a single ~200k-level
column requested ~320 GB: a MemoryError on Windows, and under numba's omp layer on Linux a silently swallowed worker
exception that returned an all-zero flag matrix (no feature ever clustered).

The clustering calls here pass ``chance_correct=False``: they pin the raw plug-in kernels' edges, whereas the default chance correction deliberately unlinks
near-unique (``K > n/2``) column pairs, which carry no evidence of dependence (``test_shap_proxy_cluster_su_chance.py``).
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.info_theory import compute_su_from_classes
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster_su import (
    _setup_su_kernel_inputs,
    cluster_correlated_features_su,
)
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster_su_joint import (
    _pair_mi_dense,
    _pair_mi_sparse,
    _pairwise_su_edges,
    su_from_classes_sparse,
)


def _low_card_bins(rng: np.random.Generator, n: int, f: int) -> dict[str, np.ndarray]:
    """Quantile-bin-like low-cardinality columns, a third of them sharing a common driver."""
    base = rng.integers(0, 6, n)
    bins = {}
    for i in range(f):
        k = int(rng.integers(2, 12))
        mix = base if i % 3 == 0 else 0
        bins[f"f{i}"] = ((mix + rng.integers(0, k, n)) % k).astype(np.int32)
    return bins


def test_high_cardinality_column_keeps_parallel_su_edges():
    rng = np.random.default_rng(0)
    n = 200_000
    bins = _low_card_bins(rng, n, 57)
    bins["f1"] = bins["f0"].copy()
    bins["id_a"] = rng.integers(0, n, n).astype(np.int32)
    bins["id_b"] = bins["id_a"].copy()
    names = list(bins)
    labels = cluster_correlated_features_su(bins, threshold=0.5, feature_names=names, use_gpu=False, chance_correct=False)
    assert labels[names.index("f0")] == labels[names.index("f1")]
    assert labels[names.index("id_a")] == labels[names.index("id_b")]


def test_high_cardinality_pair_serial_path_does_not_allocate_dense_table():
    rng = np.random.default_rng(1)
    n = 200_000
    bins = _low_card_bins(rng, n, 4)
    bins["id_a"] = rng.integers(0, n, n).astype(np.int32)
    bins["id_b"] = bins["id_a"].copy()
    names = list(bins)
    labels = cluster_correlated_features_su(bins, threshold=0.5, feature_names=names, use_parallel=False, chance_correct=False)
    assert labels[names.index("id_a")] == labels[names.index("id_b")]


def test_wide_code_space_categorical_is_relabelled_densely():
    rng = np.random.default_rng(2)
    n = 5_000
    dense = rng.integers(0, 10, n).astype(np.int32)
    wide = (dense * 50_000 + 7).astype(np.int32)
    other = ((dense + rng.integers(0, 2, n)) % 10).astype(np.int32)
    packed = _setup_su_kernel_inputs([wide, other], None)
    assert packed is not None
    assert int(packed[1][0]) == 10
    ref = _setup_su_kernel_inputs([dense, other], None)
    assert ref is not None
    np.testing.assert_array_equal(packed[0], ref[0])
    np.testing.assert_array_equal(packed[4], ref[4])


def test_sparse_joint_counting_is_bit_identical_to_dense():
    rng = np.random.default_rng(3)
    for trial in range(6):
        n = int(rng.integers(300, 6_000))
        arrs = [rng.integers(0, int(rng.integers(2, 20)), n).astype(np.int32) for _ in range(12)]
        arrs[1] = ((arrs[0] + rng.integers(0, 2, n)) % 20).astype(np.int32)
        packed = _setup_su_kernel_inputs(arrs, None)
        assert packed is not None
        bp, nb, fp, fo, _h, _cm = packed
        for thr in (0.01, 0.1, 0.4):
            np.testing.assert_array_equal(_pairwise_su_edges(*packed, thr), _pairwise_su_edges(*packed, thr, 0))
        for i in range(4):
            for j in range(i + 1, 4):
                d = _pair_mi_dense(bp[i], bp[j], nb[i], nb[j], fp, fo[i], fo[j], np.zeros(nb[i] * nb[j], np.int64), 1.0 / n)
                s = _pair_mi_sparse(bp[i], bp[j], nb[j], fp, fo[i], fo[j], np.empty(n, np.int64), 1.0 / n)
                assert d == s, (trial, i, j)
                ci, cj = arrs[i].astype(np.int64), arrs[j].astype(np.int64)
                fi, fj = np.bincount(ci, minlength=nb[i]) / n, np.bincount(cj, minlength=nb[j]) / n
                assert su_from_classes_sparse(ci, fi, cj, fj) == compute_su_from_classes(ci, fi, cj, fj)
