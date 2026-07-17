"""Regression sensor for S31 / A1#5: global np.random.shuffle removed from njit kernels.

Pre-fix sites:
* ``parallel_mi`` (permutation.py:210) called ``np.random.shuffle(classes_y_safe)`` on the
  process-global numpy RNG -- two parallel suite calls produced racing, non-deterministic
  output and no per-call ``base_seed`` could be threaded.
* ``get_fleuret_criteria_confidence`` (fleuret.py:187,191) called
  ``np.random.shuffle(data_copy[:, idx])`` likewise.

Fix: add ``base_seed`` kwarg to both kernels + replace the global-RNG sites with inline
LCG Fisher-Yates (mirrors the existing ``parallel_mi_prange`` / ``parallel_mi_besag_clifford``
LCG pattern). The same ``base_seed`` must produce identical output across calls;
different ``base_seed`` must produce different output.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.fleuret import (
    _fleuret_shuffle_col_lcg,
    get_fleuret_criteria_confidence,
)
from mlframe.feature_selection.filters.permutation import parallel_mi


def _tiny_classes_freqs():
    rng = np.random.default_rng(0)
    n = 500
    classes_x = rng.integers(0, 4, size=n).astype(np.int32)
    classes_y = rng.integers(0, 3, size=n).astype(np.int32)
    freqs_x = np.bincount(classes_x).astype(np.float64) / n
    freqs_y = np.bincount(classes_y).astype(np.float64) / n
    return classes_x, freqs_x, classes_y, freqs_y


def test_parallel_mi_same_seed_reproducible():
    classes_x, freqs_x, classes_y, freqs_y = _tiny_classes_freqs()
    nperms = 50
    out_a = parallel_mi(
        classes_x=classes_x,
        freqs_x=freqs_x,
        classes_y=classes_y,
        freqs_y=freqs_y,
        npermutations=nperms,
        original_mi=0.001,
        max_failed=nperms,
        base_seed=np.uint64(12345),
    )
    out_b = parallel_mi(
        classes_x=classes_x,
        freqs_x=freqs_x,
        classes_y=classes_y,
        freqs_y=freqs_y,
        npermutations=nperms,
        original_mi=0.001,
        max_failed=nperms,
        base_seed=np.uint64(12345),
    )
    assert out_a == out_b, f"same base_seed must yield identical (nfailed, nchecked); got {out_a} vs {out_b}"


def test_parallel_mi_different_seed_different_stream():
    """Use a target MI high enough that only ~half the permutations exceed it; that lets the seed-driven order
    materially change the running nfailed sum across short streams. With max_failed = small, distinct seeds
    bail at different iter counts."""
    classes_x, freqs_x, classes_y, freqs_y = _tiny_classes_freqs()
    # Pick a moderate original_mi so roughly half of permuted MIs exceed it.
    from mlframe.feature_selection.filters.info_theory import compute_mi_from_classes

    base_mi = compute_mi_from_classes(
        classes_x=classes_x,
        freqs_x=freqs_x,
        classes_y=classes_y,
        freqs_y=freqs_y,
        dtype=np.int32,
    )
    nperms = 200
    max_failed = 5  # forces early exit at different iter counts per seed
    seeds = [np.uint64(s) for s in (1, 7, 99, 100003, 9876543210)]
    streams = []
    for s in seeds:
        out = parallel_mi(
            classes_x=classes_x,
            freqs_x=freqs_x,
            classes_y=classes_y,
            freqs_y=freqs_y,
            npermutations=nperms,
            original_mi=float(base_mi),
            max_failed=max_failed,
            base_seed=s,
        )
        streams.append(out)
    distinct = len({(int(a), int(b)) for a, b in streams})
    assert distinct >= 2, f"distinct base_seeds must yield varied (nfailed, nchecked); got {streams}"


def test_fleuret_shuffle_col_lcg_reproducible():
    col_a = np.arange(40, dtype=np.int32)
    col_b = col_a.copy()
    state = np.uint64(42)
    s1 = _fleuret_shuffle_col_lcg(col_a, state)
    s2 = _fleuret_shuffle_col_lcg(col_b, state)
    assert np.array_equal(col_a, col_b), "same starting state must produce identical permutation"
    assert s1 == s2


def test_fleuret_shuffle_col_lcg_different_seed_diff_permutation():
    col = np.arange(40, dtype=np.int32)
    states = [np.uint64(s) for s in (1, 17, 99, 100003)]
    perms = []
    for s in states:
        c = col.copy()
        _fleuret_shuffle_col_lcg(c, s)
        perms.append(tuple(int(v) for v in c))
    assert len(set(perms)) >= 3


def test_parallel_mi_no_global_rng_drift():
    """If parallel_mi consumed the process-global numpy RNG, calling np.random.* between two
    invocations of parallel_mi(base_seed=X) would alter the second invocation's output.
    With the LCG fix, both invocations are isolated from global state.
    """
    classes_x, freqs_x, classes_y, freqs_y = _tiny_classes_freqs()
    nperms = 80
    np.random.seed(101)
    out_a = parallel_mi(
        classes_x=classes_x,
        freqs_x=freqs_x,
        classes_y=classes_y,
        freqs_y=freqs_y,
        npermutations=nperms,
        original_mi=0.0001,
        max_failed=nperms,
        base_seed=np.uint64(55555),
    )
    np.random.seed(202)
    _ = np.random.random(10000)
    out_b = parallel_mi(
        classes_x=classes_x,
        freqs_x=freqs_x,
        classes_y=classes_y,
        freqs_y=freqs_y,
        npermutations=nperms,
        original_mi=0.0001,
        max_failed=nperms,
        base_seed=np.uint64(55555),
    )
    assert out_a == out_b, f"parallel_mi must be isolated from global RNG drift; got {out_a} vs {out_b}"
