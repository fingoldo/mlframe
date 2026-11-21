"""Audit 2026-11-20 findings 34-F2, 34-F3 and 34-F4 in composite auto-base selection.

34-F2: a hint feature the finite filter drops must not take a top-k slot, and must not be returned as a candidate.
34-F3: demoted (time/spatial) candidates must not set the MRMR relevance scale, so they rank after the MRMR pass.
34-F4: the collinear replay must re-check every column exactly, so the kept mask matches the numpy reference on tied data.
"""

from types import SimpleNamespace

import numpy as np

from mlframe.training.composite.discovery._auto_base import _auto_base_step5_last_resort
from mlframe.training.composite.discovery._auto_base_prep import finite_mask_prep
from mlframe.training.composite.discovery._collinear_numba import near_collinear_keep_mask_fast
from mlframe.training.composite.discovery._eval_stats import near_collinear_keep_mask


def test_demoted_candidate_is_ranked_after_the_mrmr_pass():
    """With a demoted top-MI column, the demoted one is last; the rest keep an MRMR order that is not decided by redundancy alone."""
    rng = np.random.default_rng(0)
    n = 400
    b = rng.normal(size=n)
    x = np.column_stack([rng.normal(size=n), b, b + 1e-3 * rng.normal(size=n), rng.normal(size=n)])
    usable = ["a_demoted", "b", "c_copy_of_b", "d"]
    mi = np.array([0.5, 0.3, 0.29, 0.2])
    cfg = SimpleNamespace(base_ranking_criterion="mrmr", base_ranking_mrmr_beta=1.0, mi_nbins=10, base_max_abs_corr_with_y=0.9995)
    self_ = SimpleNamespace(config=cfg)
    finite = np.ones(n, dtype=bool)
    _thr, ranked = _auto_base_step5_last_resort(self_, {"a_demoted"}, usable, mi.copy(), x, finite)
    names = [c for _m, c in ranked]
    assert names[-1] == "a_demoted"
    assert names.index("d") < names.index("c_copy_of_b"), names


def test_hint_dropped_by_the_finite_filter_is_not_returned():
    """A hint whose column is dropped for too few finite cells does not appear in the early-result list."""
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(size=(n, 3))
    x[: int(0.95 * n), 0] = np.nan  # 'sparse' has ~5% finite cells, below the 10% cutoff
    usable = ["sparse", "f1", "f2"]
    self_ = SimpleNamespace(config=SimpleNamespace(mi_nbins=10, top_k=2))
    y = rng.normal(size=n)
    early, *_rest = finite_mask_prep(self_, usable, x, y, ["sparse"], 2)
    assert early is None or "sparse" not in early


def test_collinear_mask_fast_matches_numpy_reference_on_tied_discrete_data():
    """On discrete data with exact ties the fast mask equals the reference for several seeds."""
    for seed in range(5):
        rng = np.random.default_rng(seed)
        base = rng.integers(0, 3, size=(300, 6)).astype(np.float64)
        base[:, 3] = base[:, 0] + base[:, 1]
        fast = near_collinear_keep_mask_fast(base, corr_threshold=0.9, reference_fn=near_collinear_keep_mask)
        ref = near_collinear_keep_mask(base, corr_threshold=0.9)
        np.testing.assert_array_equal(np.asarray(fast).astype(bool), np.asarray(ref).astype(bool))
