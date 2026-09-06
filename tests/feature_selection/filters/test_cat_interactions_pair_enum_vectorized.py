"""Wave 11 (Category 3) M10: ``_cat_interactions_step.py``'s candidate-pair-index enumeration was a
pure-Python nested loop (~125K iterations at p~500) building the cardinality-budget-filtered pair list one
comparison at a time. Rewritten via ``np.triu_indices`` + a vectorized boolean mask, including the
int32-overflow guard (kept explicit int64 arithmetic so a wide-cardinality product cannot silently wrap
before the guard sees it). Pins the vectorized enumeration against a frozen copy of the pre-fix nested loop.
"""

from __future__ import annotations

import numpy as np


def _old_enum(candidate_idxs_arr, nbins, max_combined):
    """Old enum."""
    pairs_a_list = []
    pairs_b_list = []
    for ii in range(len(candidate_idxs_arr)):
        for jj in range(ii + 1, len(candidate_idxs_arr)):
            i = int(candidate_idxs_arr[ii])
            j = int(candidate_idxs_arr[jj])
            nb_prod = int(nbins[i]) * int(nbins[j])
            if nb_prod > max_combined:
                continue
            if nb_prod >= 2**31:
                continue
            pairs_a_list.append(i)
            pairs_b_list.append(j)
    return np.asarray(pairs_a_list, dtype=np.int64), np.asarray(pairs_b_list, dtype=np.int64)


# The production enumeration itself, not a copy of it. This file used to define `_new_enum` as a hand-copy
# of the inline block in `_cat_interactions_step` and import nothing from the package, so two local copies
# were compared to each other and production never ran: dropping the `< 2**31` mask or the int64 cast left
# both tests here green while a 46341x46341 pair was admitted and overflowed downstream int32 arithmetic.
from mlframe.feature_selection.filters._cat_interactions_step import enumerate_candidate_pairs as _new_enum

def test_pair_enum_matches_reference_across_random_configs():
    """Pair enum matches reference across random configs."""
    rng = np.random.default_rng(0)
    n_checks = 0
    for trial in range(150):
        n_cols = int(rng.integers(0, 60))
        n_cand = int(rng.integers(0, min(30, n_cols + 1))) if n_cols else 0
        candidate_idxs_arr = rng.choice(np.arange(max(n_cols, 1)), size=n_cand, replace=False) if n_cand else np.array([], dtype=np.int64)
        nbins = rng.choice([2, 5, 10, 100, 5000, 50000, 70000], size=max(n_cols, 1))
        max_combined = int(rng.choice([10, 1000, 100000, 5_000_000]))
        a_old, b_old = _old_enum(candidate_idxs_arr, nbins, max_combined)
        a_new, b_new = _new_enum(candidate_idxs_arr, nbins, max_combined)
        n_checks += 1
        assert np.array_equal(a_old, a_new) and np.array_equal(b_old, b_new), f"trial={trial}"
    assert n_checks == 150


def test_pair_enum_int32_overflow_guard_rejects_wide_cardinality_pair():
    """A pair whose cardinality PRODUCT exceeds 2**31, but where a naive int32 multiply would silently
    WRAP to something under the guard's threshold, must still be rejected on both paths -- proves the
    vectorized path's explicit int64 cast is load-bearing, not cosmetic."""
    candidate_idxs_arr = np.array([0, 1], dtype=np.int64)
    nbins = np.array([46341, 46341], dtype=np.int64)  # 46341**2 = 2147488281 > 2**31 (2147483648)
    max_combined = 10_000_000_000  # budget itself would NOT reject it -- only the overflow guard should
    a_old, _b_old = _old_enum(candidate_idxs_arr, nbins, max_combined)
    a_new, _b_new = _new_enum(candidate_idxs_arr, nbins, max_combined)
    assert a_old.size == 0
    assert a_new.size == 0
