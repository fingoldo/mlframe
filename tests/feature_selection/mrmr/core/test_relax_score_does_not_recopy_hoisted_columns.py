"""The RelaxMRMR score must not re-copy or re-scan the columns its caller hoisted for the whole round.

The target and the selected set are fixed across a greedy round, and the caller materialises them once as int64 codes. The score then copied
every one of them again per candidate (``astype`` copies unconditionally, even when the dtype already matches) and range-scanned each of them
again per candidate. At |S|=200 and n=1e6 that is gigabytes of transient copies and |S| full column scans, for every candidate.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._relaxmrmr_3d import assert_relax_inputs_in_range, relax_mrmr_score


@pytest.fixture
def case():
    """A candidate, a target and a selected set, all already int64 codes as the caller's hoist produces them."""
    rng = np.random.default_rng(0)
    n, K_x, K_y = 3000, 4, 3
    x = rng.integers(0, K_x, size=n).astype(np.int64)
    y = rng.integers(0, K_y, size=n).astype(np.int64)
    K_sel = [3, 4, 2]
    sel = [rng.integers(0, k, size=n).astype(np.int64) for k in K_sel]
    return x, y, sel, K_x, K_y, K_sel


def test_int64_inputs_are_not_copied(case):
    """Already-int64 columns are taken as views, so scoring allocates nothing per column."""
    x, y, sel, K_x, K_y, K_sel = case
    seen: list = []
    real_asarray = np.asarray

    def recording_asarray(a, *args, **kwargs):
        """Record whether each conversion returned the very object it was given."""
        out = real_asarray(a, *args, **kwargs)
        if isinstance(a, np.ndarray) and a.dtype == np.int64:
            seen.append(out is a)
        return out

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(np, "asarray", recording_asarray)
        relax_mrmr_score(x, sel, y, K_x, K_sel, K_y, alpha=1.0, selected_prechecked=True)
    assert seen, "no int64 conversion was observed, so this test is not looking at the right thing"
    assert all(seen), f"{seen.count(False)} of {len(seen)} int64 columns were copied rather than viewed"


def test_the_precheck_flag_skips_the_repeat_scan_but_not_the_candidate_scan(case):
    """With the fixed inputs already checked, only the candidate column is scanned per call."""
    x, y, sel, K_x, K_y, K_sel = case
    from mlframe.feature_selection.filters import _fe_batched_mi

    scanned: list = []
    real_check = _fe_batched_mi._assert_codes_in_range

    def recording_check(codes, nbins, what):
        """Record which named input each range check looked at."""
        scanned.append(what)
        return real_check(codes, nbins, what)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_fe_batched_mi, "_assert_codes_in_range", recording_check)
        relax_mrmr_score(x, sel, y, K_x, K_sel, K_y, alpha=1.0, selected_prechecked=True)
        assert scanned == ["relax_mrmr_score x_cand"], f"the fixed inputs were re-scanned: {scanned}"
        scanned.clear()
        relax_mrmr_score(x, sel, y, K_x, K_sel, K_y, alpha=1.0)
    assert scanned.count("relax_mrmr_score selected_col") == len(sel), "without the flag every selected column must still be checked"
    assert "relax_mrmr_score y" in scanned


def test_the_hoisted_check_rejects_what_the_per_candidate_check_rejected(case):
    """Moving the check must not weaken it: an out-of-range code is still refused, just one call earlier."""
    x, y, sel, K_x, K_y, K_sel = case
    bad = sel[0].copy()
    bad[7] = K_sel[0] + 5
    with pytest.raises(ValueError):
        assert_relax_inputs_in_range(y, K_y, [bad, *sel[1:]], K_sel)
    with pytest.raises(ValueError):
        relax_mrmr_score(x, [bad, *sel[1:]], y, K_x, K_sel, K_y, alpha=1.0)


def test_the_score_is_unchanged_by_skipping_the_repeat_scan(case):
    """The flag is about work, not about the answer."""
    x, y, sel, K_x, K_y, K_sel = case
    checked = relax_mrmr_score(x, sel, y, K_x, K_sel, K_y, alpha=1.0)
    prechecked = relax_mrmr_score(x, sel, y, K_x, K_sel, K_y, alpha=1.0, selected_prechecked=True)
    assert checked == prechecked
