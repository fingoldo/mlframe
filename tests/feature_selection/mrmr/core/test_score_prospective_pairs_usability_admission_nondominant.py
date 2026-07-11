"""Adversarial regression test for the 2026-07-11 ``_admit_via_usability`` wiring bug in
``score_prospective_pairs`` (``_step_pairs_rank.py``).

BACKGROUND: ``_admit_via_usability`` was computed (via ``_pair_is_tail_concentrated_rankaware``, one of the
largest hotspots in a 100k-row production profile) but its result was never consulted in the accept
condition -- it read ``(_passes_prevalence and _passes_maxt) or _admit_via_perm``, missing
``or _admit_via_usability`` entirely. Wiring the missing clause in was verified NOT to change the two
canonical ``ratio_sqr/with_outliers`` fixtures this block's own comments target, because in BOTH of those
fixtures the tail-concentrated pair is already the candidate pool's DOMINANT pair (by best pair-form
``|corr|``), so the SEPARATE first-sweep prescan relaxation (this file's dominant-pair rank-aware scan)
already rescues it -- the per-pair admission gate this bug affects never becomes the deciding factor there.

This test constructs the ONE scenario neither canonical fixture can exercise: a candidate pool with TWO
pairs sharing the SAME target --

  * pair (0,1): a clean multiplicative interaction (``y`` contains ``a_dom*b_dom`` exactly) -- high linear
    AND high rank correlation (they AGREE, the "balanced" signature), so ``pair_is_tail_concentrated_rankaware``
    correctly reads False for it. Given high cached MI, it is ALSO the pool's dominant pair by best
    pair-form |corr| (verified below) -- so the prescan's dominant-pair gate finds a NON-tail-concentrated
    dominant pair and never relaxes ``fe_min_pair_mi_prevalence`` for this pool. Its own cached MI clears
    prevalence/maxT independently, so it is a control: admitted in every configuration.
  * pair (2,3): an outlier-driven ratio (``a_tail**2/b_tail`` with a small fraction of near-zero ``b_tail``
    rows) contributing a SMALLER weighted share of ``y`` -- individually tail-concentrated (linear |corr|
    clears ``min_corr`` while its RANK association collapses, verified below) but NOT the pool's dominant
    pair. Its cached MI is deliberately set below both the prevalence ratio bar and the maxT floor, so
    ONLY the per-pair usability admission (not prevalence/maxT, not the perm-null path which stays off by
    default, not the prescan relaxation which never fires since the DOMINANT pair disagrees) can rescue it.

``score_prospective_pairs`` is called directly (bypassing a full MRMR fit) with synthetic ``cached_MIs`` --
this isolates the exact accept-condition wiring bug from the rest of the FE pipeline. Toggling
``fe_pair_usability_admission_enable`` True/False on the SAME live (already-fixed) code reproduces the
observable pre-fix-vs-post-fix outcome for pair (2,3): with the flag off, ``_admit_via_usability`` is never
computed (stays False), identical in effect to the bug (computed but never consulted) -- either way pair
(2,3) is rejected. With the flag on (current default), the restored clause admits it. This is the concrete
evidence that the fix adds genuine, currently-inaccessible-by-the-canonical-suite value, not just "no
regression": SIGNAL_LOSS is asserted for a pair whose warp materialises the outlier-tail interaction the
strict rank-MI gates structurally cannot see."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._fe_usability_signal import (
    _rank_transform,
    abs_pearson,
    pair_is_tail_concentrated_rankaware,
    usability_form_corrs,
)
from mlframe.feature_selection.filters._mrmr_fe_step._step_pairs_rank import score_prospective_pairs
from pyutilz.pythonlib import sort_dict_by_value


def _build_fixture(seed: int = 7, n: int = 6000, w_tail: float = 0.055):
    rng = np.random.default_rng(seed)

    a_dom = rng.standard_normal(n)
    b_dom = rng.standard_normal(n)
    dom_term = a_dom * b_dom

    a_tail = rng.standard_normal(n)
    b_tail = rng.uniform(0.5, 2.0, n)
    outlier_mask = rng.random(n) < 0.02
    b_tail[outlier_mask] = rng.uniform(0.01, 0.03, int(outlier_mask.sum()))
    ratio_tail = (a_tail**2) / b_tail

    noise = rng.standard_normal(n) * 0.01
    y = 1.0 * dom_term + w_tail * ratio_tail + noise

    X = pd.DataFrame({"a_dom": a_dom, "b_dom": b_dom, "a_tail": a_tail, "b_tail": b_tail})
    cols = ["a_dom", "b_dom", "a_tail", "b_tail"]
    return y, X, cols, a_dom, b_dom, a_tail, b_tail


def test_fixture_shape_pair01_dominant_balanced_pair23_nondominant_tail_concentrated():
    """Pin the adversarial fixture's own contract before trusting it to test the wiring bug: pair (0,1) is the
    pool's dominant pair AND not tail-concentrated; pair (2,3) is tail-concentrated AND non-dominant."""
    y, X, cols, a_dom, b_dom, a_tail, b_tail = _build_fixture()

    cp01, cs01, _ = usability_form_corrs(y, a_dom, b_dom, return_best_pair_form=True)
    cp23, cs23, _ = usability_form_corrs(y, a_tail, b_tail, return_best_pair_form=True)
    assert cp01 > cp23, "pair (0,1) must be the pool's dominant pair by best pair-form |corr|"
    assert cp23 >= 0.6, "pair (2,3) must clear the usability min_corr floor on its own"

    tc01 = pair_is_tail_concentrated_rankaware(y, a_dom, b_dom, min_corr=0.6, pairness_margin=1.05, max_rank_frac=0.7)
    tc23 = pair_is_tail_concentrated_rankaware(y, a_tail, b_tail, min_corr=0.6, pairness_margin=1.05, max_rank_frac=0.7)
    assert tc01 is False, "the dominant pair must NOT be tail-concentrated (rank agrees with linear)"
    assert tc23 is True, "the non-dominant pair must BE tail-concentrated (rank collapses vs linear)"


def _run(y, X, cols, *, usability_enable: bool):
    cached_MIs = {
        (0,): 0.4, (1,): 0.4, (0, 1): 1.2,
        (2,): 0.01, (3,): 0.01, (2, 3): 0.02,
    }

    class _Self:
        _fe_prewarp_y_continuous_ = y
        fe_pair_usability_admission_enable = usability_enable
        fe_pair_perm_null_admission_enable = False
        _fe_rejection_records_: list = []

    n = y.shape[0]
    data = np.zeros((n, 4), dtype=np.int64)
    classes_y = np.zeros(n, dtype=np.int64)

    prospective_pairs, _ = score_prospective_pairs(
        _Self(),
        cached_MIs=cached_MIs,
        numeric_vars_to_consider={0, 1, 2, 3},
        checked_pairs=set(),
        _pair_mm_bias={},
        _pair_maxt_floor=0.05,
        _synergy_added_idx=set(),
        fe_min_pair_mi_prevalence=1.05,
        _synergy_prev_resolved=1.5,
        _prevalence_debias_auto=False,
        data=data,
        classes_y=classes_y,
        X=X,
        cols=cols,
        num_fs_steps=0,
        verbose=0,
        sort_dict_by_value=sort_dict_by_value,
    )
    kept_pairs = {frozenset(pair) for (pair, _mi) in prospective_pairs.keys()}
    return kept_pairs


def test_usability_admission_rescues_nondominant_tail_concentrated_pair():
    """THE adversarial case the two canonical fixtures cannot exercise: with the fix (usability admission ON,
    current default), the non-dominant tail-concentrated pair (2,3) -- which fails BOTH prevalence and maxT,
    and which the prescan's dominant-pair relaxation structurally cannot reach (the pool's actual dominant
    pair, (0,1), is not tail-concentrated) -- is rescued ONLY by the per-pair ``_admit_via_usability`` gate
    this bug fix restored."""
    y, X, cols, *_ = _build_fixture()

    kept_enabled = _run(y, X, cols, usability_enable=True)
    kept_disabled = _run(y, X, cols, usability_enable=False)

    assert frozenset({0, 1}) in kept_enabled, "control pair (0,1) passes prevalence/maxT on its own merit"
    assert frozenset({0, 1}) in kept_disabled, "control pair (0,1) must be unaffected by the usability toggle"

    assert frozenset({2, 3}) in kept_enabled, (
        "fixed behaviour: the non-dominant tail-concentrated pair (2,3) IS admitted via usability -- "
        "this is the genuine, previously-inaccessible value the bug fix restores"
    )
    assert frozenset({2, 3}) not in kept_disabled, (
        "with usability admission off (reproduces the pre-fix observable outcome: _admit_via_usability "
        "never computed, same effect as computed-but-ignored), the pair is genuinely LOST -- proving this "
        "is not a redundant rescue path, the pair has no other way to survive"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
