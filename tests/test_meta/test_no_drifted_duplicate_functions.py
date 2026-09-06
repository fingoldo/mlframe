"""Copies of one function must not drift apart.

A helper copy-pasted into several modules keeps working, so nothing forces the copies to stay in step and a
later fix reaches whichever ones the author happened to open. That is how eight modules of the
residual-band transformer cluster each ended up with their own ``_fit_baseline_predict``, four of them
returning honest out-of-fold predictions and four still fitting and predicting on the same rows -- mean
|residual| 0.2092 in-sample against 0.2968 out-of-fold, with 244 of 400 rows landing in a different
quintile band, and every column those modules emit derived from that judgement.

The groups below are the ones the check reports today, and they are now of two kinds, each labelled:

* CONSOLIDATED -- the bodies already delegate to one shared implementation and differ only in the arguments
  they pass. The check still reports them because it groups by body SHAPE, and a parameterised wrapper looks
  like its siblings. This is the intended end state, not work left over.
* DELIBERATE -- copies that must NOT be merged, because something about the copy is load-bearing. Each names
  the test that fails if someone consolidates it anyway.

A name with neither label is a consolidation candidate. Adding a name without a reason is visibly wrong.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# name -> why it is still a copy. Every entry is a consolidation candidate, tracked in
# audits/full_audit_2026-09-05/_TRACKER.md; none of them is a judged-legitimate duplicate.
KNOWN_DUPLICATE_GROUPS = {
    # DELIBERATE -- trainer's pickers read USE_*_SHIM from THEIR OWN module, which is what makes
    # `monkeypatch.setattr(trainer, "USE_LGB_DATASET_REUSE_SHIM", False)` flip dispatch. Delegating to the
    # factory would silently detach that documented toggle; pinned by
    # tests/training/test_trainer_local_shim_pickers_are_deliberate.py.
    "_xgb_classifier_cls": "DELIBERATE: reads the trainer-local shim constant so the documented monkeypatch toggle works",
    "_xgb_regressor_cls": "DELIBERATE: same trainer-local binding as _xgb_classifier_cls",
    "_lgb_classifier_cls": "DELIBERATE: same trainer-local binding, plus a module-qualified LGBMClassifier reference",
    "_lgb_regressor_cls": "DELIBERATE: same trainer-local binding as _lgb_classifier_cls",
    # DELIBERATE -- already the right shape: two thin wrappers over the shared
    # composite/_booster_margin.inner_raw_margin, differing in every argument (classifier vs regressor
    # attrs, keep_2d True vs False).
    "_inner_raw_margin": "DELIBERATE: thin wrappers over the shared inner_raw_margin, differing in every argument",
    # CONSOLIDATED -- one implementation each; the wrappers differ only in what they pass to it.
    "_fit_bgmm_and_sample": "CONSOLIDATED into transformer/_bgmm_sample.fit_bgmm_and_sample; wrappers pass only the caller label",
    "_block_size": "CONSOLIDATED into shap_proxied_fs/_gpu_block_size; wrappers pass their own cache key, entry field and default",
    "_agg_func_for_stat": "CONSOLIDATED into filters/_agg_stat_helpers; wrappers pass their own valid-stat set (composite has `count`, grouped does not)",
    "_global_value_for_stat": "CONSOLIDATED into filters/_agg_stat_helpers, alongside _agg_func_for_stat",
    "_frame_columns": "CONSOLIDATED into composite/_frame_columns; the _incremental wrapper wraps the shared list in a set for O(1) membership",
}


def test_no_new_drifted_duplicate_functions():
    """Fail on any near-duplicate group beyond the ones recorded above.

    `_benchmarks` and the frozen `_cpx36_baseline` are excluded: a frozen copy is meant to keep the shape it
    was frozen with, which is the entire point of comparing against it.
    """
    from py_ci_shared.drifted_duplicate_functions import assert_no_drifted_duplicate_functions

    assert_no_drifted_duplicate_functions(
        [REPO_ROOT / "src"],
        exclude=("_benchmarks", "_cpx36_baseline"),
        allow=KNOWN_DUPLICATE_GROUPS,
    )


def test_the_recorded_groups_still_exist():
    """A name that no longer drifts must be removed from the list, or it stops being a shrink list.

    Without this the dict only ever grows: a consolidation would silently leave a dead entry behind, and the
    next reader could not tell which names are real work and which are archaeology.
    """
    from py_ci_shared.drifted_duplicate_functions import find_drifted_duplicate_functions

    reported = {g.name for g in find_drifted_duplicate_functions([REPO_ROOT / "src"], exclude=("_benchmarks", "_cpx36_baseline"))}
    stale = sorted(set(KNOWN_DUPLICATE_GROUPS) - reported)
    assert not stale, (
        f"these are no longer reported as near-duplicates: {stale}. Either they were consolidated -- in which "
        "case drop them from KNOWN_DUPLICATE_GROUPS -- or the copies DRIFTED APART far enough to stop matching, "
        "which is the failure this whole file exists to catch. Check which before editing the list."
    )


def test_every_recorded_group_carries_a_reason():
    """An entry without a reason is a baseline entry, which is how the original eight-copy group survived."""
    missing = sorted(name for name, reason in KNOWN_DUPLICATE_GROUPS.items() if not reason.strip())
    assert not missing, f"recorded without a reason: {missing}"
