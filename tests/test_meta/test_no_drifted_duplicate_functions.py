"""Copies of one function must not drift apart.

A helper copy-pasted into several modules keeps working, so nothing forces the copies to stay in step and a
later fix reaches whichever ones the author happened to open. That is how eight modules of the
residual-band transformer cluster each ended up with their own ``_fit_baseline_predict``, four of them
returning honest out-of-fold predictions and four still fitting and predicting on the same rows -- mean
|residual| 0.2092 in-sample against 0.2968 out-of-fold, with 244 of 400 rows landing in a different
quintile band, and every column those modules emit derived from that judgement.

The groups below are the ones the check reports today. Each is a real near-duplicate rather than a false
positive, so this is a shrink list, not a baseline: the reason is recorded next to each name so that
removing one is a matter of doing the consolidation, and adding a name without a reason is visibly wrong.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# name -> why it is still a copy. Every entry is a consolidation candidate, tracked in
# audits/full_audit_2026-09-05/_TRACKER.md; none of them is a judged-legitimate duplicate.
KNOWN_DUPLICATE_GROUPS = {
    "_xgb_classifier_cls": "trainer.py kept a copy after the model-factory carve; differs only in local import aliases",
    "_xgb_regressor_cls": "same carve leftover as _xgb_classifier_cls",
    "_lgb_classifier_cls": "same carve leftover, plus a module-qualified LGBMClassifier reference",
    "_lgb_regressor_cls": "same carve leftover as _lgb_classifier_cls",
    "_fit_bgmm_and_sample": "bgmm_multiscale and bgmm_virtual fit the same mixture; not yet compared numerically",
    "_block_size": "three shap-proxy GPU modules size their blocks the same way",
    "_recover_cb_feature_names": "_predict_guards and cb/_cb_pool recover CatBoost feature names identically",
    "_agg_func_for_stat": "_composite_group_agg_fe and _grouped_agg_fe share the stat-to-aggregate mapping",
    "_global_value_for_stat": "the twin of _agg_func_for_stat, in the same two modules",
    "_frame_columns": "three composite-discovery modules read a frame's column list the same way",
    "_available_ram_bytes": "a filters module and a shap-proxy module probe free RAM the same way",
    "_inner_raw_margin": "composite classification and glm compute the same inner margin",
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
    assert not stale, f"these no longer drift and should be dropped from KNOWN_DUPLICATE_GROUPS: {stale}"


def test_every_recorded_group_carries_a_reason():
    """An entry without a reason is a baseline entry, which is how the original eight-copy group survived."""
    missing = sorted(name for name, reason in KNOWN_DUPLICATE_GROUPS.items() if not reason.strip())
    assert not missing, f"recorded without a reason: {missing}"
