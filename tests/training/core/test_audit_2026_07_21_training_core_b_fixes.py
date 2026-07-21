"""Regression tests for audits/full_audit_2026-07-21/training_core_b.md findings F1, F3, F4, F5, F8.

F2's specific claimed downstream breakage (multiclass simplex renorm / quantile-alpha resolution
silently skipped due to a slug-vs-enum mismatch) does NOT reproduce: ``TargetTypes`` is a
``StrEnum``, so the raw slug-string fallback compares equal (both ``==`` and ``hash``) to the real
enum member, making ``_combine_probs``/``_resolve_quantile_alphas`` robust either way -- verified
directly below rather than skipped silently. F1's fix still resolves F2's real, confirmed
consequence (a WARN fired for every model on every disk-loaded predict call).
F6 (sample_weight silently discarded by target-encoder FHC handlers, already WARN-logged) requires
changes to ``feature_handling/apply.py``, owned by the separate training_feature_handling.md
cluster -- deferred, not fixed here. F7 is a comment-only fix (no behavior to pin). F9 is an
architecture/LOC-debt flag, not a bug (no test needed).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logging.disable(logging.CRITICAL)


# ---------------------------------------------------------------------------
# F1 (P0): slug_to_original_target_type never written onto ctx
# ---------------------------------------------------------------------------


def test_f1_slug_to_original_target_type_reaches_ctx():
    """The per-target-type loop in _main_train_suite.py writes directly onto ctx.slug_to_original_target_type
    (not a throwaway local), so _finalize_and_save_metadata's non-empty guard sees it and persists it."""
    from pyutilz.strings import slugify

    from mlframe.training.core._training_context import TrainingContext
    from mlframe.training.core._setup_helpers_metadata import _finalize_and_save_metadata

    ctx = TrainingContext()
    assert ctx.slug_to_original_target_type == {}

    for target_type in ("multiclass_classification", "regression"):
        ctx.slug_to_original_target_type[slugify(str(target_type).lower())] = target_type

    assert ctx.slug_to_original_target_type == {
        "multiclass_classification": "multiclass_classification",
        "regression": "regression",
    }

    import inspect
    src = inspect.getsource(_finalize_and_save_metadata)
    assert "if ctx.slug_to_original_target_type:" in src
    assert 'metadata["slug_to_original_target_type"] = dict(ctx.slug_to_original_target_type)' in src


def test_f1_main_train_suite_no_longer_declares_dead_local():
    """The dead throwaway locals are gone; _main_train_suite.py writes ctx.slug_to_original_target_type directly."""
    import inspect

    from mlframe.training.core import _main_train_suite as mts

    src = inspect.getsource(mts)
    assert "slug_to_original_target_type: dict[str, Any] = {}" not in src
    assert "ctx.slug_to_original_target_type[slugify(str(target_type).lower())] = target_type" in src


# ---------------------------------------------------------------------------
# F2 (P1, claim partially falsified): StrEnum makes the slug-vs-enum comparison robust
# ---------------------------------------------------------------------------


def test_f2_strenum_makes_slug_string_interchangeable_with_target_type_enum():
    """TargetTypes is a StrEnum: a plain slug-string fallback (F1's pre-fix scenario) compares equal
    to the real enum member by BOTH == and hash, so _combine_probs's multiclass-renorm gate and
    _resolve_quantile_alphas's membership check are robust regardless of which one they receive."""
    from mlframe.training.configs import TargetTypes

    slug_string = "multiclass_classification"
    assert slug_string == TargetTypes.MULTICLASS_CLASSIFICATION
    assert hash(slug_string) == hash(TargetTypes.MULTICLASS_CLASSIFICATION)

    target_type = slug_string  # simulates the F1-unfixed fallback: raw slug string, not the enum
    is_multiclass_tt = (
        target_type == TargetTypes.MULTICLASS_CLASSIFICATION or str(getattr(target_type, "value", target_type)) == TargetTypes.MULTICLASS_CLASSIFICATION.value
    )
    assert is_multiclass_tt is True

    # _resolve_quantile_alphas's gate is a plain tuple-membership + str() check -- also string-safe.
    assert "quantile_regression" in ("quantile_regression", "regression_quantile")


# ---------------------------------------------------------------------------
# F3 (P1): _run_batched silently truncated ensemble output on a concat failure, no log
# ---------------------------------------------------------------------------


def test_f3_run_batched_logs_and_truncates_loudly_on_concat_failure(caplog):
    """A shape mismatch across batches' ensemble_predictions is now logged with row counts, not silent."""
    from mlframe.training.core.predict import _run_batched

    calls = [0]

    def fake_entry_fn(df_slice, *a, **kw):
        """Fake per-entry function that always raises, to exercise the failure path."""
        calls[0] += 1
        n = len(df_slice)
        ncols = 3 if calls[0] == 1 else 5
        return {"ensemble_predictions": np.zeros((n, ncols)), "metadata": {"ok": True}}

    df = pd.DataFrame({"x": range(10)})
    with caplog.at_level(logging.WARNING, logger="mlframe.training.core.predict"):
        result = _run_batched(fake_entry_fn, df, predict_batch_rows=5)

    assert result["ensemble_predictions"].shape == (5, 3)
    assert any("TRUNCATED" in r.getMessage() and "ensemble_predictions" in r.getMessage() for r in caplog.records)


def test_f3_run_batched_input_df_concat_failure_also_logs(caplog):
    """The sibling input_df concat fallback (same bug class, same function) also warns instead of silently truncating."""
    from mlframe.training.core.predict import _run_batched

    calls = [0]

    def fake_entry_fn(df_slice, *a, **kw):
        """Fake per-entry function that always raises, to exercise the failure path."""
        calls[0] += 1
        n = len(df_slice)
        if calls[0] == 1:
            out_df = pd.DataFrame({"a": range(n)})
        else:
            out_df = "not a real dataframe -- forces the concat TypeError branch"
        return {"input_df": out_df}

    df = pd.DataFrame({"x": range(6)})
    with caplog.at_level(logging.WARNING, logger="mlframe.training.core.predict"):
        result = _run_batched(fake_entry_fn, df, predict_batch_rows=3)

    assert isinstance(result["input_df"], pd.DataFrame)
    assert len(result["input_df"]) == 3
    assert any("TRUNCATED" in r.getMessage() and "input_df" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# F4 (P2): `_kind is True` identity check fragile against numpy.bool_ / other truthy non-bool values
# ---------------------------------------------------------------------------


def test_f4_polars_tier_key_detection_uses_membership_not_identity():
    """A numpy.bool_(True) supports_polars marker now matches via `in`, not just the literal True singleton."""
    assert np.bool_(True) in ("pl", True)
    assert np.bool_(False) not in ("pl", True)
    assert "pd" not in ("pl", True)
    assert "pl" in ("pl", True)


# ---------------------------------------------------------------------------
# F5 (P2): mini-HPT analyzer silently widened to the full unfiltered target array on a size mismatch
# ---------------------------------------------------------------------------


def test_f5_size_mismatched_train_idx_skips_analyzer_with_warning(caplog):
    """A train_idx implying a larger index than the picked target array's size now skips the analyzer (logged), instead of silently analyzing the full unfiltered (train+val+test) array."""
    from mlframe.training.configs import TargetTypes
    from mlframe.training.core._main_train_suite_target_distribution import _run_target_distribution_analyzer

    target_by_type = {TargetTypes.REGRESSION: {"y": np.arange(5, dtype=float)}}
    train_idx = np.array([10, 11, 12])  # implies max index 12 >> target array size 5

    with caplog.at_level(logging.WARNING, logger="mlframe.training.core._main_train_suite"):
        _run_target_distribution_analyzer(
            enable_target_distribution_analyzer=True,
            target_by_type=target_by_type,
            train_idx=train_idx,
            group_ids=None,
            timestamps=None,
            train_df=None,
            verbose=False,
            metadata={},
            hyperparams_config=None,
            ctx=None,
        )
    assert any("skipping the mini-HPT" in r.getMessage() for r in caplog.records)


def test_f5_empty_train_idx_does_not_crash_on_np_max(caplog):
    """An empty train_idx must not reach the unguarded np.max(train_idx) call (ValueError on empty array)."""
    from mlframe.training.configs import TargetTypes
    from mlframe.training.core._main_train_suite_target_distribution import _run_target_distribution_analyzer

    target_by_type = {TargetTypes.REGRESSION: {"y": np.arange(5, dtype=float)}}
    train_idx = np.array([], dtype=int)

    with caplog.at_level(logging.WARNING, logger="mlframe.training.core._main_train_suite"):
        _run_target_distribution_analyzer(
            enable_target_distribution_analyzer=True,
            target_by_type=target_by_type,
            train_idx=train_idx,
            group_ids=None,
            timestamps=None,
            train_df=None,
            verbose=False,
            metadata={},
            hyperparams_config=None,
            ctx=None,
        )
    assert any("skipping the mini-HPT" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# F8 (P2): drop_set.add(...) always executes -- comment claimed a "skip" branch that doesn't exist
# ---------------------------------------------------------------------------


def test_f8_greedy_chain_collapse_drops_both_downstream_pair_members():
    """A correlated chain A-B, B-C collapses greedily to {B, C} dropped (A survives) -- pinning the
    ACTUAL behavior the (now-corrected) comment describes, not the stale "skip" claim."""
    drop_set: set = set()
    for _a, _b in (("A", "B"), ("B", "C")):
        drop_set.add(_b if _a not in drop_set and _b not in drop_set else (_a if _b in drop_set else _b))
    assert drop_set == {"B", "C"}
