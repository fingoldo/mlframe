"""Wave-16 P0 sensors: silent broad-except swallows that masked data corruption.

Four sites where ``except Exception: pass`` (no logger.warning, no re-raise)
silently degraded production behaviour:

1. ``_phase_train_one_target.py:1127-1130`` -- per-frame ``apply_drops`` in the
   pre-screen loop swallowed failure silently. If ONE frame's ``apply_drops``
   raised, that frame kept the pre-screen-dropped columns while siblings had
   them removed; downstream training hit "feature missing" with no signal.

2. ``feature_handling/apply.py:552`` -- ``_input_content_token`` returned a
   literal ``0`` on hash failure. The ``InMemoryKey`` then collided across
   distinct train frames, replaying one suite's fitted encoder for another
   suite's data.

3. ``feature_handling/apply.py:583`` -- ``_target_content_token`` returned a
   literal ``0`` on hash failure. The target-mean / WoE encoder cache slot
   collided across distinct targets in a multi-target suite, reusing target-1's
   fitted state for target-2.

4. ``feature_handling/target_encoders.py:143`` -- ``_objectwise_isnull``
   returned an all-False mask on per-row introspection failure, so the target
   encoder silently treated None / NaN rows as VALID encoded values, averaging
   their non-numeric content into the per-category mean / WoE.

5. ``preprocessing/transforms.py:200`` -- the pd.NA-clearing branch silently
   failed when the extension-dtype cast raised, leaving pd.NA in the frame for
   CatBoost to crash on later.

Each sensor exercises the bug shape AND asserts the post-fix observable
(WARN-log fires / no all-False mask / id-based fallback / no zero token).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest


# ---- Site #1: pre-screen per-frame apply_drops ------------------------------


def test_pre_screen_apply_drops_failure_is_atomic_no_partial_drop(caplog, monkeypatch):
    """A per-mirror ``apply_drops`` failure must leave EVERY frame untouched (atomic), not partially dropped.

    The pre-screen stages all mirror drops into a dict and only reassigns ctx attributes once every mirror
    succeeds; a single failure raises before any reassignment, the outer except skips the whole pre-screen and
    logs it. Behavioural assertion: inject a failure on the second mirror and confirm no frame lost its dropped
    column (rollback by construction), no columns are recorded as dropped, and the skip is logged.
    """
    import pandas as pd
    from types import SimpleNamespace
    import mlframe.feature_selection.pre_screen as _ps
    from mlframe.training.core._phase_train_one_target_pre_screen import _maybe_run_unsupervised_pre_screen

    real_apply_drops = _ps.apply_drops
    calls = {"n": 0}

    def flaky_apply_drops(frame, drops):
        """Flaky apply drops."""
        calls["n"] += 1
        if calls["n"] == 2:  # second mirror fails
            raise RuntimeError("synthetic per-mirror apply_drops failure")
        return real_apply_drops(frame, drops)

    monkeypatch.setattr(_ps, "apply_drops", flaky_apply_drops)

    train_df = pd.DataFrame({"good": np.arange(20.0), "const": np.ones(20)})
    val_df = pd.DataFrame({"good": np.arange(10.0), "const": np.ones(10)})
    fs_cfg = SimpleNamespace(
        pre_screen_unsupervised=True,
        pre_screen_variance_threshold=0.0,
        pre_screen_null_fraction_threshold=0.99,
    )
    ctx = SimpleNamespace(
        feature_selection_config=fs_cfg,
        _pre_screen_done=False,
        _pre_screen_dropped_cols=None,
        target_by_type={},
        cat_features=None,
        text_features=None,
        embedding_features=None,
        group_id_col=None,
        ts_field=None,
        extractor=None,
        features_and_targets_extractor=None,
        split_config=None,
        filtered_train_df=train_df,
        filtered_val_df=val_df,
        train_df_pd=None,
        val_df_pd=None,
        test_df_pd=None,
        train_df_polars=None,
        val_df_polars=None,
        test_df_polars=None,
        verbose=0,
        metadata={},
    )

    with caplog.at_level(logging.WARNING, logger="mlframe.training.core._phase_train_one_target"):
        _maybe_run_unsupervised_pre_screen(ctx, {})

    # Atomic rollback: NO frame may have lost the 'const' column despite the train mirror's drop succeeding first.
    assert list(ctx.filtered_train_df.columns) == ["good", "const"], "train mirror was mutated despite a sibling-mirror failure -> non-atomic partial drop"
    assert list(ctx.filtered_val_df.columns) == ["good", "const"], "val mirror was mutated despite the apply_drops failure"
    # No columns recorded as dropped, and the skip-on-error path must log.
    assert ctx._pre_screen_dropped_cols == []
    assert any("skipped due to error" in r.message for r in caplog.records), "the atomic-failure path must log that the pre-screen was skipped"


# ---- Site #2/#3: apply.py content-token fallbacks --------------------------


class _UnhashableTarget:
    """Object that breaks every reasonable hashing path so _target_content_token
    falls through to the except branch."""

    @property
    def shape(self):  # poisoned attr access used inside the hash path
        """Shape."""
        raise RuntimeError("synthetic hash failure")

    def __iter__(self):
        raise RuntimeError("not iterable either")


def test_target_content_token_fallback_uses_id_not_zero(caplog):
    """Pre-fix the except branch returned literal 0 → all targets collide in
    the InMemoryKey cache. Post-fix: id-based fallback + WARN log."""
    from mlframe.training.feature_handling.apply import _target_content_token

    t1 = _UnhashableTarget()
    t2 = _UnhashableTarget()
    with caplog.at_level(logging.WARNING, logger="mlframe.training.feature_handling.apply"):
        token1 = _target_content_token(t1)
        token2 = _target_content_token(t2)

    assert token1 != 0, "fallback must not return literal 0 (caches collide)"
    assert token2 != 0
    assert token1 != token2, (
        "two distinct target objects must get distinct tokens; pre-fix both returned 0 and target-1's fitted encoder replayed for target-2."
    )
    # WARN log fires so operators see the fallback path was taken.
    assert any("_target_content_token: hash failed" in rec.message for rec in caplog.records), f"expected WARN log; got: {[r.message for r in caplog.records]}"


def test_text_column_content_token_fallback_uses_id_not_zero(caplog):
    """Pre-fix the except branch returned literal 0 → distinct frames collided
    in the text-encoder cache. Post-fix: id-based fallback + WARN log."""
    from mlframe.training.feature_handling.apply import _text_column_content_token

    import pandas as pd

    class _ExplodingDf(pd.DataFrame):
        """Pandas subclass whose column access raises - forces the hash
        path into the except branch (the function recognises us as a
        pandas DataFrame via isinstance, then explodes on __getitem__)."""

        def __getitem__(self, key):
            raise RuntimeError(f"synthetic column access failure for {key!r}")

    bad1 = _ExplodingDf({"col_a": [1, 2, 3]})
    bad2 = _ExplodingDf({"col_a": [4, 5, 6]})
    with caplog.at_level(logging.WARNING, logger="mlframe.training.feature_handling.apply"):
        t1 = _text_column_content_token(bad1, "col_a")
        t2 = _text_column_content_token(bad2, "col_a")

    assert t1 != 0, "fallback must not return literal 0 (caches collide)"
    assert t2 != 0
    assert t1 != t2, "two distinct DataFrames must get distinct tokens; pre-fix both returned 0 and one suite's fitted text encoder replayed for another."
    assert any("_text_column_content_token: hash failed" in rec.message for rec in caplog.records), (
        f"expected WARN log; got: {[r.message for r in caplog.records]}"
    )


# ---- Site #4: target-encoders _objectwise_isnull ---------------------------


def test_objectwise_isnull_handles_normal_object_array():
    """Objectwise isnull handles normal object array."""
    from mlframe.training.feature_handling.target_encoders import _objectwise_isnull

    arr = np.array([1.0, None, float("nan"), "x"], dtype=object)
    mask = _objectwise_isnull(arr)
    np.testing.assert_array_equal(mask, [False, True, True, False])


def test_objectwise_isnull_falls_back_to_pandas_isna(monkeypatch):
    """When per-row ``is None`` ufunc fails, the function must fall back to
    pandas.isna - NOT return an all-False mask that silently treats null rows
    as valid (the pre-fix bug)."""
    from mlframe.training.feature_handling import target_encoders as te

    # Build the input BEFORE monkeypatching so the helper's per-row listcomp
    # is the only te.np.array() consumer affected.
    arr = np.array([1.0, None, float("nan"), "x"], dtype=object)

    real_array = te.np.array
    armed = {"first": True}

    def _array(*args, **kwargs):
        """Array."""
        if armed["first"]:
            armed["first"] = False
            raise RuntimeError("synthetic listcomp failure")
        return real_array(*args, **kwargs)

    monkeypatch.setattr(te.np, "array", _array)

    mask = te._objectwise_isnull(arr)
    # pandas.isna catches None AND NaN, so the True count must be >= 2.
    assert mask.sum() >= 2, (
        f"fallback should NOT return all-False mask; got {mask} with True count "
        f"{mask.sum()}. Pre-fix this silently treated null rows as valid "
        f"target-encoder inputs."
    )


def test_objectwise_isnull_raises_when_both_paths_fail(monkeypatch):
    """If BOTH per-row and pandas.isna fail, the function MUST raise -- not
    silently return an all-False mask. Loud failure > silent corruption."""
    from mlframe.training.feature_handling import target_encoders as te
    import pandas as pd

    # Construct input BEFORE patching so the helper's np.array is the only
    # call that hits the synthetic failure.
    arr = np.array([1.0, None], dtype=object)

    def _bad_array(*args, **kwargs):
        """Bad array."""
        raise RuntimeError("synthetic listcomp failure")

    def _bad_isna(*args, **kwargs):
        """Bad isna."""
        raise RuntimeError("synthetic pandas.isna failure")

    monkeypatch.setattr(te.np, "array", _bad_array)
    monkeypatch.setattr(pd, "isna", _bad_isna)

    with pytest.raises(ValueError, match="Refusing to return an all-False mask"):
        te._objectwise_isnull(arr)


# ---- Site #5: preprocessing/transforms pd.NA clearing ---------------------


def test_transforms_extension_dtype_clear_logs_on_failure(caplog, monkeypatch):
    """When the extension-dtype astype call fails, a WARN log must fire so
    operators see the pd.NA stayed in the frame (CatBoost will crash later).
    Pre-fix this branch swallowed the failure silently."""
    import pandas as pd
    from mlframe.preprocessing import transforms as tr

    df = pd.DataFrame({"x": pd.array([1, 2, None], dtype="Int64")})

    real_astype = pd.Series.astype

    def _flaky_astype(self, dtype, *a, **k):
        """Flaky astype."""
        if dtype in (np.float32, np.float64):
            raise RuntimeError("synthetic extension-dtype cast failure")
        return real_astype(self, dtype, *a, **k)

    monkeypatch.setattr(pd.Series, "astype", _flaky_astype)

    with caplog.at_level(logging.WARNING, logger="mlframe.preprocessing.transforms"):
        out = tr.prepare_df_for_catboost(df)
    # The failed cast leaves the column an extension dtype (pd.NA still present).
    assert pd.api.types.is_extension_array_dtype(out["x"].dtype)
    assert any("Could not convert extension-dtype column" in r.getMessage() for r in caplog.records), (
        f"expected WARN; got {[r.getMessage() for r in caplog.records]}"
    )
