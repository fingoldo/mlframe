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


def test_pre_screen_per_frame_apply_drops_logs_on_failure(caplog):
    """When ``apply_drops`` raises on one frame, the WARN log must fire so
    operators see the schema-drift hazard. Pre-fix this was silently swallowed.

    Source-level guard: a single behavioural fixture would need a full suite
    setup; the warning-injection shape is fixed in code at L1127-1142 and the
    text is grepped here directly so future refactors can't silently revert.
    """
    import pathlib
    import mlframe as _mlframe
    src = (
        pathlib.Path(_mlframe.__file__).resolve().parent
        / "training" / "core" / "_phase_train_one_target.py"
    ).read_text(encoding="utf-8")
    # Pre-fix shape MUST be gone:
    assert "                        try:\n                            setattr(ctx, _frame_attr, apply_drops(_f, _drops))\n                        except Exception:\n                            pass" not in src, (
        "Pre-fix per-frame `except Exception: pass` reappeared; pre-screen "
        "failures on individual frames will silently corrupt the train/val/"
        "test/polars/pandas schema mirror (wave 16 P0 regression)."
    )
    # Post-fix marker (the WARN message text must be present):
    assert "apply_drops failed for ctx.%s" in src
    assert "schema drift hazard" in src


# ---- Site #2/#3: apply.py content-token fallbacks --------------------------


class _UnhashableTarget:
    """Object that breaks every reasonable hashing path so _target_content_token
    falls through to the except branch."""

    @property
    def shape(self):  # poisoned attr access used inside the hash path
        raise RuntimeError("synthetic hash failure")

    def __iter__(self):  # noqa: D401
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
        "two distinct target objects must get distinct tokens; pre-fix both "
        "returned 0 and target-1's fitted encoder replayed for target-2."
    )
    # WARN log fires so operators see the fallback path was taken.
    assert any(
        "_target_content_token: hash failed" in rec.message
        for rec in caplog.records
    ), f"expected WARN log; got: {[r.message for r in caplog.records]}"


def test_text_column_content_token_fallback_uses_id_not_zero(caplog):
    """Pre-fix the except branch returned literal 0 → distinct frames collided
    in the text-encoder cache. Post-fix: id-based fallback + WARN log."""
    from mlframe.training.feature_handling.apply import _text_column_content_token

    import pandas as pd

    class _ExplodingDf(pd.DataFrame):
        """Pandas subclass whose column access raises - forces the hash
        path into the except branch (the function recognises us as a
        pandas DataFrame via isinstance, then explodes on __getitem__)."""

        def __getitem__(self, key):  # noqa: D401
            raise RuntimeError(f"synthetic column access failure for {key!r}")

    bad1 = _ExplodingDf({"col_a": [1, 2, 3]})
    bad2 = _ExplodingDf({"col_a": [4, 5, 6]})
    with caplog.at_level(logging.WARNING, logger="mlframe.training.feature_handling.apply"):
        t1 = _text_column_content_token(bad1, "col_a")
        t2 = _text_column_content_token(bad2, "col_a")

    assert t1 != 0, "fallback must not return literal 0 (caches collide)"
    assert t2 != 0
    assert t1 != t2, (
        "two distinct DataFrames must get distinct tokens; pre-fix both "
        "returned 0 and one suite's fitted text encoder replayed for another."
    )
    assert any(
        "_text_column_content_token: hash failed" in rec.message
        for rec in caplog.records
    ), f"expected WARN log; got: {[r.message for r in caplog.records]}"


# ---- Site #4: target-encoders _objectwise_isnull ---------------------------


def test_objectwise_isnull_handles_normal_object_array():
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
        raise RuntimeError("synthetic listcomp failure")
    def _bad_isna(*args, **kwargs):
        raise RuntimeError("synthetic pandas.isna failure")
    monkeypatch.setattr(te.np, "array", _bad_array)
    monkeypatch.setattr(pd, "isna", _bad_isna)

    with pytest.raises(ValueError, match="Refusing to return an all-False mask"):
        te._objectwise_isnull(arr)


# ---- Site #5: preprocessing/transforms pd.NA clearing ---------------------


def test_transforms_extension_dtype_clear_logs_on_failure(caplog):
    """When the extension-dtype astype call fails, a WARN log must fire so
    operators see the pd.NA stayed in the frame (CatBoost will crash later)."""
    import pathlib
    import mlframe as _mlframe
    src = (
        pathlib.Path(_mlframe.__file__).resolve().parent
        / "preprocessing" / "transforms.py"
    ).read_text(encoding="utf-8")
    # Pre-fix shape MUST be gone:
    assert "df[var] = df[var].astype(target)\n                    except Exception:\n                        pass" not in src, (
        "Pre-fix `except Exception: pass` for pd.NA-clearing branch reappeared; "
        "silent dtype-cast failure leaves pd.NA in frame, CatBoost crashes later "
        "with no upstream signal (wave 16 P0 regression)."
    )
    assert "Could not convert extension-dtype column" in src
    assert "CatBoost cannot" in src
