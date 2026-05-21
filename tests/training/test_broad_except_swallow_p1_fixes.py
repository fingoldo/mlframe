"""Wave-16 P1 sensors: silent broad-except swallows that biased metrics or hid
configuration drops.

Six sites where ``except Exception: pass`` / ``continue`` / ``return DEFAULT``
in a non-P0 surface still degraded observability or metric integrity:

1. ``dummy_baselines.py:2460`` -- per-class log_loss components silently
   dropped from ``np.mean(losses)``; reported multilabel log-loss was a
   biased average over surviving classes. Post-fix: failures kept as NaN,
   nanmean instead of mean, WARN log naming the failed classes.

2. ``preprocessing.py:197`` -- ``_has_any_infinity`` returned False on
   detection failure. Caller's fix_infinities branch was skipped, infs
   reached XGB/HGB and crashed downstream. Post-fix: return True on
   failure (force fix_infinities path) + WARN log.

3. ``composite_auto_detect.py:85, 211`` -- ``detect_time_column_candidates``
   / ``detect_group_column_candidates`` silently skipped cols whose dtype
   or array access raised. Post-fix: DEBUG log per skip so the trail is
   there but normal scans aren't WARN-spammed.

4. ``composite_screening.py:668, 993`` -- failed CV fold returned NaN
   silently; mean-aggregation over reduced K_eff with no signal. Post-fix:
   WARN-log per failed fold; NaN return preserved (caller uses nanmean).

5. ``_phase_train_one_target.py:1638-1642`` -- ``sklearn.clone(_base_for_strategy)``
   failure silently kept the original reference, sharing a (potentially
   partially-fit) selector across strategies. Post-fix: WARN log naming
   the pipeline type so operators see the fallback fired.

6. ``_training_loop.py:62-65`` -- ``np.stack`` failure on object-array
   labels silently set ``label_arr = None``, skipping MultiLogloss /
   HammingLoss configuration; CB trained with single-label default. Post-fix:
   WARN log so operators see why MultiLogloss wasn't picked.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest


# ---- #1: per-class log_loss --------------------------------------------


# 2026-05-21 monolith split: ``_train_one_target`` body lives in
# ``_phase_train_one_target_body.py``; source-pattern sensors that grep the
# parent file must also read the body sibling. Resolves the core/ dir from
# the installed package so it works regardless of where pytest is invoked.
def _read_phase_train_one_target_combined():
    import pathlib
    import mlframe as _mlframe
    _core = pathlib.Path(_mlframe.__file__).resolve().parent / "training" / "core"
    return (
        (_core / "_phase_train_one_target.py").read_text(encoding="utf-8")
        + "\n"
        + (_core / "_phase_train_one_target_body.py").read_text(encoding="utf-8")
    )



def test_multilabel_log_loss_failed_class_warns_and_uses_nanmean(caplog):
    """Construct a closure mirroring the multilabel log-loss path with one
    failing class. Pre-fix, the failure was silently dropped and ``np.mean``
    ran over surviving classes. Post-fix: NaN appended, nanmean used,
    WARN-log fires."""
    K = 3

    def _ll_mock(y, p, labels):  # noqa: D401
        # Fail on class index 1, succeed otherwise.
        if y.shape[0] > 0 and y[0] == -1:
            raise ValueError("synthetic class-1 failure")
        return float(y.shape[0])

    # Inline the post-fix body so the sensor is independent of the live
    # ``fn`` factory which depends on a primary_metric branch.
    yi = np.array([
        [0, -1, 1],
        [1, -1, 0],
    ])
    pi = np.zeros_like(yi, dtype=float)

    losses: list[float] = []
    failed: list[tuple[int, str]] = []
    for k in range(K):
        try:
            losses.append(float(_ll_mock(yi[:, k], pi[:, k], labels=[0, 1])))
        except Exception as _e:
            failed.append((k, str(_e)))
            losses.append(float("nan"))

    # 2 surviving classes (k=0, k=2) each report y.shape[0]=2.
    assert losses[0] == 2.0
    assert np.isnan(losses[1])
    assert losses[2] == 2.0
    assert len(failed) == 1 and failed[0][0] == 1
    # nanmean over [2.0, nan, 2.0] = 2.0; biased mean over [2.0, 2.0] would
    # also = 2.0 but in a real failure the surviving class subset isn't
    # representative (the WARN log makes the bias visible).
    assert float(np.nanmean(losses)) == 2.0

    # Source-level guard that the live function uses the post-fix idiom.
    # The multilabel log-loss helper moved to the ``_dummy_bootstrap.py``
    # sibling during the dummy_baselines monolith split; the parent + every
    # sibling is searched so the WARN-log shape sensor stays valid.
    import pathlib
    import mlframe as _mlframe
    _train = pathlib.Path(_mlframe.__file__).resolve().parent / "training"
    _files = [_train / "dummy_baselines.py", *_train.glob("_dummy_*.py")]
    src = "\n".join(p.read_text(encoding="utf-8") for p in _files if p.exists())
    assert "multilabel log-loss: %d/%d class component(s) failed" in src, (
        "Wave 16 P1 regression: per-class log_loss WARN log shape gone."
    )
    assert "return float(np.nanmean(losses))" in src, (
        "Wave 16 P1 regression: still using mean() instead of nanmean() over "
        "per-class log-loss; biased average over surviving classes returns."
    )


# ---- #2: _has_any_infinity -------------------------------------------------


def test_has_any_infinity_returns_true_on_detection_failure(caplog):
    """Pre-fix returned False on detection failure -> infs reached XGB/HGB.
    Post-fix returns True (force the fix_infinities path) + WARN log."""
    import pathlib
    import mlframe as _mlframe
    src = (
        pathlib.Path(_mlframe.__file__).resolve().parent
        / "training" / "preprocessing.py"
    ).read_text(encoding="utf-8")
    # Both branches (inner specific + outer broad) must force True + WARN.
    assert "returning True to force the " in src, (
        "Wave 16 P1 regression: _has_any_infinity reverted to silent False "
        "return on detection failure; infs will reach XGB/HGB with no signal."
    )
    # The pre-fix outer shape (silent broad-except return False) must be gone.
    assert "except Exception:\n        return False" not in src


# ---- #3: detect_time/group skips -------------------------------------------


def test_detect_time_column_candidates_logs_per_skip(caplog):
    """Per-col skip emits DEBUG so operators can see why a col was bypassed."""
    import pathlib
    import mlframe as _mlframe
    src = (
        pathlib.Path(_mlframe.__file__).resolve().parent
        / "training" / "composite_auto_detect.py"
    ).read_text(encoding="utf-8")
    assert 'detect_time_column_candidates: skipping col=%r' in src
    assert 'detect_group_column_candidates: skipping col=%r' in src


def test_detect_skip_emits_debug_when_col_raises(caplog):
    """Behavioural: a column that raises on dtype access emits a DEBUG line."""
    from mlframe.training import composite_auto_detect as cad
    import pandas as pd

    class _BadCol(pd.DataFrame):
        """pandas DataFrame whose __getitem__ raises -> dtype access fails."""
        def __getitem__(self, key):
            if isinstance(key, str):
                raise RuntimeError(f"synthetic dtype failure for {key!r}")
            return super().__getitem__(key)

    df = _BadCol({"col_x": [1.0, 2.0, 3.0]})
    with caplog.at_level(logging.DEBUG, logger="mlframe.training.composite_auto_detect"):
        result = cad.detect_time_column_candidates(df)
    assert result == []
    assert any(
        "detect_time_column_candidates: skipping col=" in rec.message
        for rec in caplog.records
    ), f"expected DEBUG log; got: {[r.message for r in caplog.records]}"


# ---- #4: composite_screening CV fold -------------------------------------


def test_composite_screening_failed_fold_warns(caplog):
    """Failed CV fold returns NaN AND emits a WARN so operators see the
    effective fold count is reduced. The WARN-emitting block moved from
    composite_screening.py to the sibling _composite_screening_tiny.py
    during the screening monolith split; check both locations."""
    import pathlib
    import mlframe as _mlframe
    root = pathlib.Path(_mlframe.__file__).resolve().parent / "training"
    candidates = [root / "composite_screening.py", root / "_composite_screening_tiny.py"]
    src_combined = ""
    for p in candidates:
        if p.exists():
            src_combined += p.read_text(encoding="utf-8")
            src_combined += "\n"
    assert "composite_screening: tiny-model CV fold failed" in src_combined, (
        "Wave 16 P1 regression: failed CV fold no longer emits a WARN; "
        "Screening RMSE silently biased toward well-behaved folds."
    )


# ---- #5: clone() fallback -------------------------------------------------


def test_clone_failure_warns_with_pipeline_type(caplog):
    """sklearn.clone fallback path WARN-logs the pipeline type."""
    import pathlib
    import mlframe as _mlframe
    src = _read_phase_train_one_target_combined()
    assert "sklearn.clone failed for base_pipeline" in src
    # Message wraps at `reusing ` / `original reference.` -- match either piece.
    assert "reusing " in src and "original reference" in src
    # Pre-fix shape (silent pass) MUST be gone in this surface.
    assert "except Exception:\n                    # Non-BaseEstimator" not in src


# ---- #6: multilabel CB label_arr stack -----------------------------------


def test_multilabel_cb_stack_failure_warns(caplog):
    """np.stack failure on object-array labels warns; pre-fix label_arr=None
    silently bypassed MultiLogloss config."""
    import pathlib
    import mlframe as _mlframe
    src = (
        pathlib.Path(_mlframe.__file__).resolve().parent
        / "training" / "_training_loop.py"
    ).read_text(encoding="utf-8")
    assert "multilabel CB auto-config: failed to stack label rows" in src
    assert "single-label default loss" in src
