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


# ---- #1: per-class log_loss --------------------------------------------


# Monolith-split compat: ``_train_one_target`` body delegates to multiple
# siblings (body + ensembling tail + polars fastpath + pre-screen gate).
# Source-pattern sensors that grep the parent file must also read every
# sibling so they still match relocated code.
def _read_phase_train_one_target_combined():
    """Read phase train one target combined."""
    import pathlib
    import mlframe as _mlframe

    _core = pathlib.Path(_mlframe.__file__).resolve().parent / "training" / "core"
    return "\n".join(
        (_core / nm).read_text(encoding="utf-8")
        for nm in (
            "_phase_train_one_target.py",
            "_phase_train_one_target_body.py",
            "_phase_train_one_target_ensembling.py",
            "_phase_train_one_target_polars_fastpath.py",
            "_phase_train_one_target_pre_screen.py",
            "_phase_train_one_target_model_setup.py",
        )
        if (_core / nm).exists()
    )


def test_multilabel_log_loss_failed_class_warns_and_uses_nanmean(caplog):
    """Construct a closure mirroring the multilabel log-loss path with one
    failing class. Pre-fix, the failure was silently dropped and ``np.mean``
    ran over surviving classes. Post-fix: NaN appended, nanmean used,
    WARN-log fires."""
    K = 3

    def _ll_mock(y, p, labels):
        # Fail on class index 1, succeed otherwise.
        """Ll mock."""
        if y.shape[0] > 0 and y[0] == -1:
            raise ValueError("synthetic class-1 failure")
        return float(y.shape[0])

    # Inline the post-fix body so the sensor is independent of the live
    # ``fn`` factory which depends on a primary_metric branch.
    yi = np.array(
        [
            [0, -1, 1],
            [1, -1, 0],
        ]
    )
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

    _baselines = pathlib.Path(_mlframe.__file__).resolve().parent / "training" / "baselines"
    src = "\n".join(p.read_text(encoding="utf-8") for p in sorted(_baselines.glob("*.py")))
    assert "multilabel log-loss: %d/%d class component(s) failed" in src, "Wave 16 P1 regression: per-class log_loss WARN log shape gone."
    assert "return float(np.nanmean(losses))" in src, (
        "Wave 16 P1 regression: still using mean() instead of nanmean() over per-class log-loss; biased average over surviving classes returns."
    )


# ---- #2: _has_any_infinity -------------------------------------------------


def test_has_any_infinity_returns_true_on_detection_failure(caplog, monkeypatch):
    """When inf-detection fails entirely (the float-column helper raises an
    unexpected error), the detector must return True (force the fix_infinities
    path) + WARN -- pre-fix it silently returned False so infs reached XGB/HGB."""
    import pandas as pd
    from mlframe.training import preprocessing as pp

    def _boom(_d):
        """Boom."""
        raise RuntimeError("synthetic detection failure")

    monkeypatch.setattr(pp, "_pandas_float_like_columns", _boom)
    with caplog.at_level(logging.WARNING, logger="mlframe.training.preprocessing"):
        result = pp._frame_contains_inf(pd.DataFrame({"f0": [1.0, 2.0, 3.0]}))
    assert result is True
    assert any("detection failed" in r.getMessage() for r in caplog.records), (
        f"expected WARN on detection failure; got {[r.getMessage() for r in caplog.records]}"
    )


def test_has_any_infinity_true_on_numpy_conversion_failure(caplog, monkeypatch):
    """Inner branch: a float-like column whose ``to_numpy`` raises must also
    force True + WARN (nullable-Float coercion failure must not pass infs)."""
    import pandas as pd
    from mlframe.training import preprocessing as pp

    df = pd.DataFrame({"f0": [1.0, 2.0, 3.0]})

    class _BadNum:
        """Groups tests covering bad num."""
        shape = (3, 1)

        def to_numpy(self, *a, **k):
            """To numpy."""
            raise ValueError("synthetic numpy conversion failure")

    monkeypatch.setattr(pp, "_pandas_float_like_columns", lambda d: ["f0"])
    real_getitem = pd.DataFrame.__getitem__
    monkeypatch.setattr(
        pd.DataFrame,
        "__getitem__",
        lambda self, key: _BadNum() if isinstance(key, list) else real_getitem(self, key),
    )
    with caplog.at_level(logging.WARNING, logger="mlframe.training.preprocessing"):
        result = pp._frame_contains_inf(df)
    assert result is True
    assert any("numpy conversion failed" in r.getMessage() for r in caplog.records)


# ---- #3: detect_time/group skips -------------------------------------------


def test_detect_group_column_candidates_logs_per_skip(caplog):
    """A column whose access raises emits a DEBUG skip line (group detector), so operators
    can see why a candidate was bypassed instead of it vanishing silently."""
    import pandas as pd
    from mlframe.training.composite.discovery import auto_detect as cad

    class _BadGet(pd.DataFrame):
        """Groups tests covering bad get."""
        def __getitem__(self, key):
            if key == "boom":
                raise RuntimeError("synthetic get_col failure")
            return super().__getitem__(key)

    df = _BadGet({"boom": [1, 2, 3, 4, 5, 6], "ok": [1, 1, 2, 2, 3, 3]})
    with caplog.at_level(logging.DEBUG, logger="mlframe.training.composite.discovery.auto_detect"):
        result = cad.detect_group_column_candidates(df, candidate_columns=["boom", "ok"])
    assert all(name != "boom" for name, _ in result)
    assert any("detect_group_column_candidates: skipping col=" in r.message for r in caplog.records), (
        f"expected DEBUG skip log; got: {[r.message for r in caplog.records]}"
    )


def test_detect_skip_emits_debug_when_col_raises(caplog):
    """Behavioural: a column that raises on dtype access emits a DEBUG line."""
    from mlframe.training.composite.discovery import auto_detect as cad
    import pandas as pd

    class _BadCol(pd.DataFrame):
        """pandas DataFrame whose __getitem__ raises -> dtype access fails."""

        def __getitem__(self, key):
            if isinstance(key, str):
                raise RuntimeError(f"synthetic dtype failure for {key!r}")
            return super().__getitem__(key)

    df = _BadCol({"col_x": [1.0, 2.0, 3.0]})
    with caplog.at_level(logging.DEBUG, logger="mlframe.training.composite.discovery.auto_detect"):
        result = cad.detect_time_column_candidates(df)
    assert result == []
    assert any("detect_time_column_candidates: skipping col=" in rec.message for rec in caplog.records), (
        f"expected DEBUG log; got: {[r.message for r in caplog.records]}"
    )


# ---- #4: composite_screening CV fold -------------------------------------


def test_composite_screening_failed_fold_warns(caplog):
    """Failed CV fold returns NaN AND emits a WARN so operators see the
    effective fold count is reduced. The WARN-emitting block moved from
    composite_screening.py to the sibling _composite_screening_tiny.py
    during the screening monolith split; check both locations."""
    import pathlib
    import mlframe as _mlframe

    root = pathlib.Path(_mlframe.__file__).resolve().parent / "training" / "composite" / "discovery"
    candidates = [root / "screening.py", root / "_screening_tiny.py"]
    src_combined = ""
    for p in candidates:
        if p.exists():
            src_combined += p.read_text(encoding="utf-8")
            src_combined += "\n"
    assert "composite_screening: tiny-model CV fold failed" in src_combined, (
        "Wave 16 P1 regression: failed CV fold no longer emits a WARN; Screening RMSE silently biased toward well-behaved folds."
    )


# ---- #5: clone() fallback -------------------------------------------------


def test_clone_failure_warns_with_pipeline_type(caplog):
    """sklearn.clone fallback path WARN-logs the pipeline type."""
    src = _read_phase_train_one_target_combined()
    assert "sklearn.clone failed for base_pipeline" in src
    # Message wraps at `reusing ` / `original reference.` -- match either piece.
    assert "reusing " in src and "original reference" in src
    # Pre-fix shape (silent pass) MUST be gone in this surface.
    assert "except Exception:\n                    # Non-BaseEstimator" not in src


# ---- #6: multilabel CB label_arr stack -----------------------------------


def test_multilabel_cb_stack_failure_warns(caplog):
    """A ragged object-array label set makes the row-stack raise; pre-fix this
    silently set label_arr=None (single-label default loss) with no signal.
    Post-fix: WARN + the model is NOT reconfigured to MultiLogloss."""
    from mlframe.training._training_loop import _ensure_cb_multilabel_loss

    set_calls = []

    class CatBoostClassifier:  # name drives the type-name gate in prod
        """Groups tests covering cat boost classifier."""
        def get_param(self):
            """Get param."""
            return {"loss_function": None}

        def set_params(self, **kw):
            """Set params."""
            set_calls.append(kw)

    # Ragged rows -> np.array(obj.tolist()) raises -> stack-failure branch.
    ragged = np.empty(2, dtype=object)
    ragged[0] = np.array([1, 0])
    ragged[1] = np.array([1, 0, 1])

    model = CatBoostClassifier()
    with caplog.at_level(logging.WARNING, logger="mlframe.training._training_loop"):
        _ensure_cb_multilabel_loss(model, ragged)
    assert any("failed to stack label rows" in r.getMessage() for r in caplog.records)
    # MultiLogloss must NOT have been configured (label_arr stayed None).
    assert all("MultiLogloss" not in str(c) for c in set_calls)


def test_multilabel_cb_stack_success_sets_multilogloss(caplog):
    """Control: a uniform-width 2D object-array stacks cleanly and DOES configure
    MultiLogloss -- proves the WARN branch above is the failure path, not a no-op."""
    from mlframe.training._training_loop import _ensure_cb_multilabel_loss

    set_calls = []

    class CatBoostClassifier:
        """Groups tests covering cat boost classifier."""
        def get_param(self):
            """Get param."""
            return {"loss_function": None}

        def set_params(self, **kw):
            """Set params."""
            set_calls.append(kw)

    target = np.array([[1, 0], [0, 1], [1, 1]])
    _ensure_cb_multilabel_loss(CatBoostClassifier(), target)
    assert any(c.get("loss_function") == "MultiLogloss" for c in set_calls)
