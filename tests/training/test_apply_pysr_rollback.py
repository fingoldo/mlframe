"""Sensor: PySR equation predict failures must NOT leave train_df with a column
that val_df / test_df don't have (schema drift would crash downstream fit with a
cryptic feature-count mismatch).

Pre-fix: the per-equation loop in _apply_pysr_fe wrapped train+val+test column
assignments in one ``try ... except Exception: continue``. If predict succeeded
on train but raised on val, train_df kept ``pysr__<hash>__<seed>``, val_df
didn't, and ``new_cols`` never appended the column -- downstream code thought
the column was absent yet train_df.columns contained it.

Post-fix: any predict failure rolls back the column from EVERY frame where it
was already written, logs a warning, and continues to the next equation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd


class _FakePySRModel:
    """Stand-in for a fitted PySRRegressor: ``equations_`` plus a ``predict(df, index)`` that can be told to fail on one frame size."""

    def __init__(self, n_equations: int, *, fail_rows: int | None = None, fail_index: int | None = None):
        """Build ``n_equations`` equations (descending score) and the optional (frame size, equation index) failure trigger."""
        self.equations_ = pd.DataFrame(
            {"equation": [f"eq_{i}" for i in range(n_equations)], "score": [float(n_equations - i) for i in range(n_equations)], "complexity": 1}
        )
        self.feature_names_in_ = np.array(["x0", "x1"], dtype=object)
        self.fail_rows = fail_rows
        self.fail_index = fail_index

    def predict(self, df, index=None):
        """Predict ``index`` as a constant column; raise for the configured (frame size, equation index) pair."""
        if self.fail_rows is not None and len(df) == self.fail_rows and index == self.fail_index:
            raise ValueError(f"simulated PySR predict failure on a {len(df)}-row frame at equation idx={index}")
        return np.full(len(df), float(index), dtype=np.float32)


def _frames():
    """Train / val / test frames of 100 / 40 / 50 rows sharing the two feature columns."""
    return tuple(pd.DataFrame({"x0": np.arange(n, dtype=np.float32), "x1": np.arange(n, dtype=np.float32)}) for n in (100, 40, 50))


def _run_apply(monkeypatch, model, train_df, val_df, test_df, out_equations):
    """Run ``_apply_pysr_fe`` with ``run_pysr_feature_engineering`` replaced by one returning ``model``."""
    from mlframe.feature_engineering import bruteforce
    from mlframe.training.configs import PreprocessingExtensionsConfig
    from mlframe.training.pipeline._pipeline_extensions_pysr import _apply_pysr_fe

    monkeypatch.setattr(bruteforce, "run_pysr_feature_engineering", lambda **kwargs: model)
    return _apply_pysr_fe(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        y_train=np.arange(len(train_df), dtype=np.float64),
        config=PreprocessingExtensionsConfig(pysr_enabled=True),
        verbose=0,
        out_equations=out_equations,
    )


def _pysr_cols(df):
    """Names of the PySR-derived columns of ``df``."""
    return {c for c in df.columns if c.startswith("pysr__")}


def test_pysr_per_equation_predict_failure_rolls_back_all_frames(monkeypatch, caplog):
    """An equation whose predict fails on val is rolled back from train, val and test; the others stay uniformly present."""
    from mlframe.utils.log_throttle import reset_throttle_counts

    train_df, val_df, test_df = _frames()
    out_equations: dict = {}
    reset_throttle_counts("pysr_equation_skipped")
    with caplog.at_level(logging.WARNING, logger="mlframe.training.pipeline"):
        new_cols = _run_apply(monkeypatch, _FakePySRModel(3, fail_rows=40, fail_index=0), train_df, val_df, test_df, out_equations)

    assert len(new_cols) == 2
    assert _pysr_cols(train_df) == _pysr_cols(val_df) == _pysr_cols(test_df) == set(new_cols)
    assert set(out_equations) == set(new_cols)
    assert sorted(out_equations.values()) == ["eq_1", "eq_2"]
    assert any("rolled back to keep splits schema-consistent" in r.getMessage() and "idx=0" in r.getMessage() for r in caplog.records)


def test_pysr_all_equations_failing_leaves_no_orphan_columns(monkeypatch):
    """When every equation fails on val, no PySR column survives on any frame and nothing is reported as added."""
    train_df, val_df, test_df = _frames()
    model = _FakePySRModel(1, fail_rows=40, fail_index=0)
    new_cols = _run_apply(monkeypatch, model, train_df, val_df, test_df, {})
    assert new_cols == []
    assert _pysr_cols(train_df) == _pysr_cols(val_df) == _pysr_cols(test_df) == set()
    assert list(train_df.columns) == ["x0", "x1"]


def test_pysr_all_succeed_baseline_no_rollback(monkeypatch, caplog):
    """When no equation fails nothing is rolled back: every column lands on all three frames carrying its own equation index."""
    train_df, val_df, test_df = _frames()
    out_equations: dict = {}
    with caplog.at_level(logging.WARNING, logger="mlframe.training.pipeline"):
        new_cols = _run_apply(monkeypatch, _FakePySRModel(3), train_df, val_df, test_df, out_equations)

    assert len(new_cols) == 3
    assert _pysr_cols(train_df) == _pysr_cols(val_df) == _pysr_cols(test_df) == set(new_cols)
    assert sorted(out_equations.values()) == ["eq_0", "eq_1", "eq_2"]
    for col in new_cols:
        idx = int(out_equations[col].split("_")[1])
        assert (train_df[col] == idx).all() and (val_df[col] == idx).all() and (test_df[col] == idx).all()
    assert not any("rolled back" in r.getMessage() for r in caplog.records)
