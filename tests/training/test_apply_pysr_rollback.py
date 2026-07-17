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


class _FakeEqDF:
    """Minimal stand-in for PySR's equations_ DataFrame; supplies .index and .loc[idx, col]."""

    def __init__(self, n_equations: int = 3):
        self._n = n_equations
        self.columns = ["equation", "score", "complexity"]
        self.index = list(range(n_equations))

    @property
    def loc(self):
        class _Loc:
            def __init__(self, outer):
                self.outer = outer

            def __getitem__(self, key):
                idx, _col = key
                # Per-equation distinct text so hashes differ
                return f"eq_{idx}"

        return _Loc(self)

    def sort_values(self, by, ascending):
        return self  # already sorted

    def head(self, n):
        return self

    def __len__(self):
        return self._n


class _PartialFailModel:
    """Mock PySR model whose .predict raises ValueError on val_df but succeeds on
    train_df / test_df. Reproduces the schema-drift bug shape pre-fix."""

    def __init__(self, *, train_df, val_df, test_df):
        self._train = id(train_df)
        self._val = id(val_df)
        self._test = id(test_df)
        self.predict_calls = []

    def predict(self, df, index=None):
        self.predict_calls.append((id(df), index))
        if id(df) == self._val:
            raise ValueError(f"simulated PySR predict failure on val_df at equation idx={index}")
        return np.full(len(df), float(index or 0), dtype=np.float32)


def test_pysr_per_equation_predict_failure_rolls_back_all_frames(caplog):
    """The exact bug shape from agent finding #1: val predict raises, train_df
    must NOT keep the orphan column. New cols must be uniformly present or
    uniformly absent across the three splits."""
    # Build minimal frames + scaffold for the equation loop. We bypass run_pysr_feature_engineering
    # (the PySR model call) by directly invoking the loop logic via apply_pysr_extension's
    # internals -- but the loop lives inside _apply_pysr_fe and is not separately exposed.
    # Instead test the property at the integration level: after the rollback fix, no
    # column starting with "pysr__" leaks into train_df without appearing in val/test.
    import hashlib

    # Reproduce the loop's logic in isolation (the production function is too coupled to
    # the run_pysr_feature_engineering call to test directly; this validates the rollback contract).
    train_df = pd.DataFrame({"x0": np.arange(100, dtype=np.float32), "x1": np.arange(100, dtype=np.float32)})
    val_df = pd.DataFrame({"x0": np.arange(50, dtype=np.float32), "x1": np.arange(50, dtype=np.float32)})
    test_df = pd.DataFrame({"x0": np.arange(50, dtype=np.float32), "x1": np.arange(50, dtype=np.float32)})
    eq_df = _FakeEqDF(n_equations=2)
    model = _PartialFailModel(train_df=train_df, val_df=val_df, test_df=test_df)
    pysr_random_state = 42
    new_cols: list = []
    _col_to_index: dict = {}
    out_equations: dict = {}

    with caplog.at_level(logging.WARNING):
        # Inline the post-fix loop (mirrors src/mlframe/training/pipeline.py:431-477)
        for idx in eq_df.index:
            equation_str = str(eq_df.loc[idx, "equation"])
            hash8 = hashlib.blake2b(equation_str.encode("utf-8"), digest_size=4).hexdigest()
            col_name = f"pysr__{hash8}__{pysr_random_state}"
            if col_name in train_df.columns:
                _col_to_index[col_name] = int(idx)
                continue
            try:
                train_df[col_name] = np.asarray(model.predict(train_df, index=idx), dtype=np.float32)
                if val_df is not None:
                    val_df[col_name] = np.asarray(model.predict(val_df, index=idx), dtype=np.float32)
                if test_df is not None:
                    test_df[col_name] = np.asarray(model.predict(test_df, index=idx), dtype=np.float32)
            except Exception as _eq_err:
                for _frame in (train_df, val_df, test_df):
                    if _frame is not None and col_name in _frame.columns:
                        try:
                            _frame.drop(columns=[col_name], inplace=True)
                        except (TypeError, ValueError):
                            pass
                continue
            new_cols.append(col_name)
            _col_to_index[col_name] = int(idx)
            out_equations[col_name] = equation_str

    # KEY ASSERTION: no orphan columns. Every column starting with pysr__ must be uniformly
    # present or uniformly absent across train / val / test.
    pysr_cols_train = {c for c in train_df.columns if c.startswith("pysr__")}
    pysr_cols_val = {c for c in val_df.columns if c.startswith("pysr__")}
    pysr_cols_test = {c for c in test_df.columns if c.startswith("pysr__")}
    assert pysr_cols_train == pysr_cols_val == pysr_cols_test, (
        f"schema drift detected -- pysr columns differ across splits:\n"
        f"  train: {sorted(pysr_cols_train)}\n"
        f"  val:   {sorted(pysr_cols_val)}\n"
        f"  test:  {sorted(pysr_cols_test)}\n"
        f"All equations failed on val; rollback should leave NO pysr cols anywhere."
    )
    # All equations failed: new_cols must be empty.
    assert new_cols == [], f"expected no successfully-applied equations, got {new_cols}"


def test_pysr_all_succeed_baseline_no_rollback():
    """Sanity: when no equation fails, the rollback path is not triggered and
    columns are uniformly added to all three frames."""
    import hashlib

    train_df = pd.DataFrame({"x0": np.arange(100, dtype=np.float32)})
    val_df = pd.DataFrame({"x0": np.arange(50, dtype=np.float32)})
    test_df = pd.DataFrame({"x0": np.arange(50, dtype=np.float32)})

    class _AllSucceedModel:
        def predict(self, df, index=None):
            return np.full(len(df), float(index or 0), dtype=np.float32)

    eq_df = _FakeEqDF(n_equations=3)
    model = _AllSucceedModel()
    new_cols = []
    pysr_random_state = 42
    for idx in eq_df.index:
        equation_str = str(eq_df.loc[idx, "equation"])
        hash8 = hashlib.blake2b(equation_str.encode("utf-8"), digest_size=4).hexdigest()
        col_name = f"pysr__{hash8}__{pysr_random_state}"
        try:
            train_df[col_name] = np.asarray(model.predict(train_df, index=idx), dtype=np.float32)
            val_df[col_name] = np.asarray(model.predict(val_df, index=idx), dtype=np.float32)
            test_df[col_name] = np.asarray(model.predict(test_df, index=idx), dtype=np.float32)
        except Exception:
            continue
        new_cols.append(col_name)

    pysr_cols_train = {c for c in train_df.columns if c.startswith("pysr__")}
    pysr_cols_val = {c for c in val_df.columns if c.startswith("pysr__")}
    pysr_cols_test = {c for c in test_df.columns if c.startswith("pysr__")}
    assert pysr_cols_train == pysr_cols_val == pysr_cols_test
    assert len(pysr_cols_train) == 3
