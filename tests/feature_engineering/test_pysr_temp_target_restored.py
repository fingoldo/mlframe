"""Regression sensor for S03: ``_apply_pysr_fe`` injects ``_pysr_y_`` as a
temporary target column on the caller-passed ``train_df`` and only drops it in
a ``finally`` block. The injection happens BEFORE the ``try`` block, so an
exception raised by ``np.asarray(y_train).ravel()`` (e.g. ragged y, dtype
issues) - or any failure between the injection point and the ``try`` entry -
leaks the temp column back to the caller's frame, where it then poisons
downstream feature engineering as a fake numeric feature.

The fix is to move the column injection INSIDE the try block (so the ``finally``
clause unconditionally drops it on any exception, including AttributeError /
ValueError thrown by the injection itself OR by PySR / bruteforce later) and
to confirm the column does not leak when the PySR call raises.

This sensor MUST FAIL on the pre-fix code (column leaks) and PASS post-fix
(column is restored).
"""

from __future__ import annotations


import numpy as np
import pandas as pd


def _install_stub_run_pysr_feature_engineering(monkeypatch, raise_with):
    """Replace ``mlframe.feature_engineering.bruteforce.run_pysr_feature_engineering``
    with a stub that always raises ``raise_with``. ``_apply_pysr_fe`` imports
    the symbol lazily from the bruteforce module each call, so patching the
    source module routes the next call through the stub.
    """
    import mlframe.feature_engineering.bruteforce as _bf

    def _raising_stub(*_args, **_kwargs):
        raise raise_with

    monkeypatch.setattr(_bf, "run_pysr_feature_engineering", _raising_stub)


class _MinimalConfig:
    """Minimal duck-typed config object for ``_apply_pysr_fe``."""

    random_seed = 42
    pysr_sample_size = None
    pysr_top_k = None
    pysr_operator_preset = "standard"
    pysr_params_override = None


def test_pysr_temp_target_column_restored_when_pysr_raises(monkeypatch):
    """When ``run_pysr_feature_engineering`` raises, the ``_pysr_y_``
    temporary column MUST NOT leak back into the caller's ``train_df``.

    Pre-fix the injection at L178 is OUTSIDE the try block: ``np.asarray``
    failures or any other exception between injection and the try entry
    would leak the column. The fix moves injection INSIDE the try so the
    ``finally`` unconditionally drops it for EVERY exit path.
    """
    from mlframe.training.pipeline._pipeline_extensions import _apply_pysr_fe

    _install_stub_run_pysr_feature_engineering(
        monkeypatch,
        raise_with=RuntimeError("simulated PySR failure"),
    )

    train_df = pd.DataFrame(
        {
            "x0": np.arange(100, dtype=np.float32),
            "x1": np.arange(100, dtype=np.float32),
        }
    )
    val_df = pd.DataFrame(
        {
            "x0": np.arange(50, dtype=np.float32),
            "x1": np.arange(50, dtype=np.float32),
        }
    )
    test_df = pd.DataFrame(
        {
            "x0": np.arange(50, dtype=np.float32),
            "x1": np.arange(50, dtype=np.float32),
        }
    )
    y_train = np.arange(100, dtype=np.float32)
    cols_before = list(train_df.columns)

    new_cols = _apply_pysr_fe(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        y_train=y_train,
        config=_MinimalConfig(),
        out_transformer=None,
        verbose=False,
        out_equations=None,
    )

    assert new_cols == [], "PySR failure should return no new columns"
    assert "_pysr_y_" not in train_df.columns, (
        "S03: _apply_pysr_fe leaked _pysr_y_ into caller's train_df after "
        "PySR raised; the temp target column must always be restored. "
        f"Columns after: {list(train_df.columns)}"
    )
    assert list(train_df.columns) == cols_before, f"S03: train_df.columns changed after PySR failure: before={cols_before} after={list(train_df.columns)}"


def test_pysr_temp_target_column_restored_when_random_seed_cast_raises(monkeypatch):
    """Cover the genuine pre-fix leak window: the column is injected at L178
    BEFORE the try block. Any exception fired between injection and the try
    entry (line 183 ``int(getattr(config, "random_seed", 42))`` is the
    narrow but real window) bypasses the ``finally`` and the temp column
    leaks back to the caller's frame.

    The fix moves the injection INSIDE the try block, so the ``finally``
    drops the temp column for EVERY exit path including this one.
    """
    from mlframe.training.pipeline._pipeline_extensions import _apply_pysr_fe

    # Stub PySR so we never reach it; the test exercises the pre-PySR path.
    _install_stub_run_pysr_feature_engineering(
        monkeypatch,
        raise_with=RuntimeError("should not reach PySR"),
    )

    class _BadSeedConfig:
        """``random_seed`` is a string that survives ``getattr`` but blows up
        inside ``int(...)`` -- exact pre-fix leak window."""

        random_seed = "not_an_int"
        pysr_sample_size = None
        pysr_top_k = None
        pysr_operator_preset = "standard"
        pysr_params_override = None

    train_df = pd.DataFrame(
        {
            "x0": np.arange(100, dtype=np.float32),
            "x1": np.arange(100, dtype=np.float32),
        }
    )
    val_df = pd.DataFrame(
        {
            "x0": np.arange(50, dtype=np.float32),
        }
    )
    test_df = pd.DataFrame(
        {
            "x0": np.arange(50, dtype=np.float32),
        }
    )
    y_train = np.arange(100, dtype=np.float32)
    cols_before = list(train_df.columns)

    # Whether the call raises or returns [] depends on where exactly the
    # exception fires relative to the try/except boundary. Both are
    # acceptable; what matters is that train_df is restored either way.
    try:
        result = _apply_pysr_fe(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            y_train=y_train,
            config=_BadSeedConfig(),
            out_transformer=None,
            verbose=False,
            out_equations=None,
        )
        # Post-fix path: the bad int() cast is inside the try, caught by
        # ``except Exception``, function returns []. Pre-fix path would
        # have raised here instead AND left the column leaked.
        assert result == []
    except Exception:  # nosec B110 -- best-effort cleanup/optional step; failure here never masks this test's own assertions
        # Pre-fix path also acceptable as a possible behaviour shape; the
        # invariant under test is the no-leak guarantee below.
        pass

    assert "_pysr_y_" not in train_df.columns, (
        "S03 (random-seed-cast leak window): _apply_pysr_fe leaked _pysr_y_ "
        "into caller's train_df when an exception fired between injection "
        "and the try block. Injection must happen INSIDE try/finally. "
        f"Columns after: {list(train_df.columns)}"
    )
    assert list(train_df.columns) == cols_before, f"S03 (random-seed-cast): train_df.columns changed: before={cols_before} after={list(train_df.columns)}"
