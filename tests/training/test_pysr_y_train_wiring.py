"""Wiring test: ``pysr_enabled=True`` must actually invoke PySR.

Regression for a silent-skip bug surfaced 2026-05-15: the
``train_mlframe_models_suite`` call chain didn't thread ``y_train`` into
``apply_preprocessing_extensions``, so ``_apply_pysr_fe`` silently returned
``[]`` on the first ``if y_train is None: return []`` line. User configured
``preprocessing_extensions=PreprocessingExtensionsConfig(pysr_enabled=True)``,
expected PySR to fire, saw nothing in logs, traced 30+ minutes only to find
the wiring gap.

After the fix:
    - ``_phase_fit_pipeline`` receives ``target_by_type`` and extracts a 1-D
      y_train from the first regression target.
    - ``apply_preprocessing_extensions`` receives ``y_train=`` and forwards
      to ``_apply_pysr_fe``.
    - ``_apply_pysr_fe`` with ``y_train=None`` now ``logger.warning``s
      instead of silent-skipping, so future wiring breaks fail loudly.

This test:
    1. Direct unit-level: ``apply_preprocessing_extensions(pysr_enabled=True,
       y_train=None)`` emits the expected warning (so silent-skips can't
       sneak back in).
    2. Direct unit-level: ``apply_preprocessing_extensions(pysr_enabled=True,
       y_train=<arr>)`` invokes the PySR code path. We skip the full Julia
       fit by mocking ``run_pysr_feature_engineering`` -- the test verifies
       the WIRING (was it called?), not PySR's symbolic-regression quality.
"""

from __future__ import annotations

import logging
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from mlframe.training.pipeline import apply_preprocessing_extensions
from mlframe.training.configs import PreprocessingExtensionsConfig


def _make_frames(n=200, p=4, seed=42):
    rng = np.random.default_rng(seed)
    X_train = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    X_val = pd.DataFrame(rng.normal(size=(n // 2, p)), columns=[f"x{i}" for i in range(p)])
    X_test = pd.DataFrame(rng.normal(size=(n // 2, p)), columns=[f"x{i}" for i in range(p)])
    y = X_train["x0"] ** 2 + 0.5 * X_train["x1"] + 0.1 * rng.normal(size=n)
    return X_train, X_val, X_test, y


# ---------------------------------------------------------------------------
# 1. Without y_train, the function must warn loudly (silent-skip detector)
# ---------------------------------------------------------------------------

def test_pysr_enabled_without_y_train_warns_loudly(caplog):
    """If pysr_enabled=True but y_train=None, the user MUST see a warning.
    Silent-skip used to mask the wiring bug we hit on 2026-05-15."""
    X_train, X_val, X_test, _ = _make_frames()
    config = PreprocessingExtensionsConfig(
        pysr_enabled=True,
        pysr_params={"niterations": 1, "population_size": 5},
    )

    with caplog.at_level(logging.WARNING):
        out_train, out_val, out_test, pipeline = apply_preprocessing_extensions(
            train_df=X_train, val_df=X_val, test_df=X_test,
            config=config, verbose=1,
            y_train=None,   # <-- explicitly None
        )

    warned = any("y_train" in r.message and "pysr_enabled" in r.message
                 for r in caplog.records)
    assert warned, (
        "Silent-skip regression: apply_preprocessing_extensions must warn "
        "when pysr_enabled=True but y_train is not provided. caplog: "
        f"{[r.message for r in caplog.records]}"
    )
    # And no new columns were added.
    assert list(out_train.columns) == list(X_train.columns)


# ---------------------------------------------------------------------------
# 2. With y_train, the PySR code path is reached
# ---------------------------------------------------------------------------

def test_pysr_enabled_with_y_train_calls_pysr():
    """When y_train is passed, run_pysr_feature_engineering must be invoked.

    We don't run the full Julia fit (too slow / Julia not always present in
    CI); we monkey-patch the import to a stub that records the call and
    returns a minimal fake model whose predict() returns zeros.
    """
    X_train, X_val, X_test, y = _make_frames()
    config = PreprocessingExtensionsConfig(
        pysr_enabled=True,
        pysr_params={"niterations": 1, "population_size": 5},
    )

    # Build a minimal stub: model.equations_ has 1 row (so eq_df is non-empty);
    # model.predict returns zeros.
    fake_model = MagicMock()
    fake_eq_df = pd.DataFrame({"score": [1.0], "complexity": [3], "loss": [0.1]})
    fake_eq_df.index = [0]
    fake_model.equations_ = fake_eq_df
    fake_model.predict = MagicMock(return_value=np.zeros(len(X_train)))

    call_log = []

    def fake_run_pysr(df, target_col, sample_size, encode_categoricals, verbose, pysr_params_override):
        call_log.append({
            "n_rows": len(df),
            "target_col": target_col,
            "has_target_col_in_df": target_col in df.columns,
        })
        # Wire predict() to match the split it's called on.
        def _pred(_df, index=0):
            return np.zeros(len(_df))
        fake_model.predict.side_effect = _pred
        return fake_model

    with patch("mlframe.feature_engineering.bruteforce.run_pysr_feature_engineering",
               side_effect=fake_run_pysr):
        out_train, out_val, out_test, pipeline = apply_preprocessing_extensions(
            train_df=X_train, val_df=X_val, test_df=X_test,
            config=config, verbose=0,
            y_train=np.asarray(y),
        )

    # 1. The stub was called -> PySR code path reached
    assert len(call_log) == 1, (
        "run_pysr_feature_engineering was not invoked despite pysr_enabled=True "
        "AND y_train provided. The wiring is broken."
    )
    # 2. The target was injected as a column
    assert call_log[0]["has_target_col_in_df"], (
        "PySR fit got a frame without the target column injected"
    )
    # 3. Temp target column removed after fit
    assert "_pysr_y_" not in out_train.columns


# ---------------------------------------------------------------------------
# 3. pysr_enabled=False is a clean no-op (no warning, no new cols)
# ---------------------------------------------------------------------------

def test_pysr_disabled_is_silent(caplog):
    X_train, X_val, X_test, _ = _make_frames()
    config = PreprocessingExtensionsConfig(pysr_enabled=False)
    with caplog.at_level(logging.WARNING):
        out_train, *_ = apply_preprocessing_extensions(
            train_df=X_train, val_df=X_val, test_df=X_test,
            config=config, verbose=1,
        )
    assert list(out_train.columns) == list(X_train.columns)
    # No PySR-related warnings expected
    assert not any("pysr" in r.message.lower() for r in caplog.records)
