"""PySR symbolic FE fits on the labelled train rows only and leaves the caller's frame as it was."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training._preprocessing_configs import PreprocessingExtensionsConfig


def test_pysr_is_fitted_on_the_labelled_rows(monkeypatch):
    """Pysr is fitted on the labelled rows."""
    import mlframe.feature_engineering.bruteforce as bruteforce
    from mlframe.training.pipeline import _pipeline_extensions_pysr as pysr_fe

    seen = {}

    class _Stop(Exception):
        """Sentinel raised to abort once the frame has been captured."""
        pass

    def _fake_run(df, target_col, **kwargs):
        """Capture the target values and row count, then abort."""
        seen["y"] = df[target_col].to_numpy().copy()
        seen["n"] = len(df)
        raise _Stop  # the fit itself is not under test; the helper logs it and returns []

    monkeypatch.setattr(bruteforce, "run_pysr_feature_engineering", _fake_run)
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    y = train["a"].to_numpy() * 2
    y[::4] = np.nan
    before = list(train.columns)
    pysr_fe._apply_pysr_fe(train_df=train, val_df=None, test_df=None, y_train=y, config=PreprocessingExtensionsConfig(pysr_enabled=True), verbose=0)
    assert "y" in seen, "the helper never reached run_pysr_feature_engineering"
    assert seen["n"] == 75 and np.isfinite(seen["y"]).all()
    assert list(train.columns) == before and len(train) == 100
