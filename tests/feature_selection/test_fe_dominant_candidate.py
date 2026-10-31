"""A column that carries all of y stops the FE cascades; noisy targets do not."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._fe_dominant_candidate import dominant_candidate


def _frame(n=20_000, seed=0):
    """Five independent normal columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(5)})


def test_column_that_determines_target_is_dominant():
    """A column that decides y is reported."""
    X = _frame()
    y = (X["x2"].to_numpy() > 0).astype(int)
    assert dominant_candidate(X, y) == "x2"


def test_noisy_target_is_not_dominant():
    """A noisy target has no dominant column."""
    X = _frame()
    rng = np.random.default_rng(1)
    y = ((X["x2"].to_numpy() + rng.normal(size=len(X))) > 0).astype(int)
    assert dominant_candidate(X, y) is None


def test_small_frame_never_dominant():
    """Frames below the row minimum never qualify."""
    X = _frame(n=1_000)
    assert dominant_candidate(X, (X["x0"].to_numpy() > 0).astype(int)) is None


def test_switch_off(monkeypatch):
    """The environment switch disables the check."""
    monkeypatch.setenv("MLFRAME_FE_DOMINANT_STOP", "0")
    X = _frame()
    assert dominant_candidate(X, (X["x0"].to_numpy() > 0).astype(int)) is None


def test_cascade_skips_later_stages(monkeypatch):
    """A dominant column after early_a skips early_b."""
    from mlframe.feature_selection.filters._mrmr_fit_impl import _fe_stage_cascade_run as R

    called = []
    monkeypatch.setattr(R, "_fe_stage_cascade_early_b", lambda *a, **k: called.append(1))
    X = _frame()
    y = (X["x1"].to_numpy() > 0).astype(int)
    monkeypatch.setattr(R, "_fe_stage_cascade_early_a", lambda self, X, **k: (X, list(X.columns), None, None))

    class _Self:
        """Estimator stub."""

        degenerate_columns_ = {}

    class _Rec:
        """Recipes stub: every attribute is an empty dict."""

        def __getattr__(self, name):
            """Empty recipe dict."""
            return {}

    _, raw = R._run_fe_stage_cascades(_Self(), X, y, 0, 1, y, None, None, None, _Rec())
    assert not called and list(raw) == list(X.columns)
