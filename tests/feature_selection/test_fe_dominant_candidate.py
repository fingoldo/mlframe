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
    """A dominant column after early_a skips early_b (the sequential cascade; the concurrent one starts every stage, see the next test)."""
    from mlframe.feature_selection.filters._mrmr_fit_impl import _fe_stage_cascade_run as R

    monkeypatch.setenv("MLFRAME_FE_STAGE_CONCURRENCY", "0")
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


def test_concurrent_cascade_keeps_only_early_a_on_a_dominant_candidate(monkeypatch):
    """Run concurrently, a dominant column from early_a returns early_a's frame only; a later stage that fails is ignored, as a sequential run never ran it."""
    from mlframe.feature_selection.filters._mrmr_fit_impl import _fe_stage_cascade_run as R
    from mlframe.feature_selection.filters._mrmr_fit_impl import _fe_stage_concurrency as C
    import mlframe.feature_selection.filters._fe_device_scheduler as DS

    monkeypatch.setenv("MLFRAME_FE_STAGE_CONCURRENCY", "1")
    monkeypatch.setenv("MLFRAME_FE_STAGE_CONCURRENCY_EARLY_A", "1")
    monkeypatch.setattr(DS, "plan_stage_slots", lambda weights: [None] * len(weights))
    import contextlib

    monkeypatch.setattr(DS, "stage_device_slot", lambda slot, i: contextlib.nullcontext())
    X = _frame()
    y = (X["x1"].to_numpy() > 0).astype(int)
    monkeypatch.setattr(R, "_fe_stage_cascade_early_a", lambda self, X, **k: (X, list(X.columns), None, None))

    def _boom(*a, **k):
        """A later stage that fails."""
        raise RuntimeError("later stage")

    for name in ("_fe_stage_cascade_early_b", "_fe_stage_cascade_mid_a", "_fe_stage_cascade_mid_b"):
        monkeypatch.setattr(R, name, _boom)

    class _Self:
        """Estimator stub."""

        degenerate_columns_ = {}

    class _Rec:
        """Recipes stub: every attribute is an empty dict."""

        def __getattr__(self, name):
            """Empty recipe dict."""
            return {}

    X_out, raw = R._run_fe_stage_cascades(_Self(), X, y, 0, 1, y, None, None, None, _Rec())
    assert list(X_out.columns) == list(X.columns) and list(raw) == list(X.columns)
    assert C.concurrency_enabled()
