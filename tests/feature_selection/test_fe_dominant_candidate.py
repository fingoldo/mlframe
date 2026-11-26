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
    from types import SimpleNamespace

    monkeypatch.setattr(DS, "plan_stage_slots", lambda weights, **k: [SimpleNamespace(device_id=0)] * len(weights))
    monkeypatch.setattr(C, "_synchronize_device", lambda device_id: None)  # the stages are stubs: nothing runs on a device
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


def test_blocks_of_columns_name_the_same_dominant_column(monkeypatch):
    """A frame over the block budget is scored a block of columns at a time; the dominant column is the one a single call names."""
    import numpy as np
    import pandas as pd

    from mlframe.feature_selection.filters import _fe_dominant_candidate as dom

    rng = np.random.default_rng(0)
    n = 20_000
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(5)})
    y = (X["x2"].to_numpy() > 0).astype(np.int64)  # x2 is the label itself
    one_call = dom.dominant_candidate(X, y)
    monkeypatch.setattr(dom, "_BLOCK_BYTES", n * 4 * 2)  # two columns per block
    blocked = dom.dominant_candidate(X, y)
    assert one_call == blocked == "x2"


def test_nonfinite_values_in_other_columns_do_not_hide_the_dominant_column():
    """Columns with NaN and with infinities are read as 0 there (as before the matrix was built in place); the column that carries the target is still found and a noise frame has none."""
    import numpy as np
    import pandas as pd

    from mlframe.feature_selection.filters._fe_dominant_candidate import dominant_candidate

    rng = np.random.default_rng(3)
    n = 6000
    X = pd.DataFrame(rng.standard_normal((n, 6)).astype(np.float32), columns=[f"x{i}" for i in range(6)])
    y = rng.integers(0, 3, n)
    X["carrier"] = y.astype(np.float32)
    X.loc[X.index[::50], "x1"] = np.nan
    X.loc[X.index[::70], "x2"] = np.inf
    X.loc[X.index[::90], "x3"] = -np.inf
    assert dominant_candidate(X, y) == "carrier"
    assert dominant_candidate(X.drop(columns=["carrier"]), y) is None


def _long_frame(n=300_000, seed=5):
    """Six independent normal columns of a frame long enough for the sample screen."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((n, 6)).astype(np.float32), columns=[f"x{i}" for i in range(6)])


def test_a_long_frame_is_screened_on_a_sample_and_only_the_survivors_are_scored_at_full_length(monkeypatch):
    """The first scoring sees the sample's rows and every column; the full-length scoring sees the one column that cleared the screen and names it."""
    from mlframe.feature_selection.filters import _fe_dominant_candidate as dom

    X = _long_frame()
    y = (X["x3"].to_numpy() > 0).astype(np.int64)
    seen = []
    real = dom._column_mis

    def spy(frame, names, codes):
        """Record the rows and the columns of each scoring."""
        seen.append((len(codes), list(names)))
        return real(frame, names, codes)

    monkeypatch.setattr(dom, "_column_mis", spy)
    assert dom.dominant_candidate(X, y) == "x3"
    assert [rows for rows, _ in seen] == [dom._SCREEN_ROWS, len(X)]
    assert seen[0][1] == list(X.columns) and seen[1][1] == ["x3"]


def test_a_long_frame_without_a_dominant_column_is_decided_by_the_sample_alone(monkeypatch):
    """When no column clears the screen the full-length scoring never runs."""
    from mlframe.feature_selection.filters import _fe_dominant_candidate as dom

    X = _long_frame()
    rng = np.random.default_rng(1)
    y = ((X["x2"].to_numpy() + rng.standard_normal(len(X))) > 0).astype(np.int64)
    seen = []
    real = dom._column_mis
    monkeypatch.setattr(dom, "_column_mis", lambda frame, names, codes: (seen.append(len(codes)), real(frame, names, codes))[1])
    assert dom.dominant_candidate(X, y) is None
    assert seen == [dom._SCREEN_ROWS]


def test_the_screen_never_changes_the_verdict_at_any_noise_level_around_the_bar(monkeypatch):
    """Targets that are a column with 0 to 10 percent of the labels flipped, a clean three-level target and a rare class: the verdict with the screen equals the verdict of the full-length scoring."""
    from mlframe.feature_selection.filters import _fe_dominant_candidate as dom

    X = _long_frame()
    rng = np.random.default_rng(2)
    x = X["x4"].to_numpy()
    targets = []
    for flip in (0.0, 0.0003, 0.0007, 0.001, 0.0014, 0.002, 0.005, 0.02, 0.1):
        targets.append(((x > 0) ^ (rng.random(len(x)) < flip)).astype(np.int64))
    targets.append(np.digitize(x, [-0.5, 0.5]).astype(np.int64))
    targets.append(((x > 2.3) ^ (rng.random(len(x)) < 0.0002)).astype(np.int64))  # about one row in a hundred is positive
    screened = [dom.dominant_candidate(X, y) for y in targets]
    monkeypatch.setattr(dom, "_SCREEN_MIN_ROWS", 10**12)
    exact = [dom.dominant_candidate(X, y) for y in targets]
    assert screened == exact
    assert exact[0] == "x4" and exact[8] is None
