"""The cascades that do not fit the RAM together run in the fastest order-keeping waves that do; the results come back in stage order and a wave the device
cannot hold runs one stage after another."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fit_impl import _fe_stage_waves as W

GB = float(1 << 30)


def test_the_partitions_are_every_order_keeping_split():
    """Three stages split four ways, each a cover of 0, 1, 2 in order."""
    parts = W._partitions(3)
    assert sorted(parts) == sorted([[(0, 1, 2)], [(0,), (1, 2)], [(0, 1), (2,)], [(0,), (1,), (2,)]])


def test_the_plan_is_the_fastest_partition_that_fits():
    """With room for two stages side by side the plan pairs the longest stage with its neighbour; the makespan is the sum of the waves' longest stages."""
    # one stage of a 1 GB frame holds factor / 3 = 1 GB with factor 3
    frame_rows, frame_cols = 1 << 28, 1  # 1 GB of float32 (4 bytes a value)
    weights = (14.0, 34.0, 18.0)
    waves = W.plan_stage_waves(frame_rows, frame_cols, weights, available=2.5 * GB, factor=3.0)
    assert waves == [(0,), (1, 2)]  # 14 + 34 = 48 against 34 + 18 = 52 for [(0, 1), (2,)]
    assert W.plan_stage_waves(frame_rows, frame_cols, (30.0, 10.0, 5.0), available=2.5 * GB, factor=3.0) == [(0, 1), (2,)]  # 30 + 5 against 30 + 10


def test_no_wave_of_two_fits_gives_none_and_the_whole_set_fitting_gives_one_wave():
    """Too little RAM for two stages: ``None`` (one after the other). Room for three: one wave of all."""
    frame_rows = 1 << 28
    assert W.plan_stage_waves(frame_rows, 1, available=1.5 * GB, factor=3.0) is None
    assert W.plan_stage_waves(frame_rows, 1, available=3.5 * GB, factor=3.0) == [(0, 1, 2)]


def test_the_results_come_back_in_stage_order_and_a_refused_wave_runs_one_by_one(monkeypatch):
    """Wave results are placed by stage index; ``run_stages_concurrently`` returning ``None`` runs that wave's stages in order on the estimator."""
    calls = []

    def stage(i):
        """A stage that records the order it ran in and returns its index."""
        return lambda est: (calls.append(i), i)[1]

    stages = [stage(0), stage(1), stage(2)]
    monkeypatch.setattr(W, "run_stages_concurrently", lambda est, st, w, vram_weights, n_rows: [s(est) for s in st])
    assert W.run_stage_waves(object(), stages, [(0,), (1, 2)], n_rows=10) == [0, 1, 2]
    calls.clear()
    monkeypatch.setattr(W, "run_stages_concurrently", lambda *a, **k: None)
    assert W.run_stage_waves(object(), stages, [(0, 1), (2,)], n_rows=10) == [0, 1, 2] and calls == [0, 1, 2]


def test_a_fit_whose_cascades_do_not_fit_together_selects_as_the_sequential_one(monkeypatch):
    """With the all-at-once gate refused, the waves give the same selection as cascades run one after the other."""
    from mlframe.feature_selection.filters._mrmr_fit_impl import _fe_stage_cascade_run as R
    from tests.feature_selection.conftest import make_fast_mrmr

    rng = np.random.default_rng(3)
    n = 1500
    X = pd.DataFrame(rng.standard_normal((n, 8)), columns=[f"x{i}" for i in range(8)])
    y = pd.Series(((X["x0"] * X["x1"] + 0.5 * X["x2"] + 0.3 * rng.standard_normal(n)) > 0).astype(int))
    monkeypatch.setattr(R, "memory_allows_concurrency", lambda *a: False)
    planned = []
    real_plan = W.plan_stage_waves
    monkeypatch.setattr(W, "plan_stage_waves", lambda *a, **k: planned.append(1) or [(0,), (1, 2)])
    waved = make_fast_mrmr().fit(X, y)
    assert planned, "the cascade runner did not ask for waves"
    monkeypatch.setattr(W, "plan_stage_waves", lambda *a, **k: None)
    sequential = make_fast_mrmr().fit(X, y)
    monkeypatch.setattr(W, "plan_stage_waves", real_plan)
    assert list(waved.get_feature_names_out()) == list(sequential.get_feature_names_out())
