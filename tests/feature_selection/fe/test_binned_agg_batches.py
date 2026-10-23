"""The binned-aggregate host path scores every pair in memory-bounded batches and gives the same result as one pass."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _binned_numeric_agg_batches as bat
from mlframe.feature_selection.filters._binned_numeric_agg_fe import binned_numeric_agg_with_recipes
from mlframe.feature_selection.filters._unified_fe_gate import local_mi_gate


def test_rows_per_batch_follows_the_free_memory_of_the_working_device():
    """The host budget bounds the batch; the device budget (float64 upload) bounds it further when the gate runs on the device; never below one."""
    host = bat.rows_per_batch(1_000_000, 4, 8 << 30, None)
    assert host == int(0.5 * (8 << 30) / (4 * 1_000_000 * 4))
    assert bat.rows_per_batch(1_000_000, 4, 8 << 30, 1 << 30) < host
    assert bat.rows_per_batch(1_000_000, 4, 1 << 10, None) == 1


def test_pair_batches_keep_rank_order_and_cover_every_pair(monkeypatch):
    """Consecutive batches reproduce the ranked pair list exactly; a roomy memory gives one batch."""
    pairs = [(f"g{i}", f"a{j}") for i in range(4) for j in range(4)]
    monkeypatch.setattr(bat, "_free_device_bytes", lambda n: None)
    monkeypatch.setattr(bat, "_free_host_bytes", lambda: 1 << 40)
    assert bat.pair_batches(pairs, 10_000, 4) == [pairs]
    monkeypatch.setattr(bat, "_free_host_bytes", lambda: 10_000 * 4 * 4 * 8 * 2)  # eight stored rows per batch: two pairs of four stats
    batches = bat.pair_batches(pairs, 10_000, 4)
    assert len(batches) == 8 and [p for b in batches for p in b] == pairs


def _candidates(n: int = 3000, k: int = 14, seed: int = 0) -> tuple:
    """A candidate frame where some columns carry the target and most are noise, and the raw frame the floor is anchored on."""
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame(rng.standard_normal((n, 5)).astype(np.float32), columns=[f"r{i}" for i in range(5)])
    y = (raw["r0"].to_numpy() + 0.8 * rng.standard_normal(n) > 0).astype(np.int64)
    cand = pd.DataFrame(rng.standard_normal((n, k)).astype(np.float32), columns=[f"b{i}" for i in range(k)])
    cand["b3"] = raw["r0"].to_numpy() + 0.3 * rng.standard_normal(n).astype(np.float32)
    cand["b9"] = np.tanh(raw["r0"].to_numpy()) + 0.5 * rng.standard_normal(n).astype(np.float32)
    return cand, raw, y


@pytest.mark.parametrize("batch_cols", [1, 3, 5, 14])
def test_streaming_gate_equals_one_call_survivors_order_and_rejections(batch_cols, monkeypatch):
    """Same survivors in the same order, same frame, same rejection records, for every batch width."""
    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT", raising=False)
    cand, raw, y = _candidates()
    one_records, many_records = [], []
    one = local_mi_gate(cand, y, raw_X=raw, reject_sink=lambda **kw: one_records.append((kw["candidate"], round(kw["observed"], 12), round(kw["threshold"], 12))))
    gate = bat.StreamingLocalGate(y, raw, lambda **kw: many_records.append((kw["candidate"], round(kw["observed"], 12), round(kw["threshold"], 12))))
    for start in range(0, cand.shape[1], batch_cols):
        gate.score_batch(cand.iloc[:, start : start + batch_cols])
    survivors, frame = gate.finalize()
    assert survivors == one
    assert survivors, "the fixture must have at least one survivor"
    assert list(frame.columns) == [c for c in cand.columns if c in set(one)]
    pd.testing.assert_frame_equal(frame, cand[list(frame.columns)])
    assert sorted(many_records) == sorted(one_records)


def test_end_to_end_with_a_forced_tiny_budget_equals_the_single_pass(monkeypatch):
    """One pair per batch gives the same appended columns and values as the single pass, and the streamed path really ran."""
    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT", raising=False)
    monkeypatch.delenv("MLFRAME_FE_GPU_DEVICE_BORN_BINAGG", raising=False)
    rng = np.random.default_rng(1)
    n = 6000
    X = pd.DataFrame(rng.standard_normal((n, 6)).astype(np.float32), columns=[f"c{i}" for i in range(6)])
    cell = np.digitize(X["c0"].to_numpy(), np.quantile(X["c0"].to_numpy(), np.linspace(0.1, 0.9, 9)))
    X["c3"] = (X["c3"] * 0.3 + (cell % 3).astype(np.float32) * 2.0).astype(np.float32)
    y = (((cell % 3) == 1) ^ (rng.random(n) < 0.1)).astype(np.int64)
    kwargs = dict(group_num_cols=["c0", "c1", "c2"], agg_num_cols=["c3", "c4", "c5"], max_pairs=None, redundancy_gate=False)
    single_aug, single_names, _ = binned_numeric_agg_with_recipes(X, y, **kwargs)
    assert single_names, "the fixture must append columns, or the comparison is empty"

    calls = []
    real = bat.streamed_host_candidates
    monkeypatch.setattr(bat, "streamed_host_candidates", lambda *a, **k: calls.append(1) or real(*a, **k))
    monkeypatch.setattr(bat, "_free_device_bytes", lambda n_rows: None)
    monkeypatch.setattr(bat, "_free_host_bytes", lambda: 1)  # one stored row's worth: every pair is its own batch
    batched_aug, batched_names, _ = binned_numeric_agg_with_recipes(X, y, **kwargs)
    assert calls, "the streamed host path did not run"
    assert batched_names == single_names
    pd.testing.assert_frame_equal(batched_aug, single_aug)


@pytest.mark.parametrize("top_k", [None, 2])
@pytest.mark.parametrize("explicit_floor", [None, 0.002])
def test_local_mi_gate_scores_blocks_when_the_matrix_does_not_fit_and_keeps_its_result(top_k, explicit_floor, monkeypatch):
    """With a memory budget of a few columns the gate runs block by block; the survivors, their order and the rejection records equal the single-pass gate's."""
    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT", raising=False)
    cand, raw, y = _candidates()
    single_records, blocked_records = [], []

    def sink(store):
        """A reject sink that records (candidate, observed, threshold)."""
        return lambda **kw: store.append((kw["candidate"], round(kw["observed"], 12), round(kw["threshold"], 12)))

    single = local_mi_gate(cand, y, raw_X=raw, floor=explicit_floor, top_k=top_k, reject_sink=sink(single_records))
    assert single, "the fixture must have survivors"
    monkeypatch.setattr(bat, "_free_device_bytes", lambda n_rows: None)
    monkeypatch.setattr(bat, "_free_host_bytes", lambda: int(len(cand) * 4 * 4 * 3 * 2))  # three columns per block
    assert len(bat.column_blocks(list(cand.columns), len(cand))) > 1
    blocked = local_mi_gate(cand, y, raw_X=raw, floor=explicit_floor, top_k=top_k, reject_sink=sink(blocked_records))
    assert blocked == single
    assert sorted(blocked_records) == sorted(single_records)


def test_raw_noise_floor_scored_in_blocks_equals_the_single_pass(monkeypatch):
    """The floor from the raw columns' MIs is the same number whether the matrix is scored whole or block by block."""
    from mlframe.feature_selection.filters._unified_fe_gate import raw_mi_noise_floor

    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT", raising=False)
    cand, _raw, y = _candidates(k=14)
    whole = raw_mi_noise_floor(cand, y)
    monkeypatch.setattr(bat, "_free_device_bytes", lambda n_rows: None)
    monkeypatch.setattr(bat, "_free_host_bytes", lambda: int(len(cand) * 4 * 4 * 3 * 2))
    assert len(bat.column_blocks(list(cand.columns), len(cand))) > 1
    assert raw_mi_noise_floor(cand, y) == whole
