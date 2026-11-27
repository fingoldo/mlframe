"""The chunk pipeline's prefetch runs on a worker thread; it must score a chunk with the main thread's per-fit settings (the scoring switch passed as ``use_su`` and the
thread-local fit state such as AUTO-STRICT), not with the worker thread's defaults, so a prefetched chunk equals one computed inline."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

from mlframe.feature_selection.filters import _fe_gpu_strict
from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_score_steps as S


def test_the_prefetched_chunk_is_scored_with_the_main_threads_settings(monkeypatch):
    """Chunk 0 is computed inline, chunk 1 is prefetched on the pipeline worker: both calls see use_su=True (True only on the main thread) and the main thread's AUTO-STRICT state."""
    main = threading.current_thread()
    seen = []

    def fake_compute(**kw):
        """Record the thread, the switch and the fit state the chunk was computed with."""
        seen.append((threading.current_thread() is main, kw["use_su"], dict(_fe_gpu_strict._auto_fit_state.__dict__)))
        return {("a", "b"): {"x": 1}, ("c", "d"): {"x": 2}}

    monkeypatch.setattr(S, "_compute_one_fe_chunk", fake_compute)
    _fe_gpu_strict._auto_fit_state.__dict__["probe_marker"] = "set-on-main"
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        chunk_state = {"loaded_idx": None, "pipeline_ex": ex, "pipeline_buffers": [np.zeros(2), np.zeros(2)], "pipeline_futures": {}}
        st = SimpleNamespace(_chunk_entry=None)
        S._score_one_pair_step1_reads_buffer_have(
            True, np.zeros(2), {("a", "b"): 0, ("c", "d"): 1}, ("a", "b"), chunk_state, [[("a", "b")], [("c", "d")]], {}, {}, {}, {}, 10, np.int8,
            np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64), np.array([0.5, 0.5]), 0, 0.0, None,
            lambda: threading.current_thread() is main, None, None, True, False, None, False, None, st,
        )
        chunk_state["pipeline_futures"][1].result()
    finally:
        ex.shutdown(wait=True)
        _fe_gpu_strict._auto_fit_state.__dict__.pop("probe_marker", None)
    assert [on_main for on_main, _, _ in seen] == [True, False], "chunk 1 was not prefetched on the worker"
    assert seen and all(use_su for _, use_su, _ in seen), "the prefetch scored with the worker thread's switch"
    assert seen and all(state.get("probe_marker") == "set-on-main" for _, _, state in seen), "the prefetch did not see the main thread's fit state"
