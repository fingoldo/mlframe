"""MLFRAME_SKIP_NUMBA_PREWARM, which the benchmarks set, must actually skip the warm-up."""

import mlframe.metrics._core_numba_warmup as w


def test_the_variable_skips_the_warmup(monkeypatch):
    ran = []
    monkeypatch.setattr(w, "_prewarm_numba_cache_body", lambda **kw: ran.append(kw))
    monkeypatch.setenv("MLFRAME_SKIP_NUMBA_PREWARM", "1")
    w.prewarm_numba_cache()
    assert ran == [], "fifteen benchmarks set this to keep compile time out of their timings; it was read nowhere"
    monkeypatch.delenv("MLFRAME_SKIP_NUMBA_PREWARM")
    w.prewarm_numba_cache()
    assert len(ran) == 1
