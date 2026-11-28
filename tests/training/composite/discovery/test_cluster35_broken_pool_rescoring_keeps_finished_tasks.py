"""Cluster 35 regression: when the rerank pool breaks, only the unfinished tasks are rescored (audit 2026-11-20, finding 35-F42)."""

import logging

import joblib
from joblib.externals.loky.process_executor import BrokenProcessPool

import mlframe.training.composite.discovery._tiny_rerank_process as proc


def test_finished_prefix_is_kept_and_only_the_rest_is_rescored(monkeypatch, caplog):
    """Five tasks, the pool breaks after two results: the two stay, the other three are scored serially, order preserved."""
    calls = []

    def fake_score_spec(task):
        """Record the serial rescoring calls and return a tagged result."""
        calls.append(task["name"])
        return (task["name"], {"lgb": 1.0}, {}, None)

    class _BreakingParallel:
        """A Parallel whose generator yields two finished results and then raises BrokenProcessPool."""

        def __init__(self, *a, **k):
            """Accept and ignore the joblib arguments."""

        def __call__(self, jobs):
            """Yield the first two tasks' results, then break."""
            jobs = list(jobs)

            def gen():
                """Two results, then the pool dies."""
                for _fn, args, _kwargs in jobs[:2]:
                    yield (args[0]["name"], {"lgb": 0.5}, {}, None)
                raise BrokenProcessPool("a worker died\nsecond line")

            return gen()

    monkeypatch.setattr(proc, "score_spec", fake_score_spec)
    monkeypatch.setattr(joblib, "Parallel", _BreakingParallel)
    tasks = [{"name": f"t{i}"} for i in range(5)]
    with caplog.at_level(logging.WARNING):
        out = proc.score_specs_in_processes(tasks, n_jobs=2)
    assert [r[0] for r in out] == ["t0", "t1", "t2", "t3", "t4"]
    assert out[0][1] == {"lgb": 0.5} and out[2][1] == {"lgb": 1.0}
    assert calls == ["t2", "t3", "t4"]
    assert any("2 of 5 spec(s) had finished" in r.getMessage() for r in caplog.records)
