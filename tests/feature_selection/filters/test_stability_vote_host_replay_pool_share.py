"""The host replay pool of the stability vote takes the cascade share of the cores (audit 2026-11-20-perf E-F04)."""

from mlframe.feature_selection.filters import _adaptive_nbins as nb
from mlframe.feature_selection.filters import _fe_stability_vote as sv


def test_host_replays_size_the_pool_by_the_edge_job_share_and_keep_the_order(monkeypatch):
    """The pool size comes from ``_resolve_edge_jobs`` (the share rule of every edge pool, capped at eight) and the results stay in recipe order."""
    seen = {}

    def fake_jobs(n_jobs, n_miss):
        """Record the request and give the share of one cascade of three on eight cores."""
        seen["call"] = (n_jobs, n_miss)
        return 2

    monkeypatch.setattr(nb, "_resolve_edge_jobs", fake_jobs)
    monkeypatch.setattr(sv, "_host_replay", lambda recipe, X_fold, n_rows: recipe)
    out = sv._host_replays(list(range(10)), None, 5)
    assert out == list(range(10))
    assert seen["call"] == (None, 10)
