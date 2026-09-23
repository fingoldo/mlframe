"""The pair-MI loky watchdog: two distinct bounds, and the abandoned pool's workers are killed before the retry."""

import mlframe.feature_selection.filters._mrmr_fe_step._step_pairmi as sp


def test_the_task_timeout_fires_before_the_outer_watchdog():
    """Handing both bounds the same value made joblib's task timeout unreachable: the outer clock starts first.

    The outer watchdog covers pool spawn as well, so it must leave room for it on top of the per-task bound.
    """
    assert sp._LOKY_POOL_SPAWN_GRACE > 0
    assert sp._LOKY_POOL_WALL_CLOCK_TIMEOUT == sp._LOKY_TASK_TIMEOUT + sp._LOKY_POOL_SPAWN_GRACE
    assert sp._LOKY_TASK_TIMEOUT < sp._LOKY_POOL_WALL_CLOCK_TIMEOUT


def test_killing_the_reusable_workers_issues_a_kill(monkeypatch):
    killed = {}

    class _Exec:
        def shutdown(self, kill_workers=False):
            killed["kill_workers"] = kill_workers

    import joblib.externals.loky as loky

    monkeypatch.setattr(loky, "get_reusable_executor", lambda *a, **k: _Exec())
    assert sp._kill_reusable_loky_workers() is True
    assert killed == {"kill_workers": True}, "the workers must be killed, not asked to finish their queue"


def test_a_failure_to_kill_is_reported_not_raised(monkeypatch):
    import joblib.externals.loky as loky

    def _boom(*a, **k):
        raise RuntimeError("no executor here")

    monkeypatch.setattr(loky, "get_reusable_executor", _boom)
    assert sp._kill_reusable_loky_workers() is False
