"""Cluster 29 regression: a raising post-import callback does not drop the rest, and the failure is logged (audit 2026-11-20, finding 29-F29)."""

import logging

import mlframe.utils.post_import_hook as hook


class _InnerLoader:
    """Inner loader that does nothing on exec, so only the callbacks matter."""

    def exec_module(self, module):
        """No-op module execution."""


def test_remaining_callbacks_run_after_one_raises(caplog):
    """The second callback still runs when the first raises, and the failure is logged at ERROR (29-F29)."""
    calls = []

    def _boom():
        """Simulate a failing registration."""
        raise RuntimeError("callback broke")

    def _ok():
        """Record that the later callback ran."""
        calls.append("ok")

    name = "mlframe_cluster29_fake_module"
    with hook._LOCK:
        hook._PENDING[name] = [_boom, _ok]
    loader = hook._CallbackLoader(_InnerLoader(), name)
    with caplog.at_level(logging.ERROR, logger=hook.logger.name):
        loader.exec_module(object())
    assert calls == ["ok"]
    assert any(r.levelno == logging.ERROR and name in r.getMessage() for r in caplog.records)
