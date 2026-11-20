"""Regression test: a failing kernel-tuning lookup of the device KSG-CD kernel is logged at WARNING."""

import logging

from mlframe.feature_selection.filters import _ksg_cd_gpu
from mlframe.utils.log_throttle import reset_throttle_counts


def test_ksg_cd_ktc_choose_failure_warns(monkeypatch, caplog):
    """The row-count fallback after a failed lookup must be announced at WARNING."""
    reset_throttle_counts()
    monkeypatch.setattr("mlframe.feature_selection.filters._gpu_policy.gpu_globally_disabled", lambda: False)

    class _Spec:
        """A spec whose lookup fails."""

        def choose(self, **kw):
            """Always fail."""
            raise RuntimeError("cache unreadable")

    monkeypatch.setattr(_ksg_cd_gpu, "_SPEC", _Spec())
    with caplog.at_level(logging.WARNING, logger=_ksg_cd_gpu.logger.name):
        _ksg_cd_gpu.use_device(10)
    assert any("KTC choose() failed" in r.getMessage() for r in caplog.records)
