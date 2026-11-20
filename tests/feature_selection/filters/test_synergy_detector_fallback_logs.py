"""The synergy detector warns (throttled) when the kernel-tuning-cache lookup of its null multiple fails, instead of pinning the default silently."""

import logging

from mlframe.feature_selection.filters import _synergy_detector as D
from mlframe.utils.log_throttle import reset_throttle_counts


def test_failing_ktc_lookup_warns_and_uses_the_default(monkeypatch, caplog):
    """A raising cache getter yields the default multiple and a WARNING."""
    import pyutilz.system as psys

    class _Ktc:
        """A cache module stand-in whose getter always fails."""

        @staticmethod
        def get_cached_param(_key):
            """Always fail."""
            raise RuntimeError("registry broken")

    reset_throttle_counts()
    monkeypatch.setattr(psys, "kernel_tuning_cache", _Ktc, raising=False)
    with caplog.at_level(logging.WARNING):
        got = D._lookup_null_mult()
    assert got == D._DEFAULT_NULL_MULT
    assert any(r.levelno == logging.WARNING and "kernel_tuning_cache" in r.getMessage() for r in caplog.records)
