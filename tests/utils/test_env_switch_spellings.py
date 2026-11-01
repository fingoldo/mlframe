"""Kill switches read ``0`` / ``false`` / ``off`` as off and ``on`` / ``yes`` as on (ENV4-6).

The cache-disable and clip-disable switches treated any non-empty value as on, so ``=0`` disabled the feature it was meant to leave enabled.
"""

import pytest

from mlframe.training._lgb_shim_helpers import _lgb_cache_disabled
from mlframe.training.xgb_shim import _xgb_cache_disabled


@pytest.mark.parametrize("fn,name", [(_lgb_cache_disabled, "MLFRAME_LGB_CACHE_DISABLE"), (_xgb_cache_disabled, "MLFRAME_XGB_CACHE_DISABLE")])
@pytest.mark.parametrize("value,expected", [("0", False), ("false", False), ("OFF", False), ("no", False), ("1", True), ("on", True), ("Yes", True)])
def test_cache_disable_switch_spellings(monkeypatch, fn, name, value, expected):
    """A false spelling leaves the cache enabled; a true spelling disables it."""
    monkeypatch.setenv(name, value)
    assert fn() is expected
