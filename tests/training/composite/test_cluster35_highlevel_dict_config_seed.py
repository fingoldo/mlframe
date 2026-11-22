"""Audit 2026-11-20 finding 35-F30: a dict config's random_state is the one used, not the 42 default.

``discover_and_wrap`` read the seed from the raw dict, so the report and the reproducibility claim recorded 42 while discovery ran with
the dict's value. The test captures the config handed to the discovery object and stops there.
"""

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite import highlevel
from mlframe.training.configs import CompositeTargetDiscoveryConfig


class _Stop(Exception):
    """Raised by the spy after the config has been captured."""


def test_dict_config_random_state_reaches_the_discovery_object(monkeypatch):
    """With config={'random_state': 7, ...} the discovery object receives a config whose seed is 7."""
    seen = {}

    class _SpyDiscovery:
        """Record the config and stop before any real discovery work."""

        def __init__(self, config):
            seen["config"] = config
            raise _Stop()

    monkeypatch.setattr(highlevel, "CompositeTargetDiscovery", _SpyDiscovery)
    df = pd.DataFrame({"x": np.arange(60.0), "y": np.arange(60.0)})
    with pytest.raises(_Stop):
        highlevel.discover_and_wrap(df, "y", ["x"], config={"random_state": 7, "enabled": True})
    assert isinstance(seen["config"], CompositeTargetDiscoveryConfig)
    assert seen["config"].random_state == 7
