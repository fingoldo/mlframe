"""The ``getattr`` fallback of ``fe_optimizer`` in the FE step is the constructor default (audit 2026-11-20 D-F04)."""

import inspect

from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.filters._mrmr_fe_step import _step_core_parts


def test_fe_optimizer_fallback_equals_the_constructor_default():
    """An instance built without ``__init__`` must run the optimizer ``MRMR()`` runs: the reader's literal equals the constructor default."""
    default = inspect.signature(MRMR.__init__).parameters["fe_optimizer"].default
    assert f'getattr(self, "fe_optimizer", "{default}")' in inspect.getsource(_step_core_parts)
