"""Audit 2026-11-20 findings 28-F40 and 28-F39 in the logging transformer wrappers.

28-F40: a proxy built without __init__ (as deepcopy or pickle does) raises AttributeError for its missing slot, not RecursionError.
28-F39: an RSS re-read that fails reports the RSS fields as None, not as a zero that makes the delta negative.
"""

import pytest

from mlframe.training import logging_transformers as lt


def test_proxy_without_init_raises_attribute_error_not_recursion():
    """An uninitialised proxy reads its missing _inner as AttributeError."""
    proxy = object.__new__(lt._LoggingProxy)
    with pytest.raises(AttributeError):
        _ = proxy.anything
