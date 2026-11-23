"""Cluster 39 regression: bare E5 names get the prefix, UNC paths use the UNC long form, and multi-column targets are refused (audit 2026-11-20, findings 39-F27, 39-F40, 39-F43)."""

import numpy as np
import pandas as pd
import pytest

from mlframe.training.feature_handling import system as system_mod
from mlframe.training.feature_handling._target_encoders_canon import _coerce_y_to_float64
from mlframe.training.feature_handling.hf_provider import _needs_e5_prefix

BS = chr(92)  # a single backslash, spelled out so the Windows path literals cannot be mis-escaped


def test_bare_e5_name_needs_prefix():
    """``e5-small`` (no org prefix) is detected as E5 family (39-F27)."""
    assert _needs_e5_prefix("e5-small") is True
    assert _needs_e5_prefix("intfloat/multilingual-e5-small") is True


def test_non_e5_name_does_not_need_prefix():
    """Control: an unrelated model name is not E5 (39-F27)."""
    assert _needs_e5_prefix("sentence-transformers/all-MiniLM-L6-v2") is False


def test_unc_path_uses_unc_long_form(monkeypatch):
    """A UNC share path gets the UNC long-path form, not the plain prefix before the server name (39-F40)."""
    monkeypatch.setattr(system_mod.sys, "platform", "win32")
    monkeypatch.setattr(system_mod.os.path, "abspath", lambda p: p)
    unc = BS * 2 + "server" + BS + "share" + BS + "cache"
    expected = BS * 2 + "?" + BS + "UNC" + BS + "server" + BS + "share" + BS + "cache"
    assert system_mod.long_path_safe(unc) == expected


def test_multi_column_dataframe_target_is_refused():
    """A two-column DataFrame target raises instead of being flattened to 2n values (39-F43)."""
    with pytest.raises(ValueError, match="single column"):
        _coerce_y_to_float64(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))


def test_single_column_dataframe_target_still_works():
    """Control: a one-column DataFrame coerces to a flat float64 vector (39-F43)."""
    out = _coerce_y_to_float64(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0]))
