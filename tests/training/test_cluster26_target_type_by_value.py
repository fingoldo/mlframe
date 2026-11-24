"""Cluster 26 regression: a target_type given as an enum value resolves to that type (audit 2026-11-20, finding 26-F37).

The string branch resolved only member names, so a value such as ``"multilabel_classification"`` fell through to the shape-based guess and the
caller's type was ignored. The branch now tries the value first, then the name.
"""

from pathlib import Path

from mlframe.training import _calibration_models as cm
from mlframe.training.configs import TargetTypes


def test_value_string_resolves_to_the_enum_member():
    """The enum value resolves to the matching TargetTypes member (26-F37)."""
    assert TargetTypes("multilabel_classification") is TargetTypes.MULTILABEL_CLASSIFICATION


def test_resolution_tries_value_before_name():
    """The source resolves by value first and keeps the name lookup as the fallback (26-F37)."""
    src = Path(cm.__file__).resolve().read_text(encoding="utf-8")
    assert "_resolved_ttype = TargetTypes(_ttype)" in src
    assert "getattr(TargetTypes, _ttype, None)" in src
