"""Cluster 28 regression: the honest-diagnostics key is built by one helper, with the same value as before (audit 2026-11-20, finding 28-F72).

The target/model key was spelled out three times. _entry_key gives the same string: the model_name attribute when present, otherwise the
wrapped model's class name.
"""

from pathlib import Path

from mlframe.training import honest_diagnostics as hd

_SRC = Path(hd.__file__).resolve()


class _Named:
    """Stand-in top model that carries an explicit model_name."""

    model_name = "lgbm_a"


class _Unnamed:
    """Stand-in top model without model_name, so the class name is used."""


def test_key_uses_model_name_when_present():
    """The explicit model_name appears in the key (28-F72)."""
    assert hd._entry_key("binary", "y", _Named()) == "binary/y/lgbm_a"


def test_key_falls_back_to_class_name():
    """Without model_name the wrapped class name is used, as before (28-F72)."""
    assert hd._entry_key("binary", "y", _Unnamed()) == "binary/y/_Unnamed"
