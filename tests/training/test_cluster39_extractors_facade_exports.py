"""Cluster 39 regression: the extractors facade exports only its public API (audit 2026-11-20, finding 39-F15).

``__all__`` used to list the third-party modules ``np``, ``pd`` and ``pl``, and the facade imported several unused names. The public
extractor and helper names must still import from the package.
"""

from pathlib import Path

from mlframe.training import extractors as ex

_SRC = Path(ex.__file__).resolve().read_text(encoding="utf-8")


def test_all_exports_no_third_party_modules():
    """``__all__`` no longer lists numpy, pandas or polars (39-F15)."""
    assert not {"np", "pd", "pl"} & set(ex.__all__)


def test_public_names_still_importable():
    """Every name in ``__all__`` is an attribute of the package (39-F15)."""
    for name in ex.__all__:
        assert hasattr(ex, name), name


def test_unused_typing_and_private_imports_removed():
    """The facade no longer imports the unused private helpers or typing names (39-F15)."""
    assert "_safe_int_cast_numpy" not in _SRC
    assert "_smallest_safe_int_dtype" not in _SRC
    assert "runtime_checkable" not in _SRC
