"""Model, target and pipeline names cannot steer a saved file out of its directory or onto a device name (tracker 2026-11-02, API5-1, API5-2)."""

import os

import pytest

from mlframe.training._data_helpers import _setup_model_info_and_paths
from mlframe.utils.safe_filename import safe_file_stem

_UNSAFE_CHARS = '<>:"/\\|?*\x00'


@pytest.mark.parametrize("name", ["lgb", "CB my model", "xgb__sch_3bfa2e4346", "a-b_c.d", "модель 1"])
def test_ordinary_names_pass_through_unchanged(name):
    """Letters, digits, spaces, dashes, underscores, dots and non-ASCII text are kept, so existing file names keep matching."""
    assert safe_file_stem(name) == name


@pytest.mark.parametrize("name", ["lgb/v2", "..\\x", "C:\\x", "a:b", "x*y?", 'q"r', "p|q", "a\x00b"])
def test_separators_drive_colons_and_wildcards_are_replaced(name):
    """No path separator, colon or wildcard survives."""
    stem = safe_file_stem(name)
    assert not any(ch in stem for ch in _UNSAFE_CHARS)


@pytest.mark.parametrize("name", ["CON", "nul", "COM1", "lpt9.txt"])
def test_reserved_device_names_are_prefixed(name):
    """A reserved Windows device name cannot be the stem of a file."""
    assert safe_file_stem(name).startswith("_")


@pytest.mark.parametrize("name", ["lgb/v2", "..\\..\\escape", "C:\\Windows\\x", "/abs/path", "CON"])
def test_the_model_file_path_stays_inside_the_models_directory(tmp_path, name):
    """For hostile model names the built ``.dump`` path is a direct child of data_dir/models, never outside it."""
    _, _, _, _, model_file_name = _setup_model_info_and_paths(None, name, "", "", str(tmp_path), "models")
    parent = os.path.realpath(os.path.dirname(model_file_name))
    assert parent == os.path.realpath(os.path.join(str(tmp_path), "models"))
    assert model_file_name.endswith(".dump")
