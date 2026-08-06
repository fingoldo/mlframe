"""Regression tests: LocalDiskBackend rejects cache keys that could escape ``root``."""

import pytest

from mlframe.training.feature_handling.cache_backend import LocalDiskBackend


@pytest.fixture
def backend(tmp_path):
    """A fresh LocalDiskBackend rooted in a pytest tmp_path."""
    return LocalDiskBackend(str(tmp_path))


@pytest.mark.parametrize(
    "bad_key",
    [
        "../escape",
        "..\\escape",
        "a/../../escape",
        "sub/dir",
        "sub\\dir",
        "/absolute/path",
        "C:\\absolute\\path",
        "",
    ],
)
def test_unsafe_key_raises_on_write(backend, bad_key):
    """A key containing a separator, '..' segment, or absolute path must raise ValueError, not write outside root."""
    with pytest.raises(ValueError):
        backend.write(bad_key, b"data")


@pytest.mark.parametrize(
    "bad_key",
    [
        "../escape",
        "sub/dir",
        "",
    ],
)
def test_unsafe_key_raises_on_read(backend, bad_key):
    """Read must validate too, not just write -- both go through _value_path."""
    with pytest.raises(ValueError):
        backend.read(bad_key)


def test_safe_key_still_works(backend):
    """A normal opaque single-component key is unaffected by the validation."""
    backend.write("safe_key_123", b"payload")
    assert backend.read("safe_key_123") == b"payload"
