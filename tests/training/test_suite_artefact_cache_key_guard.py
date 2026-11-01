"""The suite artefact cache keeps every file inside its directory, and creates that directory private (FS2-1, FS2-2).

A key with ``..`` or a separator used to build a path outside ``cache_dir``. The directory holds pickles that are loaded back, so it is
created with mode 0o700 where the platform honours modes.
"""

import os

import pytest

from mlframe.training.suite_artefact_cache import SuiteArtefactCache


def test_traversal_key_is_refused_and_writes_nothing_outside_the_cache(tmp_path):
    """A key that climbs out of the directory raises before any file is created."""
    cache_dir = tmp_path / "cache"
    cache = SuiteArtefactCache(cache_dir=str(cache_dir))
    with pytest.raises(ValueError, match="plain name"):
        cache._path("../escaped")
    with pytest.raises(ValueError, match="plain name"):
        cache._path("sub/dir")
    assert not (tmp_path / "escaped.pkl").exists()


def test_hex_digest_key_maps_inside_the_cache(tmp_path):
    """The keys ``SuiteKeyBuilder`` produces (hex digests) still resolve to a file inside the directory."""
    cache = SuiteArtefactCache(cache_dir=str(tmp_path / "cache"))
    path = cache._path("0123abcdef" * 4)
    assert os.path.dirname(path) == str(tmp_path / "cache")


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are not applied on Windows")
def test_cache_directory_is_created_private(tmp_path):
    """The directory is not readable by group or others, since it holds pickles that are loaded back."""
    cache_dir = tmp_path / "cache"
    cache = SuiteArtefactCache(cache_dir=str(cache_dir))
    cache._ensure_dir()
    assert (cache_dir.stat().st_mode & 0o077) == 0
