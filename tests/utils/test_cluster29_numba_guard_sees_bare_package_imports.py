"""Audit 2026-11-20 finding 29-F17: a sibling imported as a name (``from . import b``) counts as a dependency of the caching module.

Without it, changing b.py did not invalidate a.py's numba cache, which compiled the old b. The test checks both import forms are read.
"""

from mlframe.utils import numba_cache_guard as guard


def test_from_dot_import_names_are_dependencies():
    """``from . import b, c`` yields both siblings; the module form still works."""
    known = {"a", "b", "c", "d"}
    assert guard._sibling_modules("from . import b, c\n", known) == {"b", "c"}
    assert guard._sibling_modules("from .d import thing\n", known) == {"d"}
