"""numba's disk cache must notice an edit to a callee kernel that lives in a different source file than its cached caller."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from mlframe import _numba_cache_deps as deps

_RUNNER = textwrap.dedent(
    """
    import sys
    from mlframe import _numba_cache_deps as deps
    root = sys.argv[1]
    if sys.argv[2] == "track":
        deps.track_package("nbdeps_pkg", root + "/nbdeps_pkg")
        deps.install()
    sys.path.insert(0, root)
    from nbdeps_pkg import caller
    print(caller.g(1))
    """
)


def _make_package(root, callee_body):
    """Write a two-file package: ``callee.f`` (cached kernel) and ``caller.g`` (cached kernel importing it)."""
    pkg = root / "nbdeps_pkg"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "callee.py").write_text(f"from numba import njit\n\n\n@njit(cache=True)\ndef f(x):\n    return {callee_body}\n", encoding="utf-8")
    (pkg / "caller.py").write_text(
        "from numba import njit\n\nfrom nbdeps_pkg.callee import f\n\n\n@njit(cache=True)\ndef g(x):\n    return f(x) * 10\n",
        encoding="utf-8",
    )
    return pkg


def _run(root, mode, cache_dir):
    """Run the fresh-process runner against the package and return what ``g(1)`` printed."""
    env = dict(os.environ, NUMBA_CACHE_DIR=str(cache_dir), NUMBA_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", _RUNNER, str(root), mode], capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return int(out.stdout.strip().splitlines()[-1])


def test_editing_a_callee_file_invalidates_the_cached_caller(tmp_path):
    """With the dependency-aware locator the caller recompiles against the edited callee; the stock numba cache keeps serving the stale result."""
    cache_stock = tmp_path / "nbc_stock"
    cache_tracked = tmp_path / "nbc_tracked"
    root = tmp_path / "proj"
    root.mkdir()
    pkg = _make_package(root, "x + 1")
    assert _run(root, "track", cache_tracked) == 20
    assert _run(root, "stock", cache_stock) == 20
    (pkg / "callee.py").write_text("from numba import njit\n\n\n@njit(cache=True)\ndef f(x):\n    return x + 2  # edited\n", encoding="utf-8")
    assert _run(root, "track", cache_tracked) == 30
    assert _run(root, "stock", cache_stock) == 20, "stock numba is expected to serve the stale caller; if it no longer does, this guard is obsolete"


def test_unrelated_file_edit_does_not_change_the_digest(tmp_path):
    """A kernel file's dependency digest only moves when one of its callee files changes."""
    root = tmp_path / "proj"
    root.mkdir()
    pkg = _make_package(root, "x + 1")
    (pkg / "other.py").write_text("X = 1\n", encoding="utf-8")
    deps.track_package("nbdeps_pkg", str(pkg))
    caller = str(pkg / "caller.py")
    before = deps.dependency_digest(caller)
    (pkg / "other.py").write_text("X = 2\n", encoding="utf-8")
    assert deps.dependency_digest(caller) == before
    (pkg / "callee.py").write_text("from numba import njit\n\n\n@njit(cache=True)\ndef f(x):\n    return x + 5\n", encoding="utf-8")
    assert deps.dependency_digest(caller) != before
    assert deps.dependency_files(caller) == [str(pkg / "callee.py")]


def test_dependency_files_follow_reexports_in_mlframe():
    """A kernel importing a jit function through a re-exporting package resolves to the file that defines it."""
    root = deps._PACKAGES["mlframe"]
    kernel = os.path.join(root, "feature_selection", "filters", "_mi_prange_kernel.py")
    found = [os.path.relpath(p, root) for p in deps.dependency_files(kernel)]
    assert os.path.join("feature_selection", "filters", "info_theory", "_class_mi_kernels.py") in found


def test_install_is_idempotent_and_leads_the_locator_list():
    """Installing twice leaves exactly one dependency-aware locator, ahead of numba's own."""
    pytest.importorskip("numba")
    from numba.core import caching

    assert deps.install() is True and deps.install() is True
    names = [c.__name__ for c in caching.CacheImpl._locator_classes]
    assert names.count("DependencyAwareLocator") == 1
    assert names[0] == "DependencyAwareLocator"
