"""A cached numba caller whose callee lives in another file and changed is recompiled after the guard has cleared its cache; an unrelated module keeps its cache."""
import subprocess
import sys
import textwrap

from mlframe.utils.numba_cache_guard import invalidate_stale_numba_caches

CALLEE = "from numba import njit\n\n\n@njit(cache=True)\ndef bump(x):\n    return x + {delta}\n"
CALLER = "from numba import njit\n\nfrom .callee import bump\n\n\n@njit(cache=True)\ndef run(x):\n    return bump(x) * 2\n"
OTHER = "from numba import njit\n\n\n@njit(cache=True)\ndef solo(x):\n    return x - 1\n"
DRIVER = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {root!r})
    from pkg.caller import run
    from pkg.other import solo
    print(run(10), solo(10))
    """
)


def _run(root) -> tuple[int, int]:
    """Run the driver in a fresh interpreter (so numba loads from its disk cache) and return ``(run(10), solo(10))``."""
    out = subprocess.run([sys.executable, "-c", DRIVER.format(root=str(root))], capture_output=True, text=True, check=True, timeout=300).stdout.split()
    return int(out[0]), int(out[1])


def test_changed_callee_in_another_file_does_not_leave_a_stale_caller(tmp_path):
    """Without the guard the caller keeps the old callee (the documented numba limitation); after the guard the caller gives the new value and the other module's cache files are untouched."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "callee.py").write_text(CALLEE.format(delta=1), encoding="utf-8")
    (pkg / "caller.py").write_text(CALLER, encoding="utf-8")
    (pkg / "other.py").write_text(OTHER, encoding="utf-8")
    assert invalidate_stale_numba_caches(pkg) == []
    assert _run(tmp_path) == (22, 9)
    other_cache = sorted(p.name for p in (pkg / "__pycache__").glob("other.*.nb*"))
    assert other_cache
    (pkg / "callee.py").write_text(CALLEE.format(delta=5), encoding="utf-8")
    assert _run(tmp_path) == (22, 9)  # the stale caller still runs the old callee: this is what the guard exists for
    cleared = invalidate_stale_numba_caches(pkg)
    assert set(cleared) == {"callee", "caller"}
    assert _run(tmp_path) == (30, 9)
    assert sorted(p.name for p in (pkg / "__pycache__").glob("other.*.nb*")) == other_cache
    assert invalidate_stale_numba_caches(pkg) == []
