"""FS_BENCHMARKS_C-3: h1_bench.py, h1_gpu_large.py, and h2_bench.py each hardcoded a dev-machine-specific
absolute sys.path.insert(0, r"D:/Upd/Programming/PythonCodeRepository/...") -- a stale path present but
wrong on another machine would silently shadow the properly installed mlframe package for the whole
process. They must derive the path from __file__ instead."""

from __future__ import annotations

import ast
import types
from pathlib import Path

import pytest

def _find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` until a directory containing a ``src/mlframe`` package is found."""
    for candidate in (start, *start.parents):
        if (candidate / "src" / "mlframe").is_dir():
            return candidate
    raise RuntimeError("could not locate the repo root (a src/mlframe directory) above " + str(start))


_BENCH_DIR = _find_repo_root(Path(__file__).resolve()) / "src" / "mlframe" / "feature_selection" / "_benchmarks" / "wide_data_scaling"
_FILES = ("h1_bench.py", "h1_gpu_large.py", "h2_bench.py")


def _paths_inserted_by_main_guard(path: Path) -> list[str]:
    """Execute the path-setup statements of ``path``'s ``__main__`` guard against a recording ``sys`` and return the entries inserted."""
    tree = ast.parse(path.read_bytes())
    guards = [n for n in tree.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and getattr(n.test.left, "id", None) == "__name__"]
    assert len(guards) == 1, f"{path.name} must have exactly one __main__ guard"
    setup = []
    for stmt in guards[0].body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            break
        setup.append(stmt)
    assert setup, f"{path.name} has no path-setup statements ahead of its imports"
    inserted: list[str] = []
    fake_sys = types.SimpleNamespace(path=types.SimpleNamespace(insert=lambda _idx, entry: inserted.append(entry)))
    namespace = {"sys": fake_sys, "Path": Path, "__file__": str(path)}
    exec(compile(ast.Module(body=setup, type_ignores=[]), str(path), "exec"), namespace)  # nosec B102 - executes only the repo's own path-setup statements
    return inserted


@pytest.mark.parametrize("fname", _FILES)
def test_no_dev_machine_hardcoded_path_remains(fname):
    """The path each benchmark inserts into sys.path is derived from its own location: the repo ``src`` directory, never a foreign absolute path."""
    inserted = _paths_inserted_by_main_guard(_BENCH_DIR / fname)
    assert inserted, f"{fname} inserts nothing into sys.path"
    real_src = _BENCH_DIR.parents[3].resolve()
    assert real_src.name == "src" and (real_src / "mlframe").is_dir()
    assert str(real_src) in inserted, f"{fname} inserted {inserted}, expected the real src directory {real_src}"
    for entry in inserted:
        assert Path(entry).is_dir(), f"{fname} inserts {entry}, which does not exist on this machine"
        assert not entry.replace("\\", "/").startswith("D:/Upd"), f"{fname} inserts the dev-machine path {entry}"


def test_derived_src_dir_matches_the_real_src_directory():
    """Sanity: parents[4] from a file at .../src/mlframe/feature_selection/_benchmarks/wide_data_scaling/
    resolves to the actual src/ directory, confirming the parents-index used in the fix is correct."""
    sample_file = _BENCH_DIR / "h1_bench.py"
    derived_src_dir = sample_file.resolve().parents[4]
    assert derived_src_dir.name == "src"
    assert (derived_src_dir / "mlframe").is_dir()
