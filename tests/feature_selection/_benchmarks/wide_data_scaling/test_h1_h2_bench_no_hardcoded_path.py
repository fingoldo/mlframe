"""FS_BENCHMARKS_C-3: h1_bench.py, h1_gpu_large.py, and h2_bench.py each hardcoded a dev-machine-specific
absolute sys.path.insert(0, r"D:/Upd/Programming/PythonCodeRepository/...") -- a stale path present but
wrong on another machine would silently shadow the properly installed mlframe package for the whole
process. They must derive the path from __file__ instead."""

from __future__ import annotations

from pathlib import Path

def _find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` until a directory containing a ``src/mlframe`` package is found."""
    for candidate in (start, *start.parents):
        if (candidate / "src" / "mlframe").is_dir():
            return candidate
    raise RuntimeError("could not locate the repo root (a src/mlframe directory) above " + str(start))


_BENCH_DIR = _find_repo_root(Path(__file__).resolve()) / "src" / "mlframe" / "feature_selection" / "_benchmarks" / "wide_data_scaling"
_FILES = ("h1_bench.py", "h1_gpu_large.py", "h2_bench.py")


def test_no_dev_machine_hardcoded_path_remains():
    """None of the three files may contain the old hardcoded dev-machine path."""
    for fname in _FILES:
        src = (_BENCH_DIR / fname).read_text(encoding="utf-8")
        assert "D:/Upd/Programming/PythonCodeRepository" not in src, f"{fname} still has the hardcoded dev-machine path"
        assert "__file__" in src, f"{fname} must derive its sys.path insert from __file__"


def test_derived_src_dir_matches_the_real_src_directory():
    """Sanity: parents[4] from a file at .../src/mlframe/feature_selection/_benchmarks/wide_data_scaling/
    resolves to the actual src/ directory, confirming the parents-index used in the fix is correct."""
    sample_file = _BENCH_DIR / "h1_bench.py"
    derived_src_dir = sample_file.resolve().parents[4]
    assert derived_src_dir.name == "src"
    assert (derived_src_dir / "mlframe").is_dir()
