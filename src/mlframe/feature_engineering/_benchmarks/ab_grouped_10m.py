"""E2E paired A/B + identity for grouped.py counting-sort segmentation @10M.
Loads NEW (working-tree grouped.py) and OLD (a prior git revision's grouped.py, dumped to a temp file),
alternates calls, checks identity.

Run: python -m mlframe.feature_engineering._benchmarks.ab_grouped_10m [--old-rev HEAD~1] [--new PATH] [--old PATH]

Defaults: NEW resolves to this repo's own ``grouped.py`` next to this benchmark file; OLD is dumped fresh
via ``git show <old-rev>:<relative-path-to-grouped.py>`` into a tempfile, so no hardcoded/session-specific
absolute path is needed. Pass ``--old PATH`` to compare against an existing standalone file instead.
"""

import argparse
import importlib.util
import subprocess  # nosec B404 - fixed 'git show <rev>:<path>' argv list, no shell=True, no user string interpolation into the command
import sys
import tempfile
import time
from pathlib import Path

sys.modules["cupy"] = None  # type: ignore[assignment]
import scipy.stats  # noqa: F401
import numba  # noqa: F401
import numpy as np

_THIS_FILE = Path(__file__).resolve()
_GROUPED_PY = _THIS_FILE.parents[1] / "grouped.py"  # feature_engineering/grouped.py, sibling of _benchmarks/


def _dump_old_revision(old_rev: str) -> str:
    """git show <old_rev>:<repo-relative path to grouped.py> into a fresh tempfile; returns its path."""
    repo_root = subprocess.run(  # nosec B603,B607 - fixed argv, no shell, trusted git binary
        ["git", "rev-parse", "--show-toplevel"], cwd=_GROUPED_PY.parent, capture_output=True, text=True, check=True
    ).stdout.strip()
    rel_path = _GROUPED_PY.relative_to(Path(repo_root)).as_posix()
    content = subprocess.run(  # nosec B603,B607
        ["git", "show", f"{old_rev}:{rel_path}"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout
    fd, tmp_path = tempfile.mkstemp(suffix="_grouped_old.py", prefix="mlframe_ab_")
    with open(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return tmp_path


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"ab_grouped_10m: could not build a module spec for {path!r}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", default=str(_GROUPED_PY), help="Path to the NEW grouped.py (default: this repo's working-tree copy).")
    parser.add_argument("--old", default=None, help="Path to an existing OLD grouped.py (default: dump --old-rev via git show).")
    parser.add_argument("--old-rev", default="HEAD~1", help="git revision to dump grouped.py from when --old is not given (default: HEAD~1).")
    args = parser.parse_args()

    old_path = args.old if args.old is not None else _dump_old_revision(args.old_rev)
    new = load(args.new, "grouped_new")
    old = load(old_path, "grouped_old")
    n = 10_000_000
    rng = np.random.default_rng(0)
    for n_groups in (200_000, 10_000):
        gids = rng.integers(0, n_groups, size=n).astype(np.int64)
        vals = rng.standard_normal(n)
        # warm both
        new.per_group_shift(vals[:5000], gids[:5000], 1)
        old.per_group_shift(vals[:5000], gids[:5000], 1)
        new.per_group_cum_reduce(vals[:5000], gids[:5000], "sum")
        old.per_group_cum_reduce(vals[:5000], gids[:5000], "sum")
        tn, to = [], []
        for _i in range(4):
            t0 = time.perf_counter(); new.per_group_shift(vals, gids, 1); tn.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); old.per_group_shift(vals, gids, 1); to.append(time.perf_counter() - t0)
        wins = sum(1 for a, b in zip(tn, to) if a < b)
        print(f"[shift n_groups={n_groups}] NEW best={min(tn):.3f} OLD best={min(to):.3f} speedup={min(to)/min(tn):.2f}x faster {wins}/4")
        # identity (shift + cum_reduce + rolling + rank)
        a = new.per_group_shift(vals, gids, 1); b = old.per_group_shift(vals, gids, 1)
        print("  shift identity:", np.array_equal(np.nan_to_num(a, nan=-9e9), np.nan_to_num(b, nan=-9e9)))
        a = new.per_group_cum_reduce(vals, gids, "sum"); b = old.per_group_cum_reduce(vals, gids, "sum")
        print("  cum_reduce max|diff|:", float(np.nanmax(np.abs(a - b))))
        a = new.per_group_rolling_reduce(vals, gids, 5, "mean"); b = old.per_group_rolling_reduce(vals, gids, 5, "mean")
        print("  rolling_mean max|diff|:", float(np.nanmax(np.abs(np.nan_to_num(a) - np.nan_to_num(b)))))
        a = new.per_group_rank(vals, gids); b = old.per_group_rank(vals, gids)
        print("  rank max|diff|:", float(np.nanmax(np.abs(np.nan_to_num(a) - np.nan_to_num(b)))))


if __name__ == "__main__":
    main()
