"""Selection snapshot of an all-families MRMR fit, for comparing two commits in separate processes.

Two versions of the package cannot be imported into one process, so the reference commit is checked out in a worktree and this script runs there with ``PYTHONPATH=<worktree>/src``::

    PYTHONPATH=src python -m tests.feature_selection.golden.selection_snapshot OUT.json [rows] [cols]

and again on the changed tree; ``compare`` then lists every difference::

    python -m tests.feature_selection.golden.selection_snapshot --compare REF.json NEW.json

Take the reference in a CLEAN worktree (``git worktree add <dir> <commit>``): a tree with warm numba caches can give a different selection order from stale compiled kernels (seen: two reference runs agreed with each other and
disagreed with a cold-cache run of the same commit), so a warm tree is not a reference.

The snapshot holds, for a pandas input and for the same data as a polars frame: the selected feature names in order, the engineered column names in order, the ``transform`` output
column names and a digest of its values rounded to 5 significant digits (float32 sums differ in the last bits between CPU/GPU runs, the selection must not).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
for _k, _v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
    os.environ.setdefault(_k, _v)


def _digest(arr: np.ndarray) -> str:
    """Digest of ``arr`` rounded to 5 significant digits (NaN as a fixed token)."""
    a = np.asarray(arr, dtype=np.float64)
    a = np.where(np.isfinite(a), a, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(a == 0.0, 0.0, 10.0 ** (4 - np.floor(np.log10(np.abs(a)))))
    return hashlib.sha1(np.round(a * mag).astype(np.int64).tobytes(), usedforsecurity=False).hexdigest()[:16]


def _one(X, y, kw: dict) -> dict:
    """Fit once and describe the outcome."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    sel = MRMR(verbose=0, random_seed=0, n_jobs=1, **kw)
    sel.fit(X, y)
    out = sel.transform(X)
    names = [str(c) for c in getattr(out, "columns", [])]
    values = out.to_numpy() if hasattr(out, "to_numpy") else np.asarray(out)
    return {
        "support": [str(c) for c in np.asarray(sel.get_feature_names_out())] if hasattr(sel, "get_feature_names_out") else [],
        "transform_columns": names,
        "transform_digest": _digest(values),
        "transform_shape": list(values.shape),
    }


def snapshot(rows: int = 20000, cols: int = 40) -> dict:
    """The snapshot of the all-families fit on pandas and on polars input."""
    from mlframe.feature_selection._benchmarks.profiling import family_scale as F

    kw = F.all_families_kwargs()
    kw["nbins_strategy"] = "mdlp"
    X, y = F._make(rows, cols, 0)[:2]
    res = {"rows": rows, "cols": cols, "pandas": _one(X, y, kw)}
    try:
        import polars as pl

        res["polars"] = _one(pl.from_pandas(X), np.asarray(y), kw)
    except ImportError:
        res["polars"] = None
    return res


def compare(ref: dict, new: dict) -> list[str]:
    """Human-readable differences between two snapshots (empty when selection and transform agree)."""
    diffs: list[str] = []
    for kind in ("pandas", "polars"):
        a, b = ref.get(kind), new.get(kind)
        if a is None or b is None:
            continue
        for key in ("support", "transform_columns", "transform_shape", "transform_digest"):
            if a[key] != b[key]:
                if isinstance(a[key], list) and a[key] and isinstance(a[key][0], str):
                    only_a, only_b = sorted(set(a[key]) - set(b[key])), sorted(set(b[key]) - set(a[key]))
                    diffs.append(f"{kind}.{key}: only in ref {only_a[:8]}, only in new {only_b[:8]}, same set reordered={set(a[key]) == set(b[key])}")
                else:
                    diffs.append(f"{kind}.{key}: {a[key]} != {b[key]}")
    return diffs


def main(argv: list[str]) -> int:
    """CLI: write a snapshot, or compare two."""
    if argv and argv[0] == "--compare":
        diffs = compare(json.load(open(argv[1])), json.load(open(argv[2])))
        print("\n".join(diffs) if diffs else "snapshots agree")
        return 1 if diffs else 0
    out = argv[0]
    rows = int(argv[1]) if len(argv) > 1 else 20000
    cols = int(argv[2]) if len(argv) > 2 else 40
    with open(out, "w") as fh:
        json.dump(snapshot(rows, cols), fh, indent=1)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
