"""End-to-end A/B of ``ReportingConfig.async_render`` on a real ``train_mlframe_models_suite`` run.

Runs the same small multi-model suite (charts saved) with the report rendering inline (``off``), on a thread pool (``thread``) and on spawned workers
with shared memory (``process``), and checks what the queue must guarantee:

* wall time of each arm (paired in the order given; best of ``--rounds``),
* the saved artifacts are the same files, and the same bytes where the format is deterministic (an ``off`` vs ``off`` control run measures which
  files are not: plotly HTML embeds random element ids and a timestamp),
* nothing is left behind: no child process outlives the suite and no shared-memory attachment stays open in the parent.

Run:  ``python -m mlframe.reporting._benchmarks.bench_async_render_suite_ab --rows 20000 --rounds 2``
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ARMS = ("off", "thread", "process")


def _frame(rows: int, seed: int = 0) -> pd.DataFrame:
    """Binary-classification frame with a linear signal plus noise columns."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, 12))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(rows) * 0.7 > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)])
    df["target"] = y
    return df


def _digests(root: Path) -> Dict[str, str]:
    """``{relative path: sha1}`` of every file under ``root``."""
    out: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[path.relative_to(root).as_posix()] = hashlib.sha1(path.read_bytes(), usedforsecurity=False).hexdigest()
    return out


def run_arm(arm: str, rows: int, data_dir: Path) -> Tuple[float, Dict[str, object]]:
    """One suite run; returns ``(wall seconds, metadata)``."""
    from mlframe.training.configs import OutputConfig, ReportingConfig
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

    reporting = ReportingConfig(
        show_perf_chart=True,
        show_fi=True,
        async_render=(arm != "off"),
        async_render_backend=("process" if arm == "process" else "thread"),
    )
    t0 = time.perf_counter()
    _, metadata = train_mlframe_models_suite(
        df=_frame(rows),
        target_name="target",
        model_name="ab",  # the same name in every arm: it is part of the saved file names
        features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(classification_targets=["target"]),
        mlframe_models=["cb", "lgb"],
        hyperparams_config={"iterations": 60},
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        output_config=OutputConfig(data_dir=str(data_dir), models_dir="models", save_charts=True),
        reporting_config=reporting,
        verbose=0,
    )
    return time.perf_counter() - t0, metadata


def _leftovers() -> List[str]:
    """Child processes of this interpreter that are still alive, plus shared-memory attachments still open in it."""
    problems: List[str] = []
    try:
        import psutil

        problems += [f"child pid {p.pid} {p.name()}" for p in psutil.Process().children(recursive=True) if p.is_running()]
    except ImportError:
        pass
    from mlframe.reporting import _shared_arrays

    problems += [f"shm attachment {name}" for name in getattr(_shared_arrays, "_ATTACHED", {})]
    return problems


def main(argv: List[str] | None = None) -> int:
    """Run the arms and print the table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--arms", default=",".join(ARMS))
    args = parser.parse_args(argv)
    arms = [a for a in args.arms.split(",") if a]
    root = Path(tempfile.mkdtemp(prefix="bench_async_ab_"))
    walls: Dict[str, List[float]] = {a: [] for a in arms}
    trees: Dict[str, Dict[str, str]] = {}
    try:
        run_arm("off", min(args.rows, 2000), root / "warm")  # warm numba / imports out of the timings
        for rnd in range(args.rounds):
            for arm in arms:
                out = root / f"{arm}_{rnd}"
                wall, metadata = run_arm(arm, args.rows, out)
                walls[arm].append(wall)
                trees[f"{arm}_{rnd}"] = _digests(out)
                if arm != "off":
                    print(f"{arm} round {rnd}: {wall:.1f}s  async_render={metadata.get('async_render')}")
        control_a, control_b = root / "ctl_a", root / "ctl_b"
        run_arm("off", args.rows, control_a)
        run_arm("off", args.rows, control_b)
        da, db = _digests(control_a), _digests(control_b)
        unstable = sorted(k for k in da if k in db and da[k] != db[k])
        print("\n== wall seconds (best of rounds)")
        for arm in arms:
            print(f"  {arm:8s} {min(walls[arm]):7.1f}s   all={['%.1f' % w for w in walls[arm]]}")
        print(f"\n== control: off vs off differ in {len(unstable)} of {len(da)} files (the nondeterministic ones)")
        by_ext: Dict[str, int] = {}
        for k in unstable:
            by_ext[os.path.splitext(k)[1]] = by_ext.get(os.path.splitext(k)[1], 0) + 1
        print("  by extension:", by_ext)
        for arm in arms:
            if arm == "off":
                continue
            other = trees[f"{arm}_0"]
            only_off = sorted(set(trees["off_0"]) - set(other))
            only_arm = sorted(set(other) - set(trees["off_0"]))
            diff = sorted(k for k in other if k in trees["off_0"] and other[k] != trees["off_0"][k] and k not in unstable)
            print(f"\n== {arm} vs off: same file set={not only_off and not only_arm}  missing={len(only_off)} extra={len(only_arm)}  "
                  f"byte-different among the deterministic files={len(diff)}")
            for k in (only_off + only_arm + diff)[:10]:
                print("   ", k)
        leftovers = _leftovers()
        print("\n== leftovers after the suite:", leftovers or "none")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
