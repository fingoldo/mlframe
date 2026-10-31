"""Run ``family_scale`` for every ``fe_*_enable`` family (plus BASE) of one mode and size, resuming after the families already in the results file.

python -m mlframe.feature_selection._benchmarks.profiling.family_batch <n> <p> <mode> [timeout_s] [results.jsonl]

A record is kept for a family unless it was measured on a busy machine (``contaminated``) or failed for lack of memory; those are run again.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mlframe.feature_selection._benchmarks.profiling import family_scale as F
from mlframe.feature_selection._benchmarks.profiling._paths import OUT_DIR


def finished_families(out: Path, n: int, p: int, mode: str) -> set:
    """Families with a usable record for this size and mode in the results file."""
    done: set = set()
    if not out.exists():
        return done
    for line in out.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("n") == n and r.get("p") == p and r.get("mode") == mode and not r.get("contaminated") and r.get("status") not in ("MEMORY_ERROR", "SKIPPED_MEMORY"):
            done.add(r["family"])
    return done


def main() -> None:
    """Run the families one after another as child processes and print one line per family."""
    n, p, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    timeout = int(sys.argv[4]) if len(sys.argv) > 4 else 900
    out = Path(sys.argv[5]) if len(sys.argv) > 5 else OUT_DIR / f"family_{n}x{p}_{mode}.jsonl"
    names = ["BASE"] + [f[3:-7] for f in F._flag_names()]
    done = finished_families(out, n, p, mode)
    for i, fam in enumerate(names):
        if fam in done:
            continue
        cmd = [sys.executable, "-I", F.__file__, "run", "--n", str(n), "--p", str(p), "--mode", mode, "--family", fam, "--timeout", str(timeout), "--out", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(f"[{i + 1}/{len(names)}] {r.stdout.strip().splitlines()[-1][:200] if r.stdout.strip() else r.stderr[-200:]}", flush=True)
    print("BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
