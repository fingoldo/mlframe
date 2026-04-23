"""Drive up to 10_000 unique fuzz combos across many master_seeds.

Each pytest invocation reads ``FUZZ_SEED`` and enumerates 150 combos
deterministically. We sweep seeds, launch one pytest subprocess per
seed, accumulate the JSONL rows into
``tests/training/_fuzz_results.jsonl``, and stop when we've covered
>= ``--target`` unique canonical combos or exhausted the seed budget.

Background: the fuzz suite hits a native SIGSEGV after ~70-90 combos
in a single pytest invocation (state accumulation — known issue
documented in ``test_fuzz_suite.py``). Spawning a fresh subprocess
per seed sidesteps this entirely: each subprocess gets a clean
process image, so 150 combos × N seeds are fully exercised.

Usage (foreground for dev):
    python -m mlframe.tests.training.run_fuzz_10k --target 10000

or just run as a script:
    python tests/training/run_fuzz_10k.py --target 10000

Progress is printed every seed; JSONL accumulates across seeds. New
fail classes (error_class values never seen before) are highlighted
in the summary line so a human can decide whether to interrupt.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent  # mlframe/
RESULTS_LOG = HERE / "_fuzz_results.jsonl"


def _count_unique_combos() -> int:
    if not RESULTS_LOG.exists():
        return 0
    seen = set()
    with RESULTS_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            # Canonical key proxy — short_id is sha-prefixed over
            # canonical_key, so same short_id ⇒ same canonical combo.
            seen.add(r.get("short_id"))
    return len(seen)


def _summarize_since(start_offset: int) -> dict:
    """Summarize rows appended after start_offset (bytes) in the JSONL."""
    totals = {"pass": 0, "fail": 0, "xfail": 0, "xpass": 0, "skip": 0, "?": 0}
    fails_by_class: dict[str, int] = {}
    if not RESULTS_LOG.exists():
        return {"totals": totals, "fails_by_class": fails_by_class}
    with RESULTS_LOG.open("rb") as f:
        f.seek(start_offset)
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            totals[r.get("outcome", "?")] = totals.get(r.get("outcome", "?"), 0) + 1
            if r.get("outcome") == "fail":
                cls = r.get("error_class", "?")
                fails_by_class[cls] = fails_by_class.get(cls, 0) + 1
    return {"totals": totals, "fails_by_class": fails_by_class}


def _file_size() -> int:
    return RESULTS_LOG.stat().st_size if RESULTS_LOG.exists() else 0


def _all_fail_classes() -> set[str]:
    """Every error_class ever seen across the JSONL — used to flag NEW
    classes that appear in a given batch."""
    classes: set[str] = set()
    if not RESULTS_LOG.exists():
        return classes
    with RESULTS_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("outcome") == "fail":
                cls = r.get("error_class")
                if cls:
                    classes.add(cls)
    return classes


def _run_one_seed(seed: int, per_seed_timeout_s: int) -> tuple[int, str]:
    """Run pytest with FUZZ_SEED=seed. Returns (rc, tail_text).

    Sets -p no:randomly so the parametrize order is fixed, and uses
    --tb=no -q because the per-combo outcomes go to the JSONL (we
    don't need pytest's own traceback echo for every fail).
    """
    env = dict(os.environ)
    env["FUZZ_SEED"] = str(seed)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/training/test_fuzz_suite.py::test_fuzz_train_mlframe_models_suite",
        "--no-cov", "-s", "-p", "no:randomly", "-q", "--tb=no",
    ]
    p = subprocess.Popen(
        cmd, cwd=str(REPO_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    tail_lines: list[str] = []
    t_start = time.time()
    try:
        # Stream output but only keep tail (memory-frugal); kill on timeout.
        for line in p.stdout:  # type: ignore[union-attr]
            if tail_lines and len(tail_lines) > 60:
                tail_lines.pop(0)
            tail_lines.append(line.rstrip("\n"))
            if time.time() - t_start > per_seed_timeout_s:
                p.kill()
                tail_lines.append(
                    f"[driver] per-seed timeout {per_seed_timeout_s}s reached, killed."
                )
                break
        p.wait(timeout=30)
    except Exception as e:
        try:
            p.kill()
        except Exception:
            pass
        tail_lines.append(f"[driver] exception: {e}")
    return p.returncode or 0, "\n".join(tail_lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=10_000,
                    help="stop after this many unique canonical combos covered (default 10_000)")
    ap.add_argument("--max-seeds", type=int, default=200,
                    help="hard cap on number of seeds to iterate (default 200 → 30k combo cap)")
    ap.add_argument("--seed-start", type=int, default=2026_04_22,
                    help="first master_seed to run (default 20260422)")
    ap.add_argument("--per-seed-timeout-s", type=int, default=1800,
                    help="kill a single seed's pytest after this many seconds (default 1800)")
    args = ap.parse_args()

    print(f"[driver] target={args.target} unique combos, max_seeds={args.max_seeds}, "
          f"seed_start={args.seed_start}, per_seed_timeout_s={args.per_seed_timeout_s}")
    initial = _count_unique_combos()
    print(f"[driver] existing unique combos in JSONL: {initial}")

    t_campaign = time.time()
    for i in range(args.max_seeds):
        seed = args.seed_start + i
        already = _count_unique_combos()
        if already >= args.target:
            print(f"[driver] target reached ({already} >= {args.target}); stopping.")
            return 0

        size_before = _file_size()
        known_classes = _all_fail_classes()
        t_seed = time.time()
        print(f"\n[driver] === seed {seed} (batch {i+1}/{args.max_seeds}, "
              f"{already}/{args.target} combos so far) ===", flush=True)
        rc, tail = _run_one_seed(seed, args.per_seed_timeout_s)
        dur = time.time() - t_seed

        summary = _summarize_since(size_before)
        after = _count_unique_combos()
        delta = after - already
        new_fail_classes = set(summary["fails_by_class"]) - known_classes

        # Highlight lines
        print(f"[driver] seed {seed} rc={rc} duration={dur:.1f}s "
              f"added={delta} combos, total={after}", flush=True)
        print(f"[driver]   outcomes: {summary['totals']}", flush=True)
        if summary["fails_by_class"]:
            print(f"[driver]   fails by class: {summary['fails_by_class']}", flush=True)
        if new_fail_classes:
            print(f"[driver]   *** NEW FAIL CLASSES: {sorted(new_fail_classes)} ***", flush=True)
        if rc != 0:
            # Pytest non-zero is expected (fails emitted). Native SIGSEGV
            # may leave rc=3221225477 or similar. Keep going — another
            # seed in a fresh process image will pick up.
            print(f"[driver]   (pytest rc={rc}; tail below)", flush=True)
            print("\n".join(tail.splitlines()[-15:]), flush=True)

    elapsed = time.time() - t_campaign
    final = _count_unique_combos()
    print(f"\n[driver] DONE. Total unique combos: {final}, elapsed: {elapsed:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
