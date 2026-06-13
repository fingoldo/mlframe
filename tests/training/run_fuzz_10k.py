"""Drive up to 10_000 unique fuzz combos across many master_seeds.

Each pytest invocation reads ``FUZZ_SEED`` and enumerates 150 combos
deterministically. We sweep seeds, launch one pytest subprocess per
seed, accumulate the JSONL rows into
``tests/training/_fuzz_results.jsonl``, and stop when we've covered
>= ``--target`` unique canonical combos or exhausted the seed budget.

Background: the fuzz suite hits a native SIGSEGV after ~70-90 combos
in a single pytest invocation (state accumulation -- known issue
documented in ``test_fuzz_suite.py``). Spawning a fresh subprocess
per seed sidesteps this entirely: each subprocess gets a clean
process image, so 150 combos x N seeds are fully exercised.

Usage (foreground for dev):
    python -m mlframe.tests.training.run_fuzz_10k --target 10000

or just run as a script:
    python tests/training/run_fuzz_10k.py --target 10000

Progress is printed every seed; JSONL accumulates across seeds. New
fail classes (error_class values never seen before) are highlighted
in the summary line so a human can decide whether to interrupt.

Combo-isolated mode (--isolate-combos, RECOMMENDED for bug-hunts)
----------------------------------------------------------------
The per-seed mode above runs ~150 combos back-to-back in ONE pytest
process. Heavy combos (CatBoost / BorutaShap / SHAP / full-MRMR-FE)
corrupt process memory, so later combos fail non-deterministically
(random ValueError/RuntimeError/ShapeError that PASS in isolation) or
the whole process hits a Windows ``access violation`` / hangs. The
per-seed wall-timeout could not reliably kill a hung native child.

``--isolate-combos`` spawns a FRESH pytest subprocess per (seed, combo)
-- one combo per process, zero cross-combo state/memory carryover -- with
a HARD wall-timeout that kills the whole process TREE (psutil, with a
``taskkill /F /T`` fallback on Windows). A combo that crashes / hangs is
logged (``outcome=native_crash`` / ``timeout``) and the sweep continues
to the next combo in a fresh process. Pass/fail rows are written by the
child itself via ``log_combo_outcome`` (identical schema); the driver
writes only the timeout / native_crash rows the dead child could not.

    python tests/training/run_fuzz_10k.py --isolate-combos --target 12 \
        --per-combo-timeout-s 180
"""
from __future__ import annotations

import argparse
import orjson
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent  # mlframe/
RESULTS_LOG = HERE / "_fuzz_results.jsonl"

# Native crash signatures on the child's return code. On Windows an access
# violation surfaces as the unsigned STATUS_ACCESS_VIOLATION (0xC0000005 =
# 3221225477) or its signed two's-complement (-1073741819); other STATUS_*
# codes appear as large unsigned / negative rcs too. On POSIX a fatal signal
# surfaces as a negative rc (-signum). We treat any of these as a native crash
# distinct from pytest's own rc==1 (tests failed) / rc==0 (passed).
_WIN_ACCESS_VIOLATION = 0xC0000005


def _count_unique_combos() -> int:
    if not RESULTS_LOG.exists():
        return 0
    seen = set()
    with RESULTS_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = orjson.loads(line)
            except Exception:
                continue
            # Canonical key proxy -- short_id is sha-prefixed over
            # canonical_key, so same short_id => same canonical combo.
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
                r = orjson.loads(line)
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
    """Every error_class ever seen across the JSONL -- used to flag NEW
    classes that appear in a given batch."""
    classes: set[str] = set()
    if not RESULTS_LOG.exists():
        return classes
    with RESULTS_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = orjson.loads(line)
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
        # --run-fuzz is REQUIRED: conftest.pytest_collection_modifyitems skips all fuzz-marked
        # tests without it (a no-op run otherwise -- 0 combos generated). Do NOT add --fast/-m:
        # the fuzz test is slow_only, which --fast mode deselects.
        "--run-fuzz",
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


def _kill_process_tree(pid: int) -> None:
    """Kill ``pid`` and ALL its descendants. This is the crux the old
    per-seed timeout got wrong: ``subprocess.run(timeout=)`` / ``Popen.kill()``
    only signal the direct child, leaving native worker subprocesses (CatBoost
    GPU helpers, joblib loky workers, etc.) orphaned and the wall-time
    uncapped. We kill the whole tree.

    Strategy: psutil (cross-platform, terminates children first then parent,
    waits, then SIGKILLs survivors). Falls back to ``taskkill /F /T`` on
    Windows / ``os.killpg`` on POSIX if psutil is unavailable.
    """
    try:
        import psutil

        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        procs = parent.children(recursive=True)
        procs.append(parent)
        for p in procs:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        # Reap so the OS releases handles; survivors get a second SIGKILL.
        _gone, alive = psutil.wait_procs(procs, timeout=10)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        return
    except Exception:
        pass
    # Fallback: OS-native tree kill.
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, check=False,
            )
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    except Exception:
        pass


def _classify_native_crash(rc: int) -> bool:
    """True when ``rc`` looks like a native crash (access violation / segfault)
    rather than a clean pytest exit (0 pass / 1 tests-failed / 2-5 usage)."""
    if rc < 0:  # POSIX fatal signal (-signum) or signed Windows STATUS_*
        return True
    if rc == (_WIN_ACCESS_VIOLATION & 0xFFFFFFFF):  # 3221225477
        return True
    # Any large rc (Windows STATUS_* are >= 0xC0000000) that is not a normal
    # pytest exit code is a native abort.
    if rc >= 0xC0000000:
        return True
    return False


def _run_one_combo(seed: int, short_id: str, per_combo_timeout_s: int) -> tuple[int, str, bool]:
    """Spawn a FRESH pytest selecting exactly one combo via ``-k <short_id>``.

    One combo per process => zero cross-combo state/memory carryover. Returns
    ``(rc, tail_text, timed_out)``. On wall-timeout the whole process tree is
    killed (see ``_kill_process_tree``) and ``timed_out=True``.

    ``PYTHONFAULTHANDLER=1`` is set in the child env so a hung child dumps
    Python stacks (to its stderr, captured into the tail) before the wall-kill
    -- this distinguishes a genuine Python-level hang from a native AV.
    """
    env = dict(os.environ)
    env["FUZZ_SEED"] = str(seed)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/training/test_fuzz_suite.py",
        "-k", short_id,
        "--run-fuzz",
        "--no-cov", "-s", "-p", "no:randomly", "-p", "no:cacheprovider",
        "-q", "--tb=line",
    ]
    kwargs: dict = dict(
        cwd=str(REPO_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    if os.name != "nt":
        kwargs["start_new_session"] = True  # own process group for killpg fallback
    p = subprocess.Popen(cmd, **kwargs)
    tail_lines: list[str] = []
    timed_out = False

    # Wall-timeout watchdog: a background timer fires the tree-kill at exactly
    # ``per_combo_timeout_s`` REGARDLESS of stdout activity. This is the crux the
    # old per-seed timeout got wrong -- it only checked the clock when a new line
    # arrived, so a SILENT hung child (no output) would never trip it (the 2.5h
    # hang). The watchdog kills on wall-time even with zero output; the stdout
    # read loop then ends naturally when the killed pipe closes.
    import threading
    _killed = threading.Event()

    def _watchdog() -> None:
        _killed.set()
        _kill_process_tree(p.pid)

    timer = threading.Timer(per_combo_timeout_s, _watchdog)
    timer.daemon = True
    timer.start()
    try:
        for line in p.stdout:  # type: ignore[union-attr]
            if len(tail_lines) > 80:
                tail_lines.pop(0)
            tail_lines.append(line.rstrip("\n"))
        # Drain + reap. If the watchdog already tree-killed, wait returns fast.
        p.wait(timeout=30)
        if _killed.is_set():
            timed_out = True
            tail_lines.append(
                f"[driver] per-combo wall-timeout {per_combo_timeout_s}s reached -- killed process tree."
            )
    except subprocess.TimeoutExpired:
        # The streamed read finished but the process is still alive (e.g. a
        # hung native finalizer holding stdout open closed but not exiting).
        timed_out = True
        _kill_process_tree(p.pid)
        try:
            p.wait(timeout=10)
        except Exception:
            pass
        tail_lines.append("[driver] child did not exit after stdout close -- tree-killed.")
    except Exception as e:
        _kill_process_tree(p.pid)
        tail_lines.append(f"[driver] exception while running combo: {e}")
    finally:
        timer.cancel()
    rc = p.returncode if p.returncode is not None else 0
    return rc, "\n".join(tail_lines), timed_out


def _run_isolate(args: argparse.Namespace) -> int:
    """Combo-isolated bug-hunt sweep: one fresh subprocess per (seed, combo)."""
    sys.path.insert(0, str(REPO_ROOT))
    from tests.training._fuzz_combo import enumerate_combos

    print(f"[driver] ISOLATE mode: target={args.target} combos, "
          f"per_combo_timeout_s={args.per_combo_timeout_s}, "
          f"seed_start={args.seed_start}, max_seeds={args.max_seeds}", flush=True)

    done = 0
    totals: dict[str, int] = {}
    t_campaign = time.time()
    for i in range(args.max_seeds):
        if done >= args.target:
            break
        seed = args.seed_start + i
        combos = enumerate_combos(target=150, master_seed=seed)
        for combo in combos:
            if done >= args.target:
                break
            short_id = combo.short_id()
            size_before = _file_size()
            t0 = time.time()
            rc, tail, timed_out = _run_one_combo(seed, short_id, args.per_combo_timeout_s)
            dur = time.time() - t0
            wrote_row = _file_size() > size_before

            if timed_out:
                outcome = "timeout"
                # The child died before logging; the driver records the row so
                # a hung combo is never silently lost from the sweep.
                if not wrote_row:
                    _log_driver_row(combo, seed, outcome, dur,
                                    error_class="WallTimeout",
                                    error_summary=f"per-combo wall-timeout {args.per_combo_timeout_s}s; process tree killed",
                                    tail=tail)
            elif _classify_native_crash(rc):
                outcome = "native_crash"
                if not wrote_row:
                    _log_driver_row(combo, seed, outcome, dur,
                                    error_class="NativeCrash",
                                    error_summary=f"native crash (rc={rc}; access violation / segfault)",
                                    tail=tail)
            elif rc == 0:
                outcome = "pass"  # child logged its own row
            else:
                # rc==1 (pytest: tests failed). The child's except-branch
                # already logged the fail row with the real error_class/summary.
                outcome = "fail"
                if not wrote_row:
                    # Defensive: collection error / fail before log_combo_outcome ran.
                    _log_driver_row(combo, seed, outcome, dur,
                                    error_class="PytestNonZero",
                                    error_summary=f"pytest rc={rc}, no combo row emitted (collection/early error)",
                                    tail=tail)

            totals[outcome] = totals.get(outcome, 0) + 1
            done += 1
            flag = ""
            if outcome in ("timeout", "native_crash"):
                flag = f"  *** {outcome.upper()} ***"
            print(f"[driver] [{done}/{args.target}] seed={seed} {short_id} "
                  f"rc={rc} {dur:.1f}s -> {outcome}{flag}", flush=True)
            if outcome in ("timeout", "native_crash", "fail"):
                for ln in tail.splitlines()[-8:]:
                    print(f"[driver]     | {ln}", flush=True)

    elapsed = time.time() - t_campaign
    print(f"\n[driver] ISOLATE DONE. combos={done}, outcomes={totals}, "
          f"elapsed={elapsed:.1f}s.", flush=True)
    return 0


def _log_driver_row(combo, seed: int, outcome: str, dur: float,
                    error_class: str, error_summary: str, tail: str) -> None:
    """Append a results row from the DRIVER for combos whose child died before
    logging (timeout / native_crash). Reuses the exact JSONL schema."""
    row = {
        **combo.to_json(),
        "outcome": outcome,
        "duration_s": round(dur, 3),
        "master_seed": int(seed),
        "error_class": error_class,
        "error_summary": error_summary[:300],
        "extra": {"isolated": True, "driver_logged": True, "tail": "\n".join(tail.splitlines()[-15:])[:1500]},
    }
    try:
        RESULTS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with RESULTS_LOG.open("ab") as f:
            f.write(orjson.dumps(row, option=orjson.OPT_SORT_KEYS) + b"\n")
    except OSError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=10_000,
                    help="stop after this many unique canonical combos covered (default 10_000)")
    ap.add_argument("--max-seeds", type=int, default=200,
                    help="hard cap on number of seeds to iterate (default 200 -> 30k combo cap)")
    ap.add_argument("--seed-start", type=int, default=2026_04_22,
                    help="first master_seed to run (default 20260422)")
    ap.add_argument("--per-seed-timeout-s", type=int, default=1800,
                    help="kill a single seed's pytest after this many seconds (default 1800)")
    ap.add_argument("--isolate-combos", action="store_true",
                    help="RECOMMENDED bug-hunt mode: spawn a fresh pytest subprocess per (seed, combo) "
                         "with a hard per-combo wall-timeout that kills the whole process tree. A "
                         "crashing/hanging combo cannot poison or stall the others.")
    ap.add_argument("--per-combo-timeout-s", type=int, default=180,
                    help="(isolate mode) hard wall-timeout per combo; on expiry the whole process tree "
                         "is killed and the combo is logged as outcome=timeout (default 180)")
    args = ap.parse_args()

    if args.isolate_combos:
        return _run_isolate(args)

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
        # Retry once on fast-fail: when pytest exits nonzero under
        # 120 s AND produced zero new combo rows, the cause is almost
        # always a flaky import during collection (shap, transformers --
        # seen 2026-04-23 on Windows in ~50% of subprocess spawns). A
        # fresh subprocess usually succeeds. Cap at one retry to avoid
        # burning wallclock on genuinely-broken seeds.
        dur = time.time() - t_seed
        if rc != 0 and dur < 120 and _file_size() == size_before:
            print(f"[driver]   fast-fail detected ({dur:.1f}s, no combos logged) "
                  f"-- retrying seed {seed} once.", flush=True)
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
            # may leave rc=3221225477 or similar. Keep going -- another
            # seed in a fresh process image will pick up.
            print(f"[driver]   (pytest rc={rc}; tail below)", flush=True)
            print("\n".join(tail.splitlines()[-15:]), flush=True)

    elapsed = time.time() - t_campaign
    final = _count_unique_combos()
    print(f"\n[driver] DONE. Total unique combos: {final}, elapsed: {elapsed:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
