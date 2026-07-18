"""Fix E — nightly-style master_seed rotation driver.

Rotates ``FUZZ_SEED`` across N iterations, runs the combo fuzz suite
once per seed, aggregates fails, and prints a summary. Each row in
``_fuzz_results.jsonl`` carries its originating ``master_seed`` (see
``log_combo_outcome`` in ``_fuzz_combo.py``) so post-hoc analysis can
separate seed-specific escapes from seed-invariant bugs.

Usage
-----

```bash
# Rotate over the next 5 calendar days' seeds, starting today.
python tests/training/run_fuzz_seed_rotation.py --n 5

# Rotate over explicit seeds.
python tests/training/run_fuzz_seed_rotation.py --seeds 1 2 3

# CI-style: one run, seed picked from $FUZZ_SEED if set else today's date.
python tests/training/run_fuzz_seed_rotation.py --n 1
```

Designed to be called from a GitHub Actions ``schedule`` cron job:

```yaml
# .github/workflows/fuzz-nightly.yml (sketch)
on:
  schedule:
    - cron: '0 3 * * *'
jobs:
  fuzz:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - run: pytest mlframe/tests/training/test_fuzz_suite.py --no-cov -p no:randomly
        env:
          FUZZ_SEED: ${{ github.run_number }}
```
"""

from __future__ import annotations

import argparse
import datetime as dt
import orjson
import os
import subprocess  # nosec B404 -- test-only local trusted subprocess invocation (fixed argv, no shell, no untrusted input)
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]  # tests/training/.. = mlframe/ ; parents[2] = repo root
_RESULTS = _HERE / "_fuzz_results.jsonl"
_SEED_SUMMARY = _HERE / "_fuzz_seed_summary.jsonl"


def _run_one_seed(seed: int, extra_pytest_args: list[str]) -> tuple[int, int, int]:
    """Run the combo fuzz suite once with FUZZ_SEED=seed. Returns (passed, failed, xfailed)."""
    env = os.environ.copy()
    env["FUZZ_SEED"] = str(seed)
    # Use pytest's JUnit XML for structured results; pure stdout is fragile.
    junit_path = _HERE / f"_junit_seed_{seed}.xml"
    cmd = [sys.executable, "-m", "pytest", str(_HERE / "test_fuzz_suite.py"), "--no-cov", "-p", "no:randomly", f"--junitxml={junit_path}", "--tb=line", *extra_pytest_args]
    print(f"\n=== Running seed {seed} ===")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, env=env, cwd=str(_REPO))  # nosec B603 -- fixed local argv (sys.executable/git + literal args), no shell, no untrusted input
    passed = failed = xfailed = 0
    if junit_path.exists():
        # Minimal parse — count testcase elements with/without failure child.
        import xml.etree.ElementTree as ET  # nosec B405 -- parses this run's own locally-generated pytest JUnit report, never untrusted/network XML

        try:
            tree = ET.parse(junit_path)  # nosec B314 -- parses this run's own locally-generated pytest JUnit report, never untrusted/network XML
            for tc in tree.iter("testcase"):
                if tc.find("failure") is not None or tc.find("error") is not None:
                    failed += 1
                elif tc.find("skipped") is not None:
                    xfailed += 1  # conservatively bucket skipped+xfail together
                else:
                    passed += 1
        except Exception as e:
            print(f"  junit parse failed: {e}")
    return passed, failed, xfailed


def _seeds_for_today(n: int) -> list[int]:
    """Returns n consecutive YYYYMMDD-based seeds starting from today, so fuzz runs rotate deterministically by date."""
    base = int(dt.date.today().strftime("%Y%m%d"))
    return [base + i for i in range(n)]


def main() -> int:
    """CLI entry point: runs the fuzz suite against explicit or date-rotated seeds and reports pass/fail/xfail counts."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", help="explicit seed list")
    ap.add_argument("--n", type=int, default=1, help="rotate the next N YYYYMMDD-based seeds")
    ap.add_argument("--extra", nargs="*", default=[], help="extra pytest args (e.g. -k c00)")
    args = ap.parse_args()

    seeds = args.seeds if args.seeds else _seeds_for_today(args.n)
    results: list[dict] = []
    for seed in seeds:
        passed, failed, xfailed = _run_one_seed(seed, args.extra)
        row = {
            "seed": seed,
            "passed": passed,
            "failed": failed,
            "xfailed_or_skipped": xfailed,
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
        }
        results.append(row)
        print(f"  seed={seed}: passed={passed} failed={failed} xfailed/skipped={xfailed}")
        with _SEED_SUMMARY.open("ab") as f:
            f.write(orjson.dumps(row, option=orjson.OPT_SORT_KEYS) + b"\n")

    total_pass = sum(r["passed"] for r in results)
    total_fail = sum(r["failed"] for r in results)
    print(f"\n=== Rotation summary across {len(seeds)} seeds ===")
    print(f"  Total passed: {total_pass}")
    print(f"  Total failed: {total_fail}")
    print(f"  Per-seed failure rate: {[r['failed'] for r in results]}")
    return 1 if total_fail > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
