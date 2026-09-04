"""Phase 0 driver: the real-data leg the pre-registration's kill criterion is decided on.

Two modes, and the distinction is binding rather than cosmetic:

`--mode dev` runs on seeds from the development range, where thresholds and parameters may be tuned freely.
`--mode confirm` runs on the reserved report-only range; tuning anything against those seeds is a
pre-registration violation, so this mode exists to make the two runs impossible to confuse in the output.

Resumable: cells already present in the JSONL are skipped, so a run killed by machine pressure continues
where it stopped rather than restarting.

    python -m mlframe.feature_selection._benchmarks.fs_hybrid.run_phase0 --mode dev --seeds 1
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import List, Sequence

logger = logging.getLogger(__name__)

# Development range: tune freely. Reserved range: report only, never tuned against (pre-registration section 6).
DEV_SEED_BASE = 0
CONFIRM_SEED_BASE = 1000

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_results")


def _seeds(mode: str, count: int) -> List[int]:
    """Return `count` seeds from the range the mode is allowed to use."""
    base = DEV_SEED_BASE if mode == "dev" else CONFIRM_SEED_BASE
    return [base + i for i in range(int(count))]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the real-data leg and return the number of cells executed."""
    parser = argparse.ArgumentParser(description="Phase 0 real-data leg")
    parser.add_argument("--mode", choices=("dev", "confirm"), default="dev")
    parser.add_argument("--seeds", type=int, default=1, help="how many dataset seeds from the mode's range")
    parser.add_argument("--include-ineligible", action="store_true", help="also run beds excluded from the kill criterion")
    parser.add_argument("--arms", default="", help="comma-separated arm subset; empty means the full roster")
    parser.add_argument("--retry-failed", action="store_true", help="re-run cells recorded as anything other than ok")
    parser.add_argument("--out", default="", help="results JSONL; defaults to a mode-specific file under _results/")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from ._real_beds import real_bed_scenarios
    from .run_experiment import run_grid

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out = args.out or os.path.join(RESULTS_DIR, f"phase0_{args.mode}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    seeds = _seeds(args.mode, args.seeds)
    beds = real_bed_scenarios(include_ineligible=args.include_ineligible)

    wanted = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    roster = None
    if wanted:
        # Only meaningful for a smoke: a subset roster is built for one width and reused, whereas the default
        # path rebuilds per bed so the fixed-cardinality arms match each bed's own feature count.
        from ._arms import build_arm_roster

        widest = max(int(gen(seeds[0])[0].shape[1]) for _, gen in beds)
        full = build_arm_roster(widest, random_state=seeds[0])
        roster = {name: factory for name, factory in full.items() if name in wanted or name.startswith("random-")}
        logger.warning("arm subset requested: fixed-cardinality arms are sized for the widest bed (%s features)", widest)

    logger.info("mode=%s seeds=%s beds=%s out=%s", args.mode, seeds, [n for n, _ in beds], out)
    executed = run_grid(scenarios=beds, roster=roster, dataset_seeds=seeds, cv_seeds=(0,), results_path=out, resume=True, retry_failed=args.retry_failed)
    logger.info("cells executed this run: %s", executed)
    return executed


if __name__ == "__main__":
    raise SystemExit(0 if main() >= 0 else 1)
