"""Driver for the SCM leg: the same roster and protocol on beds whose answer key comes from their graph.

The three legs answer different questions and are kept in separate results files on purpose:

* the **real** leg (`run_phase0`) asks whether selection pays on data nobody generated, where no ground
  truth exists and the only comparator is `all-features`;
* the **adversarial** leg (`run_synth_control`) asks whether the real leg's verdict is about real data or
  about the downstream model, using beds whose truth is hand-written alongside their generator;
* this leg asks what a method RECOVERS when the answer key is derived from the structural model that
  produced the data, so support recovery is a fact rather than an annotation, and each bed's ceiling is
  calibrated rather than incidental.

Pooling them would average over three different questions, which is why the runner writes each to its own
file and the analysis reports them separately.

    python -m mlframe.feature_selection._benchmarks.fs_hybrid.run_scm_beds --seeds 20
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Sequence

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_results")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the SCM leg and return the number of cells executed."""
    parser = argparse.ArgumentParser(description="Benchmark leg over the SCM scenario library")
    parser.add_argument("--seeds", type=int, default=20, help="how many dataset seeds from the development range")
    parser.add_argument("--rows", type=int, default=0, help="rows per bed; 0 keeps the module default")
    parser.add_argument("--include-null", action="store_true", help="also run the null beds, which carry no relevant column")
    parser.add_argument("--retry-failed", action="store_true", help="re-run cells recorded as anything other than ok")
    parser.add_argument("--out", default="", help="results JSONL; defaults to _results/scm_beds.jsonl")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from ._scm_beds import SCM_BED_ROWS, scm_bed_scenarios
    from .run_experiment import run_grid

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out = args.out or os.path.join(RESULTS_DIR, "scm_beds.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    beds = scm_bed_scenarios(include_null=args.include_null, n_samples=int(args.rows) or SCM_BED_ROWS)
    seeds = list(range(int(args.seeds)))

    logger.info("scm leg: seeds=%s beds=%s out=%s", len(seeds), [name for name, _ in beds], out)
    executed = run_grid(scenarios=beds, dataset_seeds=seeds, cv_seeds=(0,), results_path=out, resume=True, retry_failed=args.retry_failed)
    logger.info("cells executed this run: %s", executed)
    return executed


if __name__ == "__main__":
    raise SystemExit(0 if main() >= 0 else 1)
