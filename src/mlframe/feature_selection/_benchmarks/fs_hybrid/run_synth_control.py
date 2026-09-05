"""Synthetic control point for the Phase 0 verdict: the same arms and protocol on beds whose truth is known.

Phase 0 ran the pre-registered roster against seven eligible real beds and the kill criterion fired: with
lightgbm downstream, no arm beat the ``all-features`` null at any K on four to seven of them. That result has
two readings, and they call for opposite next steps:

* feature selection does not pay on **real** data at these widths, because the real beds carry dense,
  genuinely predictive correlation structure a strong model exploits by itself; or
* feature selection does not pay for a **strong model**, full stop, in which case the criterion says nothing
  about real data and the whole leg was mis-designed.

Only a bed with known truth separates them. Here the informative columns are constructed, so
``oracle-informative`` is available as an arm and the achievable gap over ``all-features`` is a measured
quantity rather than an assumption. If ``all-features`` also wins on beds where an oracle subset exists and
is known to carry the signal, the verdict is about the downstream model and must be read that way. If the
oracle wins here while nothing wins on the real beds, the verdict stands as a statement about real data.

The nine eligible beds are the adversarial roster minus the three null gates: a null bed carries no signal,
so "did any arm beat all-features" has no meaning on it, exactly as in the real leg's eligibility rule.

Seeds come from the development range: this run is diagnostic and interpretive, it decides no
pre-registered hypothesis, and spending reserved seeds on it would be a waste of the report-only range.

    python -m mlframe.feature_selection._benchmarks.fs_hybrid.run_synth_control --seeds 20
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import TYPE_CHECKING, List, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime, the runner imports this module's caller
    from .run_experiment import ScenarioGen

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_results")


def control_bed_scenarios(include_gates: bool = False) -> List[Tuple[str, "ScenarioGen"]]:
    """Return the known-truth beds, gate nulls excluded unless asked for."""
    from .adversarial_scenarios import ADVERSARIAL_SCENARIOS, GATE_SCENARIOS

    names = [name for name in ADVERSARIAL_SCENARIOS if include_gates or name not in GATE_SCENARIOS]
    return [(name, (lambda fn: (lambda seed: fn(seed)))(ADVERSARIAL_SCENARIOS[name])) for name in names]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the synthetic control grid and return the number of cells executed."""
    parser = argparse.ArgumentParser(description="Phase 0 synthetic control leg")
    parser.add_argument("--seeds", type=int, default=20, help="how many dataset seeds from the development range")
    parser.add_argument("--include-gates", action="store_true", help="also run the null gate beds")
    parser.add_argument("--retry-failed", action="store_true", help="re-run cells recorded as anything other than ok")
    parser.add_argument("--out", default="", help="results JSONL; defaults to _results/phase0_synth_control.jsonl")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from .run_experiment import run_grid

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out = args.out or os.path.join(RESULTS_DIR, "phase0_synth_control.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    seeds = list(range(int(args.seeds)))
    beds = control_bed_scenarios(include_gates=args.include_gates)

    logger.info("synthetic control: seeds=%s beds=%s out=%s", len(seeds), [name for name, _ in beds], out)
    executed = run_grid(scenarios=beds, dataset_seeds=seeds, cv_seeds=(0,), results_path=out, resume=True, retry_failed=args.retry_failed)
    logger.info("cells executed this run: %s", executed)
    return executed


if __name__ == "__main__":
    raise SystemExit(0 if main() >= 0 else 1)
