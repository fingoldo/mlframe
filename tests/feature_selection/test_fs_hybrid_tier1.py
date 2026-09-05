"""Tier-1 benchmark gate: a handful of real cells, with committed numbers, on every run of this suite.

This repository contains roughly two hundred `bench_*` and `profile_*` scripts and fourteen workflows, and
none of them runs a benchmark. Results land in a gitignored directory, get read once, and the next change to
a selector is measured against nobody. A one-off campaign shares that fate by construction, so the smallest
possible slice of the benchmark runs here instead, with the numbers it produced written down.

What it pins is chosen so that only a real change can move it:

* the difficulty of each bed, through the `all-features` null's downstream AUC. If a generator changes, this
  moves first, and everything computed on that bed afterwards is a different question.
* a known blindness: on `xor3_plus_marginal_decoy` a marginal-greedy selector recovers NONE of the three
  parity operands, because each has zero marginal mutual information with the target by construction while
  the decoy has plenty. Recovery above zero here means either the bed stopped being a parity bed or the arm
  stopped being marginal -- both worth a failing test.
* the variance-sort tripwire, which must not look like a selector. It ignores the target entirely, so
  whenever it recovers a real support the bed has leaked variance ordering into relevance.

**Updating these numbers is allowed and is not an add-only ratchet.** When a deliberate change moves a band,
re-run this file, replace the entry, and say in the commit message which change moved it and why the new
value is right. What is not allowed is widening a band to make a failure go away without that sentence.

Timing: the six cells themselves cost about fifteen seconds in a warm process. A cold one pays another
thirty-odd for first-call compilation of the numba kernels and the downstream booster, which the rest of the
suite pays anyway. `mrmr` is exercised only on the narrow parity bed, where it costs two seconds; on the
wider bed its feature-engineering machinery costs forty-six, which is what would have made this gate too
slow to keep and therefore not a gate at all.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Set, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from mlframe.feature_selection._benchmarks.fs_hybrid.adversarial_scenarios import group_additive, xor3_plus_marginal_decoy
from mlframe.feature_selection._benchmarks.fs_hybrid.run_experiment import CellSpec, build_arm_roster, run_cell

# Measured on the committed generators at seed 0, n=800. Bands are wide enough for library-version drift and
# narrow enough that a changed bed or a broken arm falls outside them.
NULL_AUC_BANDS: Dict[str, Tuple[float, float]] = {
    "group_additive": (0.58, 0.71),
    "xor3_plus_marginal_decoy": (0.74, 0.86),
}
# The parity operands carry zero marginal information by construction, so a marginal selector cannot find
# them. This is an equality against zero on purpose: any recovery at all is a structural change.
MARGINAL_RECOVERY_ON_PARITY = 0.0
# variance-sort never sees the target. Recovering half of a support is already more than chance deserves.
VARIANCE_SORT_RECOVERY_CEILING = 0.55
TIER1_SEED = 0
TIER1_N = 800
K_LABEL = "1k"

BEDS: Dict[str, Tuple[Callable[[int], Any], Tuple[str, ...]]] = {
    "group_additive": (lambda seed: group_additive(seed=seed, n=TIER1_N, n_group=6, n_noise=14), ("all-features", "univariate-mi", "variance-sort")),
    "xor3_plus_marginal_decoy": (lambda seed: xor3_plus_marginal_decoy(seed=seed, n=TIER1_N, n_noise=17), ("all-features", "univariate-mi", "mrmr")),
}


def _run_bed(bed: str) -> Dict[str, Dict[str, Any]]:
    """Run one bed's arms at the tier-1 seed and return `{arm: record}`."""
    gen, arms = BEDS[bed]
    x, y, truth = gen(TIER1_SEED)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=TIER1_SEED, stratify=y)
    roster = build_arm_roster(int(x.shape[1]), random_state=TIER1_SEED)
    out: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        spec = CellSpec(scenario=bed, arm=arm, dataset_seed=TIER1_SEED, cv_seed=0, protocol_version="tier1", config={})
        out[arm] = run_cell(spec, roster[arm], x_train, np.asarray(y_train), x_test, np.asarray(y_test), truth)
    return out


@pytest.fixture(scope="module")
def cells() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Run the whole tier-1 grid once and share it across the assertions."""
    return {bed: _run_bed(bed) for bed in BEDS}


def _auc(record: Dict[str, Any], model: str = "lightgbm") -> float:
    """Return one cell's downstream AUC at the pinned K label."""
    return float(record["scores"][K_LABEL]["models"][model]["roc_auc"])


def _recovery(record: Dict[str, Any]) -> float:
    """Return the fraction of the declared relevant columns the cell's selection contains."""
    relevant: Set[str] = {str(c) for c in (record.get("truth_relevant") or [])}
    block = (record.get("selected") or {}).get(K_LABEL) or {}
    assert block.get("status") == "ok", f"expected a stored selection, got {block.get('status')!r}"
    chosen = {str(c) for c in block.get("columns", ())}
    assert relevant, "the bed must declare a relevant set for recovery to mean anything"
    return len(chosen & relevant) / len(relevant)


class TestEveryCellRuns:
    """Nothing below means anything if a cell silently failed."""

    def test_all_cells_complete(self, cells: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """A crashed cell is a failing gate, not a missing row."""
        failures = {(bed, arm): rec.get("error") for bed, arms in cells.items() for arm, rec in arms.items() if rec["status"] != "ok"}
        assert not failures, failures

    def test_selections_are_stored(self, cells: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """Recovery is unmeasurable without them, so their absence must fail here rather than downstream."""
        for bed, arms in cells.items():
            for arm, rec in arms.items():
                status = ((rec.get("selected") or {}).get(K_LABEL) or {}).get("status")
                expected = "all_features" if arm == "all-features" else "ok"
                assert status == expected, f"{bed}/{arm}: selection status {status!r}"


class TestBedDifficulty:
    """The null's score is the bed's difficulty; if it moves, every later comparison changed meaning."""

    @pytest.mark.parametrize("bed", sorted(BEDS))
    def test_null_auc_inside_its_committed_band(self, bed: str, cells: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """`all-features` on each bed stays inside the band this file records."""
        low, high = NULL_AUC_BANDS[bed]
        auc = _auc(cells[bed]["all-features"])
        assert low <= auc <= high, f"{bed}: all-features AUC {auc:.4f} outside [{low}, {high}]"


class TestKnownBlindness:
    """The result the parity bed exists to produce."""

    def test_marginal_selectors_recover_none_of_the_parity_operands(self, cells: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """Zero marginal information means a marginal-greedy ranking cannot see the operands at all."""
        for arm in ("univariate-mi", "mrmr"):
            recovery = _recovery(cells["xor3_plus_marginal_decoy"][arm])
            assert recovery == MARGINAL_RECOVERY_ON_PARITY, f"{arm} recovered {recovery:.3f} of the parity operands"

    def test_the_decoy_is_what_they_take_instead(self, cells: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """The bed only tests blindness if something marginally attractive is there to be taken."""
        block = (cells["xor3_plus_marginal_decoy"]["univariate-mi"].get("selected") or {})[K_LABEL]
        chosen = {str(c) for c in block["columns"]}
        assert chosen, "the arm selected nothing, so it cannot have preferred the decoy"


class TestControlArm:
    """variance-sort is a tripwire, and a tripwire that starts winning is the alarm."""

    def test_variance_sort_does_not_recover_a_real_support(self, cells: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """It never sees the target; recovering the support would mean the bed leaked variance ordering."""
        recovery = _recovery(cells["group_additive"]["variance-sort"])
        assert recovery <= VARIANCE_SORT_RECOVERY_CEILING, f"variance-sort recovered {recovery:.3f} of a real support"


class TestGateStaysAffordable:
    """A gate that grows slow gets deleted, so its cost is asserted rather than hoped for."""

    def test_one_bed_runs_in_under_a_minute(self) -> None:
        """Re-running a single bed stays well inside the tier-1 budget."""
        started = time.perf_counter()
        _run_bed("xor3_plus_marginal_decoy")
        assert time.perf_counter() - started < 60.0


def test_beds_are_frames_with_the_declared_width() -> None:
    """The generators still produce what the bands were measured on, so a band cannot silently change bed."""
    for bed, (gen, _) in BEDS.items():
        x, y, _truth = gen(TIER1_SEED)
        assert isinstance(x, pd.DataFrame) and len(x) == TIER1_N, bed
        assert len(np.unique(np.asarray(y))) == 2, bed
