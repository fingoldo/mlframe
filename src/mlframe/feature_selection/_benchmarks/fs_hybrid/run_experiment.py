"""Phase 0 runner for the pre-registered feature-selection benchmark (`docs/BENCHMARK_PREREGISTRATION.md`).

This benchmark is designed and run by the author of one of the arms it judges. The scenario distribution
is not a sample from any real problem population.

What one cell is::

    outer:  honest holdout, cut ONCE per (scenario, dataset_seed)
    arm:    arm.fit(train)  ->  a feature ranking (see `_matched_k`)
    score:  downstream PANEL {logistic, LightGBM} on the holdout, at matched K = 1x/2x/5x the target-set
            size, plus the arm's self-chosen K reported separately

`all-features` is the null hypothesis, not a baseline row: it is run as an arm on every single cell so
that every other arm has a paired partner on the identical holdout. The aggregation in `analyze.py` scores
each arm as a paired per-`dataset_seed` difference against it and reports every scenario where nothing
clears it as `FS does not pay here`.

Cost is `n_model_fits`, which is deterministic. Wall-clock is recorded but advisory: this host routinely
runs over a hundred python processes, so every figure using it must say so.

Results are JSONL, one object per cell, resumable: the cell key is a sha256 over the canonically encoded
cell spec, and a cell that fails writes its status (`error` / `timeout` / `crashed` / `oom`) rather than
disappearing from the file.
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ._cell_store import JsonlCellStore
from ._leaderboard import NULL_ARM
from ._matched_k import SELF_CHOSEN_K, Ranking, cut_at_k, k_grid_for_bed, ranking_from_arm_result
from ._panel import PANEL_MEMBERS, assert_wrapper_estimator_differs, base_rate_scores, fit_and_score_panel, normalized_skill
from ._protocol_types import PROTOCOL_VERSION, CellSpec, classify_exception

logger = logging.getLogger(__name__)

__all__ = ["OUT_DIR", "RESULTS_PATH", "build_arm_roster", "run_cell", "run_grid", "main"]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results")
RESULTS_PATH = os.path.join(OUT_DIR, "protocol_results.jsonl")

# Development range per the pre-registration's reserved seeds; [1000..1099] is report-only.
DEV_DATASET_SEEDS: Tuple[int, ...] = (0, 1, 2)
# `cv_seed` is a NUISANCE axis: replication budget never goes here. More than one value only buys the
# selection-instability spread reported alongside the headline.
CV_SEEDS: Tuple[int, ...] = (0,)

HOLDOUT_FRACTION = 0.4

ScenarioGen = Callable[[int], Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]]


def _declared_target_size(truth: Dict[str, Any], n_features: int) -> Optional[int]:
    """Return the pre-declared primary target-set size, or `None` when the bed declares none.

    Synthetic beds carry it in `truth["base"]`; a bed may also state it outright as
    `truth["declared_target_size"]`. A real bed has no ground truth and declares neither -- that is not an
    error, it selects the absolute K grid instead (pre-registration section 3a). What stays forbidden is
    GUESSING a target size from the data, which would let the K grid be chosen after seeing results.
    """
    base = truth.get("base")
    if base is not None:
        return int(len(base))
    declared = truth.get("declared_target_size")
    if declared is None:
        return None
    return int(min(int(declared), n_features))


def build_arm_roster(n_features: int, *, k: Optional[int] = None, random_state: int = 0) -> Dict[str, Callable[[], Any]]:
    """Return `{arm_name: factory}` from the real roster in `_arms`, with `all-features` as the null hypothesis.

    Delegates rather than duplicating: `_arms` is where each arm's verified `score_kind` lives, and a second
    roster here would drift from it silently -- the arm would keep running while its declared score kind no
    longer matched what it returns, which is exactly what the `ArmResult` contract exists to make impossible.
    """
    from ._arms import build_arm_roster as _real_roster

    roster: Dict[str, Callable[[], Any]] = dict(_real_roster(n_features, k=k, random_state=random_state))
    if NULL_ARM not in roster:
        raise ValueError(f"the arm roster must contain the null hypothesis {NULL_ARM!r}; got {sorted(roster)}")
    return roster


# Internal estimator per wrapper arm, so `assert_wrapper_estimator_differs` can refuse a tautological cell.
WRAPPER_INTERNAL_ESTIMATOR: Dict[str, Optional[str]] = {
    "rfecv_lgbm": "lightgbm",
    "rfecv_logit": "logistic",
}


def _fit_arm(factory: Callable[[], Any], x_train: pd.DataFrame, y_train: np.ndarray, cv_seed: int) -> Tuple[Any, float, float]:
    """Fit one arm, returning `(result_or_fitted_selector, wall_seconds, process_seconds)`.

    An arm exposing a `cv_seed` attribute receives the nuisance seed; the rest simply ignore it.

    Two shapes are accepted. `_arms.BaseArm` exposes `run(X, y) -> ArmResult` and times itself around the
    selector call only, so its own numbers are tighter than anything measured out here and are preferred.
    A bare sklearn-style object exposing `fit` is timed here and returned as-is, which is what the protocol
    tests use; without that fallback every duck-typed test double would have to grow a `run`.
    """
    arm = factory()
    if hasattr(arm, "cv_seed"):
        setattr(arm, "cv_seed", cv_seed)
    runner = getattr(arm, "run", None)
    if callable(runner):
        result = runner(x_train, y_train)
        return result, float(getattr(result, "wall_time_s", 0.0)), float(getattr(result, "process_time_s", 0.0))
    wall0, proc0 = time.perf_counter(), time.process_time()
    arm.fit(x_train, y_train)
    return arm, time.perf_counter() - wall0, time.process_time() - proc0


def _selection_sets(
    ranking: Ranking, target_size: Optional[int], n_features: int, constant_selection: bool = False
) -> Tuple[Dict[str, Optional[List[str]]], str]:
    """Build `({K label: selected columns}, k_grid_mode)` for the matched-K grid plus the self-chosen-K row.

    `constant_selection` is for the `all-features` null hypothesis, whose selection is the whole column
    set at every K label: it is the paired partner every other arm is differenced against, so it must
    carry a value in every matched-K row rather than being skipped as unrankable.

    The mode is returned, not just used, because a synthetic `2k` and a real `k20` label answer different
    questions -- pooling them would average over two different denominators.
    """
    sets: Dict[str, Optional[List[str]]] = {SELF_CHOSEN_K: list(ranking.selected)}
    grid, mode = k_grid_for_bed(target_size, n_features)
    for label, k in grid.items():
        sets[label] = list(ranking.selected) if constant_selection else cut_at_k(ranking, k)
    return sets, mode


def run_cell(
    spec: CellSpec,
    factory: Callable[[], Any],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    truth: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one cell and return its record. Never raises: a failure is recorded with its status."""
    record: Dict[str, Any] = dict(spec.as_dict())
    record["cell_key"] = spec.key()
    record["host_contended"] = True
    record["panel"] = list(PANEL_MEMBERS)
    try:
        assert_wrapper_estimator_differs(spec.arm, WRAPPER_INTERNAL_ESTIMATOR.get(spec.arm))
        feature_names = [str(c) for c in x_train.columns]
        target_size = _declared_target_size(truth, len(feature_names))
        record["target_size"] = target_size

        arm, wall_s, proc_s = _fit_arm(factory, x_train, y_train, spec.cv_seed)
        record["wall_time_s"] = round(wall_s, 3)
        record["process_time_s"] = round(proc_s, 3)

        ranking = ranking_from_arm_result(arm, feature_names)
        record["score_kind"] = ranking.score_kind
        record["ranking_coverage"] = round(ranking.coverage, 4)
        record["n_selected_self"] = len(ranking.selected)

        base_rate = base_rate_scores(y_train, y_test)
        record["base_rate"] = base_rate

        scores: Dict[str, Any] = {}
        total_fits = int(getattr(arm, "n_model_fits_", 0) or 0)
        selection_sets, k_grid_mode = _selection_sets(ranking, target_size, len(feature_names), constant_selection=(spec.arm == NULL_ARM))
        record["k_grid_mode"] = k_grid_mode
        for label, cols in selection_sets.items():
            if cols is None:
                # `score_kind == "none"`: the arm supplies no order, so it has no matched-K row at all.
                # Synthesising one would silently compare a different statistic against every other arm.
                scores[label] = {"skipped": "no_ranking"}
                continue
            block = fit_and_score_panel(x_train, y_train, x_test, y_test, cols)
            block["n_features"] = len(cols)
            block["skill"] = {
                member: normalized_skill(metrics["brier"], base_rate["brier"])
                for member, metrics in block["models"].items()
                if "brier" in metrics
            }
            total_fits += int(block.pop("n_model_fits", 0))
            block.pop("base_rate", None)
            scores[label] = block

        record["scores"] = scores
        record["n_model_fits"] = total_fits
        record["status"] = "ok"
    except BaseException as exc:  # a crashed cell is data, not an absence -- record and continue
        record["status"] = classify_exception(exc)
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()[-2000:]
        logger.warning("cell %s/%s seed=%s failed (%s): %s", spec.scenario, spec.arm, spec.dataset_seed, record["status"], exc)
    return record


def _default_scenarios() -> List[Tuple[str, ScenarioGen]]:
    """Return the scenario list: the synthetic beds by default, real cached beds when requested."""
    spec = os.environ.get("FS_HYBRID_SCENARIOS", "default").strip()
    if spec in ("", "default"):
        from .synth import make_dataset

        return [("default", lambda seed: make_dataset(n_samples=5000, seed=seed))]

    from .hard_synth2 import HARD_SCENARIOS

    names = list(HARD_SCENARIOS) if spec == "all" else [s.strip() for s in spec.split(",") if s.strip()]
    out: List[Tuple[str, ScenarioGen]] = []
    for name in names:
        if name not in HARD_SCENARIOS:
            raise SystemExit(f"unknown scenario {name!r}; available: default, all, {list(HARD_SCENARIOS)}")
        out.append((name, (lambda fn: (lambda seed: fn(seed)))(HARD_SCENARIOS[name])))
    return out


def run_grid(
    scenarios: Optional[Sequence[Tuple[str, ScenarioGen]]] = None,
    roster: Optional[Dict[str, Callable[[], Any]]] = None,
    dataset_seeds: Sequence[int] = DEV_DATASET_SEEDS,
    cv_seeds: Sequence[int] = CV_SEEDS,
    results_path: str = RESULTS_PATH,
    resume: bool = True,
) -> int:
    """Run the whole grid, appending one JSONL record per cell. Returns the number of cells executed."""
    scenarios = list(scenarios if scenarios is not None else _default_scenarios())
    if roster is not None and NULL_ARM not in roster:
        raise ValueError(f"the roster must contain the null hypothesis {NULL_ARM!r} on every cell")

    store = JsonlCellStore(results_path)
    done = store.completed_keys() if resume else set()
    executed = 0

    for scenario_name, gen in scenarios:
        for dataset_seed in dataset_seeds:
            x_all, y_all, truth = gen(int(dataset_seed))
            x_train, x_test, y_train, y_test = train_test_split(
                x_all, y_all, test_size=HOLDOUT_FRACTION, random_state=int(dataset_seed), stratify=y_all
            )
            # Built per scenario, not once for the grid: the fixed-cardinality arms (random-k, variance-sort)
            # need this bed's feature count, and a roster carried over from a wider bed would ask them for
            # more columns than exist here.
            cell_roster = dict(roster) if roster is not None else build_arm_roster(int(x_all.shape[1]), random_state=int(dataset_seed))
            if NULL_ARM not in cell_roster:
                raise ValueError(f"the roster must contain the null hypothesis {NULL_ARM!r} on every cell")
            for arm_name, factory in cell_roster.items():
                for cv_seed in cv_seeds:
                    spec = CellSpec(
                        scenario=scenario_name,
                        arm=arm_name,
                        dataset_seed=int(dataset_seed),
                        cv_seed=int(cv_seed),
                        protocol_version=PROTOCOL_VERSION,
                        config={"holdout_fraction": HOLDOUT_FRACTION, "panel": list(PANEL_MEMBERS)},
                    )
                    if spec.key() in done:
                        continue
                    record = run_cell(spec, factory, x_train, np.asarray(y_train), x_test, np.asarray(y_test), truth)
                    store.append(record)
                    executed += 1
                    logger.info(
                        "cell %s/%s seed=%s cv=%s -> %s (fits=%s)",
                        scenario_name,
                        arm_name,
                        dataset_seed,
                        cv_seed,
                        record["status"],
                        record.get("n_model_fits"),
                    )
    return executed


def main() -> None:
    """Run the grid with the default scenarios, roster and dev seeds, resuming from any existing file."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(OUT_DIR, exist_ok=True)
    executed = run_grid()
    logger.info("DONE executed=%d results=%s", executed, RESULTS_PATH)


if __name__ == "__main__":
    main()
