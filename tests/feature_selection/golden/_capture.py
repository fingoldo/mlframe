"""Golden snapshot capture / replay helpers.

Capture pre-refactor baseline at etap 0a (no keyed cached_MIs values --
collisions in the pre-B12 ``arr2str`` make those untrustworthy).

Capture intermediate baseline at etap 9 (post-cleanup, keyed values are
collision-safe and may be compared with the rtol policy below).

rtol policy (from third-round numerical audit):
* single entropy values  -> rtol=1e-12, atol=0
* cached conditional MI  -> rtol=1e-9,  atol=1e-12  (4-term subtraction
  accumulates ~2e-9 error for n=10^4; numba recompile may exacerbate)
"""

from __future__ import annotations

import orjson
from pathlib import Path
from typing import Any

import numpy as np

GOLDEN_DIR = Path(__file__).parent
PRE_REFACTOR_DIR = GOLDEN_DIR / "pre_refactor"
INTERMEDIATE_DIR = GOLDEN_DIR / "intermediate"


def _extract_attr(mrmr, name: str, default=None):
    """Fetch an optional fitted attribute from an MRMR instance, tolerating older attribute-name variants."""
    return getattr(mrmr, name, default)


def capture_pre_refactor(mrmr, scenario_name: str, seed: int) -> dict[str, Any]:
    """Capture only the trustworthy fields at etap 0a.

    Per the plan: NO keyed cached_MIs values. Only support_set + counters +
    cache sizes + per-iteration counters where exposed.
    """
    support = _extract_attr(mrmr, "support_")
    support_list = sorted(support.tolist()) if support is not None else None

    cached_MIs = _extract_attr(mrmr, "_cached_MIs") or _extract_attr(mrmr, "cached_MIs_")
    cached_cond = _extract_attr(mrmr, "_cached_cond_MIs") or _extract_attr(mrmr, "cached_cond_MIs_")

    return {
        "scenario": scenario_name,
        "seed": seed,
        "support_set": support_list,
        "n_features_in": _extract_attr(mrmr, "n_features_in_"),
        "n_features_selected": _extract_attr(mrmr, "n_features_"),
        "cache_sizes": {
            "MIs": len(cached_MIs) if cached_MIs is not None else None,
            "cond_MIs": len(cached_cond) if cached_cond is not None else None,
        },
    }


def capture_intermediate(mrmr, scenario_name: str, seed: int) -> dict[str, Any]:
    """Capture pre-refactor fields PLUS keyed cached values (post-B12 format).

    Used at etap 9 onwards. Float values stored as Python floats (JSON-safe).
    """
    base = capture_pre_refactor(mrmr, scenario_name, seed)
    cached_MIs = _extract_attr(mrmr, "_cached_MIs") or _extract_attr(mrmr, "cached_MIs_")
    cached_cond = _extract_attr(mrmr, "_cached_cond_MIs") or _extract_attr(mrmr, "cached_cond_MIs_")
    base["cached_MIs"] = {str(k): float(v) for k, v in cached_MIs.items()} if cached_MIs is not None else None
    base["cached_cond_MIs"] = {str(k): float(v) for k, v in cached_cond.items()} if cached_cond is not None else None
    expected_gains = _extract_attr(mrmr, "_expected_gains")
    if expected_gains is not None:
        base["expected_gains"] = {int(k): float(v) for k, v in expected_gains.items()}
    return base


def save_snapshot(snap: dict[str, Any], target_dir: Path) -> Path:
    """Write a captured snapshot dict to ``<target_dir>/<scenario>.json``, sorted-key for stable diffs."""
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{snap['scenario']}.json"
    path.write_text(orjson.dumps(snap, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS).decode(), encoding="utf-8")
    return path


def load_snapshot(scenario_name: str, target_dir: Path) -> dict[str, Any]:
    """Load a previously captured snapshot dict for the given scenario from ``target_dir``."""
    path = target_dir / f"{scenario_name}.json"
    return orjson.loads(path.read_text(encoding="utf-8"))


def assert_support_equivalent(
    actual_support: list[int],
    expected: dict[str, Any],
    actual_expected_gains: dict[int, float] | None = None,
    tie_eps: float = 1e-6,
) -> None:
    """Tied-score-aware support comparator.

    Plain equality first. If they differ, every differing candidate must have
    a gain within ``tie_eps`` of an accepted one (ranking ambiguity).
    """
    actual_set = set(actual_support)
    expected_set = set(expected["support_set"])
    if actual_set == expected_set:
        return
    diff = actual_set ^ expected_set
    expected_gains = expected.get("expected_gains")
    if expected_gains is None or actual_expected_gains is None:
        raise AssertionError(f"support_ differs and no expected_gains captured for tie-break: diff={sorted(diff)}")
    accepted_gains = list(expected_gains.values())
    for cand in diff:
        cand_gain = actual_expected_gains.get(cand) or expected_gains.get(str(cand))
        if cand_gain is None:
            raise AssertionError(f"candidate {cand} has no gain recorded for tie-break")
        nearest = min((abs(cand_gain - g) for g in accepted_gains), default=float("inf"))
        if nearest >= tie_eps:
            raise AssertionError(f"candidate {cand} (gain={cand_gain}) not tied with accepted set (nearest delta={nearest})")


def assert_cached_close(
    actual_cached: dict[str, float],
    expected_cached: dict[str, float],
    rtol: float,
    atol: float,
    skip_collision_keys: bool = False,
) -> None:
    """Compare cached MI dicts using ``np.isclose`` with the per-tier rtol/atol.

    ``skip_collision_keys=True`` is set for pre-refactor baselines whose keys
    suffer ``arr2str`` collisions; those entries are non-authoritative.
    """
    if skip_collision_keys:
        return
    for key, expected_val in expected_cached.items():
        if key not in actual_cached:
            raise AssertionError(f"missing cached key: {key}")
        actual_val = actual_cached[key]
        if not np.isclose(actual_val, expected_val, rtol=rtol, atol=atol):
            raise AssertionError(
                f"cached value drift on key {key}: expected={expected_val}, actual={actual_val}, |diff|={abs(actual_val - expected_val)} > rtol*|expected|+atol"
            )
