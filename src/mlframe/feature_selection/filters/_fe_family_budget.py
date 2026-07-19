"""Shapley-flavored compute-budgeting of MRMR's FE generator families (gt_07).

Game framing: FE generator families (pairwise cross-basis, triplet/quadruplet cross-basis, adaptive-
arity cross-basis, smart-polynom pairs, categorical pair/triple crosses, dispersion/rare-category/
conditional-residual families, ...) are players; the value of a coalition is the quality of the
selected feature set built from their combined outputs. A family's CREDIT is the Shapley-style
attribution of realized feature importance to whichever family produced each surviving column; ROI =
credit / wall-cost. Reallocating next-fit's per-family candidate budget proportional to ROI is a
mechanism-design-flavored compute market: families "earn" future compute by producing surviving,
important features, instead of every family always getting an equal (or unconditional) share
regardless of whether its output is ever kept.

HONEST SIMPLIFICATION (v1, stated explicitly per the plan): full Shapley over families (retraining
selection once per family coalition) is unnecessary and expensive -- each generated column already
receives an importance score from the selector (MRMR's ``mrmr_gain``, exposed per-survivor via
``fe_provenance_``), and columns map to families via their recipe kind (see :func:`_recipe_kind_to_family`
-- NOT via name-string parsing, which is provably ambiguous for this codebase's FE families, see that
function's docstring). Family credit = sum of surviving columns' importance is the ADDITIVE
approximation; it inherits Shapley's meaning only to the extent the underlying selector importances do
(MRMR relevance is itself an additive/greedy approximation, not an exact game value). A ``credit="loo"``
upgrade (leave-one-family-out re-selection deltas, the middle ground between additive and full Shapley)
is specced as future opt-in work for datasets where families' outputs are strongly redundant --
additive credit double-counts value shared between two families producing near-identical survivors;
LOO would not. Not implemented in v1.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

_BUDGET_CACHE_DIR = Path(os.environ.get("MLFRAME_FE_BUDGET_CACHE_DIR", str(Path.home() / ".cache" / "mlframe" / "fe_family_budget")))

# Recipe-kind -> family, at the SAME granularity as the wall-time ledger's own family names
# (``_fe_family_timing.py``'s ``@fe_timed("...")`` call sites), NOT the coarser ``origin`` bucket
# ``_mrmr_fe_provenance.py`` uses (that bucket merges orth_pair_cross/orth_triplet_cross/
# orth_quadruplet_cross/orth_spline/orth_fourier/... into one "hybrid_orth" origin -- too coarse for
# per-family ROI, which needs to distinguish triplet's wall-cost from quadruplet's).
_RECIPE_KIND_TO_FAMILY: dict[str, str] = {
    "orth_pair_cross": "orth_pair",
    "orth_triplet_cross": "triplet",
    "orth_quadruplet_cross": "quadruplet",
    "hermite_pair": "smart_polynom",
    "cat_pair_cross": "cat_pair",
    "cat_triple_cross": "cat_triple",
    "rare_category": "extra_fe_dispersion",
    "conditional_residual": "extra_fe_dispersion",
    "conditional_dispersion": "extra_fe_dispersion",
    "rankgauss": "extra_fe_dispersion",
}

_ORIGINAL_FAMILY = "_original"


def _recipe_kind_to_family(recipe_kind: Optional[str]) -> str:
    """Map an ``fe_provenance_`` row's recipe kind to a wall-ledger-granularity family name.

    KNOWN ARCHITECTURAL AMBIGUITY (not a parsing bug -- verified against the actual FE generator
    source, not guessed): adaptive-arity's winning arity-2/3/4 columns explicitly REUSE the
    orth_pair_cross/orth_triplet_cross/orth_quadruplet_cross recipe kinds respectively (see
    ``_orthogonal_adaptive_arity_fe.py``'s module docstring: "arity-2 winners reuse the Layer 22
    orth_pair_cross recipe" etc) -- there is NO recipe-kind-level (or column-name-level; both were
    checked) signal distinguishing an adaptive-arity column from a same-arity fixed-family column.
    When ``fe_hybrid_orth_adaptive_arity`` is enabled ALONGSIDE the fixed-arity families, its
    surviving columns' credit is attributed to the fixed-arity family sharing its winning arity, NOT
    to "adaptive_arity" -- meaning adaptive_arity's WALL cost is tracked separately (it has its own
    ``@fe_timed("adaptive_arity")`` site) but its CREDIT silently flows elsewhere, understating its
    own ROI and overstating the fixed-arity families'. This is a real, currently-unresolvable
    coupling in the recipe system (fixing it would mean adding a NEW recipe kind per adaptive-arity
    winner, out of scope for this plan) -- documented here rather than hidden, and callers relying on
    adaptive_arity's specific ROI should treat it as a floor (true value >= reported).
    """
    if recipe_kind is None:
        return _ORIGINAL_FAMILY
    return _RECIPE_KIND_TO_FAMILY.get(recipe_kind, "engineered_other")


def family_credit(
    fe_provenance: Any,
    *,
    gain_col: str = "mrmr_gain",
    kind_col: str = "mechanism_details",
    origin_col: str = "origin",
    credit: str = "additive",
) -> dict[str, float]:
    """Aggregate per-family credit from an MRMR ``fe_provenance_`` DataFrame (one row per surviving column).

    ``credit="additive"`` (v1, the only implemented mode): family credit = sum of ``gain_col`` over
    every surviving row whose recipe kind maps to that family (:func:`_recipe_kind_to_family`).
    Columns whose ``kind_col`` doesn't parse to a known recipe kind (raw input features, or an
    unrecognised engineered kind) are bucketed under ``"_original"``/``"engineered_other"``
    respectively and excluded from budget reallocation by the caller.

    ``mechanism_details`` is a stringified dict (``_mrmr_fe_provenance.py``'s ``_safe_str``); the
    recipe kind is recovered from its ``'kind': '...'`` entry via a targeted regex rather than a full
    parse (the stringification is one-way, not round-trippable JSON) -- verified against real
    provenance output in the accompanying unit test, not guessed.

    Raises ``NotImplementedError`` for ``credit="loo"`` (specced as v2, not implemented).
    """
    if credit == "loo":
        raise NotImplementedError("family_credit: credit='loo' (leave-one-family-out) is specced as v2 future work, not implemented.")
    if credit != "additive":
        raise ValueError(f"family_credit: unsupported credit mode {credit!r}, expected 'additive' (or 'loo', not yet implemented)")

    import re

    result: dict[str, float] = {}
    if fe_provenance is None or len(fe_provenance) == 0:
        return result

    kind_pattern = re.compile(r"'kind':\s*'([^']*)'")
    for _idx, row in fe_provenance.iterrows():
        raw_gain = row.get(gain_col, 0.0)
        gain = 0.0 if raw_gain is None or (isinstance(raw_gain, float) and np.isnan(raw_gain)) else float(raw_gain)
        if str(row.get(origin_col, "")) == "raw":
            family = _ORIGINAL_FAMILY
        else:
            details = str(row.get(kind_col, "") or "")
            m = kind_pattern.search(details)
            recipe_kind = m.group(1) if m else None
            family = _recipe_kind_to_family(recipe_kind)
        result[family] = result.get(family, 0.0) + max(gain, 0.0)
    return result


def family_roi(credit: dict[str, float], wall: dict[str, Any]) -> dict[str, Optional[float]]:
    """ROI = credit / max(wall_seconds, eps) per family present in ``wall`` (the ``get_fe_family_wall()`` snapshot).

    Families with recorded wall but zero credit get ROI ``0.0`` (they ran and produced nothing that
    survived). Families never yet run (absent from ``wall``, or present with 0 invocations) get ROI
    ``None`` -- a family that has never had a chance to prove itself must not be starved by
    :func:`reallocate_budgets` before its first trial (the explore/exploit floor+exploration terms
    exist precisely for this).
    """
    eps = 1e-9
    result: dict[str, Optional[float]] = {}
    for family_name, wall_entry in wall.items():
        wall_seconds, n_invocations = wall_entry[0], wall_entry[1]
        if n_invocations <= 0:
            result[family_name] = None
            continue
        result[family_name] = float(credit.get(family_name, 0.0)) / max(float(wall_seconds), eps)
    return result


def reallocate_budgets(
    roi: dict[str, Optional[float]],
    *,
    base_budget: dict[str, float],
    floor: float = 0.1,
    smoothing: float = 0.5,
    exploration: float = 0.1,
) -> dict[str, float]:
    """Proportional-to-ROI budget reallocation with a MANDATORY floor and an exploration reserve.

    Families present in ``base_budget`` but ABSENT from ``roi`` (never run) get the shared
    ``exploration`` reserve split evenly among them, never zero -- a family that never got a chance
    to run must not be permanently starved. Families with a known (non-``None``) ROI split the
    remaining ``1 - exploration`` budget mass proportional to their ROI, each floored at
    ``floor * base_budget[family]`` (a family that scored zero ROI once must still keep a minimum
    stake -- target regimes change between datasets, and a permanently-zeroed family can never
    redeem itself; this is the explore/exploit tension the floor+exploration terms encode).
    ``smoothing`` EMA-blends the newly-computed allocation with the PREVIOUS ``base_budget`` (``0`` =
    keep the old budget unchanged, ``1`` = fully adopt the new proportional-to-ROI allocation) to
    damp oscillation across successive fits.

    Returns a dict over the SAME keys as ``base_budget``, values summing to ``sum(base_budget.values())``
    (budget mass is conserved, only reallocated).
    """
    if not base_budget:
        return {}
    total_mass = sum(base_budget.values())
    if total_mass <= 0.0:
        return dict(base_budget)

    never_run = [f for f in base_budget if roi.get(f) is None]
    scored = [f for f in base_budget if roi.get(f) is not None]

    exploration_mass = exploration * total_mass if never_run else 0.0
    remaining_mass = total_mass - exploration_mass

    floors = {f: floor * base_budget[f] for f in base_budget}
    floor_mass_scored = sum(floors[f] for f in scored)
    proportional_mass = max(remaining_mass - floor_mass_scored, 0.0)

    roi_values: dict[str, float] = {}
    for f in scored:
        roi_f = roi[f]
        if roi_f is not None:
            roi_values[f] = max(float(roi_f), 0.0)
    roi_total = sum(roi_values.values())

    new_budget: dict[str, float] = {}
    for f in scored:
        proportional_share = (roi_values[f] / roi_total * proportional_mass) if roi_total > 0.0 else (proportional_mass / len(scored) if scored else 0.0)
        new_budget[f] = floors[f] + proportional_share
    for f in never_run:
        new_budget[f] = (exploration_mass / len(never_run)) if never_run else 0.0

    # Renormalize to exactly conserve total_mass (floor+proportional+exploration bookkeeping can
    # drift by float error, or floor_mass_scored alone can exceed remaining_mass at extreme floor
    # values -- always end on an exact-sum guarantee rather than a "should be close" one).
    new_total = sum(new_budget.values())
    if new_total > 0.0:
        new_budget = {f: v / new_total * total_mass for f, v in new_budget.items()}

    smoothed = {f: (1.0 - smoothing) * base_budget[f] + smoothing * new_budget.get(f, base_budget[f]) for f in base_budget}
    return smoothed


def dataset_fingerprint(n_features: int, column_names: Any) -> str:
    """Stable short hash of a dataset's shape + column names, for keying persisted budgets.

    REQUIRED (not optional) per the plan's own risk section: budgets learned on one dataset applied
    unconditionally to an unrelated dataset is a silent correctness bug (that dataset's family
    usefulness may be entirely different) -- this fingerprint gates cross-dataset carryover.
    """
    names_joined = "|".join(sorted(str(c) for c in column_names))
    digest = hashlib.sha256(f"{n_features}:{names_joined}".encode()).hexdigest()[:16]
    return digest


def persist_budgets(budgets: dict[str, float], *, cache_key: str = "mlframe.fe_family_budget", fingerprint: Optional[str] = None) -> None:
    """Persist ``budgets`` to a local JSON cache file keyed by ``cache_key`` + ``fingerprint``.

    Uses a plain local JSON cache under ``MLFRAME_FE_BUDGET_CACHE_DIR`` (default
    ``~/.cache/mlframe/fe_family_budget``), NOT ``pyutilz.performance.kernel_tuning.cache`` -- that
    cache's API (``get_or_tune``/``lookup``/``update``, keyed by HARDWARE fingerprint) is built for
    per-hardware kernel-parameter tuning, not per-DATASET arbitrary key-value persistence; a prior
    attempt to (mis)use a simple ``.get(key, default=...)`` shape against it elsewhere in this repo
    was already identified as dead/non-existent API and removed (see
    ``shap_proxied_fs/_shap_proxied_resolvers.py``'s ``_resolve_brute_force_max_features`` docstring)
    -- this module does not repeat that mistake.
    """
    _BUDGET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    file_key = f"{cache_key}.{fingerprint}" if fingerprint else cache_key
    path = _BUDGET_CACHE_DIR / f"{file_key}.json"
    try:
        path.write_text(json.dumps(budgets, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        logger.warning("persist_budgets: failed to write %s (%s); budget learning will restart next fit.", path, exc)


def load_budgets(*, cache_key: str = "mlframe.fe_family_budget", fingerprint: Optional[str] = None) -> Optional[dict[str, float]]:
    """Load previously-persisted budgets for ``cache_key`` + ``fingerprint``, or ``None`` if absent/unreadable/corrupt."""
    file_key = f"{cache_key}.{fingerprint}" if fingerprint else cache_key
    path = _BUDGET_CACHE_DIR / f"{file_key}.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        return {str(k): float(v) for k, v in raw.items()}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("load_budgets: failed to read/parse %s (%s); starting from equal-split budgets.", path, exc)
        return None


__all__ = [
    "family_credit",
    "family_roi",
    "reallocate_budgets",
    "dataset_fingerprint",
    "persist_budgets",
    "load_budgets",
]
