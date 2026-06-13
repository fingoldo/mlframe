"""ROW-ARGMAX + CONDITIONAL-GATE relationship detection FE (wired into MRMR; argmax default ON, gate default OFF / opt-in -- the gate's select sweep is an n-driven wide-frame cost blow-up the column-count budget cannot bound, so it ships validated but off-by-default).

Two multi-column operators the rich catalog cannot express for the MI / linear-downstream selector (frontier discovery pass 2,
confirmed by ``_benchmarks/bench_frontier_candidates`` + ``bench_conditional_gate_detection``):

* ROW-ARGMAX -- ``argmax_row(a, b, c)`` = which column is the row maximum (an ordinal / comparison pattern). A tree gets the
  pairwise comparisons for free, but the MI / linear path sees only marginal columns + pairwise diffs; no single shipped column
  equals the 3-way argmax code (+0.55 single-column MI lift over the best shipped op). ZERO free params, detector-clean (negative
  lift on smooth / noise / ordinary-interaction controls). Triples only -- a 2-col argmax == sign of the diff, already shipped.

* CONDITIONAL-GATE -- a REGIME SWITCH ``c > tau ? a : b`` (select) and a MASKED interaction ``1[c > tau] * a`` (mask): two raw
  features routed / masked by a THIRD feature's data-dependent threshold. The shipped ``conditional_residual`` is ``a - E[a|bin(c)]``
  (a residual, not a value-selecting switch); the hinge basis is univariate; a raw product ``a*c`` is a smooth bilinear surface,
  not a hard switch. On a true regime target the gate MI (+0.55 select / +0.31 mask over the best shipped op) dwarfs every existing op.

GATE HARDENING (the discovery caveat): the prototype gate detector gated only vs the RAW single-operand MI floor, so it ALSO fired
on ``smooth`` / ``ordinary_mul`` controls (a hard threshold can partly reconstruct an XOR-sign regime; bench measured FP lift
+0.17 smooth / +0.32 ordinary_mul). The production detector gates the engineered MI vs the BEST-EXISTING-OP MI on the SAME operands
-- the max MI over the cheap arithmetic ops a selector already has (product, ratio, diff, min, max), mirroring the
``bench_frontier_candidates`` baseline -- NOT the raw single-operand MI. With that floor the gate clears 0 false-positives on
smooth / noise / ordinary_mul at p=30 over 3 seeds (``bench_conditional_gate_wideframe``), while the regime targets still respond.

Design mirrors ``_pairwise_modular_fe`` / ``_integer_lattice_fe``: CHEAP-FIRST scan + a dual ``_responded`` gate -- the engineered
column's MI must beat the operand baseline by ``_MIN_MARGIN`` AND a 12-permutation null upper band (so a non-structured frame
injects nothing). For the gate the threshold ``tau`` is found by a ~17-point quantile scan over the gating column and FROZEN in the
recipe; for argmax there is no parameter (one column per integer-eligible / continuous triple). Replay is a pure function of X
(argmax = ``np.argmax(stack, axis=1)``; gate = ``np.where(c>tau, a, b)`` / ``(c>tau)*a`` with the frozen tau) -- no y, no fitted
state beyond tau, so transform() is leak-free + deterministic + train/test bit-identical.

Cost: ~99% of each scan is the shipped binned-MI kernel (``_mi`` -> ``_mi_classif_batch``). The per-tau gate residue MIs batch into
ONE ``_mi_classif_batch`` call per (mode, a, b, gate) over the tau grid (bit-identical to per-tau -- the kernel bins each column
independently, only the per-call dispatch overhead is amortised). The 12-perm null is EARLY-REJECTED: computed only for a candidate
that already clears the baseline margin (``_responded`` needs BOTH, so a margin-failing candidate can never respond -> its null is
skipped, stored +inf -- bit-identical by short-circuit). Budget-guarded on wide frames (``max_cols`` skips the whole sweep).

cProfile (n=2000, p=15, enabled path, measured): the batched plug-in MI dispatch is the top hotspot (argmax ~99%, gate ~97% of the
scan wall); the argmax stack + ``np.argmax`` and the gate ``np.where`` / quantile arithmetic are <3% combined -- the per-candidate
build is dwarfed by the (already-tuned, batched) MI kernel. No further actionable caller-side speedup beyond the per-candidate tau
batching + early-reject already applied. See ``_benchmarks/bench_conditional_gate_wideframe``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Sequence

import numpy as np

from ._pairwise_modular_fe import _mi

logger = logging.getLogger(__name__)

__all__ = [
    "ArgmaxHit",
    "GateHit",
    "GATE_MODES",
    "ROW_ARGMAX_PREFIX",
    "CONDITIONAL_GATE_PREFIX",
    "apply_row_argmax",
    "apply_conditional_gate",
    "cheap_row_argmax_scan",
    "cheap_conditional_gate_scan",
    "best_existing_op_mi",
    "detect_row_argmax",
    "detect_conditional_gate",
    "engineered_name_row_argmax",
    "engineered_name_conditional_gate",
    "build_row_argmax_recipe",
    "build_conditional_gate_recipe",
    "hybrid_row_argmax_fe_with_recipes",
    "hybrid_conditional_gate_fe_with_recipes",
]

ROW_ARGMAX_PREFIX = "argmax"
CONDITIONAL_GATE_PREFIX = "gate"

GATE_MODES = ("select", "mask")

# Quantile grid for the tau scan: skip the extreme tails (a tau at q<=0.05 / q>=0.95 leaves one branch nearly empty, so the
# gate degenerates to a single column already on the raw list). 17 interior quantiles is enough to land near a true tau.
_TAU_QUANTILES = tuple(np.round(np.linspace(0.1, 0.9, 17), 4))

# Margin the engineered column's MI must beat its operand baseline by (mirrors _pairwise_modular_fe._MIN_MARGIN). Below it the
# selector can already recover the signal from a raw / cheap-op column, so the engineered column adds no genuine structure.
_MIN_MARGIN = 0.02


def apply_row_argmax(X, cols: Sequence[str]) -> np.ndarray:
    """Replay one row-argmax column: the integer index (0..k-1) of the row-maximum over the source columns.

    Pure function of X (no y, no fitted state) -> transform() is leak-free + deterministic + train/test bit-identical. Output
    float64 in ``{0, .., k-1}``; ties resolve to the first max (numpy ``argmax`` semantics), stable across fit / replay."""
    if len(cols) < 2:
        raise ValueError(f"row-argmax needs >= 2 source columns; got {tuple(cols)!r}")
    stk = np.stack([np.asarray(X[c], dtype=np.float64) for c in cols], axis=1)
    return np.argmax(stk, axis=1).astype(np.float64)


def apply_conditional_gate(X, mode: str, cols: Sequence[str], tau: float) -> np.ndarray:
    """Replay one conditional-gate column with the FROZEN threshold ``tau``.

    ``select`` (cols = (a, b, c)): ``c > tau ? a : b`` -- a regime switch routing a / b by the gating column c.
    ``mask`` (cols = (a, c)): ``1[c > tau] * a`` -- a active only where c > tau.

    Pure function of (source columns, tau) -- no y, no fitted state beyond the recipe-frozen tau -> leak-free + train/test exact."""
    if mode == "select":
        if len(cols) != 3:
            raise ValueError(f"conditional-gate 'select' needs exactly 3 source columns (a, b, c); got {tuple(cols)!r}")
        a, b, c = (np.asarray(X[cn], dtype=np.float64) for cn in cols)
        return np.where(c > float(tau), a, b)
    if mode == "mask":
        if len(cols) != 2:
            raise ValueError(f"conditional-gate 'mask' needs exactly 2 source columns (a, c); got {tuple(cols)!r}")
        a, c = (np.asarray(X[cn], dtype=np.float64) for cn in cols)
        return (c > float(tau)).astype(np.float64) * a
    raise ValueError(f"conditional-gate mode must be one of {GATE_MODES}; got {mode!r}")


def best_existing_op_mi(arrs: dict, names: Sequence[str], yi: np.ndarray, nbins: int) -> float:
    """Max binned-MI over the cheap operators a selector already has on the given operands: raw columns + pairwise
    product / ratio / diff + row-max / row-min. This is the HARDENED baseline both detectors must beat -- the prototype gated only
    vs the raw single-operand MI, which let a hard threshold reconstruct an XOR-sign regime on smooth / ordinary_mul controls AND
    let a spurious row-argmax clear the floor on an ordinary-multiplicative target (false positives, measured in
    ``bench_conditional_gate_wideframe``). Mirrors the ``bench_frontier_candidates`` 'best existing op' reference.

    All candidate columns stack into ONE batched ``_mi_classif_batch`` call (bit-identical to per-candidate -- the kernel bins
    each column independently, only the per-call dispatch overhead is amortised); cProfile showed the per-``_mi`` dispatch was the
    dominant mlframe-side cost of the hardened floor, so batching is the actionable caller-side lever."""
    from ._orthogonal_univariate_fe import _mi_classif_batch

    names = list(names)
    cols_arr = [np.asarray(arrs[c], dtype=np.float64) for c in names]
    cands = list(cols_arr)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            u, v = cols_arr[i], cols_arr[j]
            cands.append(u * v)
            cands.append(u - v)
            cands.append(u / (np.abs(v) + 1e-6))
    stk = np.stack(cols_arr, axis=1)
    cands.append(stk.max(axis=1))
    cands.append(stk.min(axis=1))
    mat = np.column_stack(cands).astype(np.float64, copy=False)
    mis = _mi_classif_batch(np.ascontiguousarray(mat), yi.astype(np.int64), nbins=nbins)
    return float(np.max(mis))


def _responded(feat_mi: float, baseline: float, null_hi: float, min_margin: float = _MIN_MARGIN) -> bool:
    """Gate: the engineered column's MI must clear BOTH the operand baseline (by ``min_margin``) AND the permutation-null upper
    band. Mirrors ``_pairwise_modular_fe._responded`` (``baseline`` plays the smooth-basis floor role)."""
    return (feat_mi - baseline) >= min_margin and feat_mi > null_hi


def _perm_null_hi(feat: np.ndarray, y: np.ndarray, nbins: int, n_perm: int = 12, seed: int = 0, z: float = 3.0) -> float:
    """Upper band (mean + z*std) of the fixed feature's MI under y permutation -- the noise reference the feature MI must clear.
    The feature is fixed; only y is shuffled (cheap, n_perm small)."""
    rng = np.random.default_rng(seed)
    yi = np.asarray(y).astype(np.int64)
    vals = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        vals[i] = _mi(feat, yi[rng.permutation(yi.size)], nbins=nbins)
    return float(vals.mean() + z * vals.std())


def _gate_grid_mi(feats: np.ndarray, yi: np.ndarray, nbins: int) -> np.ndarray:
    """MI of every column of the (n, k) tau-grid feature matrix vs y, in one batched kernel call (bit-identical to per-column --
    ``_mi_classif_batch`` bins each column independently, only the per-call dispatch overhead is amortised)."""
    from ._orthogonal_univariate_fe import _mi_classif_batch

    return np.asarray(_mi_classif_batch(np.ascontiguousarray(feats), yi, nbins=nbins), dtype=np.float64)


def _is_argmax_eligible(x: np.ndarray) -> bool:
    """True iff the column is finite numeric (int or float) -- argmax / gate need an order, not an integer lattice."""
    a = np.asarray(x)
    if not np.issubdtype(a.dtype, np.number):
        return False
    return bool(np.all(np.isfinite(a[np.isfinite(a)] if a.size else a)))


@dataclass(frozen=True)
class ArgmaxHit:
    """One row-argmax candidate: a column triple, its argmax-code MI, the HARDENED best-existing-op floor on the triple, the
    null band. The floor is the best-existing-op MI (raw / product / ratio / diff / min / max), NOT the raw single-operand MI:
    on an ordinary-multiplicative target (``(a*b)>0``) some triple's argmax code clears the raw floor by ~0.02 (a false positive
    at scale, measured in ``bench_conditional_gate_wideframe``), but the product ``a*b`` itself dwarfs it -- so gating vs the
    best-existing-op floor keeps argmax detector-clean (0 FP at p=30 over 3 seeds incl. the ordinary_mul control)."""

    cols: tuple[str, ...]
    feat_mi: float
    operand_floor: float
    null_hi: float

    @property
    def margin_over_operands(self) -> float:
        return self.feat_mi - self.operand_floor

    @property
    def responded(self) -> bool:
        return _responded(self.feat_mi, self.operand_floor, self.null_hi)


@dataclass(frozen=True)
class GateHit:
    """One conditional-gate candidate: mode + source columns + the FROZEN best tau, its engineered MI, the HARDENED
    best-existing-op baseline + the null band."""

    mode: str
    cols: tuple[str, ...]
    tau: float
    feat_mi: float
    baseline_mi: float  # best-existing-op MI on the operands = the hardened floor the gate must beat
    null_hi: float

    @property
    def margin_over_baseline(self) -> float:
        return self.feat_mi - self.baseline_mi

    @property
    def responded(self) -> bool:
        return _responded(self.feat_mi, self.baseline_mi, self.null_hi)


def cheap_row_argmax_scan(
    X,
    y: np.ndarray,
    cols: Optional[Sequence[str]] = None,
    *,
    nbins: int = 12,
    seed: int = 0,
    max_triples: int = 40,
) -> list[ArgmaxHit]:
    """Cheap first-pass scan for row-argmax structure over column TRIPLES of X.

    For each triple compute the argmax-code MI vs y; keep the hit when it beats the HARDENED best-existing-op MI on the triple
    (max over raw / product / ratio / diff / min / max) by ``_MIN_MARGIN`` AND a 12-perm null band. The hardened floor (not the
    raw single-operand MI) is what keeps argmax clean on the ordinary-multiplicative control at scale. Pairs are skipped (a 2-col
    argmax == sign of the diff, already on the diff list). Budgeted by ``max_triples``. The null is early-rejected (computed only
    for triples already clearing the operand margin)."""
    import pandas as pd  # noqa: F401  (X may be pandas or polars; we pull ndarrays)

    if cols is None:
        cols = [c for c in X.columns if _is_argmax_eligible(np.asarray(X[c]))]
    else:
        cols = [c for c in cols if c in X.columns and _is_argmax_eligible(np.asarray(X[c]))]
    yi = np.asarray(y).astype(np.int64)
    arrs = {c: np.asarray(X[c], dtype=np.float64) for c in cols}

    hits: list[ArgmaxHit] = []
    budget = int(max_triples)
    for tri in combinations(cols, 3):
        if budget <= 0:
            break
        budget -= 1
        operand_floor = best_existing_op_mi(arrs, tri, yi, nbins)
        feat = np.argmax(np.stack([arrs[c] for c in tri], axis=1), axis=1).astype(np.float64)
        feat_mi = _mi(feat, yi, nbins=nbins)
        # Early-reject: the null only matters for a triple already clearing the operand margin (``_responded`` needs BOTH).
        if (feat_mi - operand_floor) >= _MIN_MARGIN:
            null_hi = _perm_null_hi(feat, yi, nbins, seed=seed)
        else:
            null_hi = float("inf")
        hits.append(ArgmaxHit(tuple(tri), feat_mi, operand_floor, null_hi))

    hits.sort(key=lambda h: h.margin_over_operands, reverse=True)
    return hits


def cheap_conditional_gate_scan(
    X,
    y: np.ndarray,
    cols: Optional[Sequence[str]] = None,
    *,
    nbins: int = 12,
    seed: int = 0,
    max_gate_cols: int = 5,
) -> list[GateHit]:
    """Cheap first-pass scan for conditional-gate structure (regime-switch + masked-interaction) over X.

    The gating column ``c`` ranges over the eligible columns; for ``mask`` the active column ``a`` ranges over the rest; for
    ``select`` the ordered pair ``(a, b)`` ranges over the remaining distinct pairs. For each candidate the best tau is found by a
    ~17-point quantile scan over c (the per-tau residue MIs batch into one kernel call), then the best-tau column is gated vs the
    HARDENED best-existing-op baseline (max MI over raw / product / ratio / diff / min / max on the candidate's operands) by
    ``_MIN_MARGIN`` AND a 12-perm null band. ``max_gate_cols`` bounds the gating-column fan-out (the (a, b) select sweep is
    O(p^3); the cap keeps it budgeted -- the budget guard in the orchestrator skips the whole sweep on wide frames). The null is
    early-rejected (computed only for a candidate already clearing the hardened baseline)."""
    import pandas as pd  # noqa: F401

    if cols is None:
        cols = [c for c in X.columns if _is_argmax_eligible(np.asarray(X[c]))]
    else:
        cols = [c for c in cols if c in X.columns and _is_argmax_eligible(np.asarray(X[c]))]
    cols = list(cols)[: int(max_gate_cols)]
    yi = np.asarray(y).astype(np.int64)
    arrs = {c: np.asarray(X[c], dtype=np.float64) for c in cols}
    # Cache the hardened best-existing-op baseline per operand set (it does not depend on tau / mode).
    _baseline_cache: dict[tuple[str, ...], float] = {}

    def _baseline(operands: tuple[str, ...]) -> float:
        key = tuple(sorted(operands))
        if key not in _baseline_cache:
            _baseline_cache[key] = best_existing_op_mi(arrs, key, yi, nbins)
        return _baseline_cache[key]

    hits: list[GateHit] = []
    for cgate in cols:
        cv = arrs[cgate]
        taus = np.quantile(cv, _TAU_QUANTILES)
        others = [cn for cn in cols if cn != cgate]
        # mask: one active column a (cols = (a, c)); baseline over {a, c}.
        for a in others:
            av = arrs[a]
            feats = np.empty((cv.shape[0], len(taus)), dtype=np.float64)
            for j, tau in enumerate(taus):
                feats[:, j] = (cv > tau).astype(np.float64) * av
            grid = _gate_grid_mi(feats, yi, nbins)
            best_j = int(np.argmax(grid))
            best_mi, best_tau = float(grid[best_j]), float(taus[best_j])
            baseline = _baseline((a, cgate))
            if (best_mi - baseline) >= _MIN_MARGIN:
                null_hi = _perm_null_hi(feats[:, best_j], yi, nbins, seed=seed)
            else:
                null_hi = float("inf")
            hits.append(GateHit("mask", (a, cgate), best_tau, best_mi, baseline, null_hi))
        # select: ordered (a, b), cols = (a, b, c); baseline over {a, b, c}.
        for a in others:
            for b in others:
                if a == b:
                    continue
                av, bv = arrs[a], arrs[b]
                feats = np.empty((cv.shape[0], len(taus)), dtype=np.float64)
                for j, tau in enumerate(taus):
                    feats[:, j] = np.where(cv > tau, av, bv)
                grid = _gate_grid_mi(feats, yi, nbins)
                best_j = int(np.argmax(grid))
                best_mi, best_tau = float(grid[best_j]), float(taus[best_j])
                baseline = _baseline((a, b, cgate))
                if (best_mi - baseline) >= _MIN_MARGIN:
                    null_hi = _perm_null_hi(feats[:, best_j], yi, nbins, seed=seed)
                else:
                    null_hi = float("inf")
                hits.append(GateHit("select", (a, b, cgate), best_tau, best_mi, baseline, null_hi))

    hits.sort(key=lambda h: h.margin_over_baseline, reverse=True)
    return hits


def detect_row_argmax(
    X, y: np.ndarray, cols: Optional[Sequence[str]] = None, *,
    top_k: int = 4, nbins: int = 12, seed: int = 0,
):
    """Cheap-first scan; returns the list of responded row-argmax hits (each a dict with the source triple, MI + margin), capped
    at ``top_k``. Empty when nothing responds -- a non-argmax frame detects nothing."""
    hits = cheap_row_argmax_scan(X, y, cols, nbins=nbins, seed=seed)
    out = []
    for h in hits:
        if not h.responded:
            continue
        out.append({"cols": h.cols, "feat_mi": h.feat_mi, "operand_floor": h.operand_floor, "margin": h.margin_over_operands})
        if len(out) >= int(top_k):
            break
    return out


def detect_conditional_gate(
    X, y: np.ndarray, cols: Optional[Sequence[str]] = None, *,
    top_k: int = 4, nbins: int = 12, seed: int = 0,
):
    """Cheap-first scan; returns the list of responded conditional-gate hits (each a dict with mode, source columns, frozen tau,
    MI + margin), capped at ``top_k``. Empty when nothing responds -- the hardened baseline keeps smooth / ordinary_mul silent."""
    hits = cheap_conditional_gate_scan(X, y, cols, nbins=nbins, seed=seed)
    out = []
    for h in hits:
        if not h.responded:
            continue
        out.append({"mode": h.mode, "cols": h.cols, "tau": h.tau, "feat_mi": h.feat_mi,
                    "baseline_mi": h.baseline_mi, "margin": h.margin_over_baseline})
        if len(out) >= int(top_k):
            break
    return out


# Recipe plumbing: emit frozen EngineeredRecipe objects so the MRMR selector materialises, scores, selects, and replays the
# argmax / gate column at predict time identically. Replay is a pure function of X (argmax index; gate np.where / mask with the
# FROZEN tau) -- no y reference, so it is leak-free + deterministic + train/test bit-identical.


def engineered_name_row_argmax(cols: Sequence[str]) -> str:
    """Canonical engineered column name for one row-argmax, e.g. ``argmax__a__b__c``. Source columns join with ``__``."""
    if len(cols) < 2:
        raise ValueError(f"row-argmax needs >= 2 source columns; got {tuple(cols)!r}")
    joined = "__".join(str(c) for c in cols)
    return f"{ROW_ARGMAX_PREFIX}__{joined}"


def engineered_name_conditional_gate(mode: str, cols: Sequence[str], tau: float) -> str:
    """Canonical engineered column name for one conditional-gate column, e.g. ``gate_select__a__b__c__t0.123`` /
    ``gate_mask__a__c__t-0.4``. mode + source columns + the frozen tau fully determine the column."""
    if mode not in GATE_MODES:
        raise ValueError(f"conditional-gate mode must be one of {GATE_MODES}; got {mode!r}")
    joined = "__".join(str(c) for c in cols)
    return f"{CONDITIONAL_GATE_PREFIX}_{mode}__{joined}__t{float(tau):.6g}"


def build_row_argmax_recipe(*, name: str, cols: Sequence[str]):
    """Frozen recipe for one row-argmax column. Replay is ``np.argmax`` over the stacked source columns -- no parameters."""
    from .engineered_recipes import EngineeredRecipe

    if len(cols) < 2:
        raise ValueError(f"row-argmax needs >= 2 source columns; got {tuple(cols)!r}")
    return EngineeredRecipe(name=name, kind="row_argmax", src_names=tuple(str(c) for c in cols))


def build_conditional_gate_recipe(*, name: str, mode: str, cols: Sequence[str], tau: float):
    """Frozen recipe for one conditional-gate column. The chosen ``tau`` is FROZEN in ``extra`` for exact replay."""
    from .engineered_recipes import EngineeredRecipe

    if mode not in GATE_MODES:
        raise ValueError(f"conditional-gate mode must be one of {GATE_MODES}; got {mode!r}")
    return EngineeredRecipe(
        name=name,
        kind="conditional_gate",
        src_names=tuple(str(c) for c in cols),
        extra={"mode": str(mode), "tau": float(tau)},
    )


def hybrid_row_argmax_fe_with_recipes(
    X, y: np.ndarray, *,
    cols: Optional[Sequence[str]] = None,
    top_k: int = 4, nbins: int = 12, seed: int = 0,
    max_cols: int = 30,
):
    """Detect responded row-argmax structure and emit it as frozen, replayable ``EngineeredRecipe`` objects.

    Column-count BUDGET GUARD (the wide-frame cost is the budgeted C(p,3) triple sweep + the batched MI; see
    ``_benchmarks/bench_conditional_gate_wideframe``): when the number of eligible columns exceeds ``max_cols`` the whole sweep is
    SKIPPED (logged, never silent).

    Returns ``(appended_names, recipes)`` -- the materialised columns are NOT concatenated here (the MRMR caller appends them under
    its own RAM-safe path); each recipe replays the exact argmax column from X alone, leak-free."""
    if cols is None:
        elig = [c for c in X.columns if _is_argmax_eligible(np.asarray(X[c]))]
    else:
        elig = [c for c in cols if c in X.columns and _is_argmax_eligible(np.asarray(X[c]))]

    p = len(elig)
    if p > int(max_cols):
        logger.info(
            "row_argmax FE: %d eligible columns exceeds max_cols=%d; skipping the row-argmax sweep for budget.",
            p, int(max_cols),
        )
        return [], []

    hits = cheap_row_argmax_scan(X, y, elig, nbins=nbins, seed=seed)
    appended: list[str] = []
    recipes = []
    seen: set[str] = set()
    for h in hits:
        if not h.responded:
            continue
        name = engineered_name_row_argmax(h.cols)
        if name in seen:
            continue
        seen.add(name)
        appended.append(name)
        recipes.append(build_row_argmax_recipe(name=name, cols=h.cols))
        if len(appended) >= int(top_k):
            break
    return appended, recipes


def hybrid_conditional_gate_fe_with_recipes(
    X, y: np.ndarray, *,
    cols: Optional[Sequence[str]] = None,
    top_k: int = 4, nbins: int = 12, seed: int = 0,
    max_cols: int = 20,
):
    """Detect responded conditional-gate structure and emit it as frozen, replayable ``EngineeredRecipe`` objects (tau frozen).

    Column-count BUDGET GUARD (the select (a, b) sweep is O(p^3) x a 17-point tau scan; see
    ``_benchmarks/bench_conditional_gate_wideframe``): when the number of eligible columns exceeds ``max_cols`` the whole sweep is
    SKIPPED (logged, never silent). The detector gates vs the HARDENED best-existing-op baseline so smooth / ordinary_mul controls
    stay silent (0 FP at p=30 over 3 seeds).

    Returns ``(appended_names, recipes)`` -- the materialised columns are NOT concatenated here (the MRMR caller appends them under
    its own RAM-safe path); each recipe replays the exact gate column from X + the frozen tau alone, leak-free."""
    if cols is None:
        elig = [c for c in X.columns if _is_argmax_eligible(np.asarray(X[c]))]
    else:
        elig = [c for c in cols if c in X.columns and _is_argmax_eligible(np.asarray(X[c]))]

    p = len(elig)
    if p > int(max_cols):
        logger.info(
            "conditional_gate FE: %d eligible columns exceeds max_cols=%d; skipping the conditional-gate sweep for budget.",
            p, int(max_cols),
        )
        return [], []

    hits = cheap_conditional_gate_scan(X, y, elig, nbins=nbins, seed=seed)
    appended: list[str] = []
    recipes = []
    seen: set[str] = set()
    for h in hits:
        if not h.responded:
            continue
        name = engineered_name_conditional_gate(h.mode, h.cols, h.tau)
        if name in seen:
            continue
        seen.add(name)
        appended.append(name)
        recipes.append(build_conditional_gate_recipe(name=name, mode=h.mode, cols=h.cols, tau=h.tau))
        if len(appended) >= int(top_k):
            break
    return appended, recipes
