"""EXPERIMENTAL prototype: CONDITIONAL-GATE and ROW-ARGMAX feature operators (NOT wired into prod).

Second-pass discovery (after gcd/lcm shipped). Two multi-column operators the rich catalog still cannot express for the
MI / linear-downstream selector, confirmed by ``_benchmarks/bench_frontier_candidates`` (+0.55 / +0.55 single-column MI lift
over the best shipped operator, specific -- negative lift on smooth / noise / ordinary-interaction controls):

* GATE -- a REGIME SWITCH ``c > tau ? a : b`` (select) and a MASKED interaction ``1[c>tau] * a`` (mask): two raw features
  routed / masked by a THIRD feature's data-dependent threshold. The shipped ``conditional_residual`` is ``a - E[a|bin(c)]``
  (a *residual*, not a value-selecting switch); the hinge basis is UNIVARIATE (one column's kink), it cannot route between
  two columns; a raw product ``a*c`` is a smooth bilinear surface, not a hard switch. So on a true regime target the gate MI
  (0.69) dwarfs the best existing op (0.14, the raw product).

* ARGMAX -- ``argmax_row(a, b, c)`` = which column is the row maximum (an ordinal/comparison pattern). A tree gets the pairwise
  comparisons for free, but the MI / linear path sees only marginal columns + pairwise diffs; no single shipped column equals
  the 3-way argmax code (cand MI 1.05 vs best existing 0.49 = a single pairwise diff).

REJECTED sibling (kept here as the measured negative -- ``concordance3`` in the bench): the concordance-count #{a>b, b>c, a>c}
is FULLY captured by an existing pairwise diff (``diff_a_c``) -- +0.000 lift. Not an operator gap; do NOT add a detector.

Design mirrors ``_pairwise_modular_fe`` (cheap-first scan + dual gate). For the GATE the threshold ``tau`` is found by a coarse
quantile scan over the gating column; for ARGMAX there is no parameter (one column per integer-eligible/continuous triple).
Replay is a pure function of X (no y), so transform() is leak-free by construction -- the recipe would store only
``{a, b, gate_col, tau, mode}`` (gate) or the source triple (argmax).

Cost: each candidate is one vectorised numpy op + the shipped ``_mi`` kernel; the gate adds a small (~17-point) quantile scan
over tau. cProfile (n=2000) -> ~99% of wall is ``_mi`` (the shipped binned-MI kernel), the gate arithmetic ~1%; a prod wiring
would batch the per-tau residue MIs into one ``_mi_classif_batch`` call (mirror ``_residue_grid_mi``).
"""
from __future__ import annotations

import numpy as np

from ._pairwise_modular_fe import _mi

__all__ = [
    "GATE_MODES",
    "apply_conditional_gate",
    "apply_row_argmax",
    "scan_conditional_gate",
    "scan_row_argmax",
]

GATE_MODES = ("select", "mask")

# Quantile grid for the tau scan: skip the extreme tails (a tau at q<=0.05 / q>=0.95 leaves one branch nearly empty, so the
# "gate" degenerates to a single column already on the raw list). 17 interior quantiles is enough to land near a true tau.
_TAU_QUANTILES = tuple(np.round(np.linspace(0.1, 0.9, 17), 4))

# A gate / argmax column must beat the best operand-derived MI by this margin to count as a genuine multi-column gap (mirrors
# ``_pairwise_modular_fe._MIN_MARGIN``); below it the selector can already get the signal from a raw / pairwise column.
_MIN_MARGIN = 0.02


def apply_conditional_gate(a: np.ndarray, b: np.ndarray, c: np.ndarray, tau: float, mode: str = "select") -> np.ndarray:
    """One gate column. ``select``: ``c>tau ? a : b`` (regime switch). ``mask``: ``1[c>tau]*a`` (a active only where c>tau).

    Pure function of (a, b, c, tau) -- no y, so replay is leak-free. ``b`` is ignored for ``mask`` (pass any array / a)."""
    af = np.asarray(a, dtype=np.float64)
    cf = np.asarray(c, dtype=np.float64)
    hi = cf > tau
    if mode == "select":
        return np.where(hi, af, np.asarray(b, dtype=np.float64))
    if mode == "mask":
        return hi.astype(np.float64) * af
    raise ValueError(f"apply_conditional_gate: unknown mode {mode!r}; expected one of {GATE_MODES}")


def apply_row_argmax(cols: list[np.ndarray]) -> np.ndarray:
    """``argmax`` over a row's columns -> the integer index of the largest. Pure function of X (no y); leak-free at replay."""
    stk = np.stack([np.asarray(c, dtype=np.float64) for c in cols], axis=1)
    return np.argmax(stk, axis=1).astype(np.float64)


def _perm_null_hi(feat: np.ndarray, yi: np.ndarray, nbins: int, n_perm: int, rng, z: float = 3.0) -> float:
    """Upper band (mean + z*std) of the fixed feature's MI under y permutation -- the noise reference the feature must clear."""
    vals = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        vals[i] = _mi(feat, yi[rng.permutation(yi.size)], nbins=nbins)
    return float(vals.mean() + z * vals.std())


def scan_conditional_gate(
    X,
    y: np.ndarray,
    cols: list[str],
    *,
    nbins: int = 12,
    n_perm: int = 12,
    min_margin: float = _MIN_MARGIN,
    rng_seed: int = 0,
    max_triples: int = 40,
) -> list[dict]:
    """Cheap-first scan for regime-switch / masked-interaction structure. For each ordered (a, b, gate_col=c) triple and each
    mode, scan tau over the quantile grid of c, keep the best-tau gate, and emit a hit when its MI beats the best operand MI by
    ``min_margin`` AND a permutation-null band -- the same dual gate the modular / lattice detectors use.

    ``cols`` are the candidate column names (continuous or integer). The gating column ``c`` ranges over all cols; (a, b) over the
    remaining ordered pairs for ``select`` and over the single masked column for ``mask``. Budgeted by ``max_triples``."""
    arrs = {cn: np.asarray(X[cn], dtype=np.float64) for cn in cols}
    yi = np.asarray(y).astype(np.int64)
    raw_mi = {cn: _mi(arrs[cn], yi, nbins=nbins) for cn in cols}
    rng = np.random.default_rng(rng_seed)
    hits: list[dict] = []
    budget = max_triples
    for cgate in cols:
        cv = arrs[cgate]
        taus = np.quantile(cv, _TAU_QUANTILES)
        others = [cn for cn in cols if cn != cgate]
        # mask: one active column a (b unused)
        for a in others:
            if budget <= 0:
                break
            floor = raw_mi[a]
            best = (-1.0, None)
            for tau in taus:
                feat = apply_conditional_gate(arrs[a], arrs[a], cv, tau, "mask")
                mi = _mi(feat, yi, nbins=nbins)
                if mi > best[0]:
                    best = (mi, float(tau))
            budget -= 1
            if best[0] < floor + min_margin:
                continue
            feat = apply_conditional_gate(arrs[a], arrs[a], cv, best[1], "mask")
            null_hi = _perm_null_hi(feat, yi, nbins, n_perm, rng)
            if best[0] <= null_hi:
                continue
            hits.append(dict(mode="mask", a=a, b=None, gate_col=cgate, tau=best[1],
                             mi=float(best[0]), operand_floor=float(floor), null_hi=float(null_hi)))
        # select: ordered (a, b) -- a where c>tau else b
        for a in others:
            for b in others:
                if a == b or budget <= 0:
                    continue
                floor = max(raw_mi[a], raw_mi[b])
                best = (-1.0, None)
                for tau in taus:
                    feat = apply_conditional_gate(arrs[a], arrs[b], cv, tau, "select")
                    mi = _mi(feat, yi, nbins=nbins)
                    if mi > best[0]:
                        best = (mi, float(tau))
                budget -= 1
                if best[0] < floor + min_margin:
                    continue
                feat = apply_conditional_gate(arrs[a], arrs[b], cv, best[1], "select")
                null_hi = _perm_null_hi(feat, yi, nbins, n_perm, rng)
                if best[0] <= null_hi:
                    continue
                hits.append(dict(mode="select", a=a, b=b, gate_col=cgate, tau=best[1],
                                 mi=float(best[0]), operand_floor=float(floor), null_hi=float(null_hi)))
    hits.sort(key=lambda h: h["mi"], reverse=True)
    return hits


def scan_row_argmax(
    X,
    y: np.ndarray,
    cols: list[str],
    *,
    nbins: int = 12,
    n_perm: int = 12,
    min_margin: float = _MIN_MARGIN,
    rng_seed: int = 0,
    max_triples: int = 40,
) -> list[dict]:
    """Cheap-first scan for row-argmax structure over column triples. Emit a hit per triple whose argmax-code MI beats the best
    member's raw MI by ``min_margin`` AND a permutation-null band. Pairs are skipped (a 2-col argmax == sign of the diff, already
    on the diff list); triples are the smallest genuinely-new arity."""
    from itertools import combinations

    arrs = {cn: np.asarray(X[cn], dtype=np.float64) for cn in cols}
    yi = np.asarray(y).astype(np.int64)
    raw_mi = {cn: _mi(arrs[cn], yi, nbins=nbins) for cn in cols}
    rng = np.random.default_rng(rng_seed)
    hits: list[dict] = []
    budget = max_triples
    for tri in combinations(cols, 3):
        if budget <= 0:
            break
        budget -= 1
        floor = max(raw_mi[c] for c in tri)
        feat = apply_row_argmax([arrs[c] for c in tri])
        mi = _mi(feat, yi, nbins=nbins)
        if mi < floor + min_margin:
            continue
        null_hi = _perm_null_hi(feat, yi, nbins, n_perm, rng)
        if mi <= null_hi:
            continue
        hits.append(dict(cols=tri, mi=float(mi), operand_floor=float(floor), null_hi=float(null_hi)))
    hits.sort(key=lambda h: h["mi"], reverse=True)
    return hits
