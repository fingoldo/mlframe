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
-- the max MI over the cheap arithmetic ops a selector already has (product, ratio, diff, min, max) AND the ADDITIVE/linear
combinations (pairwise sums + the full sum of the involved columns), mirroring the ``bench_frontier_candidates`` baseline -- NOT the
raw single-operand MI. The additive terms close a specificity hole on the COMMON additive-linear target shape: a piecewise
``c>tau ? a : b`` partially reconstructs a purely additive ``y = a+b+c``, so without ``a+b`` / ``a+c`` / ``b+c`` / ``a+b+c`` in the
floor the gate fired a spurious feature on a multi-driver additive (binned-regression / 4-driver) target. With this floor the gate
clears 0 false-positives on smooth / noise / ordinary_mul / multi-driver-additive at p=30 over 3 seeds
(``bench_conditional_gate_wideframe``), while the true regime targets (no additive combo reconstructs the switch) still respond.

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
skipped, stored +inf -- bit-identical by short-circuit). RELEVANCE-PRUNED candidate set (``_rank_and_prune``): the gate select sweep
was O(p^3) (C(p,2) operand pairs x p gate cols x ~17 tau), which forced the gate OFF; it is now O(k_operand^2 * k_gate) FLAT in p --
operands ranked by raw MI vs y, the gate column by a conditional-divergence rank (a regime switch's gate column can be marginally
y-independent, so raw MI ranks it last). With k_operand=10 / k_gate=8 the added cost is comparable to modular/lattice/argmax and the
gate now defaults ON. ``max_cols`` (default 200) stays a defense-in-depth outer cap that skips the whole sweep on an absurdly wide frame.

cProfile (n=2000, p=15, enabled path, measured): the batched plug-in MI dispatch is the top hotspot (argmax ~99%, gate ~97% of the
scan wall); the argmax stack + ``np.argmax`` and the gate ``np.where`` / quantile arithmetic are <3% combined -- the per-candidate
build is dwarfed by the (already-tuned, batched) MI kernel. No further actionable caller-side speedup beyond the per-candidate tau
batching + early-reject already applied. See ``_benchmarks/bench_conditional_gate_wideframe``.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Sequence

import numba
import numpy as np

from ._pairwise_modular_fe import _mi

logger = logging.getLogger(__name__)


@numba.njit(cache=True, parallel=True, fastmath=False)
def _gate_mask_grid_njit(cv, av, taus):
    """Fused (n, n_tau) mask block: ``feats[i, j] = av[i] * (cv[i] > taus[j])``. The off-branch value ``a * 0.0``
    preserves the NaN semantics of the numpy ``av * (cv > tau)`` form (0*NaN=NaN), so the kernel is bit-identical incl NaN."""
    n = cv.shape[0]; k = taus.shape[0]
    out = np.empty((n, k), dtype=np.float64)
    for i in numba.prange(n):
        c = cv[i]; a = av[i]; off = a * 0.0
        for j in range(k):
            out[i, j] = a if c > taus[j] else off
    return out


@numba.njit(cache=True, parallel=True, fastmath=False)
def _gate_select_grid_njit(cv, av, bv, taus):
    """Fused (n, n_tau) select block: ``feats[i, j] = av[i] if cv[i] > taus[j] else bv[i]``. Pure gather (no arithmetic),
    so bit-identical to the numpy ``np.where(cv > tau, av, bv)`` form incl NaN operands."""
    n = cv.shape[0]; k = taus.shape[0]
    out = np.empty((n, k), dtype=np.float64)
    for i in numba.prange(n):
        c = cv[i]; a = av[i]; b = bv[i]
        for j in range(k):
            out[i, j] = a if c > taus[j] else b
    return out

# The per-candidate (n, 17-tau) mask/select build is fused into one njit(parallel) kernel ABOVE ``_GATE_BUILD_NJIT_MIN_N``
# (default 20000), numpy per-tau loop below. History: the fusion was first e2e-rejected at SMALL n (2026-06-13, iter53):
# isolated njit(parallel) won 1.26-2.46x over the numpy loop at n=533..12000 but LOST end-to-end (whole-scan njit 0.89-0.90x)
# because at small n the build is a tiny fraction of the scan and the kernel's prange contends with the MI prange
# (``_gate_grid_mi``). At LARGE n that flips: a full-suite profile put ``_build_feats`` at 22s tottime / 2058 calls (n=40k)
# and a paired end-to-end A/B of ``cheap_conditional_gate_scan`` @n=40k measured numpy 22.3s vs njit 20.8s = 1.07x, 3/3 wins,
# scan output BIT-IDENTICAL (705 hits both). So the fusion is gated ON only where the build is a real fraction of the scan.
# The mask off-value is ``a*0.0`` (not 0.0) to preserve the numpy off-region NaN semantics. A numpy broadcast build
# (``cv[:,None]>taus[None,:]``) was rejected outright (0.65x@n12000, the (n,17) bool temp blows cache).
# bench: _benchmarks/bench_gate_grid_njit.py.

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

# Row threshold above which _build_feats fuses the per-tau (n, 17) mask/select block into one njit(prange) kernel
# instead of the numpy per-tau loop. The isolated build kernel wins at all n, but the 2026-06-13 iter53 reject found
# it LOSES end-to-end at small n (build is a tiny fraction of the scan + its prange contends with the MI prange);
# gated ON only for large n where the build is a real fraction of the scan (validated end-to-end). Env-overridable.
_GATE_BUILD_NJIT_MIN_N = int(os.environ.get("MLFRAME_GATE_BUILD_NJIT_MIN_N", "20000"))

# Absolute floor the engineered MI must clear ABOVE the permutation-null band (not just `> null_hi`); mirrors _pairwise_modular_fe._MIN_NULL_MARGIN.
# Guards the cardinality-inflation false positive on a few-class y (a ~10-bin regression/quantized target), where a select/mask column's
# plug-in MI can sit ~0.01 nats above a z=3 null on noise; a true regime/argmax hit clears the null by a wide margin.
_MIN_NULL_MARGIN = 0.05


def apply_row_argmax(X, cols: Sequence[str]) -> np.ndarray:
    """Replay one row-argmax column: the integer index (0..k-1) of the row-maximum over the source columns.

    Pure function of X (no y, no fitted state) -> transform() is leak-free + deterministic + train/test bit-identical. Output
    float64 in ``{0, .., k-1}``; ties resolve to the first max (numpy ``argmax`` semantics), stable across fit / replay."""
    if len(cols) < 2:
        raise ValueError(f"row-argmax needs >= 2 source columns; got {tuple(cols)!r}")
    stk = np.stack([np.asarray(X[c], dtype=np.float64) for c in cols], axis=1)
    out = np.argmax(stk, axis=1).astype(np.float64)
    # Serve-time NaN policy: eligible source columns are all-finite at FIT (see _is_argmax_eligible), but a row can
    # carry a NaN at SERVE. np.argmax would return the first-NaN index -- a spurious in-distribution code the model
    # never learned. Propagate NaN instead so the downstream model treats the row as missing (LGBM/CatBoost native).
    nan_rows = ~np.isfinite(stk).all(axis=1)
    if nan_rows.any():
        out[nan_rows] = np.nan
    return np.asarray(out)


def apply_conditional_gate(X, mode: str, cols: Sequence[str], tau: float) -> np.ndarray:
    """Replay one conditional-gate column with the FROZEN threshold ``tau``.

    ``select`` (cols = (a, b, c)): ``c > tau ? a : b`` -- a regime switch routing a / b by the gating column c.
    ``mask`` (cols = (a, c)): ``1[c > tau] * a`` -- a active only where c > tau.

    Pure function of (source columns, tau) -- no y, no fitted state beyond the recipe-frozen tau -> leak-free + train/test exact."""
    # Serve-time NaN policy on the GATING column c: ``c > tau`` is False for NaN, which would silently route a
    # missing-gate row to b (select) or 0 (mask) -- a defined-but-arbitrary code the model did not learn. Propagate
    # NaN where c is non-finite so a missing gate reads as missing downstream (a / b NaN already propagate naturally).
    if mode == "select":
        if len(cols) != 3:
            raise ValueError(f"conditional-gate 'select' needs exactly 3 source columns (a, b, c); got {tuple(cols)!r}")
        a, b, c = (np.asarray(X[cn], dtype=np.float64) for cn in cols)
        res = np.where(c > float(tau), a, b)
        return np.where(np.isfinite(c), res, np.nan)
    if mode == "mask":
        if len(cols) != 2:
            raise ValueError(f"conditional-gate 'mask' needs exactly 2 source columns (a, c); got {tuple(cols)!r}")
        a, c = (np.asarray(X[cn], dtype=np.float64) for cn in cols)
        res = (c > float(tau)).astype(np.float64) * a
        return np.where(np.isfinite(c), res, np.nan)
    raise ValueError(f"conditional-gate mode must be one of {GATE_MODES}; got {mode!r}")


def best_existing_op_mi(arrs: dict, names: Sequence[str], yi: np.ndarray, nbins: int) -> float:
    """Max binned-MI over the cheap operators a selector already has on the given operands: raw columns + pairwise
    product / ratio / diff + row-max / row-min + ADDITIVE/linear combinations (pairwise sums + the full sum of all involved
    columns). This is the HARDENED baseline both detectors must beat -- the prototype gated only vs the raw single-operand MI,
    which let a hard threshold reconstruct an XOR-sign regime on smooth / ordinary_mul controls AND let a spurious row-argmax clear
    the floor on an ordinary-multiplicative target (false positives, measured in ``bench_conditional_gate_wideframe``).

    The ADDITIVE terms close the gate's additive-target specificity hole: a piecewise ``c>tau ? a : b`` partially reconstructs a
    purely additive signal (e.g. ``y = a + b + c``), so on a multi-driver additive target the gate's MI cleared a floor that knew
    only the pairwise arithmetic ops -- not the additive combinations ``a+b`` / ``a+c`` / ``b+c`` / ``a+b+c`` that a linear selector
    trivially captures. Including those sums makes a purely-additive target FAIL the floor (the additive baseline captures it) while
    a TRUE regime switch (where no additive combo reconstructs the data-dependent branch) still clears it. Mirrors the
    ``bench_frontier_candidates`` 'best existing op' reference.

    All candidate columns stack into ONE batched ``_mi_classif_batch`` call (bit-identical to per-candidate -- the kernel bins
    each column independently, only the per-call dispatch overhead is amortised); cProfile showed the per-``_mi`` dispatch was the
    dominant mlframe-side cost of the hardened floor, so batching is the actionable caller-side lever."""
    from ._orthogonal_univariate_fe import _mi_classif_batch

    names = list(names)
    # MANDATE-2 (2026-06-23): resident-GPU candidate-gen + MI route. The candidate columns are built on the
    # device + scored by the resident plug-in MI (NO host round-trip), engaged ONLY where the per-host KTC
    # crossover (_resident_candidate_mi_ktc) measured it faster than the host njit batch-MI below. On the dev
    # GTX 1050 Ti the gate's k is small (k = m + 4*C(m,2) + 2|3; m=3 -> k=14, sub-crossover) so this stays
    # CPU here; it engages for large-k / stronger GPUs. Selection-equivalent (percentile-edge vs rank binning,
    # the approved FE-PAIR trade); on any failure / no-cupy it returns None and the exact njit path runs.
    _m = len(names)
    _k = _m + 4 * (_m * (_m - 1) // 2) + (3 if _m >= 3 else 2)
    # GATE rank residency: under MLFRAME_FE_GPU_STRICT_RESIDENT the gate baseline MI must byte-match the CPU
    # njit RANK MI (the gate output gate_mask is ~50% tied zeros where edge != rank). Route the resident
    # candidate MI through the RANK kernel; on any failure it returns None and the host njit rank path runs.
    _gate_rank = _gate_rank_binning()
    try:
        from ._resident_candidate_mi_ktc import rescand_use_resident
        if rescand_use_resident(int(np.asarray(arrs[names[0]]).shape[0]), _k):
            from ._resident_candidate_mi import best_existing_op_mi_resident
            # y is a fit-constant -> host-derive y_min / n_classes once and pass them so the resident plug-in
            # skips the per-call GPU cp.min + cp.max reduction (same data -> bit-identical bincount layout).
            _yi = np.ascontiguousarray(np.asarray(yi)).astype(np.int64).ravel()
            _ym = int(_yi.min()) if _yi.size else 0
            _nc = (int(_yi.max()) - _ym + 1) if _yi.size else 1
            _r = best_existing_op_mi_resident(arrs, names, yi, nbins, y_min=_ym, n_classes=_nc, rank_binning=_gate_rank)
            if _r is not None:
                return _r
    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
        logger.debug("suppressed in _conditional_gate_fe.py:234: %s", e)
        pass
    cols_arr = [np.asarray(arrs[c], dtype=np.float64) for c in names]
    cands = list(cols_arr)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            u, v = cols_arr[i], cols_arr[j]
            cands.append(u * v)
            cands.append(u - v)
            cands.append(u / (np.abs(v) + 1e-6))
            cands.append(u + v)  # pairwise additive baseline a+b / a+c / b+c -- a linear selector captures it for free
    stk = np.stack(cols_arr, axis=1)
    cands.append(stk.max(axis=1))
    cands.append(stk.min(axis=1))
    if len(cols_arr) >= 3:
        cands.append(stk.sum(axis=1))  # full additive sum a+b+c -- the multi-driver additive signal the gate must not reconstruct
    mat = np.column_stack(cands).astype(np.float64, copy=False)
    mis = _mi_classif_batch(np.ascontiguousarray(mat), yi.astype(np.int64), nbins=nbins, rank_binning=_gate_rank)
    return float(np.max(mis))


def _responded(feat_mi: float, baseline: float, null_hi: float, min_margin: float = _MIN_MARGIN, null_margin: float = _MIN_NULL_MARGIN) -> bool:
    """Gate: the engineered column's MI must clear BOTH the operand baseline (by ``min_margin``) AND the permutation-null upper band by an
    absolute ``null_margin`` (not just ``> null_hi`` -- guards the cardinality-inflation false positive on a few-class y; see ``_MIN_NULL_MARGIN``).
    Mirrors ``_pairwise_modular_fe._responded`` (``baseline`` plays the smooth-basis floor role)."""
    return (feat_mi - baseline) >= min_margin and feat_mi > (null_hi + null_margin)


def _gate_cands_resident() -> bool:
    """Whether the SCORED gate / row-argmax candidate float is uploaded ONCE and its resident cupy handle threaded
    through the marginal MI + the 12-perm null -- so the candidate never re-crosses H2D at the ``_mi_classif_batch``
    :318 upload site (measured: 80 MB/x10 from ``cheap_row_argmax_scan``, 3 MB/x12 from ``_perm_null_hi`` on a 1M
    STRICT-resident F2 fit). Default ON under ``fe_gpu_strict_resident_enabled`` + ``_cmi_gpu_enabled`` (the same
    predicate pair the CMI-gate / raw-redundancy candidate residency uses); opt-out ``MLFRAME_FE_GATE_RESIDENT_CANDS=0``.
    Any import fault -> off (the exact host upload path runs)."""
    if os.environ.get("MLFRAME_FE_GATE_RESIDENT_CANDS", "1").strip().lower() not in ("1", "true", "on", "yes"):
        return False
    try:
        from ._gpu_strict_fe import fe_gpu_strict_resident_enabled
        from ._mi_greedy_cmi_fe import _cmi_gpu_enabled
        return bool(fe_gpu_strict_resident_enabled()) and bool(_cmi_gpu_enabled())
    except Exception as _flag_exc:
        logger.debug("gate GPU-strict-resident flag probe failed (%s); resident candidates disabled.", _flag_exc)
        return False


def _resident_cand(feat: np.ndarray, role):
    """Upload a SCORED candidate float column ONCE and return its RESIDENT cupy handle (content-keyed via
    ``resident_operand`` under ``role``), so ``_mi`` / ``_perm_null_hi`` read it device-side instead of re-uploading
    the same host float at every MI kernel call. Returns the ORIGINAL host array unchanged when residency is off, on a
    non-finite column (the resident percentile-edge binner needs an all-finite operand -> host path preserves the CPU
    NaN handling), or on ANY cupy fault -> the host MI path runs (selection-identical, no work hidden)."""
    if not _gate_cands_resident():
        return feat
    host = np.ascontiguousarray(np.asarray(feat, dtype=np.float64)).ravel()
    if not np.all(np.isfinite(host)):
        return feat
    try:
        from ._fe_resident_operands import resident_operand
        import cupy as cp  # noqa: F401  (import guard: skip residency when cupy is absent)
        return resident_operand(host, role, dtype=np.float64)
    except Exception:
        logger.debug("gate resident-candidate upload failed; host fallback", exc_info=True)
        return feat


def _argmax_resident(arrs, tri):
    """DEVICE-BORN the row-argmax candidate ``argmax_k(operand_k)`` from the RESIDENT operand columns, returning
    the resident float64 index column, or ``None`` when residency is off / any operand is non-finite / cupy faults
    (the caller then takes the host ``np.argmax`` + one-shot resident upload).

    ``cp.argmax`` returns the FIRST occurrence of the max along the axis -- identical to ``np.argmax`` -- and the
    operand floats are byte-identical to the host columns, so the argmax INDEX codes are selection-identical. The
    win: the DISTINCT argmax candidate is BORN on device from the raw operands (each uploaded once, shared via the
    content-keyed cache) and never crosses H2D, instead of a host ``np.argmax`` whose float column re-uploads."""
    if not _gate_cands_resident():
        return None
    try:
        import cupy as cp

        from ._fe_resident_operands import resident_operand
        _ops = []
        for c in tri:
            _h = np.ascontiguousarray(np.asarray(arrs[c], dtype=np.float64))
            if not np.all(np.isfinite(_h)):
                return None  # host np.argmax NaN semantics differ -> keep the host path for a non-finite operand
            _ops.append(resident_operand(_h, ("argmax_op", str(c)), dtype=cp.float64))
        return cp.argmax(cp.stack(_ops, axis=1), axis=1).astype(cp.float64)
    except Exception:
        logger.debug("device-born row-argmax failed; host fallback", exc_info=True)
        return None


def _perm_null_hi(feat, y: np.ndarray, nbins: int, n_perm: int = 12, seed: int = 0, z: float = 3.0) -> float:
    """Upper band (mean + z*std) of the fixed feature's MI under y permutation -- the noise reference the feature MI must clear.
    The feature is fixed; only y is shuffled (cheap, n_perm small).

    ``feat`` may be a host ndarray OR an ALREADY-RESIDENT cupy handle (the gate / row-argmax scorer uploads the scored
    candidate ONCE and threads the resident handle here): the fixed candidate is FIT-CONSTANT across all ``n_perm``
    shuffles, so a resident handle is uploaded once and reused for every permutation instead of re-uploading the same
    float 12x. ``_mi`` takes the resident-input branch; only y is shuffled (a fresh host int64 per perm, as before)."""
    rng = np.random.default_rng(seed)
    yi = np.asarray(y).astype(np.int64)
    vals = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        vals[i] = _mi(feat, yi[rng.permutation(yi.size)], nbins=nbins)
    return float(vals.mean() + z * vals.std())


def _gate_rank_binning() -> bool:
    """Whether the conditional-gate MI uses the RANK resident binner -- opt-in BYTE-MATCH only.

    Default OFF, AND off under the plain resident flag: the resident gate MI uses the FAST percentile-edge
    binner, which is selection-equivalent to CPU on F2 (the gate edge-vs-rank difference shifts the gate lift
    MAGNITUDE on heavily-tied operator outputs but does not flip the F2 selection). Only when the dedicated
    ``MLFRAME_FE_GPU_STRICT_BYTEMATCH`` opt-in is set (which also requires the resident path) does the gate MI
    bin by argsort equi-frequency RANK to byte-match the CPU njit rank MI -- paying an irreducible per-gate
    argsort (~1s/fit on the GTX 1050 Ti). Fast-by-default, byte-match-on-request."""
    try:
        from ._gpu_strict_fe import fe_gpu_strict_bytematch_enabled
        return bool(fe_gpu_strict_bytematch_enabled())
    except Exception as _flag_exc:
        logger.debug("gate GPU-strict-bytematch flag probe failed (%s); bytematch binning disabled.", _flag_exc)
        return False


def _gate_grid_mi(feats: np.ndarray, yi: np.ndarray, nbins: int) -> np.ndarray:
    """MI of every column of the (n, k) tau-grid feature matrix vs y, in one batched kernel call (bit-identical to per-column --
    ``_mi_classif_batch`` bins each column independently, only the per-call dispatch overhead is amortised)."""
    from ._orthogonal_univariate_fe import _mi_classif_batch

    return np.asarray(_mi_classif_batch(np.ascontiguousarray(feats), yi, nbins=nbins, rank_binning=_gate_rank_binning()), dtype=np.float64)


def _is_argmax_eligible(x: np.ndarray) -> bool:
    """True iff the column is finite numeric (int or float) -- argmax / gate need an order, not an integer lattice."""
    a = np.asarray(x)
    if not np.issubdtype(a.dtype, np.number):
        return False
    # Test the WHOLE column: a column with any NaN/inf has no total order for argmax / gating and must be excluded.
    # (The prior form ``isfinite(a[isfinite(a)])`` pre-filtered to the finite subset, so the check was always True
    # and NaN columns slipped through, producing NaN taus / first-NaN argmax codes / constant gate features.)
    return bool(a.size == 0 or np.isfinite(a).all())


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
        """MI gained by the argmax code over the hardened best-existing-op floor; the ranking key for candidate triples."""
        return self.feat_mi - self.operand_floor

    @property
    def responded(self) -> bool:
        """Whether this argmax feature's MI clears both the operand floor and the null band -- i.e. it is a genuine signal, not noise."""
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
        """MI gained by the gated feature at its frozen tau over the hardened best-existing-op baseline; the ranking key for candidate gates."""
        return self.feat_mi - self.baseline_mi

    @property
    def responded(self) -> bool:
        """Whether this gate's MI clears both the operand baseline and the null band -- i.e. it is a genuine signal, not noise."""
        return _responded(self.feat_mi, self.baseline_mi, self.null_hi)


def cheap_row_argmax_scan(
    X,
    y: np.ndarray,
    cols: Optional[Sequence[str]] = None,
    *,
    nbins: int = 12,
    seed: int = 0,
    max_triples: int = 40,
    _cols_prefiltered: bool = False,
) -> list[ArgmaxHit]:
    """Cheap first-pass scan for row-argmax structure over column TRIPLES of X.

    For each triple compute the argmax-code MI vs y; keep the hit when it beats the HARDENED best-existing-op MI on the triple
    (max over raw / product / ratio / diff / min / max) by ``_MIN_MARGIN`` AND a 12-perm null band. The hardened floor (not the
    raw single-operand MI) is what keeps argmax clean on the ordinary-multiplicative control at scale. Pairs are skipped (a 2-col
    argmax == sign of the diff, already on the diff list). Budgeted by ``max_triples``. The null is early-rejected (computed only
    for triples already clearing the operand margin).

    ``_cols_prefiltered`` is internal-only: set by callers that already built ``cols`` via ``_is_argmax_eligible`` themselves
    (e.g. ``hybrid_row_argmax_fe_with_recipes``), to skip the redundant re-check below. External callers must leave it False."""
    import pandas as pd  # noqa: F401  (X may be pandas or polars; we pull ndarrays)

    if cols is None:
        cols = [c for c in X.columns if _is_argmax_eligible(np.asarray(X[c]))]
    elif _cols_prefiltered:
        cols = [c for c in cols if c in X.columns]
    else:
        cols = [c for c in cols if c in X.columns and _is_argmax_eligible(np.asarray(X[c]))]
    # Canonicalize the eligible-column order so the budgeted triple enumeration below scans the SAME triples regardless of the caller's input column order; without this a reversed-column frame walks a different ``max_triples`` prefix and synthesizes different argmax features, making the downstream selection column-order dependent.
    cols = sorted(cols, key=lambda c: str(c))
    yi = np.asarray(y).astype(np.int64)
    arrs = {c: np.asarray(X[c], dtype=np.float64) for c in cols}

    hits: list[ArgmaxHit] = []
    budget = int(max_triples)
    # Iterate triples in a column-ORDER-INVARIANT sequence. C(p,3) far exceeds the
    # ``max_triples`` budget for any realistic p, so the budget truncates the scan;
    # if the triples were enumerated in raw input-column order, reversing the input
    # columns would feed a DIFFERENT first-``budget`` set of triples into the pool,
    # seed different argmax candidates, and change the downstream greedy selection
    # (column-order-invariance contract break). Enumerating over the NAME-sorted
    # column order makes the budgeted triple set identical under any input column
    # permutation. ``hits`` is re-sorted by margin below, so per-hit output order is
    # unaffected -- only WHICH triples survive the budget is made deterministic.
    # (``cols`` was already name-sorted above; no re-sort needed here.)
    # bench-attempt-rejected (2026-06-26): batching the per-triple operand floor (one resident MI for the whole
    # gate sweep instead of one per triple) is NOT suite-safe, in BOTH variants tried:
    #   * HOST-built batch via _mi_classif_batch -- routes on STRICT only, so under use_gpu-without-STRICT it
    #     switched the gate binning estimator (resident EDGE -> CPU RANK) vs the per-triple rescand_use_resident
    #     (KTC) routing, shifting the selected set (broke test_gpu_cpu_..._selection_identical[reg_mixed]).
    #   * RESIDENT device-built batch (best_existing_op_mi_resident_batched) -- per-column BIT-IDENTICAL to the
    #     per-triple resident floor (reg_two_pairs PASSES in isolation), but concatenating ~20 triples' device
    #     matrices changes the cupy-pool / KTC-cache state pattern enough to expose the SUITE's pre-existing
    #     order-dependent routing flakiness (reg_two_pairs FAILED only in-suite). A change that makes the GPU
    #     suite flaky is unshippable.
    # Keep the per-triple call. The gate's launch count is irreducible without first decoupling the suite from
    # KTC/pool ordering (a test-infra change, not a launch optimisation).
    for tri in combinations(cols, 3):
        if budget <= 0:
            break
        budget -= 1
        operand_floor = best_existing_op_mi(arrs, tri, yi, nbins)
        # DEVICE-BORN the argmax candidate from the RESIDENT operand columns: cp.argmax of the resident operand
        # stack returns the FIRST-max index exactly as np.argmax on the SAME floats -> selection-identical index
        # codes, and the DISTINCT candidate is generated on device from raw so it NEVER uploads (only the few base
        # operands upload once, shared via the resident cache). The resident handle is then threaded through the
        # marginal MI + the 12-perm null. Falls back to host np.argmax + a one-shot resident upload when residency
        # is off / an operand is non-finite (host argmax NaN semantics) / any cupy fault.
        feat_r = _argmax_resident(arrs, tri)
        if feat_r is None:
            feat = np.argmax(np.stack([arrs[c] for c in tri], axis=1), axis=1).astype(np.float64)
            feat_r = _resident_cand(feat, ("gate_cand_argmax", tuple(str(c) for c in tri)))
        feat_mi = _mi(feat_r, yi, nbins=nbins)
        # Early-reject: the null only matters for a triple already clearing the operand margin (``_responded`` needs BOTH).
        if (feat_mi - operand_floor) >= _MIN_MARGIN:
            null_hi = _perm_null_hi(feat_r, yi, nbins, seed=seed)
        else:
            null_hi = float("inf")
        hits.append(ArgmaxHit(tuple(tri), feat_mi, operand_floor, null_hi))

    # Canonical secondary key on the triple identity so near-ties don't break by enumeration (column) order.
    hits.sort(key=lambda h: (-h.margin_over_operands, tuple(str(c) for c in h.cols)))
    return hits


def _rank_and_prune(X, cols: Sequence[str], yi: np.ndarray, nbins: int, k_gate: int, k_operand: int) -> tuple[list[str], list[str]]:
    """Cheap-first candidate prune that makes the gate sweep O(k_operand^2 * k_gate), FLAT in p, returning (gate pool, operand pool).

    TWO DIFFERENT relevance signals, because operands and the gate column play different roles in ``c > tau ? a : b``:

    * OPERANDS a, b -- the VALUE the switch routes, so a useful operand carries marginal relevance to y. Ranked by raw binned-MI vs y
      (one batched ``_mi_classif_batch`` call) -> top-``k_operand``. Gating two pure-noise columns yields noise, so the noise tail drops.

    * GATE column c -- only decides WHICH operand, so on a pure regime switch c can be marginally INDEPENDENT of y (raw MI ~ 0): raw-MI
      ranking puts the true gate column LAST and a tight raw-MI top-k would miss it (measured: seed 7 of the gate synthetic, c ranks 28/28).
      So c is ranked by a CONDITIONAL-DIVERGENCE signal instead -- how much splitting on ``c > median(c)`` changes the operand->y MI:
      ``|MI(a*, y | c>med) - MI(a*, y | c<=med)|`` summed over the top operands a*. A column that genuinely switches the regime shows a
      large conditional divergence even at zero marginal MI; pure noise does not. One batched MI call per side ranks the whole gate pool.

    Both signals are O(p) one-shot batched MI calls; the resulting O(k_operand^2 * k_gate) sweep is independent of p."""
    from ._orthogonal_univariate_fe import _mi_classif_batch

    _rb = _gate_rank_binning()
    # Canonicalize column order so the stable argsort tie-breaks below resolve by feature name, not by the caller's input column order (reversed-column invariance).
    cols = sorted(cols, key=lambda c: str(c))
    if not cols:
        return [], []
    arrs = [np.asarray(X[c], dtype=np.float64) for c in cols]
    mat = np.ascontiguousarray(np.column_stack(arrs))
    # Class-B :311 collapse (2026-06-30): this ``mat`` is the FIT-CONSTANT raw column_stack of the (sorted)
    # candidate columns -- a pure relevance baseline re-scored across the fit. Under STRICT-residency it already
    # routes through the resident plug-in but re-uploads fresh at _orth_mi_backends:311; ride the resident-operand
    # cache so it uploads ONCE. Same (rank|edge) resident estimator the host STRICT path uses -> byte-identical
    # per-column MI -> identical ``argsort(mis)`` ranking (cols are pre-sorted at :393 so ties resolve
    # deterministically). None on any cupy failure / non-strict -> the EXACT host scorer (byte-identical default).
    from ._resident_raw_mi import resident_raw_baseline_mi

    mis = resident_raw_baseline_mi(mat, yi, ("gate_prune_raw", tuple(cols)), nbins=nbins, rank_binning=_rb)
    if mis is None:
        mis = _mi_classif_batch(mat, yi, nbins=nbins, rank_binning=_rb)
    mis = np.asarray(mis, dtype=np.float64)
    operand_order = list(np.argsort(mis)[::-1])  # marginal relevance, descending
    operand_cols = [cols[i] for i in operand_order][: max(2, int(k_operand))]

    # Gate rank: conditional-divergence of the top operands' y-MI across a c>median split. Computed against a small operand probe
    # set (the most-relevant few) so it stays cheap; summed |divergence| ranks every candidate gate column without needing c~y MI.
    n_probe = min(len(operand_cols), 3)
    probe_idx = [operand_order[i] for i in range(n_probe)]
    probe = mat[:, probe_idx]  # (n, n_probe)
    div = np.zeros(len(cols), dtype=np.float64)
    # DEVICE-BORN GATE RANK-PRUNE (Phase-1 residency, 2026-07-01). Each gate candidate splits the probe operand
    # matrix by ``cv > median(cv)`` and scores each side's MI. Without residency the probe SLICE (measured ~24 MB,
    # the _mi_classif_batch upload) AND the per-split y SUBSET (~38 MB, the 14+10 distinct hi/lo subsamples of the
    # y_mi_classif role) re-upload every iteration -- the two biggest remaining candidate-code uploads. Keep the
    # operand matrix + the label y RESIDENT and do the median split + boolean slice ON device, so both sides stay
    # on the GPU (mat_dev content-HITS the ("gate_prune_raw", cols) upload already made at ~:485 -> no extra H2D;
    # float64 -> the device median split is byte-identical to the host np.median split -> selection-identical).
    # Any cupy fault -> the exact host path below (correctness first).
    _gate_dev = None
    try:
        from ._gpu_strict_fe import fe_gpu_strict_resident_enabled as _grs
        if _grs():
            import cupy as _cp
            from ._fe_resident_operands import assemble_resident_matrix, resident_operand
            _mat_dev = assemble_resident_matrix(mat, cols, ("gate_prune_raw", tuple(cols)), dtype=_cp.float64)
            _yi_dev = resident_operand(np.ascontiguousarray(np.asarray(yi)).astype(np.int64), "y_mi_classif", dtype=np.int64)
            _gate_dev = (_cp, _mat_dev, _yi_dev, _mat_dev[:, probe_idx])
    except Exception as _gate_dev_exc:
        logger.debug("gate resident-matrix assembly failed (%s); host prune path used.", _gate_dev_exc)
        _gate_dev = None
    for gi in range(len(cols)):
        if _gate_dev is not None:
            _cp, _mat_dev, _yi_dev, _probe_dev = _gate_dev
            _cvd = _mat_dev[:, gi]
            _hi = _cvd > _cp.median(_cvd)
            # Integer row indices (not boolean masks): cp.where syncs once to size each index, but then the four
            # regime gathers (_probe_dev / _yi_dev for hi + lo) are known-size integer gathers with NO sync, and
            # the degenerate-split check reads .size (host shape, free) instead of a separate int(_hi.sum()) D2H.
            _hi_idx = _cp.where(_hi)[0]
            _lo_idx = _cp.where(~_hi)[0]
            if int(_hi_idx.size) < nbins or int(_lo_idx.size) < nbins:
                continue  # degenerate split (near-constant gate column) -> no usable regime
            mi_hi = np.asarray(_mi_classif_batch(_probe_dev[_hi_idx], _yi_dev[_hi_idx], nbins=nbins, rank_binning=_rb), dtype=np.float64)
            mi_lo = np.asarray(_mi_classif_batch(_probe_dev[_lo_idx], _yi_dev[_lo_idx], nbins=nbins, rank_binning=_rb), dtype=np.float64)
        else:
            cv = arrs[gi]
            hi = cv > np.median(cv)
            lo = ~hi
            if hi.sum() < nbins or lo.sum() < nbins:
                continue  # degenerate split (near-constant gate column) -> no usable regime
            mi_hi = np.asarray(_mi_classif_batch(np.ascontiguousarray(probe[hi]), yi[hi], nbins=nbins, rank_binning=_rb), dtype=np.float64)
            mi_lo = np.asarray(_mi_classif_batch(np.ascontiguousarray(probe[lo]), yi[lo], nbins=nbins, rank_binning=_rb), dtype=np.float64)
        div[gi] = float(np.abs(mi_hi - mi_lo).sum())
    gate_order = list(np.argsort(div)[::-1])
    gate_cols = [cols[i] for i in gate_order][: max(1, int(k_gate))]
    return gate_cols, operand_cols


def cheap_conditional_gate_scan(
    X,
    y: np.ndarray,
    cols: Optional[Sequence[str]] = None,
    *,
    nbins: int = 12,
    seed: int = 0,
    k_gate: int = 8,
    k_operand: int = 10,
    subsample_n: int = 0,
    _cols_prefiltered: bool = False,
) -> list[GateHit]:
    """Cheap first-pass scan for conditional-gate structure (regime-switch + masked-interaction) over X.

    RELEVANCE-PRUNED candidate set (the cost lever): a regime switch ``c > tau ? a : b`` is only useful when the gate column c
    carries SOME relevance to y (an irrelevant split is meaningless) and the operands a, b are among the more relevant columns
    (gating two pure-noise columns yields noise). So we rank every eligible column ONCE by raw binned-MI vs y (one batched
    ``_mi_classif_batch`` call -- the cheap primitive) and restrict the GATE columns c to the top-``k_gate`` and the OPERAND
    columns a, b to the top-``k_operand`` by that relevance. The sweep becomes C(k_operand, 2) x k_gate x tau-scan =
    O(k_operand^2 * k_gate), INDEPENDENT of p -- with k=8/10 it is a small constant whether p=30 or p=300, replacing the prior
    positional ``cols[:5]`` slice (cheap but blind: the true operands routinely fell outside the first 5 columns of a wide frame).

    For each candidate the best tau is found by a ~17-point quantile scan over c (per-tau residue MIs batch into one kernel call),
    then the best-tau column is gated vs the HARDENED best-existing-op baseline (max MI over raw / product / ratio / diff / min /
    max on the candidate's operands) by ``_MIN_MARGIN`` AND a 12-perm null band. The null is early-rejected (computed only for a
    candidate already clearing the hardened baseline). The tau-scan + hardened gate are unchanged -- only the candidate SET shrinks.

    ``_cols_prefiltered`` is internal-only: set by callers that already built ``cols`` via ``_is_argmax_eligible`` themselves
    (e.g. ``hybrid_conditional_gate_fe_with_recipes``), to skip the redundant re-check below. External callers must leave it False."""
    import pandas as pd  # noqa: F401

    if cols is None:
        cols = [c for c in X.columns if _is_argmax_eligible(np.asarray(X[c]))]
    elif _cols_prefiltered:
        cols = [c for c in cols if c in X.columns]
    else:
        cols = [c for c in cols if c in X.columns and _is_argmax_eligible(np.asarray(X[c]))]
    cols = list(cols)
    yi = np.asarray(y).astype(np.int64)

    # FAST-SEARCH SUBSAMPLE (2026-06-14). The gate DETECTION -- raw-relevance ranking, the ~17-point
    # quantile tau-scan, and the per-tau residue-MI band -- is RANK-stable under row subsampling (the
    # tau is a quantile of the gate column; the MI ranking is monotone-preserving on a representative
    # subset). NOTE (P1-7): rank-stable is NOT threshold-stable -- the absolute accept thresholds
    # (_MIN_NULL_MARGIN / _MIN_MARGIN and the permutation null) are computed on the subsample, where MI
    # has larger O(1/n) bias+variance, so a candidate sitting right at a margin can flip accept/reject
    # vs the full-n scan. This only moves BORDERLINE gate candidates and the path ships default-OFF;
    # treat subsample_n as a speed/accuracy knob, not a no-op. The returned GateHit carries only the
    # frozen tau, and the recipe replays the gate from the FULL X at materialisation time, so the emitted
    # column is full-n regardless. When subsample_n is in (0, n) we draw a seeded subset ONCE for the
    # whole scan -> the O(n) MI kernels run on the smaller set. subsample_n <= 0 / >= n keeps full-n.
    _n_rows = len(yi)
    if isinstance(subsample_n, int) and 0 < subsample_n < _n_rows:
        _sub_rng = np.random.default_rng(int(seed))
        _sub_idx = np.sort(_sub_rng.choice(_n_rows, size=int(subsample_n), replace=False))
        yi = yi[_sub_idx]
        _X_for_scan = X.iloc[_sub_idx] if hasattr(X, "iloc") else X[_sub_idx]
    else:
        _X_for_scan = X

    # Rank eligible columns by raw relevance (one batched MI call) and prune to the top-k gate / operand pools. The gate column c
    # and the operands a, b are drawn from these pools; their union is what we materialise into ``arrs``.
    gate_cols, operand_cols = _rank_and_prune(_X_for_scan, cols, yi, nbins, int(k_gate), int(k_operand))
    cols = list(dict.fromkeys(list(operand_cols) + list(gate_cols)))  # union, operands first, dedup, order-stable
    arrs = {c: np.asarray(_X_for_scan[c], dtype=np.float64) for c in cols}
    # Cache the hardened best-existing-op baseline per operand set (it does not depend on tau / mode).
    _baseline_cache: dict[tuple[str, ...], float] = {}

    def _baseline(operands: tuple[str, ...]) -> float:
        """Memoised hardened best-existing-op MI floor for a given operand set -- computed once per distinct operand combo since it is tau/mode-independent."""
        key = tuple(sorted(operands))
        if key not in _baseline_cache:
            _baseline_cache[key] = best_existing_op_mi(arrs, key, yi, nbins)
        return _baseline_cache[key]

    # BATCHED gate-MI (2026-06-17 perf): every (gate, operand-combo) builds a (n, n_tau) residue block
    # and previously scored it with its OWN ``_mi_classif_batch`` call (686+ small calls -> per-call njit
    # launch overhead + underfilled prange). The per-column MIs are INDEPENDENT, so we accumulate combos'
    # blocks into a COLUMN-BUDGET-bounded buffer, score them in ONE batched call per chunk, then take each
    # combo's argmax over its own slice. Bit-identical (same columns -> same MI -> same best_tau/best_mi;
    # the per-best-j permutation null is unchanged). Column budget bounds peak memory at any n.
    _GATE_MI_COL_BUDGET = 512
    hits: list[GateHit] = []
    # Each pending entry carries the BUILD SPEC (operand columns + taus) so the tau-grid candidates can be built
    # DEVICE-BORN under STRICT residency (no host matrix uploaded); the host fallback materialises ``feats`` from
    # the same spec. (mode, cols_tuple, operand_arrays, taus, baseline_key) where operand_arrays is (cv, av) for
    # "mask" and (cv, av, bv) for "select".
    _pending: list = []
    _pending_cols = 0

    def _build_feats(mode, operands, taus):
        """Host tau-grid block for one combo -- the exact host candidate columns (used by the host fallback MI
        and by ``_perm_null_hi`` for the best column)."""
        cv = operands[0]
        n = cv.shape[0]
        if n >= _GATE_BUILD_NJIT_MIN_N:
            _t = np.ascontiguousarray(taus, dtype=np.float64)
            _cv = np.ascontiguousarray(cv, dtype=np.float64)
            if mode == "mask":
                return _gate_mask_grid_njit(_cv, np.ascontiguousarray(operands[1], dtype=np.float64), _t)
            return _gate_select_grid_njit(_cv, np.ascontiguousarray(operands[1], dtype=np.float64), np.ascontiguousarray(operands[2], dtype=np.float64), _t)
        feats = np.empty((n, len(taus)), dtype=np.float64)
        if mode == "mask":
            av = operands[1]
            # ``av * (cv > tau)`` (bool upcast) preserves the off-region NaN semantics of the prior
            # ``(cv > tau).astype(float) * av`` (0*NaN=NaN) while skipping the explicit astype temp; ~1.1x.
            for j, tau in enumerate(taus):
                np.multiply(av, cv > tau, out=feats[:, j])
        else:  # select
            av, bv = operands[1], operands[2]
            for j, tau in enumerate(taus):
                feats[:, j] = np.where(cv > tau, av, bv)
        return feats

    # DEVICE-BORN gate-grid (2026-06-29): under STRICT residency, build the tau-grid candidates on the device
    # from resident operand columns + score per-column MI via the resident plug-in -- the host gate-grid matrix
    # is never materialised + uploaded (collapses the dominant :311 H2D). Threads the SAME rank_binning the host
    # path uses so the binning estimator never switches. Per-column bit-identical; on any cupy failure / non-strict
    # default the host ``_gate_grid_mi`` of the materialised blocks runs (byte-identical).
    def _device_born_all_mi():
        """Score all pending tau-grid specs on the device without materialising/uploading the host gate-grid matrix; returns ``None`` (triggering the host fallback) on missing strict-residency support, disabled config, or any cupy failure."""
        try:
            from ._gpu_strict_fe import fe_gpu_device_born_gate_enabled
            if not fe_gpu_device_born_gate_enabled():
                return None
            from ._resident_candidate_mi import gate_grid_mi_resident
            _yi64 = np.ascontiguousarray(np.asarray(yi)).astype(np.int64).ravel()
            _ym = int(_yi64.min()) if _yi64.size else 0
            _nc = (int(_yi64.max()) - _ym + 1) if _yi64.size else 1
            # Carry the operand COLUMN NAMES (ctup) so the resident operand cache keys stably per column (the
            # same gate / operand column recurs across many specs -> uploaded ONCE per fit, not per spec).
            specs = [(mode, ctup, operands, taus) for (mode, ctup, operands, taus, _bkey) in _pending]
            return gate_grid_mi_resident(specs, yi, nbins, rank_binning=_gate_rank_binning(), y_min=_ym, n_classes=_nc)
        except Exception as _resident_exc:
            logger.debug("gate_grid_mi_resident batch failed (%s); caller falls back to per-spec host scoring.", _resident_exc)
            return None

    def _flush():
        """Score every pending spec's tau grid in one batched call (device-born if available, else host), pick each spec's argmax tau, gate on the margin-over-baseline, and append the resulting ``GateHit``(s); clears ``_pending`` afterward."""
        nonlocal _pending, _pending_cols
        if not _pending:
            return
        all_mi = _device_born_all_mi()
        if all_mi is None:
            big = np.ascontiguousarray(np.concatenate([_build_feats(m, o, t) for (m, _c, o, t, _b) in _pending], axis=1))
            all_mi = _gate_grid_mi(big, yi, nbins)
        off = 0
        for mode, ctup, operands, taus, bkey in _pending:
            k = len(taus)
            grid = all_mi[off:off + k]; off += k
            # SELECTION OPTIMISM (mrmr_critique EX-3, DOC): tau is chosen to MAXIMISE in-sample MI over the grid, and
            # the permutation null downstream is then computed on this already-argmax-selected best-tau column, so the
            # null understates the selection-inflated MI -- a borderline candidate on data with no genuine regime
            # structure can clear the margin/null band by chance. This is in-sample OPTIMISM (inflates apparent value),
            # NOT a serving skew: tau is frozen and replays deterministically. Consistent with the framework's other
            # in-sample-MI FE gates; a nested/held-out tau selection would be strictly more honest (FUTURE).
            best_j = int(np.argmax(grid))
            best_mi, best_tau = float(grid[best_j]), float(taus[best_j])
            baseline = _baseline(bkey)
            if (best_mi - baseline) >= _MIN_MARGIN:
                # Recompute just the best column on host for the cheap permutation null (one column, not the grid).
                # bench-attempt-rejected (2026-07-05): slicing the best column out of the already-built ``big``
                # (`np.ascontiguousarray(big[:, bstart+best_j])`) measured 0.3x (SLOWER, 0.692ms vs 0.214ms/spec @100k)
                # -- ``big`` is row-major so a single column is strided and copies cache-unfriendly, while the fused
                # njit single-tau rebuild is contiguous + parallel. The rebuild is cheaper than the slice; kept.
                best_col = _build_feats(mode, operands, np.asarray([taus[best_j]], dtype=np.float64))[:, 0]
                # The best-tau column is fit-constant across the 12-perm null: upload it ONCE and thread the
                # resident handle so it never re-crosses H2D at the :318 MI upload site (host array unchanged when
                # residency is off / non-finite / cupy fault).
                best_col_r = _resident_cand(best_col, ("gate_cand_best", mode, ctup, best_tau))
                null_hi = _perm_null_hi(best_col_r, yi, nbins, seed=seed)
            else:
                null_hi = float("inf")
            hits.append(GateHit(mode, ctup, best_tau, best_mi, baseline, null_hi))
        _pending = []; _pending_cols = 0

    def _add(mode, ctup, operands, taus, bkey):
        """Queue one (mode, columns, operand-arrays, tau-grid, baseline-key) spec and flush the batch once the accumulated tau-grid column count hits ``_GATE_MI_COL_BUDGET`` (bounds peak memory at any n)."""
        nonlocal _pending_cols
        _pending.append((mode, ctup, operands, taus, bkey))
        _pending_cols += len(taus)
        if _pending_cols >= _GATE_MI_COL_BUDGET:
            _flush()

    for cgate in gate_cols:
        cv = arrs[cgate]
        taus = np.quantile(cv, _TAU_QUANTILES)
        others = [cn for cn in operand_cols if cn != cgate]
        # mask: one active column a (cols = (a, c)); baseline over {a, c}.
        for a in others:
            av = arrs[a]
            _add("mask", (a, cgate), (cv, av), taus, (a, cgate))
        # select: ordered (a, b), cols = (a, b, c); baseline over {a, b, c}.
        for a in others:
            for b in others:
                if a == b:
                    continue
                av, bv = arrs[a], arrs[b]
                _add("select", (a, b, cgate), (cv, av, bv), taus, (a, b, cgate))
    _flush()

    # Canonical secondary key on (mode, operand names) so near-ties don't break by ranking/enumeration (column) order.
    hits.sort(key=lambda h: (-h.margin_over_baseline, str(h.mode), tuple(str(c) for c in h.cols)))
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
        out.append({"mode": h.mode, "cols": h.cols, "tau": h.tau, "feat_mi": h.feat_mi, "baseline_mi": h.baseline_mi, "margin": h.margin_over_baseline})
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

    hits = cheap_row_argmax_scan(X, y, elig, nbins=nbins, seed=seed, _cols_prefiltered=True)
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
    k_gate: int = 8, k_operand: int = 10,
    max_cols: int = 200,
    subsample_n: int = 0,
):
    """Detect responded conditional-gate structure and emit it as frozen, replayable ``EngineeredRecipe`` objects (tau frozen).

    RELEVANCE-PRUNED candidate set: the sweep cost is O(k_operand^2 * k_gate * tau-scan), FLAT in p -- the gate column c is drawn
    from the top-``k_gate`` columns by raw MI vs y and the operands a, b from the top-``k_operand`` (see ``cheap_conditional_gate_scan``).
    With k=8/10 the added cost is a small constant whether p=30 or p=300, so the prior O(p^3) blow-up that forced this OFF is gone.
    The column-count BUDGET GUARD (``max_cols``, default 200) is kept as a defense-in-depth outer cap: an absurdly wide frame still
    skips the whole sweep (logged, never silent). The detector gates vs the HARDENED best-existing-op baseline so smooth / ordinary_mul
    controls stay silent (0 FP at p=30 over 3 seeds).

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

    hits = cheap_conditional_gate_scan(
        X, y, elig, nbins=nbins, seed=seed, k_gate=k_gate, k_operand=k_operand, subsample_n=subsample_n,
        _cols_prefiltered=True,
    )
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
