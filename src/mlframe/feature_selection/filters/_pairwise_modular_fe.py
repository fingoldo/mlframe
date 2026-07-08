"""Pairwise / n-way MODULAR relationship detection FE (wired into MRMR, default ON behind ``fe_pairwise_modular_enable``).

Extends the single-column ``_periodic_fe`` (``x mod period`` on a fixed calendar ladder) to the
genuinely-uncovered case: a target that is a function of an INTEGER MODULUS of a COMBINATION of
columns -- ``y = (a + b) mod m``, ``y = (a * b) mod m``, n-way parity ``y = (sum_i x_i) mod 2`` --
or a single column with a HIDDEN integer period not in the calendar ladder. Smooth bases (poly /
Fourier) need unboundedly many harmonics to fit a sawtooth residue, so they never clear the MI
floor; the exact residue ``c mod m`` recovers the signal in one column.

Design: CHEAP-FIRST / ESCALATE.

* CHEAP stage (``cheap_modular_scan``): for a small set of integer COMBINERS of the candidate
  columns (``a``, ``a+b``, ``a-b``, ``a*b``) and a COARSE modulus grid, compute binned MI of
  ``combiner mod k`` vs ``y``. A real modulus spikes MI at the true ``m`` (and its multiples);
  smooth / noise combiners stay flat. Cost is O(n * #combiners * #moduli), all integer arithmetic
  plus the existing binned-MI kernel.

* GATE (``_responded``): the best ``combiner mod k`` MI must beat BOTH a smooth-basis baseline MI
  (the raw combiner's own MI, which is what a poly/Fourier leg can at best recover) by a measured
  margin AND a permutation-null upper band (so noise never escalates).

* ESCALATE stage (``escalate_modulus``): only when the cheap stage responded, refine the modulus on
  a FINE integer grid around the coarse winner and emit the materialised ``combiner mod m`` column
  for the selector to pick. Pure arithmetic on X, no y at replay time -> leak-free.

Only integer-typed (or exactly-integer-valued float) columns are eligible -- modular structure is an
integer-lattice property; quantising a continuous column would manufacture spurious periodicity.

Cost: ~99% of the scan is the shipped binned-MI kernel (``_mi`` -> ``_mi_classif_batch`` -> ``plugin_mi_classif_batch_dispatch``).
Two caller-side optimisations cut it 2.3-3.4x (bench ``_benchmarks/bench_pairwise_modular_cost`` before/after, both bit-identical
to the prior responded-set -- verified across TP + control frames):

* BATCHED GRID (``_residue_grid_mi``): the per-modulus residue MIs (the dominant ~57% of calls) stack into ONE multi-column
  ``_mi_classif_batch`` call per effective-nbins group instead of one call per modulus. ``_mi_classif_batch`` already takes a
  ``(n, p)`` batch (per-column binning is independent), so this is pure caller batching -- the kernel API is untouched and the
  per-column MI is byte-for-byte the same; only the per-call dispatch overhead is amortised.
* EARLY-REJECT NULL: the 12-permutation null (~40% of calls) is computed ONLY for combiners that already clear the baseline
  margin (``_MIN_MARGIN``). ``_responded`` needs BOTH margin AND null, so a margin-failing combiner can never respond -> its
  null is skipped (stored as +inf). Bit-identical to the prior responded-set by short-circuit, not an approximation; on a
  non-modular frame this drops the null for all 114 combiners.

A shared per-(n, class-counts, k) null (one null reused across combiners) was REJECTED: the null band depends on the residue
bin-count vector, which differs ~5e-4 in mean / ~3e-3 per-perm between combiners at the same k -- selection-altering, so it
could flip a borderline ``residue_mi > null_hi`` decision. Kept the per-combiner null for the (few) combiners that reach it.

Generalisation audit (does this batching lever apply to OTHER FE families?): NULL RESULT. Every other MI-scoring FE family on the
default path already scores its engineered columns with ONE ``_mi_classif_batch(full_matrix, y)`` call -- the orthogonal families
(univariate / triplet / quadruplet / cluster / diff / routing / cmim / jmim / total-correlation / adaptive-arity / adaptive-degree /
three-gate / pair-cross), the mi-greedy families, ``_extra_fe_families``, and the bootstrap / periodic scorers (which route through
``score_features_by_bootstrap_mi``, one batch call per replicate). The modular scan was the sole unbatched per-column MI loop on the
default path because it materialised ``c mod k`` per-modulus inside Python. The one remaining genuine per-column MI loop --
``_orth_auto_scorer_fe.select_best_scorer_per_column`` (per column x per bootstrap x per scorer) -- is gated behind
``fe_hybrid_orth_auto_scorer_enable`` (default OFF) and interleaves plug-in MI with four per-column-native scorers (KSG / copula /
dCor / HSIC), so batching just the plug-in leg would fork the uniform scorer dispatch for an opt-in path with no default-path win.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _is_cupy_ndarray(x) -> bool:
    """True iff ``x`` is a resident cupy ndarray -- WITHOUT hard-importing cupy (no import when it is a host array).

    Used by ``_mi`` (and the conditional-gate / row-argmax scorers that thread a resident candidate handle) to
    take the resident-input branch instead of ``np.asarray`` (which raises on a cupy array). Returns False for
    any host array / when cupy is absent -> the exact host path runs."""
    if type(x).__module__.split(".", 1)[0] != "cupy":
        return False
    try:
        import cupy as cp
        return isinstance(x, cp.ndarray)
    except Exception:
        return False


__all__ = [
    "ModularHit",
    "COARSE_MODULI",
    "cheap_modular_scan",
    "escalate_modulus",
    "detect_pairwise_modular",
    "PAIRWISE_MODULAR_PREFIX",
    "engineered_name_pairwise_modular",
    "apply_pairwise_modular",
    "build_pairwise_modular_recipe",
    "hybrid_pairwise_modular_fe_with_recipes",
]

PAIRWISE_MODULAR_PREFIX = "pmod"

# Coarse modulus ladder for the cheap stage: small primes (so a hidden PRIME period spikes -- a
# prime m has no proper divisor in a composite-only grid, so it must be listed) + common composite /
# byte cycles. A composite true modulus that is a MULTIPLE of a grid entry still spikes (its divisors
# carry the signal); the escalate stage pins the exact m on a fine grid afterwards.
COARSE_MODULI: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 23, 24, 32)

# Combiner op codes: how a pair (a, b) is folded into one integer column before taking the residue.
# "self" uses column a alone (single-column hidden period); the rest are the pair combiners.
_PAIR_OPS = ("sum", "diff", "prod")


# Gate floor shared by ``_responded`` and the cheap-scan early-reject: a combiner whose residue MI does not beat its own
# raw-combiner baseline by this margin can NEVER respond (``_responded`` requires margin AND null), so the expensive
# permutation null is skipped for it -- bit-identical to ``responded`` by short-circuit, not an approximation.
_MIN_MARGIN = 0.02

# Absolute floor the residue MI must clear ABOVE the permutation-null upper band (not just `> null_hi`). The plug-in MI of a high-cardinality
# residue (large coarse modulus m -> ~m residue bins) is cardinality-inflated when y has few classes (a ~10-bin regression/quantized y); the
# z=3 null band then sits just below the inflated residue MI, so a smooth/noise combiner can squeak over by ~0.007 nats of pure chance. A true
# modular hit clears the null by >1 nat, so this absolute gap floor kills the over-fragmentation false positive while leaving real hits untouched.
_MIN_NULL_MARGIN = 0.05


def _mi(col: np.ndarray, y: np.ndarray, nbins: int = 12) -> float:
    """Binned MI of one column vs y, via the shipped classif-MI batch kernel.

    Routes the gate/operator-edge MI through the RANK resident binner under ``MLFRAME_FE_GPU_STRICT_RESIDENT``
    (default OFF -> byte-for-byte the prior path) so the STRICT gate MI byte-matches the CPU njit rank MI on
    tied operator outputs (the edge resident path would lower MI on the ~50%-tied gate_mask).

    RESIDENT-INPUT branch: a device-born caller may hand an ALREADY-RESIDENT cupy candidate column (e.g. the
    conditional-gate / row-argmax scorer, which uploads the scored candidate ONCE and reuses the resident handle
    across its marginal MI + the 12-perm null). Pass it straight to ``_mi_classif_batch`` -- which accepts a
    resident cupy input (isinstance branch, no upload) -- so the candidate float never re-crosses H2D at the
    :318 MI upload site. ``np.asarray`` on a cupy array would raise, so the resident branch MUST bypass it. Host
    columns take the exact prior ``np.asarray`` upload path -> byte-identical default."""
    from ._orthogonal_univariate_fe import _mi_classif_batch

    try:
        from ._gpu_strict_fe import fe_gpu_strict_resident_enabled
        _rb = fe_gpu_strict_resident_enabled()
    except Exception:
        _rb = False
    if _is_cupy_ndarray(col):
        arr = col  # resident (n,) or (n,1) cupy -> _mi_classif_batch reshapes 1-D to a column itself
    else:
        arr = np.asarray(col, dtype=np.float64).reshape(-1, 1)
    return float(_mi_classif_batch(arr, np.asarray(y).astype(np.int64), nbins=nbins, rank_binning=_rb)[0])


def _residue_grid_mi(c: np.ndarray, y: np.ndarray, mods: Sequence[int], nbins: int) -> np.ndarray:
    """MI of ``c mod k`` vs y for every k in ``mods``, in one batched kernel call per effective-nbins group.

    ``_residue_mi`` bins each residue at ``max(nbins, k)``; columns sharing the same effective nbins can be stacked into
    a single multi-column ``_mi_classif_batch`` call (bit-identical to per-column -- same per-column binning math, only
    the per-call dispatch overhead is amortised). Moduli are grouped by ``max(nbins, k)`` so each group is one call."""
    from ._orthogonal_univariate_fe import _mi_classif_batch

    yi = np.asarray(y).astype(np.int64)
    # SF2 :311 collapse (2026-06-30): the per-group residue matrices are the DOMINANT modular :311 H2D (one
    # (n, |group|) cp.asarray per eff-nbins group at _orth_mi_backends:139). Build ``c mod k`` DEVICE-BORN from a
    # single resident combiner column (exact integer cupy ``%``) and score each group through the SAME
    # percentile-EDGE resident estimator the host grid uses (rank_binning stays False on the grid). None on cupy
    # failure / non-strict -> the EXACT host grid below (byte-identical default path untouched).
    from ._pairwise_modular_resident import residue_grid_mis_resident

    res = residue_grid_mis_resident(c, yi, mods, nbins=nbins)
    if res is not None:
        return res
    out = np.empty(len(mods), dtype=np.float64)
    groups: dict[int, list[int]] = {}
    for idx, k in enumerate(mods):
        groups.setdefault(max(nbins, int(k)), []).append(idx)
    for eff_nbins, idxs in groups.items():
        # Build the residue matrix as (|group|, n) with CONTIGUOUS row writes (each ``np.mod`` lands in a
        # contiguous row, no strided column copy), then pass the ``.T`` F-view to the batched MI kernel --
        # bit-identical to the (n, |group|) column-build (same float64 residues, same per-column binning) but
        # avoids the strided writes + the result->column double-copy (1.19x on the host residue-grid path).
        matT = np.empty((len(idxs), c.shape[0]), dtype=np.float64)
        for j, idx in enumerate(idxs):
            matT[j] = np.mod(c, mods[idx])
        mis = _mi_classif_batch(matT.T, yi, nbins=eff_nbins)
        for j, idx in enumerate(idxs):
            out[idx] = float(mis[j])
    return out


def _is_integer_col(x: np.ndarray) -> bool:
    """True iff x is integer-typed OR exactly-integer-valued float (no fractional part), finite."""
    a = np.asarray(x)
    if np.issubdtype(a.dtype, np.integer):
        return True
    if not np.issubdtype(a.dtype, np.floating):
        return False
    finite = a[np.isfinite(a)]
    return bool(finite.size > 0 and np.all(finite == np.floor(finite)))


def _combine(arrs: Sequence[np.ndarray], op: str) -> np.ndarray:
    """Fold one to three integer columns into a single combiner column."""
    ai = np.asarray(arrs[0], dtype=np.int64)
    if op == "self":
        return ai
    bi = np.asarray(arrs[1], dtype=np.int64)
    if op == "sum":
        return ai + bi
    if op == "diff":
        return ai - bi
    if op == "prod":
        return ai * bi
    if op == "sum3":
        return ai + bi + np.asarray(arrs[2], dtype=np.int64)
    raise ValueError(f"unknown combiner op {op!r}")


def _residue_mi(c: np.ndarray, y: np.ndarray, k: int, nbins: int) -> float:
    """MI of ``c mod k`` vs y. For k <= nbins the residue is used as its own bin index (no rebin)."""
    eff = max(nbins, k)
    # SF2 :311 collapse (2026-06-30): the escalate-stage residue single-column upload is built DEVICE-BORN from
    # the resident combiner column (exact integer ``c % k``) + scored through the SAME rank resident estimator
    # host ``_mi`` uses. None on cupy failure / non-strict -> the EXACT host ``_mi`` below (byte-identical).
    try:
        from ._gpu_strict_fe import fe_gpu_strict_resident_enabled
        _rb = fe_gpu_strict_resident_enabled()
    except Exception:
        _rb = False
    from ._pairwise_modular_resident import combiner_mi_resident

    mi = combiner_mi_resident(c, y, nbins=eff, rank_binning=_rb, modulus=int(k))
    if mi is not None:
        return mi
    r = np.mod(c, k).astype(np.float64)
    return _mi(r, y, nbins=eff)


@dataclass(frozen=True)
class ModularHit:
    """One cheap-scan candidate: a combiner over given columns, its best coarse modulus + MI."""

    op: str
    cols: tuple[str, ...]
    modulus: int
    residue_mi: float
    baseline_mi: float  # raw-combiner MI = the best a smooth basis could recover
    null_hi: float  # permutation-null upper band on residue MI

    @property
    def margin_over_baseline(self) -> float:
        return self.residue_mi - self.baseline_mi

    @property
    def responded(self) -> bool:
        return _responded(self.residue_mi, self.baseline_mi, self.null_hi)


def _responded(residue_mi: float, baseline_mi: float, null_hi: float, min_margin: float = _MIN_MARGIN, null_margin: float = _MIN_NULL_MARGIN) -> bool:
    """Gate: the residue MI must clear BOTH the smooth-basis baseline (by ``min_margin``) AND the permutation-null upper band by an absolute
    ``null_margin`` (not just ``> null_hi`` -- a high-cardinality residue's cardinality-inflated plug-in MI can sit ~0.007 nats above a z=3
    null on noise; a true hit clears it by >1 nat). ``min_margin`` is the measured separation floor between a true modular hit and the best
    non-modular combiner (see ``_benchmarks/bench_modular_period_detection``)."""
    return (residue_mi - baseline_mi) >= min_margin and residue_mi > (null_hi + null_margin)


def _perm_null_hi(c: np.ndarray, y: np.ndarray, k: int, nbins: int, n_perm: int = 12, seed: int = 0, z: float = 3.0) -> float:
    """Upper band (mean + z*std) of ``c mod k`` MI under y permutation -- the noise reference the
    residue MI must clear. Cheap: n_perm small, the residue is computed once and only y is shuffled."""
    r = np.mod(c, k).astype(np.float64)
    rng = np.random.default_rng(seed)
    yi = np.asarray(y).astype(np.int64)
    eff_nbins = max(nbins, k)
    # SF2 :311 collapse (2026-06-30): the per-perm residue MIs are the dominant single-column residue uploads at
    # _orth_mi_backends:311 (one (n,1) cp.asarray per perm). MI(r; y[perm]) == MI(r[inv_perm]; y) (joint-reindex
    # invariance), so the 12 per-perm MIs are the per-column MIs of a single (n, n_perm) matrix -- scored in ONE
    # resident plug-in call against the resident y. SAME seeded permutations (bit-identical perms drawn HERE in
    # the exact loop order) + SAME (rank, under STRICT) resident estimator the host _mi uses -> byte-identical
    # per-perm MIs -> byte-identical null band. None on cupy failure / non-strict -> the EXACT host loop below.
    perms = [rng.permutation(yi.size) for _ in range(n_perm)]
    try:
        from ._gpu_strict_fe import fe_gpu_strict_resident_enabled
        _rb = fe_gpu_strict_resident_enabled()
    except Exception:
        _rb = False
    from ._pairwise_modular_resident import perm_null_residue_mis_resident

    vals = perm_null_residue_mis_resident(r, yi, perms, eff_nbins=eff_nbins, rank_binning=_rb)
    if vals is None:
        vals = np.empty(n_perm, dtype=np.float64)
        for i in range(n_perm):
            vals[i] = _mi(r, yi[perms[i]], nbins=eff_nbins)
    vals = np.asarray(vals, dtype=np.float64)
    return float(vals.mean() + z * vals.std())


def cheap_modular_scan(
    X,
    y: np.ndarray,
    cols: Optional[Sequence[str]] = None,
    *,
    moduli: Sequence[int] = COARSE_MODULI,
    max_pairs: int = 24,
    max_triples: int = 12,
    nbins: int = 12,
    seed: int = 0,
) -> list[ModularHit]:
    """Cheap first-pass scan for modular structure over integer columns of X.

    For each integer column (``self`` op), each integer pair (``sum`` / ``diff`` / ``prod``) and a
    budgeted set of integer TRIPLES (``sum3`` -- the n-way parity combiner ``(a+b+c) mod m``, which no
    pair can reach), take the residue at every coarse modulus, score MI vs y, and keep the best-modulus
    hit per combiner. The triple sweep is what makes 3-way parity ``y = (x0+x1+x2) mod 2`` detectable;
    pairwise parity of a 3-way target is independent of y, so the pair combiners stay at the null floor.
    Returns hits sorted by ``margin_over_baseline`` descending. A hit's ``.responded`` flag applies the
    measured gate. The permutation null is computed ONLY for the per-combiner best modulus (keeps the
    null cost off the inner grid loop)."""
    import pandas as pd

    if cols is None:
        cols = [c for c in X.columns if _is_integer_col(np.asarray(X[c]))]
    else:
        cols = [c for c in cols if _is_integer_col(np.asarray(X[c]))]
    # Canonicalize eligible-column order so the budgeted pair/triple enumeration scans the same combinations regardless of caller column order (reversed-column invariance).
    cols = sorted(cols, key=lambda c: str(c))
    yi = np.asarray(y).astype(np.int64)
    arrs = {c: np.asarray(X[c]) for c in cols}
    mods = [int(k) for k in moduli if int(k) >= 2]

    hits: list[ModularHit] = []

    try:
        from ._gpu_strict_fe import fe_gpu_strict_resident_enabled
        _scan_rb = fe_gpu_strict_resident_enabled()
    except Exception:
        _scan_rb = False
    from ._pairwise_modular_resident import combiner_mi_resident

    def _scan_one(op: str, cset: tuple[str, ...], c_arr: np.ndarray):
        # SF2 :311 collapse (2026-06-30): the per-combiner baseline single-column upload (one (n,1) cp.asarray per
        # combiner at _orth_mi_backends:311) is built DEVICE-BORN from the resident combiner column + scored
        # through the SAME rank resident estimator host ``_mi`` uses. None -> the EXACT host ``_mi`` below.
        base = combiner_mi_resident(c_arr, yi, nbins=nbins, rank_binning=_scan_rb, modulus=0)
        if base is None:
            base = _mi(c_arr.astype(np.float64), yi, nbins=nbins)
        grid = _residue_grid_mi(c_arr, yi, mods, nbins)
        best_i = int(np.argmax(grid))
        best_k, best_mi = mods[best_i], float(grid[best_i])
        # Early-reject: the permutation null only matters for combiners that already clear the baseline margin
        # (``_responded`` needs BOTH). A combiner below margin can never respond -> skip its 12-perm null. The null_hi
        # we store is +inf for those, which keeps ``responded`` False and is bit-identical to the full computation.
        if (best_mi - base) >= _MIN_MARGIN:
            null_hi = _perm_null_hi(c_arr, yi, best_k, nbins, seed=seed)
        else:
            null_hi = float("inf")
        hits.append(ModularHit(op, cset, best_k, best_mi, base, null_hi))

    for c in cols:
        _scan_one("self", (c,), _combine([arrs[c]], "self"))

    pair_budget = max_pairs
    for a, b in combinations(cols, 2):
        if pair_budget <= 0:
            break
        for op in _PAIR_OPS:
            _scan_one(op, (a, b), _combine([arrs[a], arrs[b]], op))
        pair_budget -= 1

    triple_budget = max_triples
    for a, b, c in combinations(cols, 3):
        if triple_budget <= 0:
            break
        _scan_one("sum3", (a, b, c), arrs[a].astype(np.int64) + arrs[b].astype(np.int64) + arrs[c].astype(np.int64))
        triple_budget -= 1

    # Canonical secondary key on (op, operand names) so near-ties don't break by enumeration (column) order.
    hits.sort(key=lambda h: (-h.margin_over_baseline, str(h.op), tuple(str(c) for c in h.cols)))
    return hits


def escalate_modulus(
    X,
    y: np.ndarray,
    hit: ModularHit,
    *,
    nbins: int = 12,
    span: int = 6,
) -> tuple[int, float, np.ndarray]:
    """Refine the modulus for a responded hit on a FINE integer grid around the coarse winner.

    Searches ``[max(2, k0 - span), k0 + span]`` plus the small-multiple ladder ``{2*k0, 3*k0}`` (a
    coarse divisor often spikes at a multiple of the true modulus). Returns ``(best_m, best_mi,
    residue_column)``; the residue column is the materialised ``combiner mod best_m`` for the selector."""
    k0 = hit.modulus
    arrs = [np.asarray(X[c]) for c in hit.cols]
    c_arr = _combine(arrs, hit.op)
    yi = np.asarray(y).astype(np.int64)

    grid = set(range(max(2, k0 - span), k0 + span + 1))
    grid.update({2 * k0, 3 * k0})
    best_m, best_mi = k0, -1.0
    for m in sorted(grid):
        mi = _residue_mi(c_arr, yi, m, nbins)
        if mi > best_mi:
            best_mi, best_m = mi, m
    residue = np.mod(c_arr, best_m).astype(np.float64)
    return best_m, best_mi, residue


def detect_pairwise_modular(
    X,
    y: np.ndarray,
    cols: Optional[Sequence[str]] = None,
    *,
    moduli: Sequence[int] = COARSE_MODULI,
    top_k: int = 4,
    nbins: int = 12,
    seed: int = 0,
):
    """End-to-end cheap-first + escalate. Returns the list of responded+refined hits (each a dict with
    the final modulus, MI, and the materialised residue column), capped at ``top_k``. Empty when
    nothing responds -- a non-modular frame escalates nothing."""
    hits = cheap_modular_scan(X, y, cols, moduli=moduli, nbins=nbins, seed=seed)
    out = []
    for h in hits:
        if not h.responded:
            continue
        best_m, best_mi, residue = escalate_modulus(X, y, h, nbins=nbins)
        out.append(
            {
                "op": h.op, "cols": h.cols, "coarse_modulus": h.modulus,
                "modulus": best_m, "residue_mi": best_mi, "baseline_mi": h.baseline_mi,
                "margin": best_mi - h.baseline_mi, "residue": residue,
            }
        )
        if len(out) >= top_k:
            break
    return out


# Recipe plumbing: emit frozen EngineeredRecipe objects so the MRMR selector materialises, scores, selects, and replays
# the residue column at predict time identically. Replay is pure integer arithmetic on X (combine columns, take mod) --
# no y reference, so it is leak-free + deterministic + train/test exact, mirroring the single-column ``_periodic_fe`` path.

_RECIPE_VALID_OPS = ("self", "sum", "diff", "prod", "sum3")


def engineered_name_pairwise_modular(op: str, cols: Sequence[str], modulus: int) -> str:
    """Canonical engineered column name for one pairwise/n-way modular residue, e.g. ``pmod_sum__a__b__m7``.

    The op + source columns + modulus fully determine the residue, so the name round-trips through ``_parse`` back to a
    recipe. Source columns are joined with ``__`` (the codebase's source-link convention) and the modulus suffixed ``m{k}``."""
    if op not in _RECIPE_VALID_OPS:
        raise ValueError(f"pairwise-modular op must be one of {_RECIPE_VALID_OPS}; got {op!r}")
    joined = "__".join(str(c) for c in cols)
    return f"{PAIRWISE_MODULAR_PREFIX}_{op}__{joined}__m{int(modulus)}"


def apply_pairwise_modular(X, op: str, cols: Sequence[str], modulus: int) -> np.ndarray:
    """Replay one pairwise/n-way modular residue: combine the source columns via ``op`` then take ``mod modulus``.

    Pure integer arithmetic on the source columns -- no y reference, no fitted state, so transform() is leak-free and
    train/test exact. Output float64 in ``[0, modulus)``; the residue is the materialised selector feature."""
    if op not in _RECIPE_VALID_OPS:
        raise ValueError(f"pairwise-modular op must be one of {_RECIPE_VALID_OPS}; got {op!r}")
    m = int(modulus)
    if m < 2:
        raise ValueError(f"pairwise-modular modulus must be >= 2; got {m!r}")
    arrs = [np.asarray(X[c]) for c in cols]
    # The fit-time eligibility scan guarantees finite integer-valued operands, but a drifted/test frame may
    # carry NaN/inf in a source column that was integer-valued on train. Casting those to int64 in _combine
    # silently yields INT64_MIN and np.mod turns it into a wrong, non-NaN residue in [0, m) -- garbage that
    # downstream cannot tell from a genuine residue. NaN-out those rows: the residue is undefined where an
    # operand is not a finite integer. On clean data the mask is all-False (byte-identical to before).
    bad = np.zeros(arrs[0].shape[0], dtype=bool)
    for arr in arrs:
        bad |= ~np.isfinite(np.asarray(arr, dtype=np.float64))
    with np.errstate(invalid="ignore"):  # non-finite rows cast to INT64_MIN then get masked to NaN below
        c_arr = _combine(arrs, op)
        out = np.mod(c_arr, m).astype(np.float64)
    if bad.any():
        out[bad] = np.nan
    return np.asarray(out)


def build_pairwise_modular_recipe(*, name: str, op: str, cols: Sequence[str], modulus: int):
    """Frozen recipe for one pairwise/n-way modular residue. Replay is pure integer arithmetic on the source columns."""
    from .engineered_recipes import EngineeredRecipe

    if op not in _RECIPE_VALID_OPS:
        raise ValueError(f"pairwise-modular op must be one of {_RECIPE_VALID_OPS}; got {op!r}")
    m = int(modulus)
    if m < 2:
        raise ValueError(f"pairwise-modular modulus must be >= 2; got {m!r}")
    return EngineeredRecipe(
        name=name,
        kind="pairwise_modular",
        src_names=tuple(str(c) for c in cols),
        extra={"op": str(op), "modulus": m},
    )


def hybrid_pairwise_modular_fe_with_recipes(
    X,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    moduli: Sequence[int] = COARSE_MODULI,
    top_k: int = 4,
    nbins: int = 12,
    seed: int = 0,
    max_int_cols: int = 30,
    max_triple_cols: int = 20,
):
    """Detect responded pairwise/n-way modular structure and emit it as frozen, replayable ``EngineeredRecipe`` objects.

    Column-count BUDGET GUARD (the wide-frame cost is O(p) self-scan + budgeted pairs/triples; see
    ``_benchmarks/bench_pairwise_modular_cost``): when the number of integer-eligible columns exceeds ``max_int_cols`` the
    whole sweep is SKIPPED (logged, never silent); when it is within ``max_int_cols`` but exceeds the tighter
    ``max_triple_cols`` the expensive C(p,3) triple sweep is dropped (pairs-only, logged).

    Returns ``(appended_names, recipes)`` -- the materialised columns are NOT concatenated here (the MRMR caller appends
    them under its own RAM-safe path); each recipe replays the exact residue column from X alone, leak-free."""
    if cols is None:
        int_cols = [c for c in X.columns if _is_integer_col(np.asarray(X[c]))]
    else:
        int_cols = [c for c in cols if c in X.columns and _is_integer_col(np.asarray(X[c]))]

    p = len(int_cols)
    if p > int(max_int_cols):
        logger.info(
            "pairwise_modular FE: %d integer-eligible columns exceeds max_int_cols=%d; skipping the "
            "pairwise/n-way modular sweep for budget.", p, int(max_int_cols),
        )
        return [], []

    max_triples = 12
    if p > int(max_triple_cols):
        logger.info(
            "pairwise_modular FE: %d integer-eligible columns exceeds max_triple_cols=%d; limiting to "
            "PAIRS-ONLY (dropping the C(p,3) triple sweep) for budget.", p, int(max_triple_cols),
        )
        max_triples = 0

    hits = cheap_modular_scan(
        X, y, int_cols, moduli=moduli, nbins=nbins, seed=seed, max_triples=max_triples,
    )
    appended: list[str] = []
    recipes = []
    seen: set[str] = set()
    for h in hits:
        if not h.responded:
            continue
        best_m, _best_mi, _residue = escalate_modulus(X, y, h, nbins=nbins)
        name = engineered_name_pairwise_modular(h.op, h.cols, best_m)
        if name in seen:
            continue
        seen.add(name)
        appended.append(name)
        recipes.append(build_pairwise_modular_recipe(name=name, op=h.op, cols=h.cols, modulus=best_m))
        if len(appended) >= int(top_k):
            break
    return appended, recipes
