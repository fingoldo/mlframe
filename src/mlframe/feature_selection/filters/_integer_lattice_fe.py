"""Pairwise INTEGER-LATTICE relationship detection FE (wired into MRMR, default ON behind ``fe_integer_lattice_enable``).

Sibling-in-spirit to ``_pairwise_modular_fe``: where the modular operator captures a hidden PERIOD of an integer
combination (``c mod m``), these capture a hidden COMMON DIVISOR (``gcd(a, b)`` -- shared factor / grid alignment),
its dual the least-common-multiple (``lcm(a, b)``), or a bit-level co-occurrence of integer codes (``a & b`` --
shared-flag interaction). Smooth bases (poly / Fourier / RBF / spline) and the existing arithmetic ops
(mul/add/sub/div/mod) provably cannot express ``gcd``: it is number-theoretic, non-smooth, and non-monotone in either
argument (gcd(6,9)=3 but gcd(6,10)=2), so a poly/Fourier leg never clears the MI floor on a shared-factor target.

bitwise-XOR is EXCLUDED: the bench (``_benchmarks/bench_integer_lattice_fe``) measured its lift at ~0.09 -- the modular
residue operator already captures low-bit parity, so XOR is redundant on the default path (see the pinned non-edge test
``test_biz_val_xor_lowbits_is_redundant_with_modular``). bitwise-OR was likewise dropped as a near-mirror of AND with no
measured complementary edge; only the three measured-distinct operators ship.

Design mirrors the modular operator: CHEAP-FIRST scan over integer-eligible column PAIRS (no triples -- gcd/lcm/AND are
binary, there is no n-way analogue the way ``(a+b+c) mod 2`` parity is for modular), each op-pair gated by a dual
``_responded`` test -- the engineered column's MI must beat BOTH operands' raw MI by ``_MIN_MARGIN`` AND a 12-permutation
null upper band (so a non-lattice frame injects nothing). Each responded hit becomes a frozen, replayable
``EngineeredRecipe``: replay is pure integer arithmetic on X (``np.gcd`` / ``np.lcm`` / ``np.bitwise_and`` on the
integer-cast operands), no y reference -> leak-free + deterministic + train/test bit-identical.

Only integer-typed (or exactly-integer-valued float) columns are eligible (reused ``_is_integer_col`` from the modular
module) -- the lattice structure is an integer property; quantising a continuous column would manufacture spurious shared
factors.

Cost: ~99% of the scan is the shipped binned-MI kernel (``_mi`` -> ``_mi_classif_batch``). The lever that keeps it cheap
is the BATCHED grid MI (``_lattice_grid_mi``): the three op-columns for one pair stack into ONE multi-column
``_mi_classif_batch`` call instead of one call per op (bit-identical to per-column -- ``_mi_classif_batch`` already bins
each column independently, only the per-call dispatch overhead is amortised). The 12-perm null is EARLY-REJECTED: it is
computed only for op-columns that already clear the ``_MIN_MARGIN`` floor (``_responded`` needs BOTH margin AND null, so
a margin-failing op can never respond -> its null is skipped, stored as +inf -- bit-identical by short-circuit). The
sweep is pairs-only (O(C(p,2)) integer ufuncs), strictly cheaper than the modular pairs+triples sweep, and budget-guarded
on wide frames (``max_int_cols``).

cProfile (n=2000, p=24 int cols, enabled path, measured): the batched plug-in MI dispatch is the top hotspot (~37% of
wall) and the integer ufunc + cast work (``_lattice_column`` / ``_to_int`` / gcd-reduce) is the SECOND (~30%, NOT
negligible at this shape -- the MI kernel is fast here so the per-pair build is a real fraction). Applied lever: a
per-pair ``_lattice_columns_for_pair`` casts both operands to int ONCE and reuses the gcd for the lcm op (instead of
re-casting + recomputing gcd once per op), bit-identical (same arithmetic); on the profiled shape it cut the per-pair
build ~2x and the full enabled scan ~15-20%. Remaining hotspot is the shipped, already-tuned MI plug-in path (also
batched per-pair into one call) -- no further actionable caller-side speedup. See ``bench_integer_lattice_wideframe``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Sequence

import numpy as np

from ._pairwise_modular_fe import _is_integer_col, _mi

logger = logging.getLogger(__name__)

__all__ = [
    "LatticeHit",
    "INTEGER_LATTICE_OPS",
    "cheap_integer_lattice_scan",
    "detect_integer_lattice",
    "INTEGER_LATTICE_PREFIX",
    "engineered_name_integer_lattice",
    "apply_integer_lattice",
    "build_integer_lattice_recipe",
    "hybrid_integer_lattice_fe_with_recipes",
]

INTEGER_LATTICE_PREFIX = "il"

# The measured-distinct operators (bench_integer_lattice_fe): gcd is the genuine 6.97x edge, lcm is its number-theoretic
# dual, bitwise_and is the marginal 1.23x flag-co-occurrence edge. XOR (~0.09, redundant with modular residue) and OR
# (mirror of AND, no measured complementary edge) are excluded from the default sweep.
INTEGER_LATTICE_OPS: tuple[str, ...] = ("gcd", "lcm", "bitwise_and")

# Margin a lattice column must beat each operand's own MI by (mirrors _pairwise_modular_fe._MIN_MARGIN). A column whose
# MI does not exceed the better operand by this much carries no structure the selector cannot already get from the raw column.
_MIN_MARGIN = 0.02

# Absolute floor the engineered MI must clear ABOVE the permutation-null band (not just `> null_hi`); mirrors _pairwise_modular_fe._MIN_NULL_MARGIN.
# A high-cardinality op (lcm of two mid-range ints, bitwise on wide operands) has cardinality-inflated plug-in MI when y has few classes
# (a ~10-bin regression/quantized y), so the z=3 null sits just below it and a noise pair can squeak over by ~0.01 nats; a true gcd/lcm/AND
# hit clears the null by a wide margin, so this absolute gap floor kills the over-fragmentation false positive while leaving real hits untouched.
_MIN_NULL_MARGIN = 0.05


def _to_int(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest int64 view of an exactly-integer-valued column (eligibility already checked by caller).

    A non-finite entry (only possible on a drifted REPLAY frame -- fit data is finite by eligibility) casts
    to INT64_MIN; the caller masks those rows to NaN afterward, so the intermediate value is discarded. The
    errstate silences the harmless 'invalid value in cast' warning that would otherwise fire on that row."""
    with np.errstate(invalid="ignore"):
        return np.rint(np.asarray(x, dtype=np.float64)).astype(np.int64)


def _nonfinite_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rows where either operand is NOT a finite number. The fit-time eligibility scan guarantees finite
    integer-valued operands, so on training data this is all-False (the lattice computation is then
    byte-identical to the un-masked form). At REPLAY a drifted/test frame may carry NaN/inf in a source
    column that was integer-valued on train -- casting those to int64 silently yields INT64_MIN and feeds
    garbage (a wrong, non-NaN value) into gcd/lcm/and. We instead NaN-out those rows: the lattice feature
    is undefined where an operand is not a finite integer, and a NaN is the honest, downstream-droppable
    signal for that, never INT64_MIN garbage."""
    return ~(np.isfinite(np.asarray(a, dtype=np.float64)) & np.isfinite(np.asarray(b, dtype=np.float64)))


def _lattice_column(a: np.ndarray, b: np.ndarray, op: str) -> np.ndarray:
    """One integer-lattice column from a pair. Pure function of (a, b) -- leak-free at replay (no y)."""
    bad = _nonfinite_mask(a, b)
    ai, bi = _to_int(a), _to_int(b)
    if op == "gcd":
        out = np.gcd(ai, bi).astype(np.float64)
    elif op == "lcm":
        # lcm overflows int64 for large coprime pairs; compute |a|*|b|/gcd in float to keep it finite + monotone.
        g = np.gcd(ai, bi)
        safe_g = np.where(g == 0, 1, g)
        out = (np.abs(ai.astype(np.float64)) * np.abs(bi.astype(np.float64))) / safe_g
    elif op == "bitwise_and":
        out = np.bitwise_and(ai, bi).astype(np.float64)
    else:
        raise ValueError(f"integer-lattice op must be one of {INTEGER_LATTICE_OPS}; got {op!r}")
    if bad.any():
        out[bad] = np.nan
    return out


def _lattice_columns_for_pair(a: np.ndarray, b: np.ndarray, ops: Sequence[str]) -> np.ndarray:
    """All op-columns for one pair, casting both operands to int ONCE and sharing gcd between the gcd/lcm ops.

    Bit-identical to per-op ``_lattice_column`` (same arithmetic) but the int-cast (per-pair, not per-op) and the gcd
    (reused by lcm) are computed once instead of once per op -- the hot-loop lever the per-column entry cannot exploit."""
    bad = _nonfinite_mask(a, b)  # all-False on the finite-integer fit data -> byte-identical there
    ai, bi = _to_int(a), _to_int(b)
    g = np.gcd(ai, bi) if ("gcd" in ops or "lcm" in ops) else None
    mat = np.empty((a.shape[0], len(ops)), dtype=np.float64)
    for j, op in enumerate(ops):
        if op == "gcd":
            mat[:, j] = g.astype(np.float64)
        elif op == "lcm":
            # lcm overflows int64 for large coprime pairs; compute |a|*|b|/gcd in float to keep it finite + monotone.
            safe_g = np.where(g == 0, 1, g)
            mat[:, j] = (np.abs(ai.astype(np.float64)) * np.abs(bi.astype(np.float64))) / safe_g
        elif op == "bitwise_and":
            mat[:, j] = np.bitwise_and(ai, bi).astype(np.float64)
        else:
            raise ValueError(f"integer-lattice op must be one of {INTEGER_LATTICE_OPS}; got {op!r}")
    if bad.any():
        mat[bad, :] = np.nan  # operand not a finite integer at replay -> column undefined, never INT64_MIN garbage
    return mat


def _lattice_grid_mi(a: np.ndarray, b: np.ndarray, y: np.ndarray, ops: Sequence[str], nbins: int) -> np.ndarray:
    """MI of every op-column for one pair vs y, in one batched kernel call.

    The op-columns share the same nbins, so they stack into a single multi-column ``_mi_classif_batch`` call
    (bit-identical to per-column -- same per-column binning math, only the per-call dispatch overhead is amortised)."""
    from ._orthogonal_univariate_fe import _mi_classif_batch

    yi = np.asarray(y).astype(np.int64)
    mat = _lattice_columns_for_pair(a, b, ops)
    return np.asarray(_mi_classif_batch(mat, yi, nbins=nbins), dtype=np.float64)


def _perm_null_hi(feat: np.ndarray, y: np.ndarray, nbins: int,
                  n_perm: int = 12, seed: int = 0, z: float = 3.0) -> float:
    """Upper band (mean + z*std) of the pre-computed feature's MI under y permutation -- the noise reference the
    feature MI must clear. The feature is fixed; only y is shuffled (cheap, n_perm small)."""
    rng = np.random.default_rng(seed)
    yi = np.asarray(y).astype(np.int64)
    vals = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        vals[i] = _mi(feat, yi[rng.permutation(yi.size)], nbins=nbins)
    return float(vals.mean() + z * vals.std())


def _responded(feat_mi: float, operand_floor: float, null_hi: float,
               min_margin: float = _MIN_MARGIN, null_margin: float = _MIN_NULL_MARGIN) -> bool:
    """Gate: the engineered column's MI must clear BOTH the better operand's raw MI (by ``min_margin``) AND the permutation-null upper band by
    an absolute ``null_margin`` (not just ``> null_hi`` -- guards the cardinality-inflation false positive on a few-class y; see ``_MIN_NULL_MARGIN``).
    Mirrors ``_pairwise_modular_fe._responded`` (operand_floor plays the baseline role)."""
    return (feat_mi - operand_floor) >= min_margin and feat_mi > (null_hi + null_margin)


@dataclass(frozen=True)
class LatticeHit:
    """One cheap-scan candidate: an op over a column pair, its engineered-column MI, the operand floor + null band."""

    op: str
    cols: tuple[str, str]
    feat_mi: float
    operand_floor: float  # max raw MI of the two operands = the best a non-lattice op could already recover
    null_hi: float        # permutation-null upper band on the engineered-column MI

    @property
    def margin_over_operands(self) -> float:
        return self.feat_mi - self.operand_floor

    @property
    def responded(self) -> bool:
        return _responded(self.feat_mi, self.operand_floor, self.null_hi)


def cheap_integer_lattice_scan(
    X,
    y: np.ndarray,
    cols: Optional[Sequence[str]] = None,
    *,
    ops: Sequence[str] = INTEGER_LATTICE_OPS,
    nbins: int = 12,
    seed: int = 0,
) -> list[LatticeHit]:
    """Cheap first-pass scan for integer-lattice structure over integer pairs of X.

    For each integer pair and each op (gcd / lcm / bitwise_and) compute the engineered column's binned MI vs y in one
    batched grid call, and keep the best-op hit per pair. Returns hits sorted by ``margin_over_operands`` descending; a
    hit's ``.responded`` flag applies the measured dual gate (margin + permutation null). The permutation null is
    computed ONLY for the per-pair best op when it already clears the baseline margin (early-reject keeps the null cost
    off the inner grid loop)."""
    import pandas as pd  # noqa: F401  (X may be pandas or polars; we pull ndarrays)

    if cols is None:
        cols = [c for c in X.columns if _is_integer_col(np.asarray(X[c]))]
    else:
        cols = [c for c in cols if _is_integer_col(np.asarray(X[c]))]
    yi = np.asarray(y).astype(np.int64)
    arrs = {c: np.asarray(X[c]) for c in cols}
    raw_mi = {c: _mi(arrs[c].astype(np.float64), yi, nbins=nbins) for c in cols}
    ops = tuple(ops)

    hits: list[LatticeHit] = []
    for a, b in combinations(cols, 2):
        aa, bb = arrs[a], arrs[b]
        operand_floor = max(raw_mi[a], raw_mi[b])
        grid = _lattice_grid_mi(aa, bb, yi, ops, nbins)
        best_i = int(np.argmax(grid))
        best_op, best_mi = ops[best_i], float(grid[best_i])
        # Early-reject: the 12-perm null only matters for ops that already clear the operand margin (``_responded`` needs
        # BOTH). An op below margin can never respond -> skip its null (stored +inf), bit-identical to the full compute.
        if (best_mi - operand_floor) >= _MIN_MARGIN:
            feat = _lattice_column(aa, bb, best_op)
            null_hi = _perm_null_hi(feat, yi, nbins, seed=seed)
        else:
            null_hi = float("inf")
        hits.append(LatticeHit(best_op, (a, b), best_mi, operand_floor, null_hi))

    hits.sort(key=lambda h: h.margin_over_operands, reverse=True)
    return hits


def detect_integer_lattice(
    X,
    y: np.ndarray,
    cols: Optional[Sequence[str]] = None,
    *,
    ops: Sequence[str] = INTEGER_LATTICE_OPS,
    top_k: int = 4,
    nbins: int = 12,
    seed: int = 0,
):
    """Cheap-first scan; returns the list of responded hits (each a dict with the op, source columns, engineered-column
    MI + margin), capped at ``top_k``. Empty when nothing responds -- a non-lattice frame detects nothing."""
    hits = cheap_integer_lattice_scan(X, y, cols, ops=ops, nbins=nbins, seed=seed)
    out = []
    for h in hits:
        if not h.responded:
            continue
        out.append(
            {
                "op": h.op, "cols": h.cols, "feat_mi": h.feat_mi,
                "operand_floor": h.operand_floor, "margin": h.margin_over_operands,
            }
        )
        if len(out) >= top_k:
            break
    return out


# Recipe plumbing: emit frozen EngineeredRecipe objects so the MRMR selector materialises, scores, selects, and replays
# the lattice column at predict time identically. Replay is pure integer arithmetic on X (cast both operands to int, take
# gcd / lcm / bitwise_and) -- no y reference, so it is leak-free + deterministic + train/test exact, mirroring the
# pairwise-modular recipe replay.


def engineered_name_integer_lattice(op: str, cols: Sequence[str]) -> str:
    """Canonical engineered column name for one lattice op, e.g. ``il_gcd__a__b``.

    The op + source columns fully determine the column. Source columns are joined with ``__`` (the codebase's
    source-link convention)."""
    if op not in INTEGER_LATTICE_OPS:
        raise ValueError(f"integer-lattice op must be one of {INTEGER_LATTICE_OPS}; got {op!r}")
    joined = "__".join(str(c) for c in cols)
    return f"{INTEGER_LATTICE_PREFIX}_{op}__{joined}"


def apply_integer_lattice(X, op: str, cols: Sequence[str]) -> np.ndarray:
    """Replay one integer-lattice column: cast both source columns to int64 then apply ``op`` (gcd / lcm / bitwise_and).

    Pure integer arithmetic on the source columns -- no y reference, no fitted state, so transform() is leak-free and
    train/test exact. Output float64; the column is the materialised selector feature."""
    if op not in INTEGER_LATTICE_OPS:
        raise ValueError(f"integer-lattice op must be one of {INTEGER_LATTICE_OPS}; got {op!r}")
    if len(cols) != 2:
        raise ValueError(f"integer-lattice op needs exactly 2 source columns; got {tuple(cols)!r}")
    a, b = (np.asarray(X[c]) for c in cols)
    return _lattice_column(a, b, op)


def build_integer_lattice_recipe(*, name: str, op: str, cols: Sequence[str]):
    """Frozen recipe for one integer-lattice column. Replay is pure integer arithmetic on the source columns."""
    from .engineered_recipes import EngineeredRecipe

    if op not in INTEGER_LATTICE_OPS:
        raise ValueError(f"integer-lattice op must be one of {INTEGER_LATTICE_OPS}; got {op!r}")
    return EngineeredRecipe(
        name=name,
        kind="pairwise_integer_lattice",
        src_names=tuple(str(c) for c in cols),
        extra={"op": str(op)},
    )


def hybrid_integer_lattice_fe_with_recipes(
    X,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    ops: Sequence[str] = INTEGER_LATTICE_OPS,
    top_k: int = 4,
    nbins: int = 12,
    seed: int = 0,
    max_int_cols: int = 30,
):
    """Detect responded integer-lattice structure and emit it as frozen, replayable ``EngineeredRecipe`` objects.

    Column-count BUDGET GUARD (the wide-frame cost is O(C(p,2)) pairs of integer ufuncs + the batched MI; see
    ``_benchmarks/bench_integer_lattice_cost``): when the number of integer-eligible columns exceeds ``max_int_cols`` the
    whole sweep is SKIPPED (logged, never silent). The sweep is pairs-only -- gcd/lcm/AND are binary, with no n-way
    analogue -- so there is no separate triple budget (cheaper than the modular pairs+triples sweep).

    Returns ``(appended_names, recipes)`` -- the materialised columns are NOT concatenated here (the MRMR caller appends
    them under its own RAM-safe path); each recipe replays the exact lattice column from X alone, leak-free."""
    if cols is None:
        int_cols = [c for c in X.columns if _is_integer_col(np.asarray(X[c]))]
    else:
        int_cols = [c for c in cols if c in X.columns and _is_integer_col(np.asarray(X[c]))]

    p = len(int_cols)
    if p > int(max_int_cols):
        logger.info(
            "integer_lattice FE: %d integer-eligible columns exceeds max_int_cols=%d; skipping the "
            "pairwise integer-lattice sweep for budget.", p, int(max_int_cols),
        )
        return [], []

    hits = cheap_integer_lattice_scan(X, y, int_cols, ops=ops, nbins=nbins, seed=seed)
    appended: list[str] = []
    recipes = []
    seen: set[str] = set()
    for h in hits:
        if not h.responded:
            continue
        name = engineered_name_integer_lattice(h.op, h.cols)
        if name in seen:
            continue
        seen.add(name)
        appended.append(name)
        recipes.append(build_integer_lattice_recipe(name=name, op=h.op, cols=h.cols))
        if len(appended) >= int(top_k):
            break
    return appended, recipes
