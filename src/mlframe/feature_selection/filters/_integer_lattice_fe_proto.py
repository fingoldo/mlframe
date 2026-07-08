"""EXPERIMENTAL prototype: integer-lattice pairwise operators (gcd / lcm / bitwise AND-OR-XOR).

NOT wired into prod. Sibling-in-spirit to ``_pairwise_modular_fe`` -- where the modular operator captures a hidden
PERIOD of an integer combination, these capture a hidden COMMON DIVISOR (``gcd(a, b)`` -- shared factor / grid
alignment) or a bit-level INTERACTION of integer codes (``a & b``, ``a ^ b`` -- flag co-occurrence, parity-of-shared-bits).

Smooth bases (poly / Fourier / RBF / spline) and the existing arithmetic ops (mul/add/sub/div/mod) provably cannot
express these: ``gcd`` is number-theoretic (non-smooth, non-monotone in either argument -- gcd(6,9)=3 but gcd(6,10)=2),
and bitwise XOR is the canonical zero-marginal interaction at the BIT level (each bit position is an independent XOR,
so ``a ^ b`` has near-zero pairwise corr with both a and b yet fully determines a step target keyed on shared low bits).
The commented-out ``np.gcd`` / ``np.lcm`` lines in ``feature_engineering.create_binary_transformations`` confirm these
were intended but dropped because the numpy ufuncs reject float32 inputs; this prototype handles the int-cast explicitly.

Eligibility mirrors the modular operator: integer-typed (or exactly-integer-valued float) columns only -- the lattice
structure is an integer property; quantising a continuous column would manufacture spurious shared factors.

Cost model: each op is one vectorised integer ufunc on (n,) + the shipped binned-MI kernel (the same ``_mi`` used by
the modular scan). Cheap-first gate: emit a column only when its MI beats BOTH operands' raw MI by a margin AND a
permutation-null band -- so a non-lattice frame injects nothing.

cProfile (n=2000, 6 cols, 5 reps): ~99% of wall is the shipped ``_mi_classif_batch`` kernel; the integer arithmetic is
~0.7%. When wiring into prod the per-op MI calls should be batched into one ``_mi_classif_batch`` per pair (mirror
``_pairwise_modular_fe._residue_grid_mi``) -- the prototype scores per-op for clarity, which the prod path need not.
"""
from __future__ import annotations

import numpy as np

from ._pairwise_modular_fe import _is_integer_col, _mi

__all__ = [
    "INTEGER_LATTICE_OPS",
    "apply_integer_lattice",
    "scan_integer_lattice_pairs",
]

INTEGER_LATTICE_OPS = ("gcd", "lcm", "and", "or", "xor")

# Margin a lattice column must beat each operand's own MI by (mirrors _pairwise_modular_fe._MIN_MARGIN). A column whose
# MI does not exceed the better operand by this much carries no structure the selector cannot already get from the raw column.
_MIN_MARGIN = 0.02


def _to_int(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest int64 view of an exactly-integer-valued column (eligibility already checked by caller)."""
    return np.rint(np.asarray(x, dtype=np.float64)).astype(np.int64)


def _perm_null_hi(feat: np.ndarray, yi: np.ndarray, nbins: int, n_perm: int, rng, z: float = 3.0) -> float:
    """Upper band (mean + z*std) of the pre-computed feature's MI under y permutation -- the noise reference the
    feature MI must clear. The feature is fixed; only y is shuffled (cheap, n_perm small)."""
    vals = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        vals[i] = _mi(feat, yi[rng.permutation(yi.size)], nbins=nbins)
    return float(vals.mean() + z * vals.std())


def apply_integer_lattice(a: np.ndarray, b: np.ndarray, op: str) -> np.ndarray:
    """One integer-lattice column from a pair. Pure function of (a, b) -- leak-free at replay (no y)."""
    ai, bi = _to_int(a), _to_int(b)
    if op == "gcd":
        return np.asarray(np.gcd(ai, bi).astype(np.float64))
    if op == "lcm":
        # lcm can overflow int64 for large coprime pairs; clip the product term via float to keep it finite + monotone.
        g = np.gcd(ai, bi)
        safe_g = np.where(g == 0, 1, g)
        return np.asarray((np.abs(ai.astype(np.float64)) * np.abs(bi.astype(np.float64))) / safe_g)
    if op == "and":
        return np.asarray(np.bitwise_and(ai, bi).astype(np.float64))
    if op == "or":
        return np.asarray(np.bitwise_or(ai, bi).astype(np.float64))
    if op == "xor":
        return np.asarray(np.bitwise_xor(ai, bi).astype(np.float64))
    raise ValueError(f"apply_integer_lattice: unknown op {op!r}; expected one of {INTEGER_LATTICE_OPS}")


def scan_integer_lattice_pairs(
    X: np.ndarray,
    y: np.ndarray,
    col_names: list[str],
    ops: tuple[str, ...] = INTEGER_LATTICE_OPS,
    nbins: int = 12,
    n_perm: int = 12,
    min_margin: float = _MIN_MARGIN,
    rng_seed: int = 0,
) -> list[dict]:
    """Cheap-first scan over integer-typed column pairs. Returns one hit dict per (op, pair) that beats BOTH operand
    MIs by ``min_margin`` AND a permutation-null upper band -- the same dual gate the modular operator uses.

    ``X`` is (n, p) numeric; only columns passing ``_is_integer_col`` are eligible. Bitwise/gcd are symmetric so each
    unordered pair is scanned once.
    """
    yi = np.asarray(y).astype(np.int64)
    n, p = X.shape
    elig = [j for j in range(p) if _is_integer_col(X[:, j])]
    raw_mi = {j: _mi(X[:, j], yi, nbins=nbins) for j in elig}
    hits: list[dict] = []
    for ii in range(len(elig)):
        for jj in range(ii + 1, len(elig)):
            a_idx, b_idx = elig[ii], elig[jj]
            a, b = X[:, a_idx], X[:, b_idx]
            operand_floor = max(raw_mi[a_idx], raw_mi[b_idx])
            for op in ops:
                feat = apply_integer_lattice(a, b, op)
                mi = _mi(feat, yi, nbins=nbins)
                if mi < operand_floor + min_margin:
                    continue
                null_hi = _perm_null_hi(feat, yi, nbins=nbins, n_perm=n_perm, rng=np.random.default_rng(rng_seed))
                if mi <= null_hi:
                    continue
                hits.append(
                    dict(op=op, a=col_names[a_idx], b=col_names[b_idx], mi=float(mi), operand_floor=float(operand_floor), null_hi=float(null_hi)),
                )
    hits.sort(key=lambda h: h["mi"], reverse=True)
    return hits
