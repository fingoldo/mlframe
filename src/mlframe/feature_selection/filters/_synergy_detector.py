"""Cheap data-dependent synergy detector for the ``redundancy_aggregator='auto'`` gate.

WHY
---
``fleuret.py`` documents that the default Fleuret/CMIM redundancy gate REJECTS synergistic features (an operand useless alone but
informative jointly with an already-selected partner). The synergy-aware JMIM aggregator (Bennasar 2015) recovers them, but
benchmarks show JMIM OVER-SELECTS correlated decoys on additive/main-effect data (precision/parsimony regression). So JMIM must
engage ONLY when the data actually contains synergy. This module is the cheap pre-fit probe that decides.

SIGNATURE OF SYNERGY
--------------------
A pair ``(X, Z)`` is synergistic for target ``y`` when their JOINT carries materially MORE information about ``y`` than EITHER
marginal: ``I({X,Z}; Y) >> max(I(X;Y), I(Z;Y))``. For pure XOR/sign-product the marginals are ~0 while the joint is large. We
score this excess with the Miller-Madow-corrected joint MI (``joint_synergy_mi``, which keeps a noise pair's joint near zero), so
a noise grid does NOT masquerade as synergy.

DECISION (subsample, bounded cost)
-----------------------------------
On a random row subsample we quantize columns to integer codes, then over a bounded number of random feature PAIRS compute the
synergy excess ``joint - max(marg_x, marg_z)``. We declare the data synergistic when the BEST pair's excess clears a data-derived
threshold: a multiple of the per-pair noise scale estimated from the SAME estimator on label-permuted targets (an analytic null),
so the threshold adapts to n / cardinality / class balance rather than being a hardcoded magic constant. The multiple
(``excess_null_mult``) is read from kernel_tuning_cache so it can be re-tuned per dataset/HW without code change.
"""
from __future__ import annotations

import numpy as np

from ._fe_synergy_screen import _pair_mm_mi_njit, joint_synergy_mi  # noqa: F401


def _pair_mm_mi(code_x: np.ndarray, code_y: np.ndarray, yc: np.ndarray, min_rows_per_cell: float = 5.0) -> float:
    """Fused-njit equivalent of ``joint_synergy_mi(code_x, code_y, yc)``.

    Builds the dense mixed-radix ``(kx*ky) x kt`` joint histogram in one O(n) pass (no
    ``np.unique`` densify, no ``np.add.at`` scatter) and returns the SAME Miller-Madow-
    corrected joint MI. ~9.6x faster per call at the detector's (n=4000, nbins=8) shape;
    output matches ``joint_synergy_mi`` to FP reduction order (~5e-16), so the synergy
    verdict is unchanged (selection-equivalent). Bench:
    ``_benchmarks/bench_synergy_detector_pair_mi_njit.py``."""
    cx = np.ascontiguousarray(np.asarray(code_x).astype(np.int64).ravel())
    cy = np.ascontiguousarray(np.asarray(code_y).astype(np.int64).ravel())
    yt = np.ascontiguousarray(np.asarray(yc).astype(np.int64).ravel())
    n = cx.shape[0]
    if n == 0 or cy.shape[0] != n or yt.shape[0] != n:
        return 0.0
    kx = int(cx.max()) + 1
    ky = int(cy.max()) + 1
    kt = int(yt.max()) + 1
    return float(_pair_mm_mi_njit(cx, cy, yt, kx, ky, kt, float(min_rows_per_cell)))

# kernel_tuning_cache key + conservative default multiple of the permuted-null scale.
_TUNING_KEY = "mrmr_synergy_auto_excess_null_mult"
_DEFAULT_NULL_MULT = 3.0


def _quantize(col: np.ndarray, nbins: int, rng: np.random.Generator) -> np.ndarray:
    """Integer-bin a 1-D column. Low-cardinality columns (<=nbins distinct) are factorised directly so
    XOR/parity bits keep their exact 0/1 codes; continuous columns get quantile bins."""
    col = np.asarray(col, dtype=np.float64).ravel()
    finite = col[np.isfinite(col)]
    uniq = np.unique(finite)
    if uniq.size <= nbins:
        # direct factorisation -- preserves discrete bits exactly
        lut = {v: i for i, v in enumerate(uniq.tolist())}
        out = np.array([lut.get(v, 0) for v in col], dtype=np.int64)
        return out
    qs = np.quantile(finite, np.linspace(0, 1, nbins + 1)[1:-1])
    return np.clip(np.searchsorted(qs, col), 0, nbins - 1).astype(np.int64)


def _excess_for_pairs(codes: list[np.ndarray], marg: list[float], yc: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    """Max INTERACTION INFORMATION ``I({X,Z};Y) - I(X;Y) - I(Z;Y)`` over the given pairs.

    This co-information is the correct synergy signature: it is POSITIVE only when the joint carries
    information NEITHER marginal explains (XOR / sign-product), and NEGATIVE/zero for redundant pairs
    (two noisy views of the same driver, the additive-regime decoy trap), whose joint merely re-recovers
    a shared signal already counted in both marginals. Using ``joint - max(marg)`` instead would
    false-positive on redundancy (two views of one driver jointly beat either single view) -- measured,
    rejected. Returns the max over pairs (can be <=0 when no pair is synergistic)."""
    best = -np.inf
    for i, j in pairs:
        joint = _pair_mm_mi(codes[i], codes[j], yc)
        exc = joint - marg[i] - marg[j]
        if exc > best:
            best = exc
    return float(best)


def detect_synergy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_rows: int = 4000,
    max_features: int = 60,
    max_pairs: int = 400,
    n_null: int = 3,
    nbins: int = 8,
    random_seed: int = 0,
) -> tuple[bool, dict]:
    """Cheap pre-fit probe: is ``(X, y)`` synergistic enough to warrant the JMIM aggregator?

    Returns ``(is_synergistic, info)``. ``info`` carries the measured best real-pair excess, the
    permuted-null excess scale, the data-derived threshold and the null multiple used (for explain/logging).
    Bounded cost: subsample rows/features/pairs, a handful of label permutations for the null.

    STATISTICAL CAVEATS (this is a coarse OPT-IN gate, reached only when ``redundancy_aggregator='auto'``; the constructor default is
    ``None`` = plain Fleuret, so a default fit never runs it):
      * The interaction-information excess ``I({X,Z};Y) - I(X;Y) - I(Z;Y)`` is built from Miller-Madow-corrected plug-in joint MIs, which still carry a residual UPWARD bias of
        order ``(kx-1)(ky-1)(kt-1)/(2n)`` at small n / high cardinality. Both the real and permuted-null excess inherit the same bias, so taking ``real_excess`` over a multiple of
        the null scale cancels most of it -- but a very high-cardinality pair at small n can still read a small spurious positive excess.
      * The null uses only ``n_null`` (default 3) label permutations and takes the MAX excess over those runs, so the null scale is a HIGH-VARIANCE max-over-3 estimate. The
        threshold is deliberately a multiple of it (``null_mult``, from kernel_tuning_cache) with an ``eps`` floor to stay conservative. Raising ``n_null`` shrinks that variance
        but is NOT default-bumped: a larger budget RAISES the max-over-runs null scale (and so the threshold), which is selection-altering for the gate decision, plus it costs
        ``n_null x pp`` extra marginal-MI passes per call. Pass a larger ``n_null`` explicitly if the gate flickers on a borderline dataset."""
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] < 2 or X.shape[0] < 50:
        return False, {"reason": "too_small"}
    rng = np.random.default_rng(int(random_seed))
    n, p = X.shape

    # row subsample
    if n > max_rows:
        ridx = rng.choice(n, size=max_rows, replace=False)
        Xs, ys = X[ridx], np.asarray(y)[ridx]
    else:
        Xs, ys = X, np.asarray(y)

    # feature subsample
    if p > max_features:
        fidx = rng.choice(p, size=max_features, replace=False)
        Xs = Xs[:, fidx]
    pp = Xs.shape[1]

    # target codes (quantize if continuous/regression)
    yc = _quantize(ys, nbins, rng)
    if np.unique(yc).size < 2:
        return False, {"reason": "degenerate_target"}

    codes = [_quantize(Xs[:, j], nbins, rng) for j in range(pp)]
    # per-feature marginal MI = joint of the column with a constant (collapses to plain MI(X;Y))
    const = np.zeros(Xs.shape[0], dtype=np.int64)
    marg = [_pair_mm_mi(codes[j], const, yc) for j in range(pp)]

    # bounded random pair set
    all_pairs = [(i, j) for i in range(pp) for j in range(i + 1, pp)]
    if len(all_pairs) > max_pairs:
        sel = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        pairs = [all_pairs[k] for k in sel.tolist()]
    else:
        pairs = all_pairs

    real_excess = _excess_for_pairs(codes, marg, yc, pairs)

    # analytic-style null: permute the target, recompute marginals + best excess; take the scale (max over runs)
    null_excess = 0.0
    for _ in range(int(max(1, n_null))):
        yc_perm = yc[rng.permutation(yc.size)]
        marg_p = [_pair_mm_mi(codes[j], const, yc_perm) for j in range(pp)]
        e = _excess_for_pairs(codes, marg_p, yc_perm, pairs)
        if e > null_excess:
            null_excess = e

    # data-derived threshold: a multiple of the permuted-null excess scale (read from kernel_tuning_cache).
    null_mult = _DEFAULT_NULL_MULT
    try:
        from pyutilz.system import kernel_tuning_cache  # noqa: F401
        null_mult = float(_lookup_null_mult())
    except Exception:  # nosec B110 - optional dependency import guard
        pass
    # floor the null scale so a perfectly-clean permuted null (excess==0) still needs a non-trivial real excess.
    eps = 1e-4
    threshold = null_mult * max(null_excess, eps)
    is_syn = real_excess > threshold
    return bool(is_syn), {
        "real_excess": float(real_excess),
        "null_excess": float(null_excess),
        "threshold": float(threshold),
        "null_mult": float(null_mult),
        "n_pairs": len(pairs),
        "n_features": pp,
    }


def _lookup_null_mult() -> float:
    """Read the synergy-excess null multiple from kernel_tuning_cache (data-derived, no hardcoded magic).

    Falls back to the conservative default when the cache has no calibrated entry. Kept tiny + isolated so
    the import is optional and detection still works without pyutilz's tuning cache present."""
    try:
        from pyutilz.system import kernel_tuning_cache as ktc
        getter = getattr(ktc, "get_cached_param", None) or getattr(ktc, "get", None)
        if getter is not None:
            val = getter(_TUNING_KEY)
            if val is not None:
                return float(val)
    except Exception:  # nosec B110 - best-effort path
        pass
    return _DEFAULT_NULL_MULT


__all__ = ["detect_synergy"]
