"""Shared stratified row-subsampling for the feature-engineering subsamplers (R1, 2026-06-18).

The FE pair-MI sweep (``_feature_engineering_pairs._pairs_core``), the polynom-pair CMA-ES
inner search (``polynom_pair_fe``) and the pure-form-retention pool/CV builder
(``_fe_pure_form_retention``) all lower the row count before the (expensive) screen via a PLAIN
uniform ``rng.choice(n, size, replace=False)``. Uniform draws ignore the target distribution:

  * CLASSIFICATION with a rare class -- at a small ``size`` a uniform draw can omit ALL rows of a
    1%-frequency class, so the MI / linear-usability screen never sees the class the engineered
    feature is meant to separate.
  * REGRESSION with a heavy-tailed / skewed target -- uniform under-represents the tails, exactly
    the region a nonlinear pair form is most likely to be usable for.

``stratified_subsample_idx`` is a drop-in replacement that returns SORTED row indices (mirroring
the ``np.sort(rng.choice(...))`` the call sites already used) with target-aware allocation, and
degrades gracefully (single class / constant y / size>=n) back to a uniform / full-index draw.
"""
from __future__ import annotations

from typing import Optional

import logging

import numpy as np

_logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

try:
    from numba import njit

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba is a hard dep in this repo but stay safe
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):
        """No-op stand-in for ``numba.njit`` when numba is unavailable: returns the function unchanged so the decorated kernels still run (as plain Python) instead of raising at import time."""
        def _wrap(fn):
            """Identity wrapper matching ``njit``'s call-with-kwargs-then-decorate form."""
            return fn

        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _wrap

# Number of quantile bins for the regression-target stratification. Ten bins preserve both tails
# (top/bottom decile) without over-fragmenting the proportional per-bin allocation at small size.
_FE_STRATIFY_REG_BINS: int = 10


# ---------------------------------------------------------------------------------------------------
# NUMBA KERNELS (R2, 2026-06-18). Deterministic per-call inline-LCG RNG (thread-safe: no process-global
# np.random state, so concurrent joblib-threading callers cannot clobber each other's stream). Exact
# bit-match to the numpy version is NOT required (the
# default flips to ON anyway) -- the kernels only guarantee DETERMINISM (same seed+inputs -> same
# indices across runs/processes) and the SAME stratification GUARANTEES: per-class proportional with
# >=min(2,count)/class (classification), proportional across y-quantile bins preserving the tails
# (regression). Repo policy keeps the pure-numpy version below as a FALLBACK (non-numeric clf labels
# are factorised to codes before the kernel; any unsupported / degenerate case routes to numpy).
# ---------------------------------------------------------------------------------------------------


@njit(cache=True)
def _partial_fisher_yates(members, k, lcg_seed):
    """Draw ``k`` distinct elements from ``members`` via a partial Fisher-Yates shuffle driven by a
    per-call inline LCG (thread-safe: no process-global ``np.random`` state, so concurrent threading
    callers cannot clobber each other's stream). Deterministic for a fixed ``lcg_seed`` + inputs. For
    ``k >= len(members)`` the whole array is returned.
    """
    m = members.copy()
    nm = m.shape[0]
    if k >= nm:
        return m
    state = np.uint64(lcg_seed) | np.uint64(1)
    for i in range(k):
        state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
        j = i + int(state >> np.uint64(33)) % (nm - i)
        tmp = m[i]
        m[i] = m[j]
        m[j] = tmp
    return m[:k]


@njit(cache=True)
def _strat_clf_kernel(codes, n_classes, size, n, seed):
    """Per-class proportional allocation with a >=min(2,count) per-class floor (classification).

    ``codes`` are integer class codes in ``[0, n_classes)``. Deterministic for a fixed ``seed`` and
    inputs. Returns the (unsorted) drawn row indices.
    """
    counts = np.zeros(n_classes, dtype=np.int64)
    for i in range(n):
        counts[codes[i]] += 1
    # Membership index lists, packed contiguously with per-class offsets.
    offsets = np.zeros(n_classes + 1, dtype=np.int64)
    for c in range(n_classes):
        offsets[c + 1] = offsets[c] + counts[c]
    packed = np.empty(n, dtype=np.int64)
    cursor = offsets[:-1].copy()
    for i in range(n):
        c = codes[i]
        packed[cursor[c]] = i
        cursor[c] += 1

    # Proportional allocation, floored at min(2, count), capped at count.
    alloc = np.empty(n_classes, dtype=np.int64)
    total = 0
    for c in range(n_classes):
        cnt = counts[c]
        prop = np.int64(np.floor(cnt / n * size))
        floor_c = 2 if cnt >= 2 else cnt
        a = prop if prop > floor_c else floor_c
        if a > cnt:
            a = cnt
        alloc[c] = a
        total += a

    out = np.empty(total, dtype=np.int64)
    pos = 0
    for c in range(n_classes):
        mem = packed[offsets[c] : offsets[c + 1]]
        # per-stratum LCG seed mixes the caller seed with the class index (golden-ratio constant) so
        # strata are independent yet fully determined by ``seed`` -- no process-global RNG involved.
        drawn = _partial_fisher_yates(mem, alloc[c], np.uint64(seed) + np.uint64(c) * np.uint64(11400714819323198485))
        for j in range(drawn.shape[0]):
            out[pos] = drawn[j]
            pos += 1
    return out[:pos]


@njit(cache=True)
def _strat_reg_kernel(bin_ids, n_bins, size, n, seed):
    """Proportional draw across precomputed quantile-bin ids (regression). Tails preserved.

    ``bin_ids`` map each row to a bin in ``[0, n_bins)`` (the dedicated NaN/inf bin included). Each
    NON-EMPTY bin gets a >=1 floor so the tails are never wholly dropped. Deterministic for a fixed
    ``seed``. Returns the (unsorted) drawn row indices.
    """
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(n):
        counts[bin_ids[i]] += 1
    offsets = np.zeros(n_bins + 1, dtype=np.int64)
    for b in range(n_bins):
        offsets[b + 1] = offsets[b] + counts[b]
    packed = np.empty(n, dtype=np.int64)
    cursor = offsets[:-1].copy()
    for i in range(n):
        b = bin_ids[i]
        packed[cursor[b]] = i
        cursor[b] += 1

    alloc = np.empty(n_bins, dtype=np.int64)
    total = 0
    for b in range(n_bins):
        cnt = counts[b]
        if cnt == 0:
            alloc[b] = 0
            continue
        prop = np.int64(np.floor(cnt / n * size))
        a = prop if prop > 1 else 1
        if a > cnt:
            a = cnt
        alloc[b] = a
        total += a

    out = np.empty(total, dtype=np.int64)
    pos = 0
    for b in range(n_bins):
        if counts[b] == 0:
            continue
        mem = packed[offsets[b] : offsets[b + 1]]
        drawn = _partial_fisher_yates(mem, alloc[b], np.uint64(seed) + np.uint64(b + 1) * np.uint64(11400714819323198485))
        for j in range(drawn.shape[0]):
            out[pos] = drawn[j]
            pos += 1
    return out[:pos]


def _derive_seed(rng) -> int:
    """Derive a deterministic non-negative int32-range seed for the numba RNG from ``rng``.

    Draws one integer from the (seeded) caller generator so the kernel seed is reproducible from the
    caller's seed yet differs across distinct call sites that pass freshly-seeded generators.
    """
    try:
        return int(rng.integers(0, 2**31 - 1))
    except Exception:
        return 0


def stratified_subsample_idx(rng, y, size: int, *, is_clf: bool) -> np.ndarray:
    """Return SORTED row indices of a target-stratified subsample of ``y`` of (about) ``size`` rows.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded generator; all randomness (per-stratum draws + the degenerate uniform fallback)
        is taken from it so the subsample is reproducible from the caller's seed.
    y : array-like
        1D target. Classification labels (any hashable/int dtype) for ``is_clf=True``; a continuous
        array for ``is_clf=False``.
    size : int
        Target subsample size. ``size >= n`` (or ``size <= 0``) returns ``np.arange(n)`` (no draw).
    is_clf : bool
        ``True`` -> per-class PROPORTIONAL allocation with a guaranteed ``min(2, count)`` per class
        (so a rare class is never wholly dropped). ``False`` -> proportional draw across
        ``_FE_STRATIFY_REG_BINS`` y-quantile bins (preserves the tails).

    Returns
    -------
    np.ndarray
        Sorted ``int64`` row indices into ``y``. Length is approximately ``size`` (per-stratum
        rounding + the >=1/class guarantee can nudge it by a few rows; never exceeds ``n``).

    Robustness
    ----------
    A single class, a constant / all-non-finite continuous y, or any unexpected error falls back to
    a plain uniform draw (``np.sort(rng.choice(n, size, replace=False))``) so the helper can ALWAYS
    stand in for the legacy uniform site without raising.

    Dispatch
    --------
    When numba is available and ``y`` is representable as integer codes (classification) / numeric
    quantile bins (regression), the deterministic ``@njit`` kernels (``_strat_clf_kernel`` /
    ``_strat_reg_kernel``) do the per-stratum draw; otherwise the pure-numpy path below runs. Both
    honour the SAME stratification guarantees; the njit path is seeded deterministically from ``rng``.
    """
    y_arr = np.asarray(y).ravel()
    n = y_arr.shape[0]
    size = int(size)
    if size <= 0 or size >= n:
        return np.arange(n, dtype=np.int64)

    def _uniform() -> np.ndarray:
        """Plain (non-stratified) random draw of ``size`` row indices, sorted for cache-friendly downstream gathers; the fallback used when stratification isn't applicable or fails."""
        return np.sort(rng.choice(n, size=size, replace=False)).astype(np.int64, copy=False)

    try:
        if is_clf:
            # Per-class proportional allocation, guaranteeing min(2, count) rows per class so a
            # rare class is never wholly dropped. Factorise to int codes (handles non-numeric labels)
            # so the njit kernel can consume them.
            classes, inv = np.unique(y_arr, return_inverse=True)
            n_classes = classes.shape[0]
            if n_classes <= 1:
                return _uniform()
            # HIGH-CARDINALITY GUARD: the per-class ``min(2, count)`` floor guarantees a rare class is
            # never wholly dropped, but when the label is (near-)continuous / very high cardinality the
            # floors alone SUM TO ~n -- the "subsample" then returns ~n rows and silently DEFEATS the
            # screen subsample (a 998327-row "30k" draw returned 998327 rows in the wellbore-shape repro).
            # When the floored minimum cannot fit the budget, stratification is meaningless: fall back to a
            # plain uniform draw of exactly ``size`` rows (a target with more distinct labels than the row
            # budget is effectively continuous). ``inv`` is already computed, so the floor sum is O(n) cheap.
            _cnts = np.bincount(inv.astype(np.int64, copy=False), minlength=n_classes)
            _floor_total = int(np.minimum(_cnts, 2).sum())
            if _floor_total > size:
                return _uniform()
            if _HAVE_NUMBA:
                codes = np.ascontiguousarray(inv.astype(np.int64))
                idx = _strat_clf_kernel(codes, int(n_classes), size, int(n), _derive_seed(rng))
                if idx.shape[0] == 0:
                    return _uniform()
                return np.sort(idx).astype(np.int64, copy=False)
            members = [np.flatnonzero(inv == c) for c in range(n_classes)]
            counts = np.array([m.shape[0] for m in members], dtype=np.int64)
            # Proportional target per class, floored at the per-class minimum.
            alloc = np.maximum(
                np.minimum(counts, 2),
                np.floor(counts.astype(np.float64) / n * size).astype(np.int64),
            )
            alloc = np.minimum(alloc, counts)  # cannot draw more than a class holds
            picked = []
            for m, k in zip(members, alloc):
                if k >= m.shape[0]:
                    picked.append(m)
                else:
                    picked.append(rng.choice(m, size=int(k), replace=False))
            idx = np.concatenate(picked)
        else:
            finite = np.isfinite(y_arr)
            if finite.sum() < 2 or np.nanmin(y_arr[finite]) == np.nanmax(y_arr[finite]):
                return _uniform()
            # Quantile bins over the FINITE values; non-finite rows form their own stratum so they
            # are still representable (and never crash the digitize).
            yv = y_arr.copy()
            edges = np.quantile(yv[finite], np.linspace(0.0, 1.0, _FE_STRATIFY_REG_BINS + 1))
            edges = np.unique(edges)
            if edges.shape[0] < 2:
                return _uniform()
            bin_ids = np.digitize(yv, edges[1:-1], right=False)
            bin_ids[~finite] = edges.shape[0]  # dedicated bin for NaN/inf rows
            if _HAVE_NUMBA:
                # Remap bin ids to a contiguous [0, n_bins) range for the kernel.
                uniq = np.unique(bin_ids)
                remap = np.zeros(int(bin_ids.max()) + 1, dtype=np.int64)
                remap[uniq] = np.arange(uniq.shape[0], dtype=np.int64)
                bin_codes = np.ascontiguousarray(remap[bin_ids].astype(np.int64))
                idx = _strat_reg_kernel(bin_codes, int(uniq.shape[0]), size, int(n), _derive_seed(rng))
                if idx.shape[0] == 0:
                    return _uniform()
                return np.sort(idx).astype(np.int64, copy=False)
            uniq_bins = np.unique(bin_ids)
            members = [np.flatnonzero(bin_ids == b) for b in uniq_bins]
            counts = np.array([m.shape[0] for m in members], dtype=np.int64)
            alloc = np.maximum(
                np.minimum(counts, 1),
                np.floor(counts.astype(np.float64) / n * size).astype(np.int64),
            )
            alloc = np.minimum(alloc, counts)
            picked = []
            for m, k in zip(members, alloc):
                if k >= m.shape[0]:
                    picked.append(m)
                else:
                    picked.append(rng.choice(m, size=int(k), replace=False))
            idx = np.concatenate(picked)
    except Exception:
        return _uniform()

    if idx.shape[0] == 0:
        return _uniform()
    return np.sort(idx).astype(np.int64, copy=False)


# Auto-ON heuristic thresholds (R1). Classification: turn stratification ON when the SMALLEST class
# is rarer than this fraction of n (a uniform draw at a lowered size risks dropping it). Regression:
# turn ON when the target is heavy-tailed / skewed beyond these bounds.
_FE_STRATIFY_MIN_CLASS_FRACTION: float = 0.10
_FE_STRATIFY_REG_ABS_SKEW: float = 2.0
_FE_STRATIFY_REG_TAIL_IQR_RATIO: float = 6.0


def resolve_shared_fe_subsample_idx(y: np.ndarray, n_rows: int, size: int, *, is_clf: bool, stratify_knob: object, random_seed: Optional[int]) -> np.ndarray | None:
    """Resolve the ONE shared FE subsample row-index set for a whole fit (the single reused draw).

    The FE screen / pair-search / polynom / permutation-null floors all bound their cost by row
    subsampling, and were historically drawing the subsample INDEPENDENTLY (different seeds / per-pair
    draws) -- wasteful re-sampling and inconsistent (stages saw different rows). This produces ONE draw
    per fit, reproducible from ``random_seed``, so every consumer SLICES the SAME rows instead of
    re-drawing. The permutation-null floors then ride the SAME ``size`` as the candidates they gate, so
    the floor (a chance-max MI threshold whose scale is ~1/n) is the matched estimator for the screened
    candidates -- not the historically full-n floor that was both the large-n cost / OOM source and a
    looser threshold than the subsampled candidates it gated.

    Returns ``None`` when ``n_rows <= size`` (nothing to subsample -> consumers use full n, byte-identical
    to legacy small-n behaviour) or on any failure. Otherwise SORTED int64 indices, ~``size`` long.
    """
    try:
        n = int(n_rows)
        size = int(size)
        if size <= 0 or n <= size:
            return None
        base = 0x9E3779B9 if random_seed is None else (int(random_seed) & 0x7FFFFFFF)
        rng = np.random.default_rng(base)
        stratify = _resolve_fe_subsample_stratify(stratify_knob, y, is_clf=is_clf)
        if stratify:
            return stratified_subsample_idx(rng, np.asarray(y), size, is_clf=is_clf)
        return np.sort(rng.choice(n, size=size, replace=False)).astype(np.int64, copy=False)
    except Exception as exc:
        # A silent None here makes every downstream FE/MI consumer run at FULL n -- a ~33x cost blow-up at
        # n~1M that is invisible in a profile (it just looks slow). Log at WARNING so a failure in the
        # (group-aware / stratified) resolve is diagnosable instead of a mystery full-n screen. Still fall
        # back (best-effort optimisation), but never silently.
        _logger.warning("resolve_shared_fe_subsample_idx failed; FE screen will run at FULL n (no subsample): %r", exc, exc_info=True)
        return None


def _resolve_fe_subsample_stratify(stratify_knob, y, *, is_clf: bool) -> bool:
    """Resolve the tri-state ``fe_subsample_stratify`` knob to a concrete bool for a fit.

    ``True`` / ``False`` are returned verbatim (explicit user intent). ``None`` (the DEFAULT)
    triggers the auto-ON heuristic: stratify only when a uniform subsample would plausibly LOSE
    target structure -- a small minimum-class fraction (classification) or a heavy-tailed / skewed
    target (regression). Otherwise OFF, so the common dense-class / well-behaved-target path stays
    byte-identical to the legacy uniform draw. Any error in the heuristic -> OFF (legacy)."""
    if stratify_knob is True:
        return True
    if stratify_knob is False:
        return False
    # stratify_knob is None -> AUTO.
    try:
        y_arr = np.asarray(y).ravel()
        n = y_arr.shape[0]
        if n < 2:
            return False
        if is_clf:
            _, counts = np.unique(y_arr, return_counts=True)
            if counts.shape[0] <= 1:
                return False
            return bool(counts.min() / n < _FE_STRATIFY_MIN_CLASS_FRACTION)
        # Regression: heavy-tail / skew detection on the finite values.
        finite = y_arr[np.isfinite(y_arr)] if y_arr.dtype.kind == "f" else y_arr.astype(np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size < 8:
            return False
        std = float(finite.std())
        if std <= 0.0:
            return False
        mean = float(finite.mean())
        skew = float(np.mean(((finite - mean) / std) ** 3))
        if abs(skew) >= _FE_STRATIFY_REG_ABS_SKEW:
            return True
        q1, med, q3 = np.quantile(finite, [0.25, 0.5, 0.75])
        iqr = float(q3 - q1)
        if iqr <= 0.0:
            return False
        tail = max(float(finite.max() - med), float(med - finite.min()))
        return bool(tail / iqr >= _FE_STRATIFY_REG_TAIL_IQR_RATIO)
    except Exception:
        return False
