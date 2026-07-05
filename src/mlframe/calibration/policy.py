"""Auto-select the best post-hoc calibrator by OOF ECE with bootstrap CI tiebreak.

The ``pick_best_calibrator`` helper benches a small palette of binary calibrators
(Sigmoid / Isotonic / Beta / Spline) on the OOF-train probabilities, computes the
ECE point estimate plus a percentile bootstrap CI (1000 resamples by default), and
returns the calibrator that minimises ECE — with a Kull-2017 default-rule fallback
(Isotonic for n_oof >= 1000, Beta otherwise) when the candidate CIs overlap so the
choice is non-arbitrary on small-sample / nearly-tied OOF fits.

Wire-up: ``post_calibrate_model`` consults ``CalibrationConfig.policy_auto_pick``
(default True) and threads the chosen calibrator into the meta-model fit. The
chosen calibrator + its CI is also stamped into the metadata report so a reviewer
can see at a glance which method the suite picked and how confident the OOF ECE
estimate is.

References:
  - Kull, Filho, Flach (AISTATS 2017) "Beta calibration".
  - Niculescu-Mizil & Caruana ICML 2005 "Predicting good probabilities".
  - Naeini et al. AAAI 2015 (ECE binning).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from mlframe.evaluation.bootstrap import _ci_from_samples, _jackknife_metric

logger = logging.getLogger(__name__)


# 10 bins beats 15 in 14/18 (scenario x n) cells on RMSE-vs-ground-truth and
# has the lowest mean RMSE (0.0097 vs 0.0109); 15 over-binned -> upward bias at
# small n. Bench: _benchmarks/bench_ece_nbins.py.
DEFAULT_ECE_NBINS: int = 10
DEFAULT_N_BOOTSTRAP: int = 1000
DEFAULT_ALPHA: float = 0.05
SMALL_SAMPLE_THRESHOLD: int = 1000

CANDIDATE_NAMES: tuple[str, ...] = ("Sigmoid", "Isotonic", "Beta", "Spline")

# RAM ceiling (bytes) for the one-time (n_bootstrap x n) int64 resample-index matrix in
# ``_build_resample_indices``. The matrix draw-order is pinned (chunking / dtype change benched +
# rejected, see that docstring), so instead of silently allocating an oversized matrix we raise a
# clear MemoryError with the projected size + remediation before the ``np.empty`` alloc. Default
# ~1 GiB; override at runtime via ``MLFRAME_CALIBRATION_RESAMPLE_MAX_BYTES`` (read per-call).
DEFAULT_RESAMPLE_MATRIX_MAX_BYTES: int = 1 << 30


def _resample_matrix_max_bytes() -> int:
    """RAM ceiling (bytes) for the resample-index matrix; env-overridable per call."""
    raw = os.environ.get("MLFRAME_CALIBRATION_RESAMPLE_MAX_BYTES")
    if raw is None or not raw.strip():
        return DEFAULT_RESAMPLE_MATRIX_MAX_BYTES
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "MLFRAME_CALIBRATION_RESAMPLE_MAX_BYTES=%r is not an int; using default %d",
            raw, DEFAULT_RESAMPLE_MATRIX_MAX_BYTES,
        )
        return DEFAULT_RESAMPLE_MATRIX_MAX_BYTES


try:
    from numba import njit as _njit  # type: ignore
    _HAS_NUMBA = True
except Exception:  # pragma: no cover  -- numba is a hard dep, keep guard for static analysers
    _HAS_NUMBA = False
    def _njit(*args, **kwargs):  # type: ignore
        def _decorator(fn):
            return fn
        return _decorator


@_njit(cache=True, nogil=True)
def _ece_score_numba_serial(y: np.ndarray, p: np.ndarray, n_bins: int) -> float:
    """Single-pass per-bin reduction (12-15x vs numpy bincount on n=2k..1M).

    Streams ``(p[i], y[i])`` once, computing per-bin counts + sums in
    fixed-size float64 accumulators. Parallel variant exists in
    ``profiling/bench_ece_score_variants.py`` but pays prange overhead
    that the per-iter scalar work cannot amortise on n<1M -- the serial
    kernel wins on every size in the bench.
    """
    n = p.size
    sum_p = np.zeros(n_bins, dtype=np.float64)
    sum_y = np.zeros(n_bins, dtype=np.float64)
    n_finite = 0.0
    for i in range(n):
        pi = p[i]
        yi = y[i]
        if not (np.isfinite(pi) and np.isfinite(yi)):
            continue
        b = int(pi * n_bins)
        if b >= n_bins:
            b = n_bins - 1
        elif b < 0:
            b = 0
        sum_p[b] += pi
        sum_y[b] += yi
        n_finite += 1.0
    if n_finite == 0.0:
        return float("nan")
    total = 0.0
    for b in range(n_bins):
        diff = sum_y[b] - sum_p[b]
        if diff < 0.0:
            diff = -diff
        total += diff
    return total / n_finite


@_njit(cache=True, nogil=True)
def _ece_score_idx_numba_serial(y: np.ndarray, p: np.ndarray, idx: np.ndarray, n_bins: int) -> float:
    """Idx-aware ECE: gather ``y[idx[i]]`` / ``p[idx[i]]`` INSIDE the bin loop.

    Identical reduction to ``_ece_score_numba_serial`` but the resample gather
    is fused into the njit loop, so the bootstrap caller never materialises a
    per-resample ``y[idx]`` / ``p[idx]`` Python slice. Bit-identical by
    construction: equal-width binning is order-independent (no argsort/tie
    break), so gathering inside vs slicing outside reduces the exact same
    per-bin sums. Mirrors the idx-aware AUC resampler (gather inside the njit loop).
    """
    m = idx.shape[0]
    sum_p = np.zeros(n_bins, dtype=np.float64)
    sum_y = np.zeros(n_bins, dtype=np.float64)
    n_finite = 0.0
    for k in range(m):
        j = idx[k]
        pi = p[j]
        yi = y[j]
        if not (np.isfinite(pi) and np.isfinite(yi)):
            continue
        b = int(pi * n_bins)
        if b >= n_bins:
            b = n_bins - 1
        elif b < 0:
            b = 0
        sum_p[b] += pi
        sum_y[b] += yi
        n_finite += 1.0
    if n_finite == 0.0:
        return float("nan")
    total = 0.0
    for b in range(n_bins):
        diff = sum_y[b] - sum_p[b]
        if diff < 0.0:
            diff = -diff
        total += diff
    return total / n_finite


@_njit(cache=True, nogil=True)
def _scan_binary01_f64(a: np.ndarray) -> int:
    """Single O(n) pass: 1 iff every finite value is exactly 0.0 or 1.0 AND both classes are present, else 0.

    Fast-path detector for the float64 bootstrap-ECE hot path. honest_diagnostics casts labels to float64 ONCE
    upstream, so every resample's ``y_true_f64[idx]`` reaches ``_normalize_binary_labels`` as float64 and MISSES
    the integer ``min==0/max==1`` short-circuit -- it fell through to ``np.unique`` (an O(n log n) sort + isfinite
    alloc) on ALL ~6000 resamples. This scan replaces that with one branch-only pass, no allocation. NaN is ignored
    (the ECE kernel skips non-finite rows); a non-0/1 finite value (incl. +-inf) returns 0 so the caller falls back
    to the exact np.unique path (which filters inf via isfinite and handles the remap contract)."""
    saw0 = False
    saw1 = False
    for i in range(a.size):
        v = a[i]
        if v != v:  # NaN
            continue
        if v == 0.0:
            saw0 = True
        elif v == 1.0:
            saw1 = True
        else:
            return 0
    if saw0 and saw1:
        return 1
    return 0


def _normalize_binary_labels(y: np.ndarray) -> np.ndarray:
    """Validate + map ``y`` to {0, 1} for the ECE kernel (which is correct only there).

    The numba ECE kernel computes per-bin accuracy as ``mean(y in bin)``, valid only for
    ``y in {0, 1}``. Non-0/1 encodings ({-1,+1}, {1,2}, ...) make that accuracy meaningless
    and the reported ECE silently wrong, so -- mirroring the explicit 0/1 guard
    ``auc_variance`` / ``delong_test`` enforce -- require exactly two distinct FINITE values
    and map the larger to 1, the smaller to 0. Already-{0,1} input is returned as an int
    array unchanged in ordering (max==1 -> positive stays 1). NaNs in ``y`` are ignored for
    the distinct-value check (the kernel skips non-finite rows) but must leave >= 2 finite
    values, else there is no valid binary label set and we raise.
    """
    arr = np.asarray(y).ravel()
    # Fast path for the overwhelmingly common already-{0,1} integer/bool labels (every bootstrap resample gathers the
    # same 0/1 base vector): an integer array with min == 0 and max == 1 can only hold the values {0, 1}, and both are
    # present (min hits 0, max hits 1), so it is EXACTLY the two-distinct-0/1 case -- no np.unique sort needed. This skips
    # the O(n log n) sort that ran on all ~12k resamples of a bootstrap ECE CI (7x on the normalisation at n=30k).
    if arr.dtype.kind in "iub" and arr.size and arr.min() == 0 and arr.max() == 1:
        # ``copy=False`` returns the already-int64 array itself (no full-n copy) -- the ECE kernel reads it
        # read-only, and the overwhelmingly common bootstrap resample is an int64 {0,1} gather (2.97x here).
        return arr.astype(np.int64, copy=False)
    # Fast path for the float64 bootstrap-ECE hot path: labels are cast to float64 ONCE upstream
    # (honest_diagnostics), so each resample gathers a float64 0.0/1.0 vector that misses the integer
    # short-circuit above and used to pay np.unique's O(n log n) sort every resample. A single branch-only
    # njit scan confirms all-{0.0,1.0}-both-present and returns the array unchanged (0.0/1.0 sum bit-identically
    # in the float64 ECE accumulator, so the ECE is identical to the int-cast general-path result); anything
    # else (0.5, inf, one class) returns 0 and falls through to the exact np.unique remap/raise path below.
    if arr.dtype == np.float64 and arr.size and _scan_binary01_f64(arr) == 1:
        return arr
    finite = arr[np.isfinite(arr.astype(np.float64))] if arr.dtype.kind == "f" else arr
    uniq = np.unique(finite)
    if uniq.size != 2:
        raise ValueError(
            f"_ece_score: y_true must have exactly 2 distinct (finite) label values for binary " f"ECE; got {uniq.size} distinct value(s): {uniq.tolist()[:10]}"
        )
    if uniq[0] == 0 and uniq[1] == 1:
        return arr.astype(np.int64, copy=False)
    hi = uniq.max()
    return (arr == hi).astype(np.int64)


def _ece_score(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = DEFAULT_ECE_NBINS) -> float:
    """Equal-width ECE over ``n_bins`` for binary probability ``p_pred[:, 1]`` or 1-D ``p_pred``.

    ``y_true`` is validated + normalised to {0, 1} (see ``_normalize_binary_labels``): the ECE
    kernel's per-bin accuracy ``mean(y in bin)`` is correct only for 0/1 labels, so non-0/1
    encodings ({-1,+1}, {1,2}, ...) are remapped (larger value -> 1) and inputs without exactly
    two distinct finite labels raise -- mirroring the 0/1 guard in ``auc_variance``/``delong_test``.

    Standard ECE: ``sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|`` over equal-width
    confidence bins on ``[0, 1]``. Returns nan when ``p_pred`` is empty or all-nan.

    Binning scheme is EQUAL-WIDTH on the fixed ``[0, 1]`` grid (bin index ``int(p * n_bins)``). This
    differs from the EQUAL-MASS scheme in ``calibration/quality.estimate_calibration_quality_binned``
    and the data-adaptive [min,max]-span scheme in
    ``metrics/calibration/_calibration_metrics.compute_ece_and_brier_decomposition``; the three ECE
    numbers are NOT comparable across schemes on the same inputs (different axis partitions), so
    compare ECE only within one scheme.

    iter309 (2026-05-26): numba single-pass reduction kernel. The
    iter308 ``np.bincount`` rewrite was 3.38x faster than the per-bin
    Python loop; this numba kernel is another ~12-15x faster than the
    bincount path because the per-i computation collapses to one branch
    + one integer cast + three accumulator updates, all inside a tight
    numba loop with no temporary arrays. Bench
    ``profiling/bench_ece_score_variants.py``:
      n=2k:    0.115 ms (numpy)   ->  0.008 ms (numba)  14.7x
      n=20k:   0.758 ms           ->  0.064 ms          11.9x
      n=200k:  9.413 ms           ->  0.636 ms          14.8x
      n=1M:   51.530 ms           ->  3.445 ms          15.0x
    Parallel variant tried and rejected: prange overhead dominates the
    per-iter scalar work; serial wins on every n in the bench.
    Numerical equivalence verified to <1e-12 vs the bincount path.

    Equivalence math: ``sum_b (count_b/n) * |conf_b - acc_b|`` with
    ``conf_b = sum_p_b / count_b`` and ``acc_b = sum_y_b / count_b``
    reduces to ``(1/n) * sum_b |sum_y_b - sum_p_b|`` because the count_b
    cancels between the per-bin weight times the per-bin magnitude.
    """
    # lead-ece-wrapper: skip the asarray/ravel/ascontiguousarray coercion when
    # the caller already passes a contiguous 1-D float64 array (the bootstrap
    # hot path always does). Guarded -- any other dtype/shape/non-contiguity
    # keeps the full coercion below, so behaviour is unchanged for all callers.
    if (
        isinstance(p_pred, np.ndarray) and p_pred.dtype == np.float64
        and p_pred.ndim == 1 and p_pred.flags.c_contiguous
        and isinstance(y_true, np.ndarray) and y_true.ndim == 1 and y_true.flags.c_contiguous
    ):
        if p_pred.size == 0 or y_true.size != p_pred.size:
            return float("nan")
        y_norm = _normalize_binary_labels(y_true)
        return _ece_score_numba_serial(y_norm, p_pred, int(n_bins))
    p = np.asarray(p_pred, dtype=np.float64)
    if p.ndim == 2 and p.shape[1] >= 2:
        p = p[:, 1]
    p = np.ascontiguousarray(p.ravel())
    # iter598: dropped the unconditional ``dtype=np.float64`` cast on
    # y_true (same pattern as iter595/596/597). The kernel only uses
    # ``yi`` in ``sum_y[b] += yi`` where sum_y is float64; mixed-dtype
    # numba dispatch widens at the accumulator, identical to the upfront
    # cast result. Bench n=100k: int64 1.40x, int8 1.27x, float64 0.99x
    # (no harm); n=25k int64 (bootstrap typical) 1.33x. Bit-equivalent.
    if p.size == 0:
        return float("nan")
    y = np.ascontiguousarray(_normalize_binary_labels(y_true))
    if y.size != p.size:
        return float("nan")
    return _ece_score_numba_serial(y, p, int(n_bins))


def _fit_calibrator(name: str, calib_p: np.ndarray, calib_y: np.ndarray) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Fit ``name`` on ``(calib_p, calib_y)``; return ``predict_proba_pos(p_test) -> p_test_calibrated`` or ``None``.

    Optional deps (betacal for Beta, ml_insights for Spline) are guarded; absent dep
    silently drops that candidate from the bench so the policy still runs with the
    remaining baseline (Sigmoid / Isotonic via sklearn).
    """
    p = np.asarray(calib_p, dtype=np.float64)
    if p.ndim == 2 and p.shape[1] >= 2:
        p = p[:, 1]
    p = p.reshape(-1, 1)
    y = np.asarray(calib_y).ravel()
    try:
        if name == "Sigmoid":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=1e10, solver="lbfgs")
            clf.fit(p, y)
            def _apply_sigmoid(probs: np.ndarray) -> np.ndarray:
                q = np.asarray(probs, dtype=np.float64)
                if q.ndim == 2 and q.shape[1] >= 2:
                    q = q[:, 1]
                return clf.predict_proba(q.reshape(-1, 1))[:, 1]
            return _apply_sigmoid
        if name == "Isotonic":
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(p.ravel(), y)
            def _apply_iso(probs: np.ndarray) -> np.ndarray:
                q = np.asarray(probs, dtype=np.float64)
                if q.ndim == 2 and q.shape[1] >= 2:
                    q = q[:, 1]
                return iso.predict(q.ravel())
            return _apply_iso
        if name == "Beta":
            try:
                from betacal import BetaCalibration
            except ImportError:
                logger.debug("pick_best_calibrator: betacal not installed; skipping Beta")
                return None
            beta = BetaCalibration(parameters="abm")
            beta.fit(p, y)
            def _apply_beta(probs: np.ndarray) -> np.ndarray:
                q = np.asarray(probs, dtype=np.float64)
                if q.ndim == 2 and q.shape[1] >= 2:
                    q = q[:, 1]
                out = beta.predict(q.reshape(-1, 1))
                return np.asarray(out).ravel()
            return _apply_beta
        if name == "Spline":
            try:
                import ml_insights as mli
            except ImportError:
                logger.debug("pick_best_calibrator: ml_insights not installed; skipping Spline")
                return None
            spline = mli.SplineCalib()
            spline.fit(p.ravel(), y)
            def _apply_spline(probs: np.ndarray) -> np.ndarray:
                q = np.asarray(probs, dtype=np.float64)
                if q.ndim == 2 and q.shape[1] >= 2:
                    q = q[:, 1]
                return np.asarray(spline.predict(q.ravel())).ravel()
            return _apply_spline
    except Exception as exc:
        logger.warning("pick_best_calibrator: %s fit failed: %s", name, exc)
        return None
    return None


def _build_resample_indices(
    n: int,
    n_bootstrap: int,
    stratify: Optional[np.ndarray],
    random_state: Optional[int],
) -> np.ndarray:
    """Build the (n_bootstrap, resample_len) resample-index matrix ONCE.

    Mirrors ``bootstrap_metric``'s RNG draw order EXACTLY (same seed, same
    per-class ``rng.integers`` order in unique-sorted class order, int64) so a
    candidate evaluated on these indices yields the bit-identical CI a fresh
    per-candidate ``bootstrap_metric`` call would. ``pick_best_calibrator`` reuses
    this single matrix across all calibrator candidates instead of regenerating
    the identical resample per candidate (every candidate shares n / stratify /
    seed; only y_pred differs).

    RAM ceiling: the matrix is ``n_bootstrap x resample_len`` **int32** ->
    ``4 * n_bootstrap * n`` bytes (~0.4 GB at n=200k, ~1 GB at n=500k for the
    1000-resample default). Resample indices live in ``[0, n)`` with n far below
    2**31, so int32 stores them exactly. Crucially, ``np.random.Generator.integers``
    consumes entropy based on the *range* [0, n), not the requested output dtype,
    so the int32 draws are BIT-IDENTICAL to the former int64 draws (verified) --
    the pinned bootstrap draw-order (which downstream ECE CIs depend on) is fully
    preserved while peak RAM is halved. This is a ONE-TIME shared allocation reused
    across ALL candidates, so it does not compound.

    A proactive size guard computes the projected ``4 * n_bootstrap * n`` bytes
    BEFORE any allocation and raises ``MemoryError`` (with the projected GiB +
    advice to lower ``n_bootstrap``) once it exceeds ``_resample_matrix_max_bytes()``
    (default ~1 GiB, override via ``MLFRAME_CALIBRATION_RESAMPLE_MAX_BYTES``), so an
    oversized request fails early and explicitly instead of OOM-ing the process.
    """
    projected_bytes = 4 * int(n_bootstrap) * int(n)
    ceiling = _resample_matrix_max_bytes()
    if projected_bytes > ceiling:
        raise MemoryError(
            f"_build_resample_indices: projected resample-index matrix is "
            f"{projected_bytes / (1 << 30):.2f} GiB (n_bootstrap={n_bootstrap} x n={n} x 4 bytes), "
            f"exceeding the {ceiling / (1 << 30):.2f} GiB ceiling. Lower n_bootstrap, or raise "
            f"MLFRAME_CALIBRATION_RESAMPLE_MAX_BYTES if the RAM is available."
        )
    rng = np.random.default_rng(random_state)
    if stratify is None:
        # int32 indices: exact for n < 2**31 and bit-identical draws to int64
        # (Generator.integers keys entropy off the range, not the dtype).
        out = np.empty((n_bootstrap, n), dtype=np.int32)
        for b in range(n_bootstrap):
            out[b] = rng.integers(0, n, size=n, dtype=np.int32)
        return out
    stratify = np.asarray(stratify).ravel()
    groups = {int(c): np.flatnonzero(stratify == c) for c in np.unique(stratify)}
    _groups_list = list(groups.values())
    _class_sizes = np.array([g.shape[0] for g in _groups_list], dtype=np.int64)
    _class_offsets = np.empty(_class_sizes.shape[0] + 1, dtype=np.int64)
    _class_offsets[0] = 0
    _class_offsets[1:] = np.cumsum(_class_sizes)
    _total_n = int(_class_sizes.sum())
    out = np.empty((n_bootstrap, _total_n), dtype=np.int32)
    # FUTURE: this stratified resample nests a Python loop over (n_bootstrap x n_classes) with a per-class
    # rng.integers + fancy-index gather. A fully vectorized rewrite (draw all per-class random offsets in one
    # rng.integers call shaped (n_bootstrap, _sz), gather without the per-bootstrap Python loop) is possible but
    # changes the rng draw ORDER -> different bootstrap indices -> not bit-identical to the current per-(b,c) draw
    # sequence, which downstream ECE bootstrap CIs are pinned to. Deferred: the win is a one-time resample-table
    # build (not a per-candidate hot path), and the identity risk on the seeded draw order is not worth it here.
    for b in range(n_bootstrap):
        for _c in range(_class_sizes.shape[0]):
            _sz = int(_class_sizes[_c])
            _rand = rng.integers(0, _sz, size=_sz, dtype=np.int64)
            out[b, _class_offsets[_c] : _class_offsets[_c + 1]] = _groups_list[_c][_rand]
    return out


def _bootstrap_ece_with_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx_matrix: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    alpha: float,
    n_bins: Optional[int] = None,
    method: str = "bca",
) -> dict[str, Any]:
    """Bootstrap CI for one candidate using a PRE-BUILT index matrix.

    Numerically identical to ``bootstrap_metric`` (same indices -> same per-resample metric -> same CI endpoints),
    but the resample matrix is generated once outside the candidate loop and shared, removing the per-candidate
    RNG + regen cost. The CI endpoints are reduced through the SAME ``_ci_from_samples`` machinery
    ``bootstrap_metric``/``bootstrap_metrics`` use, so the default bias-corrected accelerated (BCa) interval and its
    z0 / acceleration jackknife are honoured here too -- not the legacy raw-percentile interval.

    When ``n_bins`` is supplied the per-resample metric is the fused idx-aware ECE kernel
    ``_ece_score_idx_numba_serial``, which gathers ``y[idx]`` / ``p[idx]`` inside the njit bin loop -- removing the
    per-resample Python-level ``y_true[idx]`` / ``y_pred[idx]`` fancy-index copy entirely. Bit-identical to the
    slice-based ``metric_fn`` path (equal-width binning is order-independent). ``metric_fn`` still produces the point
    estimate AND the BCa acceleration jackknife (slice-based, exactly as ``bootstrap_metric`` does), so any
    caller-specific ECE config flows through unchanged.
    """
    point = float(metric_fn(y_true, y_pred))
    n_bootstrap = idx_matrix.shape[0]
    samples = np.empty(n_bootstrap, dtype=np.float64)
    valid = 0
    if n_bins is not None:
        yb = np.ascontiguousarray(np.asarray(y_true).ravel())
        pb = np.ascontiguousarray(np.asarray(y_pred, dtype=np.float64).ravel())
        nb = int(n_bins)
        for b in range(n_bootstrap):
            idx = idx_matrix[b]
            v = _ece_score_idx_numba_serial(yb, pb, idx, nb)
            if not np.isfinite(v):
                continue
            samples[valid] = v
            valid += 1
    else:
        for b in range(n_bootstrap):
            idx = idx_matrix[b]
            try:
                v = float(metric_fn(y_true[idx], y_pred[idx]))
            except Exception as exc:
                logger.debug("bootstrap-ECE: metric_fn on resample %d failed; skipping resample: %r", b, exc, exc_info=True)
                continue
            if not np.isfinite(v):
                continue
            samples[valid] = v
            valid += 1
    if valid == 0:
        raise ValueError("pick_best_calibrator: all resamples failed for a candidate")
    samples = samples[:valid]
    jackknife = _jackknife_metric(y_true, y_pred, metric_fn) if method == "bca" else None
    lo, hi = _ci_from_samples(samples, point, alpha, method, jackknife)
    return {"point": point, "lo": lo, "hi": hi}


def _cis_overlap(ci_a: tuple[float, float], ci_b: tuple[float, float]) -> bool:
    """True if two percentile CIs overlap (closed intervals)."""
    lo_a, hi_a = ci_a
    lo_b, hi_b = ci_b
    return not (hi_a < lo_b or hi_b < lo_a)


def _stratified_inner_folds(y: np.ndarray, n_splits: int, random_state: Optional[int]) -> list[np.ndarray]:
    """Return ``n_splits`` stratified held-out index arrays for BINARY ``y``.

    Each candidate is fitted on the complement of a fold and scored on the fold, so the reported ECE reflects
    generalisation to rows the calibrator never saw -- not the same-data interpolation that lets Isotonic drive
    its in-sample ECE to ~0.

    Contract: the calibration candidates are binary-only, so ``y`` MUST have exactly two distinct classes.
    A ``>2``-class input raises ``ValueError`` -- the previous code silently fell through to a GLOBAL (un-stratified)
    shuffle despite the ``_stratified_`` name, which could hand a fold a class the fitted calibrator never saw.
    """
    n = y.shape[0]
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError(
            f"_stratified_inner_folds: binary calibration requires exactly 2 classes; " f"got {classes.size} distinct value(s): {classes.tolist()[:10]}"
        )
    rng = np.random.default_rng(random_state)
    fold_of = np.empty(n, dtype=np.int64)
    for c in classes:
        members = np.flatnonzero(y == c)
        rng.shuffle(members)
        fold_of[members] = np.arange(members.shape[0]) % n_splits
    return [np.flatnonzero(fold_of == k) for k in range(n_splits)]


def _heldout_ece_inner_cv(
    name: str,
    oof_p_pos: np.ndarray,
    oof_y: np.ndarray,
    folds: Sequence[np.ndarray],
    n_bins: int,
) -> Optional[tuple[float, list[float]]]:
    """Mean held-out ECE for ``name`` plus the per-fold held-out ECEs: fit on each fold's complement, score on the
    held-out fold.

    Returns ``(mean_heldout_ece, per_fold_eces)`` -- the per-fold list lets the caller build a CI from the SAME
    held-out resampling that produced the point estimate, instead of an in-sample bootstrap CI that does not bracket
    the held-out number. Returns ``None`` when the candidate cannot be fitted (missing optional dep), so the caller
    drops it; a single failed fold is skipped and the remaining folds still average.
    """
    n = oof_y.shape[0]
    all_idx = np.arange(n)
    scores: list[float] = []
    for held in folds:
        if held.shape[0] < 1:
            continue
        train_mask = np.ones(n, dtype=bool)
        train_mask[held] = False
        train_idx = all_idx[train_mask]
        if train_idx.shape[0] < 2:
            continue
        apply_fn = _fit_calibrator(name, oof_p_pos[train_idx], oof_y[train_idx])
        if apply_fn is None:
            return None
        try:
            cal = np.clip(np.asarray(apply_fn(oof_p_pos[held]), dtype=np.float64).ravel(), 0.0, 1.0)
        except Exception as exc:
            logger.warning("pick_best_calibrator: inner-CV %s predict failed: %s", name, exc)
            continue
        v = _ece_score(oof_y[held], cal, n_bins=n_bins)
        if np.isfinite(v):
            scores.append(float(v))
    if not scores:
        return None
    return float(np.mean(scores)), scores


def _heldout_ece_ci(point: float, fold_eces: Sequence[float], alpha: float) -> tuple[float, float]:
    """CI for the mean held-out ECE from the per-fold held-out distribution.

    Built from the SAME held-out resampling as ``point`` so the interval brackets the reported number (the prior
    code paired this held-out point with an in-sample bootstrap CI that did not). With few folds we use a
    Student-t interval ``mean +- t_{k-1} * SE`` where ``SE = std(fold_eces, ddof=1)/sqrt(k)`` and ``t_{k-1}`` is
    the Student-t quantile with ``k-1`` degrees of freedom -- the correct small-sample quantile when the SE is
    itself estimated from the same k folds (the prior normal-z quantile understated the interval width at the
    typical k=5). A single fold has no spread so the CI degenerates to the point. The interval is centred on the
    per-fold mean (== ``point``), guaranteeing containment.
    """
    arr = np.asarray(list(fold_eces), dtype=np.float64)
    k = arr.size
    mean = float(np.mean(arr))
    if k < 2:
        return (mean, mean)
    from scipy.stats import t as _student_t

    se = float(np.std(arr, ddof=1)) / np.sqrt(k)
    tq = float(_student_t.ppf(1.0 - alpha / 2.0, k - 1))
    half = tq * se
    return (max(0.0, mean - half), mean + half)


def _emit_reliability_plot(
    candidates: Mapping[str, dict[str, Any]],
    oof_probs: np.ndarray,
    oof_y: np.ndarray,
    plot_path: str,
    n_bins: int = DEFAULT_ECE_NBINS,
) -> Optional[str]:
    """Render a reliability diagram for every candidate alongside the raw OOF curve.

    Routed through the shared ``build_reliability_overlay_spec`` + renderer pipeline
    (a multi-series LinePanelSpec: perfect diagonal + raw OOF + per-candidate curves)
    so the reliability diagram has ONE implementation across the suite. Returns the
    absolute path on success, ``None`` if the render dependency is missing or the
    write fails.
    """
    try:
        from mlframe.reporting.charts.calibration import build_reliability_overlay_spec
        from mlframe.reporting.output import parse_plot_output_dsl
        from mlframe.reporting.renderers import render_and_save
    except ImportError as exc:
        logger.warning("pick_best_calibrator: reporting stack unavailable; skipping reliability plot: %s", exc)
        return None
    try:
        os.makedirs(os.path.dirname(os.path.abspath(plot_path)) or ".", exist_ok=True)
    except OSError as exc:
        logger.warning("pick_best_calibrator: could not create plot dir for %s: %s", plot_path, exc)
        return None

    raw_p = np.asarray(oof_probs, dtype=np.float64)
    if raw_p.ndim == 2 and raw_p.shape[1] >= 2:
        raw_p = raw_p[:, 1]
    raw_p = raw_p.ravel()
    y = np.asarray(oof_y, dtype=np.float64).ravel()

    calibrated = {name: np.asarray(info["calibrated_probs"]).ravel() for name, info in candidates.items() if info.get("calibrated_probs") is not None}
    labels = {name: f"{name} ECE={info['ece_mean']:.4f}" for name, info in candidates.items() if info.get("calibrated_probs") is not None}

    spec = build_reliability_overlay_spec(
        raw_p, y, calibrated_probs=calibrated, series_labels=labels, n_bins=n_bins,
    )

    root, ext = os.path.splitext(plot_path)
    fmt = ext.lstrip(".").lower()
    if fmt not in ("png", "pdf", "svg", "jpg", "jpeg"):
        fmt = "png"
        plot_path = root + ".png"
    try:
        render_and_save(spec, parse_plot_output_dsl(f"matplotlib[{fmt}]"), root, interactive=False)
    except OSError as exc:
        logger.warning("pick_best_calibrator: reliability render failed for %s: %s", plot_path, exc)
        return None
    return os.path.abspath(plot_path)


def pick_best_calibrator(
    probs: Optional[np.ndarray],
    y: Optional[np.ndarray],
    oof_probs: np.ndarray,
    oof_y: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    candidates: Optional[Iterable[str]] = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    n_bins: int = DEFAULT_ECE_NBINS,
    random_state: Optional[int] = 0,
    selection: str = "inner_cv",
    inner_cv_splits: int = 5,
    emit_plot: bool = False,
    plot_path: Optional[str] = None,
) -> dict[str, Any]:
    """Pick the calibrator that minimises OOF ECE (with bootstrap CI tiebreak).

    Selection-optimism fix SHIPPED: the default is now ``selection="inner_cv"`` (see the ``selection`` param
    below), which ranks each candidate by HELD-OUT inner-CV ECE, so a flexible calibrator (Isotonic) can no
    longer interpolate its own reported ECE to ~0 and ``ece_mean`` is honest. The legacy
    ``selection="same_oof"`` path -- fit AND ECE-score every candidate on the SAME oof rows, which biased the
    reported ECE optimistic by ~0.006 and picked Isotonic ~100% of the time
    (_benchmarks/bench_pick_best_calibrator_selection_bias.py) -- is kept only for replay / A-B comparison.

    Parameters
    ----------
    probs, y
        Optional held-out probs / labels. When both are given, the CHOSEN calibrator's ECE on this
        slice is reported under ``secondary_ece`` in the result; this is diagnostic-only and never
        influences the selection decision (selection stays OOF-only to keep the estimate honest).
    oof_probs, oof_y
        OOF-train probs/labels — the calibrator fit + ECE benchmark surface. Required.
    alpha
        Two-sided coverage; default 0.05 -> 95% percentile CI.
    candidates
        Iterable of calibrator names to try; defaults to ``CANDIDATE_NAMES``. Unknown
        names are skipped with a warning.
    n_bootstrap
        Resample count for the OOF ECE CI; ``DEFAULT_N_BOOTSTRAP=1000``.
    n_bins
        ECE bin count; ``DEFAULT_ECE_NBINS=10`` minimises RMSE vs ground-truth ECE.
    random_state
        Seed for the bootstrap RNG. Pin for reproducibility.
    emit_plot
        When True, render a reliability diagram for every candidate to ``plot_path``.
    plot_path
        Output path; when ``emit_plot=True`` and ``plot_path is None``, defaults to
        ``reports/calibration_<utc_ts>.png`` in the working directory.

    Returns
    -------
    dict
        ``{"chosen": <name>, "ece_mean": ..., "ece_ci": (lo, hi),
           "alternatives": {<name>: {"ece_mean", "ece_ci"}}, "rule": <selection-rule>,
           "n_oof": int, "plot_path": Optional[str],
           "secondary_ece": Optional[float]}`` (``secondary_ece`` is None unless ``probs``/``y`` given).

    selection
        ``"inner_cv"`` (default) ranks candidates by HELD-OUT ECE -- each candidate is fitted on
        ``inner_cv_splits``-1 inner folds and scored on the held-out fold, averaged -- so a flexible
        calibrator (Isotonic) cannot interpolate its own score to ~0; the chosen calibrator is then refit
        on the full OOF for deployment and the reported ``ece_mean`` is the honest held-out estimate.
        ``"same_oof"`` is the legacy path that fits AND scores every candidate on the same OOF rows
        (optimistic by ~0.006 ECE, Isotonic-biased); kept for replay / A-B comparison.
    inner_cv_splits
        Inner-CV fold count for ``selection="inner_cv"``; default 5.

    Selection rule
    --------------
    For ``selection="inner_cv"`` (default):
    1. Build ``inner_cv_splits`` stratified inner folds of the OOF.
    2. For each candidate, fit on the fold complement, score ECE on the held-out fold, average.
    3. Pick the lowest held-out-ECE candidate; refit it on the FULL OOF for deployment.
    4. Report the honest held-out ECE as ``ece_mean`` (no longer ~0 for Isotonic).

    For ``selection="same_oof"`` (legacy):
    1. Bench every candidate; compute OOF ECE + bootstrap CI.
    2. Sort by ECE mean ascending.
    3. If the top candidate's CI does NOT overlap the runner-up's, return top.
    4. Otherwise apply Kull-2017 default: Isotonic when ``n_oof >= 1000``, Beta when
       ``n_oof < 1000``; if the default isn't in the OOF-tied subset, fall through
       to the lowest-ECE candidate.
    """
    if selection not in ("inner_cv", "same_oof"):
        raise ValueError(f"pick_best_calibrator: selection must be 'inner_cv' or 'same_oof'; got {selection!r}")
    if selection == "same_oof":
        # same_oof fits AND ECE-scores every candidate on the SAME OOF rows: a flexible calibrator
        # (Isotonic) interpolates its in-sample ECE toward ~0 and is selected ~100% of the time, with a
        # reported ECE optimistic by ~0.006 vs fresh data. Prefer selection="inner_cv" for an honest
        # held-out estimate; same_oof is kept only for replay / A-B against the legacy path.
        logger.warning(
            "pick_best_calibrator: selection='same_oof' fits and scores candidates on the same OOF rows "
            "-- the reported ECE is optimistically biased (Isotonic-favouring). Use selection='inner_cv' "
            "for an honest held-out estimate."
        )
    oof_p = np.asarray(oof_probs, dtype=np.float64)
    if oof_p.ndim == 2 and oof_p.shape[1] >= 2:
        oof_p_pos = oof_p[:, 1]
    else:
        oof_p_pos = oof_p.ravel()
    oof_y_arr = np.asarray(oof_y).ravel()
    n_oof = int(oof_y_arr.shape[0])
    if oof_p_pos.shape[0] != n_oof:
        raise ValueError(f"pick_best_calibrator: oof_probs rows ({oof_p_pos.shape[0]}) do not match oof_y ({n_oof})")
    if n_oof < 4:
        raise ValueError(f"pick_best_calibrator: need at least 4 OOF rows; got n_oof={n_oof}")

    cand_names = tuple(candidates) if candidates is not None else CANDIDATE_NAMES
    unknown = [c for c in cand_names if c not in CANDIDATE_NAMES]
    if unknown:
        logger.warning("pick_best_calibrator: unknown candidate(s) ignored: %s", unknown)
    cand_names = tuple(c for c in cand_names if c in CANDIDATE_NAMES)
    if not cand_names:
        raise ValueError(f"pick_best_calibrator: no valid candidate names; allowed={CANDIDATE_NAMES}")

    results: dict[str, dict[str, Any]] = {}
    classes = np.unique(oof_y_arr)
    stratify = oof_y_arr if classes.size == 2 else None

    metric_fn = lambda _y, _p, _nb=n_bins: _ece_score(_y, _p, n_bins=_nb)

    # Build the stratified resample-index matrix ONCE: every candidate shares the
    # same n / stratify / seed and only differs in calibrated y_pred, so the
    # per-candidate ``bootstrap_metric`` call previously regenerated the identical
    # resample. One matrix reused across candidates -> same indices -> bit-identical
    # CIs at a fraction of the cost. Indices mirror bootstrap_metric's RNG order.
    idx_matrix = _build_resample_indices(n_oof, n_bootstrap, stratify, random_state)

    inner_folds: Optional[list[np.ndarray]] = None
    if selection == "inner_cv":
        inner_folds = _stratified_inner_folds(oof_y_arr, max(2, int(inner_cv_splits)), random_state)

    for name in cand_names:
        apply_fn = _fit_calibrator(name, oof_p_pos, oof_y_arr)
        if apply_fn is None:
            continue
        try:
            cal_oof = np.asarray(apply_fn(oof_p_pos), dtype=np.float64).ravel()
            cal_oof = np.clip(cal_oof, 0.0, 1.0)
        except Exception as exc:
            logger.warning("pick_best_calibrator: %s.predict on OOF failed: %s", name, exc)
            continue
        try:
            ci = _bootstrap_ece_with_indices(oof_y_arr, cal_oof, idx_matrix, metric_fn, alpha, n_bins=n_bins)
        except Exception as exc:
            logger.warning("pick_best_calibrator: bootstrap on %s failed: %s", name, exc)
            continue
        # ``rank_ece`` drives selection: held-out (honest) for inner_cv, same-OOF (legacy) otherwise.
        rank_ece = float(ci["point"])
        ece_ci = (float(ci["lo"]), float(ci["hi"]))
        if inner_folds is not None:
            ho = _heldout_ece_inner_cv(name, oof_p_pos, oof_y_arr, inner_folds, n_bins)
            if ho is None:
                continue
            rank_ece, fold_eces = ho
            # CI from the SAME held-out resampling as the point estimate, so the reported interval brackets the
            # reported number. The in-sample bootstrap CI above was paired with a held-out point and did not.
            ece_ci = _heldout_ece_ci(rank_ece, fold_eces, alpha)
        results[name] = {
            "ece_mean": rank_ece,
            "ece_ci": ece_ci,
            "calibrated_probs": cal_oof,
        }

    if not results:
        raise RuntimeError("pick_best_calibrator: every candidate calibrator failed; check optional deps " "(betacal, ml_insights) and OOF input shape.")

    ranked = sorted(results.items(), key=lambda kv: kv[1]["ece_mean"])
    chosen_name = ranked[0][0]
    selection_rule = "lowest_ece"
    if selection == "inner_cv":
        # Held-out ECE already removes the same-data optimism, so the lowest held-out ECE is the honest
        # winner; the same-OOF bootstrap CI (still reported) would mis-tie Isotonic, so no CI tiebreak here.
        selection_rule = "lowest_heldout_ece"
    elif len(ranked) > 1:
        top_ci = ranked[0][1]["ece_ci"]
        runner_ci = ranked[1][1]["ece_ci"]
        if _cis_overlap(top_ci, runner_ci):
            default_choice = "Isotonic" if n_oof >= SMALL_SAMPLE_THRESHOLD else "Beta"
            tied = [name for name, info in ranked if _cis_overlap(top_ci, info["ece_ci"])]
            if default_choice in tied:
                chosen_name = default_choice
                selection_rule = "default_isotonic" if default_choice == "Isotonic" else "default_beta"
            else:
                # Default candidate isn't in the tied subset; fall back to the lowest-mean.
                selection_rule = "lowest_ece_ci_overlap"
        else:
            selection_rule = "lowest_ece_ci_separated"

    # Secondary diagnostic: if a held-out ``probs``/``y`` pair is supplied, score the CHOSEN calibrator's ECE on it.
    # This never influences the selection above (which stays OOF-only for honesty); it is a side report so a caller can
    # see whether the OOF-picked calibrator generalises to a separate held-out slice.
    secondary_ece: Optional[float] = None
    if probs is not None and y is not None:
        try:
            sec_p = np.asarray(probs, dtype=np.float64)
            if sec_p.ndim == 2 and sec_p.shape[1] >= 2:
                sec_p = sec_p[:, 1]
            sec_p = sec_p.ravel()
            sec_y = np.asarray(y).ravel()
            if sec_p.shape[0] != sec_y.shape[0]:
                raise ValueError(f"probs rows ({sec_p.shape[0]}) do not match y ({sec_y.shape[0]})")
            chosen_apply = _fit_calibrator(chosen_name, oof_p_pos, oof_y_arr)
            if chosen_apply is not None:
                cal_sec = np.clip(np.asarray(chosen_apply(sec_p), dtype=np.float64).ravel(), 0.0, 1.0)
                secondary_ece = float(_ece_score(sec_y, cal_sec, n_bins=n_bins))
        except Exception as exc:
            logger.warning("pick_best_calibrator: secondary-ECE diagnostic failed: %s", exc)

    plot_out: Optional[str] = None
    if emit_plot:
        if plot_path is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            plot_path = os.path.join("reports", f"calibration_{ts}.png")
        plot_out = _emit_reliability_plot(results, oof_p_pos, oof_y_arr, plot_path, n_bins=n_bins)

    return {
        "chosen": chosen_name,
        "ece_mean": float(results[chosen_name]["ece_mean"]),
        "ece_ci": tuple(results[chosen_name]["ece_ci"]),
        "alternatives": {name: {"ece_mean": info["ece_mean"], "ece_ci": info["ece_ci"]} for name, info in results.items()},
        "rule": selection_rule,
        "n_oof": n_oof,
        "plot_path": plot_out,
        "secondary_ece": secondary_ece,
    }


@dataclass
class CalibrationConfig:
    """Calibration-policy knobs for ``post_calibrate_model``.

    Currently a single field; kept as a dataclass so further policy knobs (ECE bin
    count override, candidate set override, plot emission toggle) can be added
    without breaking call sites.

    Parameters
    ----------
    policy_auto_pick
        When True (default), ``post_calibrate_model`` invokes
        :func:`pick_best_calibrator` on the OOF probs and uses its decision in
        addition to (not in place of) the legacy meta-model path. The chosen
        calibrator + CI is stamped into the metrics dict under
        ``metadata["calibration_policy"]`` for downstream consumers (honest
        diagnostics report, ops dashboards).
    emit_plot
        When True, the reliability plot is rendered to ``plot_path``.
    plot_path
        Optional explicit path for the reliability plot. ``None`` = auto-generate
        ``reports/calibration_<utc_ts>.png``.
    n_bootstrap
        Bootstrap resample count for the OOF ECE CI (default 1000).
    alpha
        CI coverage; 0.05 -> 95% CI.
    candidates
        Restricted candidate set; ``None`` = ``CANDIDATE_NAMES``.
    """

    policy_auto_pick: bool = True
    emit_plot: bool = False
    plot_path: Optional[str] = None
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP
    alpha: float = DEFAULT_ALPHA
    candidates: Optional[tuple[str, ...]] = None
    selection: str = "inner_cv"
    inner_cv_splits: int = 5


__all__ = ["pick_best_calibrator", "CalibrationConfig", "CANDIDATE_NAMES", "DEFAULT_ECE_NBINS"]
