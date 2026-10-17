"""Single-metric resample helpers carved out of ``bootstrap_metric`` / ``bootstrap_metrics`` (``bootstrap.py``)
to keep those functions under the function-length gate and ``bootstrap.py`` under the file-size budget."""
from __future__ import annotations

import logging
import math
from typing import Callable, Optional

import numpy as np

from ._bootstrap_jackknife import _jackknife_mean_metric, _jackknife_metric

logger = logging.getLogger("mlframe.evaluation.bootstrap")

_isfinite = math.isfinite


def _setup_per_row_resample(jackknife_per_row: Optional[tuple], y_true: np.ndarray, y_pred: np.ndarray, n: int) -> Optional[tuple]:
    """Precompute the per-row array for the ``reduce_fn(mean(per_row))`` resample fast path (``None`` when it does not apply)."""
    # Per-row resample fast path for a metric declared as reduce_fn(mean(per_row)): every resample metric is then
    # reduce_fn(mean(per_row[idx])) -- ONE gather of the precomputed per-row array + a mean, instead of gathering BOTH
    # (y[idx], p[idx]) and recomputing the metric kernel per resample (3.3x on the RMSE resample loop at n=300k, ~1e-16
    # sum-reduction-order equivalent). Gated to requires_both_classes=False metrics (RMSE / Brier): a both-classes
    # metric (log-loss) can go NaN on a single-class resample, and only the exact metric_fn reproduces that skip, so it
    # keeps the generic path. Disabled on non-finite per-row (out-of-range inputs) -> exact metric_fn.
    _pr_resample = None
    if jackknife_per_row is not None:
        _prf0, _both0, _reduce0 = jackknife_per_row
        if not _both0:
            try:
                _prv0 = np.asarray(_prf0(y_true, y_pred), dtype=np.float64).ravel()
                if _prv0.shape[0] == n and np.all(np.isfinite(_prv0)):
                    _pr_resample = (_prv0, _reduce0)
            except Exception as _pr_exc:
                logger.debug("bootstrap_metric per-row resample setup failed (%r); using metric_fn.", _pr_exc)
    return _pr_resample


def _stratified_class_book(stratify: Optional[np.ndarray], n: int, fn_name: str) -> Optional[tuple]:
    """Per-class index book ``(groups_list, class_sizes, class_offsets, total_n)`` for stratified resampling (``None`` when unstratified)."""
    if stratify is None:
        return None
    stratify = np.asarray(stratify).ravel()
    if stratify.shape[0] != n:
        raise ValueError(f"{fn_name}: stratify length {stratify.shape[0]} must match y_true length {n}")
    groups = {int(c): np.flatnonzero(stratify == c) for c in np.unique(stratify)}
    _groups_list = list(groups.values())
    _class_sizes = np.array([g.shape[0] for g in _groups_list], dtype=np.int64)
    _total_n = int(_class_sizes.sum())
    _class_offsets = np.empty(_class_sizes.shape[0] + 1, dtype=np.int64)
    _class_offsets[0] = 0
    _class_offsets[1:] = np.cumsum(_class_sizes)
    return _groups_list, _class_sizes, _class_offsets, _total_n


def _resample_metric_samples(
    rng: np.random.Generator,
    n_bootstrap: int,
    n: int,
    strat_book: Optional[tuple],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    _pr_resample: Optional[tuple],
) -> np.ndarray:
    """Run the ``n_bootstrap`` resamples of a single metric and return the surviving (finite) samples."""
    if strat_book is not None:
        # pre-extract list+offsets once and reuse a
        # single idx buffer across all n_bootstrap iters. The listcomp +
        # np.concatenate version of this loop on c0144 1M-row binary
        # measured 8.49s tottime / 12000 calls (708us per resample) on
        # n=100k stratified resamples; the direct buffer writes drop the
        # per-iter listcomp + concat allocation + Python frame setup. RNG
        # draw order per iter is unchanged (still rng.integers per class
        # in dict-iteration order) so bit-identical reproducibility for
        # the same random_state.
        _groups_list, _class_sizes, _class_offsets, _total_n = strat_book
        _idx_buf = np.empty(_total_n, dtype=np.int64)

    samples = np.empty(n_bootstrap, dtype=np.float64)
    valid = 0
    failures = 0
    first_err: Optional[str] = None
    for _ in range(n_bootstrap):
        if strat_book is None:
            # iter464: dtype=np.int64 explicit (same 1.16x shape-validation
            # shortcut as the stratified path, iter451). The unstratified
            # path is the regression-bootstrap default (RMSE etc); this
            # fires n_bootstrap times per metric.
            idx = rng.integers(0, n, size=n, dtype=np.int64)
        else:
            # Per-class resample preserves the original class frequencies.
            # use rng.integers + index instead of
            # rng.choice(replace=True). c0091/c0141 profile showed the
            # listcomp at ~180us per call x 24000 calls = ~4.3s wall.
            # rng.integers(0, len(grp), size=len(grp)) + grp[idx] runs
            # 1.72x faster on n_class=10k (0.153ms -> 0.089ms) -- same
            # statistical semantics (uniform with-replacement resample),
            # rng.choice just has extra options-dispatch overhead.
            # bench-attempt-rejected (2026-05-27, iter417): @njit kernel
            # with per-element np.random.randint inner loop ran 0.31x
            # (570us -> 1820us / call at n=100k, 2 classes). Numpy's
            # vectorised rng.integers(0, sz, size=sz) outperforms numba's
            # per-element randint by ~6x. Don't re-try a numba per-class
            # rewrite -- the only viable path is batching all classes'
            # randoms across all bootstrap iters via ONE big rng.integers
            # call, which conflicts with the bit-identical reproducibility
            # contract for the unstratified path's RNG draw order.
            # pass dtype=np.int64 explicitly. Saves
            # 16% on the rng.integers call (n=99000 x 1000 iter: 400ms
            # -> 345ms) -- numpy skips a small piece of shape-inference
            # dispatch when the output dtype is already specified. Same
            # int64 dtype the index buffer holds (np.int64 from cumsum
            # above), so no downstream cast either.
            for _c in range(_class_sizes.shape[0]):
                _sz = int(_class_sizes[_c])
                _rand = rng.integers(0, _sz, size=_sz, dtype=np.int64)
                _idx_buf[_class_offsets[_c] : _class_offsets[_c + 1]] = _groups_list[_c][_rand]
            idx = _idx_buf  # type: ignore[assignment, unused-ignore]  # numpy stubs mis-infer rng.integers(..., size=n) as scalar-returning above; idx is always an ndarray at runtime
        # bench-attempt-rejected (2026-05-27, iter392): replacing
        # ``y_true[idx], y_pred[idx]`` with pre-allocated buffers via
        # ``np.take(y_true, idx, out=y_buf)`` ran 0.88x SLOWER on n=100k
        # float64 (1226us -> 1398us / pair). Fancy indexing's internal
        # copy is already at the C-level memcpy floor; np.take adds an
        # extra dispatch with no compensating allocation saving. Leave
        # the idiomatic form -- next agent should not re-try this.
        if _pr_resample is not None:
            _prv, _red = _pr_resample
            _mean = float(_prv[idx].mean())
            v = float(_red(_mean)) if _red is not None else _mean
        else:
            try:
                v = float(metric_fn(y_true[idx], y_pred[idx]))
            except Exception as exc:
                failures += 1
                if first_err is None:
                    first_err = f"{type(exc).__name__}: {exc}"
                logger.debug("bootstrap resample failed: %r", exc, exc_info=True)
                continue
        if not _isfinite(v):
            failures += 1
            continue
        samples[valid] = v
        valid += 1

    if valid == 0:
        raise ValueError(f"bootstrap_metric: all {n_bootstrap} resamples failed (first error: {first_err}). " "CI cannot be computed.")
    samples = samples[:valid]
    if failures > n_bootstrap // 4:
        logger.warning(
            "bootstrap_metric: %d/%d resamples failed (first: %s); CI computed over %d surviving samples may be biased.",
            failures, n_bootstrap, first_err, valid,
        )
    return samples


def _single_metric_jackknife(
    method: str,
    jackknife_per_row: Optional[tuple],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
) -> Optional[np.ndarray]:
    """Leave-one-out jackknife values for the BCa interval of one metric (``None`` for the percentile method)."""
    jackknife = None
    if method == "bca":
        # Fast algebraic O(n) jackknife when the caller declares the metric as reduce_fn(mean(per_row)) (log-loss /
        # Brier / RMSE): 517x over the generic gather jackknife at n=300k, ~1e-13 CI-equivalent. Falls back to the
        # generic gather path on any degeneracy (non-finite per-row) so the exact slow-path contract is preserved.
        if jackknife_per_row is not None:
            _prf, _both, _reduce = jackknife_per_row
            try:
                jackknife = _jackknife_mean_metric(
                    y_true, _prf(y_true, y_pred), requires_both_classes=_both, reduce_fn=_reduce,
                )
            except Exception as _jk_exc:
                logger.debug("bootstrap_metric fast jackknife failed (%r); using gather jackknife.", _jk_exc)
        if jackknife is None:
            jackknife = _jackknife_metric(y_true, y_pred, metric_fn)
    return jackknife
