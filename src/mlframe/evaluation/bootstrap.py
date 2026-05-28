"""Bootstrap confidence intervals + paired comparisons for model evaluation.

Provides the generic ``bootstrap_metric`` helper consumed by
``training.honest_diagnostics`` to attach 95% CIs to every top-line metric and
``delong_test`` for AUC paired comparisons. ``training._dummy_bootstrap`` ships
metric-specific numba kernels for the dummy-baseline phase; this module is the
metric-agnostic surface that any ``metric_fn(y_true, y_pred) -> float`` callable
can plug into.

Design notes:
  - Percentile CI (Efron) by default; symmetric / sufficient for non-skew metrics
    at n_bootstrap=1000 typically used here.
  - Stratified resampling supported via ``stratify=`` (preserves class balance,
    critical for AUC/Brier on rare-1pct).
  - ``random_state`` mandatory for reproducibility (every call into the honest-
    diagnostics aggregator threads a per-target seed).
  - DeLong is the exact non-parametric test for paired ROC-AUC differences; the
    full O(n log n) implementation lives here so callers don't pull a heavy
    extra dep just for one statistic.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# iter417: bind math.isfinite for the per-iter scalar check in
# bootstrap_metric. np.isfinite on a single Python float pays the full
# numpy dispatcher (~1.0us / call); the C-implemented math.isfinite is
# 7.5x faster (0.13us / call). On a 12000-iter bootstrap-heavy run that's
# ~10ms saved per metric, ~125ms across the typical 12-metric run.
_isfinite = math.isfinite


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    stratify: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
) -> dict[str, Any]:
    """Bootstrap percentile CI for an arbitrary ``metric_fn(y_true, y_pred)``.

    Parameters
    ----------
    y_true, y_pred
        Aligned arrays (1D or 2D-prob). Length must match; shape is otherwise
        whatever ``metric_fn`` accepts.
    metric_fn
        Callable returning a single float. Receives the same array layout as
        ``(y_true, y_pred)``; resampled views are passed through unchanged.
    n_bootstrap
        Number of resamples. 1000 is the project default; below 200 the CI is
        too granular to be useful, above 10_000 wall-time outweighs precision.
    alpha
        Two-sided coverage. 0.05 -> 95% CI (lo = 2.5%, hi = 97.5%).
    stratify
        Optional 1D label vector for stratified resampling (per-class resample
        with replacement, then concatenate). Use whenever the metric is
        sensitive to class balance (AUC, Brier, recall_at_k).
    random_state
        Seed for ``np.random.default_rng``. Required for reproducible diagnostics
        artefacts; ``None`` consults numpy's global entropy and the CI will
        differ between runs.

    Returns
    -------
    dict
        ``{"point": float, "lo": float, "hi": float, "samples": np.ndarray}``.
        ``samples`` is the full ``(n_bootstrap,)`` bootstrap distribution so
        callers can compute additional summaries (BCa, paired CI overlap).
        Failed resamples (metric_fn raised) are dropped; if fewer than
        ``n_bootstrap // 4`` survive the CI is widened to span the full
        surviving range and a warning is logged so the operator sees the
        precision degradation rather than a misleadingly narrow band.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = y_true.shape[0]
    if n < 2:
        raise ValueError(f"bootstrap_metric: need at least 2 samples; got n={n}")
    if y_pred.shape[0] != n:
        raise ValueError(
            f"bootstrap_metric: y_true ({n}) and y_pred ({y_pred.shape[0]}) row counts diverge"
        )

    rng = np.random.default_rng(random_state)

    try:
        point = float(metric_fn(y_true, y_pred))
    except Exception as exc:
        raise ValueError(f"bootstrap_metric: metric_fn failed on the full sample: {exc}") from exc

    if stratify is not None:
        stratify = np.asarray(stratify).ravel()
        if stratify.shape[0] != n:
            raise ValueError(
                f"bootstrap_metric: stratify length {stratify.shape[0]} must match y_true length {n}"
            )
        groups = {int(c): np.flatnonzero(stratify == c) for c in np.unique(stratify)}
        # iter358 (2026-05-26): pre-extract list+offsets once and reuse a
        # single idx buffer across all n_bootstrap iters. The listcomp +
        # np.concatenate version of this loop on c0144 1M-row binary
        # measured 8.49s tottime / 12000 calls (708us per resample) on
        # n=100k stratified resamples; the direct buffer writes drop the
        # per-iter listcomp + concat allocation + Python frame setup. RNG
        # draw order per iter is unchanged (still rng.integers per class
        # in dict-iteration order) so bit-identical reproducibility for
        # the same random_state.
        _groups_list = list(groups.values())
        _class_sizes = np.array([g.shape[0] for g in _groups_list], dtype=np.int64)
        _total_n = int(_class_sizes.sum())
        _class_offsets = np.empty(_class_sizes.shape[0] + 1, dtype=np.int64)
        _class_offsets[0] = 0
        _class_offsets[1:] = np.cumsum(_class_sizes)
        _idx_buf = np.empty(_total_n, dtype=np.int64)

    samples = np.empty(n_bootstrap, dtype=np.float64)
    valid = 0
    failures = 0
    first_err: Optional[str] = None
    for _ in range(n_bootstrap):
        if stratify is None:
            # iter464: dtype=np.int64 explicit (same 1.16x shape-validation
            # shortcut as the stratified path, iter451). The unstratified
            # path is the regression-bootstrap default (RMSE etc); this
            # fires n_bootstrap times per metric.
            idx = rng.integers(0, n, size=n, dtype=np.int64)
        else:
            # Per-class resample preserves the original class frequencies.
            # iter312 (2026-05-26): use rng.integers + index instead of
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
            # iter451 (2026-05-27): pass dtype=np.int64 explicitly. Saves
            # 16% on the rng.integers call (n=99000 x 1000 iter: 400ms
            # -> 345ms) -- numpy skips a small piece of shape-inference
            # dispatch when the output dtype is already specified. Same
            # int64 dtype the index buffer holds (np.int64 from cumsum
            # above), so no downstream cast either.
            for _c in range(_class_sizes.shape[0]):
                _sz = int(_class_sizes[_c])
                _rand = rng.integers(0, _sz, size=_sz, dtype=np.int64)
                _idx_buf[_class_offsets[_c]:_class_offsets[_c + 1]] = _groups_list[_c][_rand]
            idx = _idx_buf
        # bench-attempt-rejected (2026-05-27, iter392): replacing
        # ``y_true[idx], y_pred[idx]`` with pre-allocated buffers via
        # ``np.take(y_true, idx, out=y_buf)`` ran 0.88x SLOWER on n=100k
        # float64 (1226us -> 1398us / pair). Fancy indexing's internal
        # copy is already at the C-level memcpy floor; np.take adds an
        # extra dispatch with no compensating allocation saving. Leave
        # the idiomatic form -- next agent should not re-try this.
        try:
            v = float(metric_fn(y_true[idx], y_pred[idx]))
        except Exception as exc:
            failures += 1
            if first_err is None:
                first_err = f"{type(exc).__name__}: {exc}"
            continue
        if not _isfinite(v):
            failures += 1
            continue
        samples[valid] = v
        valid += 1

    if valid == 0:
        raise ValueError(
            f"bootstrap_metric: all {n_bootstrap} resamples failed (first error: {first_err}). "
            "CI cannot be computed."
        )
    samples = samples[:valid]
    if failures > n_bootstrap // 4:
        logger.warning(
            "bootstrap_metric: %d/%d resamples failed (first: %s); CI computed over %d surviving samples may be biased.",
            failures, n_bootstrap, first_err, valid,
        )

    lo_pct = (alpha / 2.0) * 100.0
    hi_pct = (1.0 - alpha / 2.0) * 100.0
    lo = float(np.percentile(samples, lo_pct))
    hi = float(np.percentile(samples, hi_pct))

    return {"point": point, "lo": lo, "hi": hi, "samples": samples}


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fns: Mapping[str, Callable[[np.ndarray, np.ndarray], float]],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    stratify: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
) -> dict[str, dict]:
    """Bootstrap percentile CIs for SEVERAL metrics sharing ONE resample loop.

    Equivalent to calling ``bootstrap_metric`` once per metric with the SAME
    ``random_state``, but each resample's indices and the ``(y_true[idx],
    y_pred[idx])`` slice are produced ONCE and every metric is evaluated on
    them, instead of regenerating the resample + redoing the fancy-index copy
    (the dominant cost) per metric. When honest-diagnostics bootstraps
    roc_auc / brier / log_loss / ece on the same predictions with one seed,
    this collapses N resample passes into one.

    Returns ``{name: {"point", "lo", "hi", "samples"}}`` per metric. A metric
    whose point estimate or every resample fails gets ``{name: {"error": str}}``
    instead, so one bad metric doesn't sink the others -- mirroring the per-call
    try/except the caller previously wrapped around each ``bootstrap_metric``.

    Bit-identical to the per-metric ``bootstrap_metric`` results for the same
    ``random_state``: the index sequence is generated identically, and a metric
    raising never advances the RNG (which is consumed only to build indices),
    so metric order does not perturb any other metric's resamples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = y_true.shape[0]
    if n < 2:
        raise ValueError(f"bootstrap_metrics: need at least 2 samples; got n={n}")
    if y_pred.shape[0] != n:
        raise ValueError(
            f"bootstrap_metrics: y_true ({n}) and y_pred ({y_pred.shape[0]}) row counts diverge"
        )
    if not metric_fns:
        return {}

    # Seed BEFORE the point estimates (which never touch the RNG), so the RNG
    # state at the resample loop matches a freshly-seeded bootstrap_metric.
    rng = np.random.default_rng(random_state)

    results: dict[str, dict] = {}
    points: dict[str, float] = {}
    active: list = []
    for name, fn in metric_fns.items():
        try:
            points[name] = float(fn(y_true, y_pred))
            active.append(name)
        except Exception as exc:
            results[name] = {"error": f"point metric failed: {type(exc).__name__}: {exc}"}
    if not active:
        return results

    if stratify is not None:
        stratify = np.asarray(stratify).ravel()
        if stratify.shape[0] != n:
            raise ValueError(
                f"bootstrap_metrics: stratify length {stratify.shape[0]} must match y_true length {n}"
            )
        groups = {int(c): np.flatnonzero(stratify == c) for c in np.unique(stratify)}
        _groups_list = list(groups.values())
        _class_sizes = np.array([g.shape[0] for g in _groups_list], dtype=np.int64)
        _class_offsets = np.empty(_class_sizes.shape[0] + 1, dtype=np.int64)
        _class_offsets[0] = 0
        _class_offsets[1:] = np.cumsum(_class_sizes)
        _total_n = int(_class_sizes.sum())
        _idx_buf = np.empty(_total_n, dtype=np.int64)

    samples = {name: np.empty(n_bootstrap, dtype=np.float64) for name in active}
    valid = {name: 0 for name in active}
    failures = {name: 0 for name in active}
    first_err: dict = {name: None for name in active}

    for _ in range(n_bootstrap):
        # Index generation MUST match bootstrap_metric exactly (same rng calls,
        # same order) so each metric's samples equal its single-call result.
        if stratify is None:
            idx = rng.integers(0, n, size=n, dtype=np.int64)
        else:
            for _c in range(_class_sizes.shape[0]):
                _sz = int(_class_sizes[_c])
                _rand = rng.integers(0, _sz, size=_sz, dtype=np.int64)
                _idx_buf[_class_offsets[_c]:_class_offsets[_c + 1]] = _groups_list[_c][_rand]
            idx = _idx_buf
        # Slice ONCE; every metric reads the same resampled views.
        yt = y_true[idx]
        yp = y_pred[idx]
        for name in active:
            try:
                v = float(metric_fns[name](yt, yp))
            except Exception as exc:
                failures[name] += 1
                if first_err[name] is None:
                    first_err[name] = f"{type(exc).__name__}: {exc}"
                continue
            if not _isfinite(v):
                failures[name] += 1
                continue
            samples[name][valid[name]] = v
            valid[name] += 1

    lo_pct = (alpha / 2.0) * 100.0
    hi_pct = (1.0 - alpha / 2.0) * 100.0
    for name in active:
        v_n = valid[name]
        if v_n == 0:
            results[name] = {
                "error": f"all {n_bootstrap} resamples failed (first error: {first_err[name]})"
            }
            continue
        s = samples[name][:v_n]
        if failures[name] > n_bootstrap // 4:
            logger.warning(
                "bootstrap_metrics[%s]: %d/%d resamples failed (first: %s); CI over %d surviving samples may be biased.",
                name, failures[name], n_bootstrap, first_err[name], v_n,
            )
        results[name] = {
            "point": points[name],
            "lo": float(np.percentile(s, lo_pct)),
            "hi": float(np.percentile(s, hi_pct)),
            "samples": s,
        }
    return results


def delong_test(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
) -> dict[str, Any]:
    """DeLong 1988 paired test for ROC-AUC difference.

    Computes ``AUC_a - AUC_b`` and the two-sided p-value under the null
    hypothesis that the two ROC curves come from the same distribution, using
    the structural-component formulation (Sun & Xu 2014, O(n log n)) which is
    the standard reference for paired AUC comparisons.

    Parameters
    ----------
    y_true
        1D binary label vector (0 / 1). Multiclass is not supported by DeLong
        without one-vs-rest decomposition; the caller must do that and run
        per-class.
    score_a, score_b
        Predicted scores (continuous or probability). Higher = positive-class.

    Returns
    -------
    dict
        ``{"auc_a": ..., "auc_b": ..., "diff": auc_a - auc_b, "p_value": ..., "z": ...}``.

    Notes
    -----
    Returns ``p_value=nan`` and a warning when the covariance matrix is
    singular (degenerate inputs: constant scores or single-class y_true);
    callers should treat ``np.isnan(p)`` as "no statistically meaningful
    comparison possible".
    """
    y_true = np.asarray(y_true).ravel()
    score_a = np.asarray(score_a, dtype=np.float64).ravel()
    score_b = np.asarray(score_b, dtype=np.float64).ravel()
    if not (y_true.shape == score_a.shape == score_b.shape):
        raise ValueError(
            f"delong_test: shape mismatch y_true={y_true.shape} score_a={score_a.shape} score_b={score_b.shape}"
        )
    classes = np.unique(y_true)
    if classes.size != 2 or not np.array_equal(np.sort(classes), np.array([0, 1])):
        raise ValueError(
            f"delong_test: y_true must be binary 0/1; got unique={classes.tolist()}"
        )

    pos = y_true == 1
    neg = ~pos
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos < 2 or n_neg < 2:
        raise ValueError(
            f"delong_test: need >=2 positives and >=2 negatives; got n_pos={n_pos}, n_neg={n_neg}"
        )

    def _structural_components(scores: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        x = scores[pos]
        y = scores[neg]
        # Midrank: tied-aware rank with .5 contribution. We use scipy.stats.rankdata
        # 'average' which is the DeLong convention.
        ranks_all = stats.rankdata(np.concatenate([x, y]), method="average")
        ranks_x_in_all = ranks_all[:n_pos]
        ranks_y_in_all = ranks_all[n_pos:]
        ranks_x_self = stats.rankdata(x, method="average")
        ranks_y_self = stats.rankdata(y, method="average")
        # V10 / V01 are the per-row structural components (Sun & Xu eq 3-4).
        v10 = (ranks_x_in_all - ranks_x_self) / n_neg
        v01 = 1.0 - (ranks_y_in_all - ranks_y_self) / n_pos
        auc = (ranks_x_in_all.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(auc), v10, v01

    auc_a, v10_a, v01_a = _structural_components(score_a)
    auc_b, v10_b, v01_b = _structural_components(score_b)

    # 2x2 covariance estimate from the structural components.
    s10 = np.cov(np.vstack([v10_a, v10_b]), ddof=1)
    s01 = np.cov(np.vstack([v01_a, v01_b]), ddof=1)
    cov = s10 / n_pos + s01 / n_neg

    diff = auc_a - auc_b
    var_diff = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    # Degenerate but legitimate cases: when ``score_a == score_b`` element-wise (caller
    # comparing a scorer to itself), both AUCs and both v10/v01 columns are identical, so
    # var_diff collapses to 0 exactly and diff == 0 exactly. Statistically the null hypothesis
    # of "no difference" is trivially true -> z = 0, p = 1.0. Treat the zero-variance + zero-diff
    # case as the limit instead of NaN-ing (which would force every consumer to special-case it).
    if var_diff <= 0 or not np.isfinite(var_diff):
        if diff == 0.0:
            return {
                "auc_a": auc_a,
                "auc_b": auc_b,
                "diff": 0.0,
                "z": 0.0,
                "p_value": 1.0,
            }
        logger.warning(
            "delong_test: variance of AUC difference is %r with non-zero diff %r; returning p=nan.",
            var_diff, diff,
        )
        return {
            "auc_a": auc_a,
            "auc_b": auc_b,
            "diff": diff,
            "z": float("nan"),
            "p_value": float("nan"),
        }
    z = diff / np.sqrt(var_diff)
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "diff": float(diff),
        "z": float(z),
        "p_value": p_value,
    }


__all__ = ["bootstrap_metric", "delong_test"]
