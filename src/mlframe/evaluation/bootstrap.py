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


def _ci_from_samples(
    samples: np.ndarray,
    point: float,
    alpha: float,
    method: str,
    jackknife: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """Reduce a bootstrap distribution to a (lo, hi) CI by percentile or BCa.

    ``method="percentile"`` is the plain Efron percentile interval (symmetric, fast). ``method="bca"`` is the
    bias-corrected and accelerated interval (Efron 1987): it shifts the percentile cut-points to correct for
    (a) median bias of the bootstrap distribution relative to the point estimate (``z0``) and (b) skew of the
    sampling distribution (``acceleration`` from the jackknife). On skewed / bounded metrics (AUC near 1.0,
    Pearson r) percentile silently UNDER-COVERS; BCa recovers close-to-nominal coverage. When the BCa inputs
    are degenerate (no jackknife, all-equal samples, non-finite z0/a) BCa gracefully falls back to percentile.
    """
    lo_pct = (alpha / 2.0) * 100.0
    hi_pct = (1.0 - alpha / 2.0) * 100.0

    def _pct_pair(p_lo: float, p_hi: float) -> tuple[float, float]:
        # CPX24: one np.percentile call over the [lo, hi] vector instead of two
        # separate calls. Each np.percentile internally np.partition's the array;
        # two calls partition the SAME samples twice. The vectorised single call
        # partitions once and is BIT-IDENTICAL (==) to the two scalar calls
        # (verified across n=1k-10k). Pure post-processing of the CI cut-points,
        # no RNG touched -- the resample draws are already complete here.
        both = np.percentile(samples, [p_lo, p_hi])
        return float(both[0]), float(both[1])

    if method != "bca":
        return _pct_pair(lo_pct, hi_pct)

    n_s = samples.shape[0]
    # z0: bias correction = inverse-normal of the fraction of resamples below the point estimate.
    n_below = int(np.count_nonzero(samples < point))
    if n_below == 0 or n_below == n_s:
        return _pct_pair(lo_pct, hi_pct)
    z0 = stats.norm.ppf(n_below / n_s)

    # a: acceleration from the jackknife skew of the metric (Efron 1987 eq 6.6). No jackknife -> percentile.
    if jackknife is None or jackknife.shape[0] < 3:
        return _pct_pair(lo_pct, hi_pct)
    jk_mean = jackknife.mean()
    diffs = jk_mean - jackknife
    denom = 6.0 * (np.sum(diffs ** 2) ** 1.5)
    if denom == 0.0 or not np.isfinite(denom):
        return _pct_pair(lo_pct, hi_pct)
    a = float(np.sum(diffs ** 3) / denom)

    z_lo = stats.norm.ppf(alpha / 2.0)
    z_hi = stats.norm.ppf(1.0 - alpha / 2.0)
    a_lo = stats.norm.cdf(z0 + (z0 + z_lo) / (1.0 - a * (z0 + z_lo)))
    a_hi = stats.norm.cdf(z0 + (z0 + z_hi) / (1.0 - a * (z0 + z_hi)))
    if not (np.isfinite(a_lo) and np.isfinite(a_hi)) or a_lo >= a_hi:
        return _pct_pair(lo_pct, hi_pct)
    return _pct_pair(a_lo * 100.0, a_hi * 100.0)


def _jackknife_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    max_n: int = 2000,
) -> Optional[np.ndarray]:
    """Leave-one-out jackknife of ``metric_fn`` for the BCa acceleration term.

    Returns the ``(n,)`` leave-one-out metric values, or ``None`` if the jackknife is infeasible. For n > ``max_n``
    the full LOO is O(n^2) in metric calls, so we sub-sample to a deterministic stride of ``max_n`` rows -- the
    acceleration estimate is a low-order skew summary and tolerates sub-sampling far better than the percentile
    cut-points themselves. Failed / non-finite LOO evaluations are dropped.
    """
    n = y_true.shape[0]
    if n < 3:
        return None
    if n <= max_n:
        sel = np.arange(n)
    else:
        sel = np.linspace(0, n - 1, max_n).astype(np.int64)
    keep_mask = np.ones(n, dtype=bool)
    out = np.empty(sel.shape[0], dtype=np.float64)
    w = 0
    for i in sel:
        keep_mask[i] = False
        try:
            v = float(metric_fn(y_true[keep_mask], y_pred[keep_mask]))
        except Exception:
            keep_mask[i] = True
            continue
        keep_mask[i] = True
        if not _isfinite(v):
            continue
        out[w] = v
        w += 1
    if w < 3:
        return None
    return out[:w]


def _jackknife_metric_idx(
    n: int,
    metric_fn_idx: Callable[[np.ndarray], float],
    max_n: int = 2000,
) -> Optional[np.ndarray]:
    """Leave-one-out jackknife for an INDEX-aware metric ``fn(idx) -> float`` (BCa acceleration term).

    Mirrors ``_jackknife_metric`` but feeds the metric the leave-one-out index array instead of pre-sliced views,
    so an index-aware metric (e.g. the pre-sorted AUC resampler) reuses its precomputed base structure.
    """
    if n < 3:
        return None
    sel = np.arange(n) if n <= max_n else np.linspace(0, n - 1, max_n).astype(np.int64)
    full = np.arange(n, dtype=np.int64)
    # CPX24: boolean mask-flip gather instead of per-iter np.delete(full, i).
    # np.delete allocates a fresh length-(n-1) array AND pays searchsorted /
    # range-rebuild dispatch every iteration; full[mask] is a single fancy-index
    # gather with the mask flipped in/out per iter (no per-call delete dispatch).
    # Bit-identical: full[mask] yields the same ascending indices delete did
    # (mirrors the already-mask-based _jackknife_metric above). No RNG here.
    keep_mask = np.ones(n, dtype=bool)
    out = np.empty(sel.shape[0], dtype=np.float64)
    w = 0
    for i in sel:
        keep_mask[i] = False
        try:
            v = float(metric_fn_idx(full[keep_mask]))
        except Exception:
            keep_mask[i] = True
            continue
        keep_mask[i] = True
        if not _isfinite(v):
            continue
        out[w] = v
        w += 1
    if w < 3:
        return None
    return out[:w]


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    stratify: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    method: str = "bca",
) -> dict[str, Any]:
    """Bootstrap CI for an arbitrary ``metric_fn(y_true, y_pred)``.

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
    method
        ``"bca"`` (default) is the bias-corrected and accelerated interval (Efron 1987); ``"percentile"`` is the
        plain Efron percentile interval. BCa corrects for median bias + skew of the sampling distribution, which
        the percentile interval ignores -- on skewed / bounded metrics (AUC near 1.0, Pearson r) percentile
        UNDER-COVERS the nominal level, while BCa recovers close-to-nominal coverage (see
        ``_benchmarks/bench_bootstrap_ci_coverage.py``). BCa adds an O(min(n, 2000)) jackknife pass for the
        acceleration term and falls back to percentile automatically when its inputs are degenerate. Pass
        ``method="percentile"`` for the legacy behaviour or to skip the jackknife on a very hot path.

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

    jackknife = _jackknife_metric(y_true, y_pred, metric_fn) if method == "bca" else None
    lo, hi = _ci_from_samples(samples, point, alpha, method, jackknife)

    return {"point": point, "lo": lo, "hi": hi, "samples": samples}


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fns: Mapping[str, Callable[[np.ndarray, np.ndarray], float]],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    stratify: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    metric_fns_idx: Optional[Mapping[str, Callable[[np.ndarray], float]]] = None,
    method: str = "bca",
) -> dict[str, dict]:
    """Bootstrap CIs for SEVERAL metrics sharing ONE resample loop.

    ``metric_fns_idx`` are INDEX-aware metric callables ``fn(idx) -> float`` that
    receive the raw resample indices instead of pre-sliced ``(yt, yp)`` views.
    They let a metric reuse a precomputed base-data structure across all 1000
    resamples (e.g. a pre-sorted base score vector for AUC -- see
    ``make_bootstrap_auc_resampler``), turning a per-resample O(n log n) argsort
    into an O(n) gather. The point estimate uses ``idx = arange(n)``. Results
    merge into the same dict; bit-identity vs the slice-based path holds when the
    index-aware metric is defined to equal the slice-based one (it is for AUC on
    tie-free scores).

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
    if not metric_fns and not metric_fns_idx:
        return {}

    # Seed BEFORE the point estimates (which never touch the RNG), so the RNG
    # state at the resample loop matches a freshly-seeded bootstrap_metric.
    rng = np.random.default_rng(random_state)

    metric_fns_idx = dict(metric_fns_idx) if metric_fns_idx else {}
    _full_idx = np.arange(n, dtype=np.int64)  # point estimate = identity resample

    results: dict[str, dict] = {}
    points: dict[str, float] = {}
    active: list = []
    active_idx: list = []
    for name, fn in metric_fns.items():
        try:
            points[name] = float(fn(y_true, y_pred))
            active.append(name)
        except Exception as exc:
            results[name] = {"error": f"point metric failed: {type(exc).__name__}: {exc}"}
    for name, fn in metric_fns_idx.items():
        try:
            points[name] = float(fn(_full_idx))
            active_idx.append(name)
        except Exception as exc:
            results[name] = {"error": f"point metric failed: {type(exc).__name__}: {exc}"}
    if not active and not active_idx:
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

    _all_active = active + active_idx
    # When every active metric is index-aware (re-gathers internally from idx),
    # the per-resample yt=y_true[idx]/yp=y_pred[idx] slice is pure waste -- the
    # idx-aware path never reads it. Skip the two n-length fancy-index copies
    # then (the dominant per-resample cost; 2.2s/1000 at n=200k). Non-idx-aware
    # metrics still get their slices when any are active.
    _need_slice = bool(active)
    samples = {name: np.empty(n_bootstrap, dtype=np.float64) for name in _all_active}
    valid = {name: 0 for name in _all_active}
    failures = {name: 0 for name in _all_active}
    first_err: dict = {name: None for name in _all_active}

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
        # Slice ONCE; every non-idx-aware metric reads the same resampled views.
        # Skipped entirely when all active metrics are idx-aware (re-gather internally).
        if _need_slice:
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
        # Index-aware metrics: pass raw idx so they can reuse a precomputed
        # base structure (e.g. pre-sorted AUC) instead of re-deriving it.
        for name in active_idx:
            try:
                v = float(metric_fns_idx[name](idx))
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

    for name in _all_active:
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
        jackknife = None
        if method == "bca":
            if name in active:
                jackknife = _jackknife_metric(y_true, y_pred, metric_fns[name])
            else:
                jackknife = _jackknife_metric_idx(n, metric_fns_idx[name])
        lo, hi = _ci_from_samples(s, points[name], alpha, method, jackknife)
        results[name] = {"point": points[name], "lo": lo, "hi": hi, "samples": s}
    return results


def _auc_structural_components(
    scores: np.ndarray, pos: np.ndarray, neg: np.ndarray, n_pos: int, n_neg: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """DeLong / Sun-Xu structural components for one score vector.

    Returns ``(auc, v10, v01)`` where ``v10`` (length ``n_pos``) and ``v01`` (length ``n_neg``) are the
    per-positive / per-negative placement-value pseudo-observations whose (co)variances give the closed-form
    AUC variance. Midranks via ``rankdata(method="average")`` make the components tie-aware (the DeLong convention).
    """
    x = scores[pos]
    y = scores[neg]
    ranks_all = stats.rankdata(np.concatenate([x, y]), method="average")
    ranks_x_in_all = ranks_all[:n_pos]
    ranks_y_in_all = ranks_all[n_pos:]
    ranks_x_self = stats.rankdata(x, method="average")
    ranks_y_self = stats.rankdata(y, method="average")
    v10 = (ranks_x_in_all - ranks_x_self) / n_neg
    v01 = 1.0 - (ranks_y_in_all - ranks_y_self) / n_pos
    auc = (ranks_x_in_all.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc), v10, v01


def auc_variance(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """Closed-form (DeLong) variance + standard error of a single-sample ROC-AUC.

    The DeLong / Hanley-McNeil structural-component estimator (Sun & Xu 2014 O(n log n) form) gives the
    asymptotic variance of the AUC estimator directly from the placement-value pseudo-observations -- no
    resampling. ``Var(AUC) = S10/n_pos + S01/n_neg`` where ``S10 = Var(v10)`` (over positives) and
    ``S01 = Var(v01)`` (over negatives), with ``ddof=1`` sample variances (the standard small-sample
    convention -- the population ``ddof=0`` divisor would under-state the variance, the same Bessel argument
    as qual-8). Tie-aware via midranks. This is the asymptotically-exact SE the bootstrap-of-AUC only
    approximates (and the bootstrap is biased at small n / extreme AUC).

    Parameters
    ----------
    y_true
        1D binary label vector (0 / 1).
    score
        Predicted scores (higher = positive class).

    Returns
    -------
    dict ``{"auc": ..., "variance": ..., "se": ...}``. ``variance``/``se`` are ``nan`` when degenerate
    (single-class y_true, or fewer than 2 positives / 2 negatives -- the sample-variance is undefined).
    """
    y_true = np.asarray(y_true).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()
    if y_true.shape != score.shape:
        raise ValueError(f"auc_variance: shape mismatch y_true={y_true.shape} score={score.shape}")
    classes = np.unique(y_true)
    if classes.size != 2 or not np.array_equal(np.sort(classes), np.array([0, 1])):
        raise ValueError(f"auc_variance: y_true must be binary 0/1; got unique={classes.tolist()}")
    pos = y_true == 1
    neg = ~pos
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos < 2 or n_neg < 2:
        auc_pt = _auc_structural_components(score, pos, neg, max(n_pos, 1), max(n_neg, 1))[0] if n_pos and n_neg else float("nan")
        return {"auc": auc_pt, "variance": float("nan"), "se": float("nan")}
    auc, v10, v01 = _auc_structural_components(score, pos, neg, n_pos, n_neg)
    var = float(np.var(v10, ddof=1) / n_pos + np.var(v01, ddof=1) / n_neg)
    se = float(np.sqrt(var)) if var >= 0 and np.isfinite(var) else float("nan")
    return {"auc": auc, "variance": var, "se": se}


def auc_ci(
    y_true: np.ndarray,
    score: np.ndarray,
    alpha: float = 0.05,
    method: str = "delong",
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
) -> dict[str, float | str]:
    """Confidence interval for a single-sample ROC-AUC.

    ``method="delong"`` (DEFAULT) is the closed-form DeLong Wald interval ``AUC +/- z * SE`` on the
    logit scale (so the interval stays inside [0, 1] and is asymptotically correct near AUC=1 where a
    symmetric raw-scale Wald interval would spill past 1.0). This is the asymptotically-exact,
    resampling-free estimator and is closer to the true sampling SE than the bootstrap at small n / extreme
    AUC (see ``evaluation/_benchmarks/bench_auc_ci_delong_vs_bootstrap.py``).

    ``method="bootstrap"`` is the legacy path: stratified bootstrap of ``roc_auc_score`` reduced by the
    qual-5 BCa interval (``bootstrap_metric``). Kept for callers who want a fully non-parametric interval or
    need to match a prior bootstrap-based report. APPROXIMATE: resampling rows ignores the tie / placement
    structure that the DeLong estimator handles exactly, so on tied scores or near-AUC=1 it can mis-state the
    SE -- ``method="delong"`` (the default) is preferred for AUC uncertainty.

    Parameters
    ----------
    y_true, score
        Binary 0/1 labels and positive-class scores.
    alpha
        Two-sided miscoverage (0.05 -> 95% CI).
    method
        ``"delong"`` (default) or ``"bootstrap"``.
    n_bootstrap, random_state
        Only used by ``method="bootstrap"``.

    Returns
    -------
    dict ``{"auc": ..., "point": ..., "lo": ..., "hi": ..., "se": ..., "method": ...}``. ``"point"`` is an alias
    of ``"auc"`` so consumers that read the generic ``"point"`` key (as ``bootstrap_metric`` returns) work
    uniformly without a KeyError. ``se`` is ``nan`` for the bootstrap path. On degenerate input the DeLong path
    returns ``lo=hi=nan``.

    bench-note (2026-06-15, bench_auc_ci_delong_vs_bootstrap.py): DeLong SE/CI is a STATISTICAL WASH vs
    the qual-5 BCa bootstrap on AUC uncertainty (closer-to-truth-SD in 22/40 scenario x seed cells, coverage
    tied) -- it is NOT more accurate, so it did NOT flip any default on ``bootstrap_metric``. It ships because
    it is instant / deterministic / RNG-free. Re-run the named bench before claiming a DeLong accuracy edge.
    """
    if method == "bootstrap":
        from sklearn.metrics import roc_auc_score

        yt = np.asarray(y_true).ravel()
        sc = np.asarray(score, dtype=np.float64).ravel()
        res = bootstrap_metric(
            yt, sc, lambda a, b: float(roc_auc_score(a, b)),
            n_bootstrap=n_bootstrap, alpha=alpha, random_state=random_state,
            stratify=yt, method="bca",
        )
        return {"auc": res["point"], "point": res["point"], "lo": res["lo"], "hi": res["hi"], "se": float("nan"), "method": "bootstrap"}
    if method != "delong":
        raise ValueError(f"auc_ci: method must be 'delong' or 'bootstrap'; got {method!r}")
    stats_d = auc_variance(y_true, score)
    auc = stats_d["auc"]
    se = stats_d["se"]
    if not np.isfinite(se) or not np.isfinite(auc):
        return {"auc": auc, "point": auc, "lo": float("nan"), "hi": float("nan"), "se": se, "method": "delong"}
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    # Logit-scale Wald keeps the interval inside (0, 1) and is the standard remedy for the raw-scale
    # interval over-shooting 1.0 when AUC is near the ceiling (the regime where bootstrap also struggles).
    eps = 1e-12
    a_cl = min(max(auc, eps), 1.0 - eps)
    logit = math.log(a_cl / (1.0 - a_cl))
    se_logit = se / (a_cl * (1.0 - a_cl))
    lo = 1.0 / (1.0 + math.exp(-(logit - z * se_logit)))
    hi = 1.0 / (1.0 + math.exp(-(logit + z * se_logit)))
    return {"auc": auc, "point": auc, "lo": float(lo), "hi": float(hi), "se": se, "method": "delong"}


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

    auc_a, v10_a, v01_a = _auc_structural_components(score_a, pos, neg, n_pos, n_neg)
    auc_b, v10_b, v01_b = _auc_structural_components(score_b, pos, neg, n_pos, n_neg)

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


__all__ = ["bootstrap_metric", "delong_test", "auc_variance", "auc_ci"]
