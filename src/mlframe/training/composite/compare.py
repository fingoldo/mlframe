"""Champion-vs-challenger model governance comparison.

Pure evaluation utilities that decide whether a challenger model is
SIGNIFICANTLY better than the incumbent champion on a held-out set --
not merely better by luck. Works for any two ``predict``-able
estimators (composite-target or plain sklearn regressors /
classifiers); no refit, no frame copy, no leakage assumptions beyond
"the supplied (X, y) is held-out and neither model saw it".

Design
------

- :func:`compare_models` scores both fitted estimators on one held-out
  ``(X, y)``, computes the per-row *loss* for each, and runs a PAIRED
  significance test on the per-row loss difference
  ``d_i = loss_champion_i - loss_challenger_i`` (positive ``d`` means
  the challenger had lower loss on that row, i.e. it is better). Pairing
  on the same rows removes between-row variance, giving far more power
  than comparing two independent score distributions.
- The default test is a **paired bootstrap** on the mean loss
  difference (resampling rows with replacement, ``n_boot`` times),
  which makes no parametric assumption about the loss distribution and
  naturally handles non-normal / heavy-tailed losses. A paired
  ``t``-test and Wilcoxon signed-rank are available via ``test=`` for
  callers who want a classical p-value.
- :func:`should_promote` layers a governance policy on top: require both
  statistical significance (``p < alpha``) AND a minimum practical
  effect (``delta >= min_effect`` on the score scale) before flagging a
  promotion -- avoids promoting a challenger that is significantly but
  trivially better.

No frame copy: predictions are pulled via ``estimator.predict(X)`` (the
estimator decides its own native frame handling); only small 1-D
ndarrays of predictions / losses are materialised. ``X`` is never
cloned, never down-converted.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Lower-is-better metrics report a LOSS directly; higher-is-better
# metrics ("accuracy", "r2") are converted to a per-row loss internally
# and the reported *score* is flipped back so the public number reads
# in its natural orientation.
_LOWER_IS_BETTER = {"rmse", "mse", "mae", "logloss"}
_HIGHER_IS_BETTER = {"accuracy", "r2"}


def _as_1d(a: Any) -> np.ndarray:
    """Materialise a prediction / target carrier as a 1-D float ndarray
    without copying a frame. Accepts ndarray, polars / pandas Series,
    lists. For 2-D probability outputs (n, 2) we take the positive-class
    column; (n, k>2) is rejected (use a custom metric for multiclass)."""
    arr = np.asarray(a)
    if arr.ndim == 2 and arr.shape[1] == 2:
        arr = arr[:, 1]
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D predictions/target, got shape {arr.shape}")
    return arr


def _per_row_loss(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> np.ndarray:
    """Per-row loss vector for a named metric (lower = better, always)."""
    if metric in ("rmse", "mse"):
        return (y_true - y_pred) ** 2
    if metric == "mae":
        return np.abs(y_true - y_pred)
    if metric == "accuracy":
        # 0/1 per-row loss; predictions may be soft -> threshold at 0.5.
        pred_lbl = (y_pred >= 0.5).astype(y_true.dtype) if not np.array_equal(y_pred, y_pred.astype(int)) else y_pred
        return (pred_lbl != y_true).astype(np.float64)
    if metric == "logloss":
        eps = 1e-15
        p = np.clip(y_pred, eps, 1.0 - eps)
        return -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
    if metric == "r2":
        # r2 has no clean per-row decomposition; use squared error as the
        # paired loss and report r2 only as the aggregate score.
        return (y_true - y_pred) ** 2
    raise ValueError(f"unknown metric {metric!r}; pass a callable for custom metrics")


def _aggregate_score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """Natural-orientation aggregate score for the public report."""
    loss = _per_row_loss(y_true, y_pred, metric)
    if metric == "rmse":
        return float(np.sqrt(loss.mean()))
    if metric == "r2":
        ss_res = float(loss.sum())
        ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if metric == "accuracy":
        return float(1.0 - loss.mean())  # accuracy = 1 - error rate
    return float(loss.mean())  # mse / mae / logloss


def _resolve_loss(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric: Union[str, Callable],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (loss_a, loss_b, score_a, score_b). For a callable metric
    the callable must accept (y_true, y_pred) and return a per-row loss
    array (lower = better); the aggregate score is its mean."""
    if callable(metric):
        loss_a = _as_1d(metric(y_true, pred_a)).astype(np.float64)
        loss_b = _as_1d(metric(y_true, pred_b)).astype(np.float64)
        return loss_a, loss_b, float(loss_a.mean()), float(loss_b.mean())
    m = metric.lower()
    loss_a = _per_row_loss(y_true, pred_a, m)
    loss_b = _per_row_loss(y_true, pred_b, m)
    return loss_a, loss_b, _aggregate_score(y_true, pred_a, m), _aggregate_score(y_true, pred_b, m)


def _paired_bootstrap_ci(
    diff: np.ndarray, n_boot: int, alpha: float, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Percentile-bootstrap CI + two-sided p-value on the mean per-row
    loss difference ``diff`` (positive mean => challenger better).

    The p-value is the bootstrap analogue: 2x the smaller tail mass of
    the resampled-mean distribution on the side of zero, i.e. how often
    the resampled mean lands on the opposite side of 0 from the observed
    mean. Returns (ci_low, ci_high, p_value) on the loss-difference
    scale (champion_loss - challenger_loss)."""
    n = diff.shape[0]
    # Row-chunk the bootstrap: drawing the (n_boot, n) index matrix + the
    # diff[idx] gather in one shot materialises two n_boot*n temporaries
    # (~16 GB at n_boot=1000, n=1e6). numpy fills rng.integers row-major,
    # so drawing the resamples in contiguous BLOCKS consumes the RNG stream
    # in the EXACT same order as the monolithic call -> bit-identical means.
    boot_means = np.empty(n_boot, dtype=np.float64)
    block = 64
    for start in range(0, n_boot, block):
        stop = min(start + block, n_boot)
        idx_blk = rng.integers(0, n, size=(stop - start, n))
        boot_means[start:stop] = diff[idx_blk].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    obs = float(diff.mean())
    # Fraction of bootstrap means on the opposite side of zero from obs.
    if obs >= 0:
        tail = float(np.mean(boot_means <= 0.0))
    else:
        tail = float(np.mean(boot_means >= 0.0))
    p_value = min(1.0, 2.0 * tail)
    return lo, hi, p_value


def compare_models(
    champion: Any,
    challenger: Any,
    X: Any,
    y: Any,
    *,
    metric: Union[str, Callable] = "rmse",
    n_boot: int = 1000,
    alpha: float = 0.05,
    test: str = "bootstrap",
    random_state: Optional[int] = 0,
) -> Dict[str, Any]:
    """Compare a fitted ``challenger`` against a fitted ``champion`` on a
    held-out ``(X, y)`` and decide if the challenger is significantly
    better.

    Parameters
    ----------
    champion, challenger : fitted estimators
        Anything with ``.predict(X)``; composite-target estimators,
        sklearn pipelines, plain regressors / classifiers all work.
    X, y : held-out data
        Neither model must have seen these rows (no-leakage contract is
        the caller's responsibility). ``X`` is passed straight to
        ``predict`` -- never copied / down-converted.
    metric : str | callable, default "rmse"
        One of ``rmse / mse / mae / r2 / accuracy / logloss``, or a
        callable ``(y_true, y_pred) -> per_row_loss`` (lower = better).
    n_boot : int, default 1000
        Bootstrap resamples for the ``bootstrap`` test.
    alpha : float, default 0.05
        Significance level for the CI and the ``challenger_wins`` flag.
    test : {"bootstrap", "ttest", "wilcoxon"}, default "bootstrap"
        Paired significance test on the per-row loss difference.
    random_state : int | None, default 0
        Seed for the bootstrap resampler (reproducible governance).

    Returns
    -------
    dict with keys
        ``champion_score``, ``challenger_score`` (natural-orientation
        scores), ``delta`` (challenger advantage, positive = better on
        the metric), ``ci_low``, ``ci_high`` (CI on the loss-difference
        scale, positive = challenger better), ``p_value``,
        ``challenger_wins`` (bool: p < alpha AND challenger actually
        better), plus ``n``, ``metric``, ``test`` for provenance.
    """
    y_true = _as_1d(y).astype(np.float64)
    pred_c = _as_1d(champion.predict(X)).astype(np.float64)
    pred_h = _as_1d(challenger.predict(X)).astype(np.float64)
    if not (y_true.shape[0] == pred_c.shape[0] == pred_h.shape[0]):
        raise ValueError("champion/challenger predictions and y must align row-wise")

    loss_c, loss_h, score_c, score_h = _resolve_loss(y_true, pred_c, pred_h, metric)

    # Per-row loss difference: positive => challenger had lower loss.
    diff = loss_c - loss_h
    mean_diff = float(diff.mean())

    higher_better = (not callable(metric)) and metric.lower() in _HIGHER_IS_BETTER
    # delta in NATURAL orientation: positive => challenger better on score.
    delta = (score_h - score_c) if higher_better else (score_c - score_h)

    rng = np.random.default_rng(random_state)
    if test == "bootstrap":
        ci_low, ci_high, p_value = _paired_bootstrap_ci(diff, n_boot, alpha, rng)
    elif test == "ttest":
        from scipy import stats

        if np.allclose(diff, diff[0]):
            p_value = 1.0
        else:
            p_value = float(stats.ttest_1samp(diff, 0.0).pvalue)
        se = float(diff.std(ddof=1)) / np.sqrt(diff.shape[0]) if diff.shape[0] > 1 else 0.0
        from scipy.stats import t as _t

        crit = float(_t.ppf(1.0 - alpha / 2.0, max(1, diff.shape[0] - 1)))
        ci_low, ci_high = mean_diff - crit * se, mean_diff + crit * se
    elif test == "wilcoxon":
        from scipy import stats

        nz = diff[diff != 0.0]
        if nz.shape[0] == 0:
            p_value = 1.0
        else:
            p_value = float(stats.wilcoxon(nz).pvalue)
        # Wilcoxon gives no CI; fall back to bootstrap percentile CI.
        ci_low, ci_high, _ = _paired_bootstrap_ci(diff, n_boot, alpha, rng)
    else:
        raise ValueError(f"unknown test {test!r}; use bootstrap/ttest/wilcoxon")

    challenger_wins = bool(p_value < alpha and mean_diff > 0.0)

    return {
        "champion_score": float(score_c),
        "challenger_score": float(score_h),
        "delta": float(delta),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
        "challenger_wins": challenger_wins,
        "n": int(y_true.shape[0]),
        "metric": metric if isinstance(metric, str) else getattr(metric, "__name__", "custom"),
        "test": test,
    }


def should_promote(
    champion: Any,
    challenger: Any,
    X: Any,
    y: Any,
    *,
    metric: Union[str, Callable] = "rmse",
    alpha: float = 0.05,
    min_effect: float = 0.0,
    n_boot: int = 1000,
    test: str = "bootstrap",
    random_state: Optional[int] = 0,
) -> Dict[str, Any]:
    """Governance convenience: promote the challenger only when it is
    BOTH statistically significant (``p < alpha``) AND clears a minimum
    practical effect (``delta >= min_effect`` on the natural score
    scale). Returns the full :func:`compare_models` dict augmented with
    ``promote`` (bool) and ``reason`` (str)."""
    res = compare_models(
        champion,
        challenger,
        X,
        y,
        metric=metric,
        n_boot=n_boot,
        alpha=alpha,
        test=test,
        random_state=random_state,
    )
    sig = res["p_value"] < alpha
    effect = res["delta"] >= min_effect and res["delta"] > 0.0
    promote = bool(res["challenger_wins"] and sig and effect)
    if not res["challenger_wins"] or not sig:
        reason = f"not significant (p={res['p_value']:.3g} >= alpha={alpha})"
    elif not effect:
        reason = f"effect too small (delta={res['delta']:.4g} < min_effect={min_effect})"
    else:
        reason = f"promote: delta={res['delta']:.4g}, p={res['p_value']:.3g}"
    res["promote"] = promote
    res["reason"] = reason
    return res
