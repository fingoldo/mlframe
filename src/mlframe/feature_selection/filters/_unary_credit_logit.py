"""Linear-usability credit of a unary transform of one feature for a BINARY target: the logit-deviance version of ``_unary_credit_stats``.

The regression criterion (Pearson information with the residual) does not transplant to a class target (research tt_b1, 2026-11-24: it names the judge-best transform in 76 of 126 cases, ANOVA and Bhattacharyya
criteria 75 and 49, equal-width MI 70, rank-binned MI 31). The criterion that does is the DEVIANCE GAIN of a one-feature logit on ``f(x_j)`` with the score of the other selected columns as an OFFSET:
``LRT_f = 2 (ll1_f - ll0)``, in units of the held-out log-loss gain (in-sample ``LRT / 2n`` = held-out gain, 0.1228 against 0.1230 nats per row). It recognised the judge-best transform in 126 of 126 gain cases
(K = 2, 3, 5), accepted nothing harmful in 244 cases and had 3 false positives in 78 controls under the parametric-bootstrap null used here.

The null is a parametric bootstrap: classes are redrawn from the identity logit on ``[offset, x_j]`` (so the null keeps the signal the identity already explains) and the same maximum over candidates of
``LRT_f - LRT_id`` is taken. Only the one-feature logits are refitted per draw (the offset is the one fitted on the observed classes), which is what makes the null cheap: every Newton step is one pass over
``(rows, candidates, draws)``. Decisions are replicated on a held-out half. Multiclass targets are not handled here (the multinomial version is a follow-up).
"""

from __future__ import annotations

import numpy as np

from ._unary_credit_stats import ALPHA, CANDIDATES, MAX_ROWS, ColumnVerdict, UnaryForm, _split, adaptive_max_pvalue, apply_form, fit_form, simplest_tied

#: Newton iterations of the one-feature logits (the problem is two-dimensional and well conditioned after standardising the candidate).
NEWTON_STEPS = 8


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    """Overflow-safe logistic function."""
    return 0.5 * (1.0 + np.tanh(0.5 * eta))


def _fit_offset_logit(columns: "list[np.ndarray]", y: np.ndarray) -> np.ndarray:
    """Linear predictor (n,) of a lightly ridged logistic regression of ``y`` on the intercept and ``columns`` (standardised), by Newton; the intercept-only logit when there are no columns."""
    n = y.shape[0]
    if columns:
        z = np.column_stack(columns).astype(np.float64)
        z = (z - z.mean(axis=0)) / np.where(z.std(axis=0) > 0, z.std(axis=0), 1.0)
        design = np.column_stack([np.ones(n), z])
    else:
        design = np.ones((n, 1))
    beta = np.zeros(design.shape[1])
    penalty = np.eye(design.shape[1]) * 1e-6 * n
    penalty[0, 0] = 0.0
    for _ in range(25):
        p = _sigmoid(design @ beta)
        grad = design.T @ (y - p) - penalty @ beta
        hess = (design * (p * (1.0 - p))[:, None]).T @ design + penalty + 1e-9 * np.eye(design.shape[1])
        step = np.linalg.solve(hess, grad)
        beta += step
        if float(np.abs(step).max()) < 1e-8:
            break
    return design @ beta


def _deviance(offset: np.ndarray, z: np.ndarray, y_draws: np.ndarray) -> np.ndarray:
    """``LRT`` of every candidate (columns of ``z``, standardised) for every class vector (columns of ``y_draws``): shape (candidates, draws).

    Per candidate and draw the logit ``offset + a + b z`` is fitted by Newton (all pairs at once) and compared with the offset-plus-intercept model."""
    _n, k = z.shape
    d = y_draws.shape[1]
    a = np.zeros((k, d))
    b = np.zeros((k, d))
    yz = y_draws[:, None, :]
    zz = z[:, :, None]
    for _ in range(NEWTON_STEPS):
        eta = offset[:, None, None] + a[None] + b[None] * zz
        p = _sigmoid(eta)
        resid = yz - p
        w = p * (1.0 - p)
        g_a = resid.sum(axis=0)
        g_b = (resid * zz).sum(axis=0)
        h_aa = w.sum(axis=0) + 1e-9
        h_ab = (w * zz).sum(axis=0)
        h_bb = (w * zz * zz).sum(axis=0) + 1e-9
        det = h_aa * h_bb - h_ab * h_ab
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        a = a + (h_bb * g_a - h_ab * g_b) / det
        b = b + (h_aa * g_b - h_ab * g_a) / det
    eta1 = offset[:, None, None] + a[None] + b[None] * zz
    ll1 = (yz * eta1 - np.logaddexp(0.0, eta1)).sum(axis=0)
    a0 = np.zeros(d)
    for _ in range(NEWTON_STEPS):
        p0 = _sigmoid(offset[:, None] + a0[None])
        a0 = a0 + (y_draws - p0).sum(axis=0) / ((p0 * (1.0 - p0)).sum(axis=0) + 1e-9)
    eta0 = offset[:, None] + a0[None]
    ll0 = (y_draws * eta0 - np.logaddexp(0.0, eta0)).sum(axis=0)
    return 2.0 * (ll1 - ll0[None])


def _standardise(values: np.ndarray) -> np.ndarray:
    """Candidate columns scaled to zero mean and unit variance (a constant column stays at zero)."""
    values = values - values.mean(axis=0)
    sd = values.std(axis=0)
    return values / np.where(sd > 0, sd, 1.0)


def _test_column(
    x: np.ndarray, y: np.ndarray, others: "list[np.ndarray]", forms: "list[UnaryForm]", level: float, rng: np.random.Generator
) -> "tuple[int, float, float, float, float]":
    """``(best candidate index, T, p, LRT_id, LRT_best)`` of one column with the other columns as an offset; T and the null draws are ``max_f (LRT_f - LRT_id)``, the p-value is decided at ``level`` with as
    few bootstrap draws as needed (``adaptive_max_pvalue``); among candidates tied with the best, the simplest is reported."""
    n = x.shape[0]
    offset = _fit_offset_logit(others, y)
    z = _standardise(np.column_stack([np.asarray(x, dtype=np.float64)] + [apply_form(f, x) for f in forms]))
    observed = _deviance(offset, z, y[:, None])[:, 0]
    gains = observed[1:] - observed[0]
    # the classes redrawn from the identity logit on [offset, x]
    p_id = _sigmoid(_fit_offset_logit([*others, np.asarray(x, dtype=np.float64)], y))

    first_batch: "list[np.ndarray]" = []

    def draw_maxima(k: int) -> np.ndarray:
        """``k`` null values of ``max_f (LRT_f - LRT_id)`` over bootstrap class vectors."""
        draws = (rng.random((n, k)) < p_id[:, None]).astype(np.float64)
        null = _deviance(offset, z, draws)
        null_gains = null[1:] - null[0][None, :]
        if not first_batch:
            first_batch.append(null_gains)
        return null_gains.max(axis=0)

    p = adaptive_max_pvalue(float(gains.max()), draw_maxima, level)
    slack = float(first_batch[0][int(np.argmax(gains))].std()) if first_batch else 0.0
    best = simplest_tied(gains, [f.name for f in forms], slack) if float(gains.max()) > 0.0 else int(np.argmax(gains))
    return best, float(gains[best]), p, float(observed[0]), float(observed[1 + best])


def credit_columns_binary(
    X: np.ndarray,
    y: np.ndarray,
    *,
    candidates: "tuple[str, ...]" = CANDIDATES,
    alpha: float = ALPHA,
    seed: int = 0,
    passes: int = 2,
) -> "list[ColumnVerdict]":
    """One verdict per column of ``X`` for a binary target ``y`` in {0, 1}: which unary form, if any, a logistic head should use instead of the identity (same protocol as ``credit_columns``)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, m = X.shape
    rng = np.random.default_rng(seed)
    if n > MAX_ROWS:
        keep = np.sort(rng.choice(n, MAX_ROWS, replace=False))
        X, y = X[keep], y[keep]
        n = MAX_ROWS
    none = ColumnVerdict(None, 0.0, 1.0, False, 0.0, 0.0, 0.0)
    if n < 600 or m == 0 or set(np.unique(y)) - {0.0, 1.0} or min(float(y.mean()), 1.0 - float(y.mean())) < 0.02:
        return [none] * m
    disc, held = _split(n, rng)
    verdicts = [none] * m
    current = [X[:, j].copy() for j in range(m)]
    for _ in range(max(1, passes)):
        for j in range(m):
            raw = X[:, j]
            if not np.isfinite(raw).all():
                continue
            forms = [f for f in (fit_form(c, raw[disc]) for c in candidates) if f is not None]
            if not forms:
                continue
            best, gain, p_raw, lrt_id, lrt_best = _test_column(raw[disc], y[disc], [current[k][disc] for k in range(m) if k != j], forms, alpha / m, rng)
            if min(1.0, p_raw * m) > alpha or gain <= 0.0:
                current[j], verdicts[j] = raw.copy(), none
                continue
            form = forms[best]
            _, gain_h, p_h, lrt_id_h, lrt_best_h = _test_column(raw[held], y[held], [current[k][held] for k in range(m) if k != j], [form], alpha, rng)
            nats_per_row = (lrt_best_h - lrt_id_h) / (2.0 * held.size)
            if not (p_h <= alpha and gain_h > 0.0 and nats_per_row > 0.0):
                current[j], verdicts[j] = raw.copy(), none
                continue
            verdicts[j] = ColumnVerdict(form, (lrt_best - lrt_id) / (2.0 * disc.size), min(1.0, p_raw * m), True, nats_per_row, lrt_id, lrt_best)
            current[j] = apply_form(form, raw)
    return verdicts
