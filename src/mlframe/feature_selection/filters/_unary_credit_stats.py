"""Linear-usability credit of a unary transform of one feature: the statistics, with no MRMR state.

Why it exists. MRMR ranks by mutual information on rank-based bins, which is the same for every monotone transform of a column and, for any transform, is ZERO conditionally on the column itself (``sin(x)`` is a
function of ``x``). So an MI selector can propose ``log(x)`` for a lognormal column or ``sin(x1)`` for ``y = sin(x1) + x2`` and can never credit them (``mrmr_gain`` is NaN; research tt_a2, tt_b2, tt_b4, 2026-11-24).
A linear or additive downstream model, however, gains a great deal from the right form. This module asks that question directly and calibrates the answer with a null, no downstream model is trained.

The statistic. For a selected column ``x_j`` let ``D`` be the other selected columns plus an intercept and ``r`` the residual of ``y`` on ``D``. For each candidate ``f`` the squared Pearson correlation
``s_f = rho^2(f(x_j) | D, r)`` is computed on ``f(x_j)`` PARTIALLED on ``D`` (Freedman-Lane: a candidate correlated with the selected columns gets credit only for what the design lacks; the fix of the
correlated-covariate failure of the plain version, 38 of 40 harmful picks to 0 of 40). The gain of the column is ``T = max_{f != id} (s_f - s_id)``. The null permutes ``r`` (re-projected on the complement of ``D``) and
takes the same maximum over candidates, so the extra freedom of choosing among candidates is paid for in the threshold. Across the ``m`` selected columns the p-values are Bonferroni-corrected, and a decision is
accepted only if the gain REPLICATES on a held-out half of the rows (its own null).

Validated by the reference implementation of research B4 (0 to 2 percent family-wise error with 100 to 500 mostly-noise columns, 148 of 149 strong transforms found, 0 of 57 linear columns given a transform).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Candidate unary forms. ``sq`` / ``abs`` / ``tanh`` / ``asinh`` are centred on the median (the fixed-zero forms capture a third of the gain on an off-centre column, B4), ``sq0`` is the plain ``x * x`` (a column centred at zero is better served by it: the median-centred square leaves a small linear component of the column to be corrected, and ties go to the simpler form); ``sin`` / ``cos`` take the raw column.
CANDIDATES = ("log", "sqrt", "recip", "sq0", "sq", "abs", "tanh", "cbrt", "asinh", "sin", "cos", "rankgauss")
#: The family-wise level of the decision.
ALPHA = 0.05
#: Rows used at most for the screen (a seeded draw above it).
MAX_ROWS = 20000


#: Candidates from the simplest to the most flexible: among candidates whose gain is within ``TIE_FRACTION`` of the best, the simplest wins (``log`` of a lognormal column is also reached by ``rankgauss``; the recipe should name the closed form).
SIMPLICITY_ORDER = ("log", "sqrt", "cbrt", "recip", "sq0", "sq", "abs", "sin", "cos", "tanh", "asinh", "rankgauss")
TIE_FRACTION = 0.95
#: Null draws come in batches; a column stops as soon as its p-value cannot reach the level any more (most columns stop after the first batch).
BATCH = 50
MAX_DRAWS = 2000


def simplest_tied(gains: np.ndarray, names: "list[str]", slack: float = 0.0) -> int:
    """Index of the simplest candidate whose gain is within ``max(slack, (1 - TIE_FRACTION) * best)`` of the best gain (the argmax when nothing ties).

    ``slack`` is the spread of the best candidate's gain under the null: candidates the data cannot tell apart from the best by more than one null standard deviation are tied, and the simplest closed form wins
    (``sin`` before ``tanh`` on a column where both are the same curve to within noise)."""
    best = float(gains.max())
    margin = max(float(slack), (1.0 - TIE_FRACTION) * best)
    tied = [i for i in range(len(names)) if gains[i] >= best - margin]
    return min(tied, key=lambda i: (SIMPLICITY_ORDER.index(names[i]) if names[i] in SIMPLICITY_ORDER else len(SIMPLICITY_ORDER), -gains[i]))


def adaptive_max_pvalue(observed: float, draw_maxima, level: float) -> float:
    """Permutation p-value ``(1 + #{null >= observed}) / (draws + 1)`` of a maximum statistic, decided against ``level`` with as few draws as needed.

    ``draw_maxima(k)`` returns ``k`` fresh null maxima. In the worst case ``2 / level`` draws are made (so a Bonferroni level over many columns is reachable at all), up to ``MAX_DRAWS``; the loop stops as soon
    as the exceedances already make the final p-value exceed ``level``. The returned value is the p-value at the point of stopping (a conservative number when stopped early).
    """
    total = int(min(MAX_DRAWS, max(BATCH, np.ceil(2.0 / max(level, 1e-9)))))
    done = exceed = 0
    while done < total:
        batch = min(BATCH, total - done)
        exceed += int((np.asarray(draw_maxima(batch)) >= observed).sum())
        done += batch
        if (1.0 + exceed) / (total + 1.0) > level:
            break
    return (1.0 + exceed) / (done + 1.0)


@dataclass(frozen=True)
class UnaryForm:
    """One unary candidate with the constants fitted on the discovery rows, so the same column can be rebuilt on any rows (a recipe)."""

    name: str
    median: float
    scale: float
    #: Sorted unique values and normal scores of the discovery rows (``rankgauss`` only; empty otherwise).
    grid_values: tuple = ()
    grid_scores: tuple = ()


def fit_form(name: str, x: np.ndarray) -> "UnaryForm | None":
    """The form ``name`` fitted on the column ``x`` (``None`` when its domain does not hold: log / reciprocal need a positive column, sqrt a non-negative one)."""
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size < 8 or float(finite.std()) <= 0.0:
        return None
    if name in ("log", "recip") and not float(finite.min()) > 0.0:
        return None
    if name == "sqrt" and not float(finite.min()) >= 0.0:
        return None
    median = float(np.median(finite))
    q1, q3 = np.percentile(finite, [25.0, 75.0])
    scale = float(q3 - q1) if q3 - q1 > 1e-12 else float(finite.std())
    if name == "rankgauss":
        from scipy.special import ndtri

        values, _inverse, counts = np.unique(finite, return_inverse=True, return_counts=True)
        mid = np.cumsum(counts) - 0.5 * counts
        return UnaryForm(name, median, scale, tuple(values.tolist()), tuple(ndtri(mid / finite.size).tolist()))
    return UnaryForm(name, median, scale)


def apply_form(form: UnaryForm, x: np.ndarray) -> np.ndarray:
    """The transformed column; non-finite results are replaced by the median of the finite ones so a replay on new rows never returns NaN."""
    x = np.asarray(x, dtype=np.float64)
    name = form.name
    with np.errstate(all="ignore"):
        if name == "log":
            out = np.log(np.maximum(x, 1e-12))
        elif name == "sqrt":
            out = np.sqrt(np.maximum(x, 0.0))
        elif name == "recip":
            out = 1.0 / np.maximum(x, 1e-12)
        elif name == "sq0":
            out = x * x
        elif name == "sq":
            out = (x - form.median) ** 2
        elif name == "abs":
            out = np.abs(x - form.median)
        elif name == "tanh":
            out = np.tanh((x - form.median) / form.scale)
        elif name == "cbrt":
            out = np.cbrt(x)
        elif name == "asinh":
            out = np.arcsinh((x - form.median) / form.scale)
        elif name == "sin":
            out = np.sin(x)
        elif name == "cos":
            out = np.cos(x)
        elif name == "rankgauss":
            out = np.interp(x, np.asarray(form.grid_values), np.asarray(form.grid_scores))
        else:
            raise ValueError(f"unknown unary form {name!r}")
    finite = np.isfinite(out)
    if not finite.all():
        out = np.where(finite, out, np.median(out[finite]) if finite.any() else 0.0)
    return out


def _orthonormal_design(columns: "list[np.ndarray]", n: int) -> np.ndarray:
    """Orthonormal basis (n, k) of the intercept and the given columns (rank-revealing QR drops a collinear one)."""
    design = np.column_stack([np.ones(n)] + [np.asarray(c, dtype=np.float64) for c in columns])
    q, r = np.linalg.qr(design)
    keep = np.abs(np.diag(r)) > 1e-9 * max(1.0, float(np.abs(r).max()))
    return q[:, keep]


@dataclass(frozen=True)
class ColumnVerdict:
    """The decision for one column: the best non-identity form, its gain over the identity (in nats of ``0.5 log((1 - s_id) / (1 - s_f))``), the family-wise p-value and the replication outcome. ``effect`` is the relative reduction of the held-out MAE of a linear model that uses the form instead of the identity (regression) or the held-out deviance gain in nats per row (binary)."""

    form: "UnaryForm | None"
    gain: float
    p_value: float
    replicated: bool
    effect: float
    identity_score: float
    best_score: float


def _scores(forms: "list[UnaryForm]", x: np.ndarray, q: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """``s`` of every candidate against every residual vector (columns of ``residuals``: the observed one first, then the permuted ones): (1 + n_candidates, n_vectors), the identity in row 0."""
    cand = np.column_stack([np.asarray(x, dtype=np.float64)] + [apply_form(f, x) for f in forms])
    cand = cand - cand.mean(axis=0)
    raw_norms = np.einsum("ij,ij->j", cand, cand)
    cand = cand - q @ (q.T @ cand)
    norms = np.einsum("ij,ij->j", cand, cand)
    res_norm = np.einsum("ij,ij->j", residuals, residuals)
    cross = cand.T @ residuals
    ok = norms > 1e-10 * np.maximum(raw_norms, 1e-300)
    return np.where(ok[:, None], cross**2 / (np.where(ok, norms, 1.0)[:, None] * np.maximum(res_norm, 1e-300)[None, :]), 0.0)


def _test_column(
    x: np.ndarray, y: np.ndarray, others: "list[np.ndarray]", forms: "list[UnaryForm]", level: float, rng: np.random.Generator
) -> "tuple[int, float, float, float, float, float]":
    """``(best candidate index, T, p, s_id, s_best, residual share)`` of one column against the residual of ``y`` on the intercept and ``others``; p is the max-over-candidates permutation p-value decided at
    ``level`` (see :func:`adaptive_max_pvalue`) and the residual share is the fraction of the variance of ``y`` the others leave. Among the candidates the null spread cannot tell from the best, the simplest is reported.
    """
    n = x.shape[0]
    # the design holds the column ITSELF: the residual is what the identity leaves, and a candidate is scored by what it explains of that leftover (an added-variable test of curvature, far more powerful than the
    # difference of two near-collinear correlations: ``sin`` against ``x`` on [-1, 1] is detectable at 3600 rows by this and not by that)
    q = _orthonormal_design([*list(others), np.asarray(x, dtype=np.float64)], n)
    yc = np.asarray(y, dtype=np.float64)
    r = yc - q @ (q.T @ yc)
    s_obs = _scores(forms, x, q, r[:, None])[:, 0]
    gains = s_obs[1:] - s_obs[0]
    first_batch: "list[np.ndarray]" = []

    def draw_maxima(k: int) -> np.ndarray:
        """``k`` null values of ``max_f (s_f - s_id)`` (the residual permuted and re-projected on the complement of the design)."""
        block = np.empty((n, k))
        for b in range(k):
            block[:, b] = r[rng.permutation(n)]
        block -= q @ (q.T @ block)
        s = _scores(forms, x, q, block)
        null_gains = s[1:] - s[0][None, :]
        if not first_batch:
            first_batch.append(null_gains)
        return null_gains.max(axis=0)

    p = adaptive_max_pvalue(float(gains.max()), draw_maxima, level)
    slack = float(first_batch[0][int(np.argmax(gains))].std()) if first_batch else 0.0
    best = simplest_tied(gains, [f.name for f in forms], slack) if float(gains.max()) > 0.0 else int(np.argmax(gains))
    share = float(r @ r) / max(float(((yc - yc.mean()) ** 2).sum()), 1e-300)
    return best, float(gains[best]), p, float(s_obs[0]), float(s_obs[1 + best]), share


def heldout_mae_gain(
    y_fit: np.ndarray,
    others_fit: "list[np.ndarray]",
    x_fit: np.ndarray,
    y_test: np.ndarray,
    others_test: "list[np.ndarray]",
    x_test: np.ndarray,
    form: UnaryForm,
) -> float:
    """Relative reduction of the MAE on the test rows of a least-squares model on ``[1, others, f(x)]`` against ``[1, others, x]``, both fitted on the fit rows: the effect of a credit in the metric the owner reads (MAE first)."""

    def mae(v_fit: np.ndarray, v_test: np.ndarray) -> float:
        """MAE of the model with the column ``v`` on the test rows."""
        design_fit = np.column_stack([np.ones(y_fit.size)] + [np.asarray(o, dtype=np.float64) for o in others_fit] + [v_fit])
        design_test = np.column_stack([np.ones(y_test.size)] + [np.asarray(o, dtype=np.float64) for o in others_test] + [v_test])
        coef = np.linalg.lstsq(design_fit, y_fit, rcond=None)[0]
        return float(np.abs(y_test - design_test @ coef).mean())

    mae_id = mae(np.asarray(x_fit, dtype=np.float64), np.asarray(x_test, dtype=np.float64))
    mae_form = mae(apply_form(form, x_fit), apply_form(form, x_test))
    return (mae_id - mae_form) / max(mae_id, 1e-300)


def _split(n: int, rng: np.random.Generator) -> "tuple[np.ndarray, np.ndarray]":
    """Random halves of the rows: discovery and replication."""
    order = rng.permutation(n)
    half = n // 2
    return np.sort(order[:half]), np.sort(order[half:])


def credit_columns(
    X: np.ndarray,
    y: np.ndarray,
    *,
    candidates: "tuple[str, ...]" = CANDIDATES,
    alpha: float = ALPHA,
    seed: int = 0,
    passes: int = 2,
) -> "list[ColumnVerdict]":
    """One verdict per column of ``X`` (the columns already selected, in the order of their rank): which unary form, if any, a linear model should use instead of the identity.

    Columns are tested one at a time against the others, which enter the design in their current form (a column credited in the first pass enters the second pass transformed, so a transform that explains
    what a partner also explains is not credited twice). ``passes`` is the number of such sweeps; the decision of the last one stands. ``y`` is a continuous target (a binary class code works).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, m = X.shape
    rng = np.random.default_rng(seed)
    if n > MAX_ROWS:
        keep = np.sort(rng.choice(n, MAX_ROWS, replace=False))
        X, y = X[keep], y[keep]
        n = MAX_ROWS
    none = ColumnVerdict(None, 0.0, 1.0, False, 0.0, 0.0, 0.0)
    if n < 400 or m == 0 or not np.isfinite(y).all() or float(y.std()) <= 0.0:
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
            others_disc = [current[k][disc] for k in range(m) if k != j]
            best, gain, p_raw, s_id, s_best, _ = _test_column(raw[disc], y[disc], others_disc, forms, alpha / m, rng)
            p_family = min(1.0, p_raw * m)
            if p_family > alpha or gain <= 0.0:
                current[j], verdicts[j] = raw.copy(), none
                continue
            best_form = forms[best]
            others_held = [current[k][held] for k in range(m) if k != j]
            _, gain_h, p_h, _, _, _ = _test_column(raw[held], y[held], others_held, [best_form], alpha, rng)
            effect = heldout_mae_gain(y[disc], others_disc, raw[disc], y[held], others_held, raw[held], best_form)
            replicated = p_h <= alpha and gain_h > 0.0 and effect > 0.0
            if not replicated:
                current[j], verdicts[j] = raw.copy(), none
                continue
            nats = 0.5 * float(np.log((1.0 - min(s_id, 0.999999)) / (1.0 - min(s_best, 0.999999))))
            verdicts[j] = ColumnVerdict(best_form, nats, p_family, True, effect, s_id, s_best)
            current[j] = apply_form(best_form, raw)
    return verdicts
