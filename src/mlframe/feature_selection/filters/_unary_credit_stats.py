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

import dataclasses
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
#: Folds of the out-of-fold replacement check that decides which candidate forms may be credited at all.
OOF_FOLDS = 5
#: Null draws come in batches; a column stops as soon as its p-value cannot reach the level any more (most columns stop after the first batch).
BATCH = 50
MAX_DRAWS = 2000


#: A candidate is tied with the best predicting one when the paired test of their out-of-fold squared errors (Diebold-Mariano, equal degrees of freedom) is below this z: the data cannot tell them apart.
TIE_Z = 2.5


def indistinguishable(errors: np.ndarray, best: int, candidates: np.ndarray, z_crit: float = TIE_Z) -> np.ndarray:
    """Mask of the ``candidates`` (a boolean mask over the columns of ``errors``) whose pointwise squared error on the leftover is not significantly larger than that of column ``best``.

    ``errors`` is (rows, forms): for each form the pointwise squared error of its model. The paired z is ``mean(d) / (sd(d) / sqrt(n))`` with ``d = e_i - e_best``, so a form that the best one beats by a
    consistent sliver (``tanh(x)`` against ``sin(x)`` on a normal column) is told apart and one that differs by noise (``sin`` against ``tanh(x / 2)`` on [-1, 1]) is not."""
    n = errors.shape[0]
    tied = np.zeros(errors.shape[1], dtype=bool)
    for i in np.flatnonzero(candidates):
        d = errors[:, i] - errors[:, best]
        spread = float(d.std(ddof=1))
        tied[i] = i == best or float(d.mean()) <= z_crit * spread / np.sqrt(n) if spread > 0.0 else True
    return tied


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


#: How the null residual vectors are drawn: ``signflip`` (every residual keeps its magnitude and gets a random sign: a wild bootstrap whose null variance is the heteroscedastic one) or ``permute`` (rows shuffled, the
#: homoscedastic null). Measured 2026-11-24 on ``y = cbrt(x1) * asinh(x2)`` with a t3 column (residual variance tied to the column): permute credited a form in 3 of 40 datasets, signflip in 0 of 40; power and the
#: homoscedastic controls are the same.
NULL_KIND = "signflip"


def null_residuals(r: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """``k`` null draws (columns) of the residual vector ``r`` under the hypothesis that the tested column explains nothing of it."""
    if NULL_KIND == "signflip":
        return np.asarray(r[:, None] * rng.choice([-1.0, 1.0], size=(r.shape[0], k)))
    block = np.empty((r.shape[0], k))
    for b in range(k):
        block[:, b] = r[rng.permutation(r.shape[0])]
    return block


def adaptive_two_pvalues(observed: "tuple[float, float]", draw_stats, level: float) -> "tuple[float, float]":
    """Permutation p-values ``(1 + #{null >= observed}) / (draws + 1)`` of two statistics computed on the SAME draws, each decided against ``level`` with as few draws as needed.

    ``draw_stats(k)`` returns two arrays of ``k`` null values. The loop stops when neither statistic can reach ``level`` any more (the exceedances of both already make their final p-values larger); a statistic
    that cannot reach it keeps its conservative p-value at the point of stopping."""
    total = int(min(MAX_DRAWS, max(BATCH, np.ceil(2.0 / max(level, 1e-9)))))
    done = 0
    exceed = [0, 0]
    while done < total:
        batch = min(BATCH, total - done)
        null_a, null_b = draw_stats(batch)
        exceed[0] += int((np.asarray(null_a) >= observed[0]).sum())
        exceed[1] += int((np.asarray(null_b) >= observed[1]).sum())
        done += batch
        if all((1.0 + e) / (total + 1.0) > level for e in exceed):
            break
    return (1.0 + exceed[0]) / (done + 1.0), (1.0 + exceed[1]) / (done + 1.0)


def curvature_basis(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Orthonormal basis (n, <=2) of the second and third powers of the robustly scaled column, partialled on the design ``q`` (which already holds the column itself): the generic curvature the omnibus test asks about.

    The scale is the median and the interquartile range (clipped at 5), so a heavy tail does not own the basis."""
    x = np.asarray(x, dtype=np.float64)
    spread = float(np.subtract(*np.percentile(x, [75, 25]))) / 1.349
    u = np.clip((x - np.median(x)) / (spread if spread > 1e-12 else max(float(x.std()), 1e-12)), -5.0, 5.0)
    basis = np.column_stack([u**2, u**3])
    basis = basis - q @ (q.T @ basis)
    qb, rr = np.linalg.qr(basis)
    return qb[:, np.abs(np.diag(rr)) > 1e-9]


@dataclass(frozen=True)
class UnaryForm:
    """One unary candidate with the constants fitted on the discovery rows, so the same column can be rebuilt on any rows (a recipe)."""

    name: str
    median: float
    scale: float
    #: Sorted unique values and normal scores of the discovery rows (``rankgauss`` only; empty otherwise).
    grid_values: tuple = ()
    grid_scores: tuple = ()
    #: A scaled variant of a base form (``tanh`` at a multiple of the base scale): it competes for the NAME of a credited column but not for the significance of the credit.
    variant: bool = False


def fit_form(name: str, x: np.ndarray, scale_mult: float = 1.0) -> "UnaryForm | None":
    """The form ``name`` fitted on the column ``x`` (``None`` when its domain does not hold: log / reciprocal need a positive column, sqrt a non-negative one); ``scale_mult`` multiplies the scale of the forms that have one (``tanh``, ``asinh``)."""
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
    scale = (float(q3 - q1) if q3 - q1 > 1e-12 else float(finite.std())) * float(scale_mult)
    if name == "rankgauss":
        from scipy.special import ndtri

        values, _inverse, counts = np.unique(finite, return_inverse=True, return_counts=True)
        mid = np.cumsum(counts) - 0.5 * counts
        return UnaryForm(name, median, scale, tuple(values.tolist()), tuple(ndtri(mid / finite.size).tolist()))
    return UnaryForm(name, median, scale)


#: The forms with a free scale are tried at these multiples of the interquartile range of the column, a factor of the square root of two apart (``tanh(x)`` and ``tanh(x / 4)`` are different shapes on the same column,
#: and a scale off by a factor of 1.5 makes ``asinh`` the better fit of a ``tanh``).
SCALE_MULTS = tuple(2.0 ** (k / 2.0) for k in range(-4, 5))
SCALED_FORMS = ("tanh", "asinh")


def candidate_forms(x: np.ndarray, names: "tuple[str, ...]" = CANDIDATES) -> "list[UnaryForm]":
    """Every candidate form fitted on ``x`` (those whose domain holds), the scaled ones at each multiple of ``SCALE_MULTS``."""
    forms = []
    for name in names:
        for mult in SCALE_MULTS if name in SCALED_FORMS else (1.0,):
            form = fit_form(name, x, mult)
            if form is not None:
                forms.append(form if mult == 1.0 else dataclasses.replace(form, variant=True))
    return forms


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
    return np.asarray(out)


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
    #: Names of the other forms the data cannot tell from the credited one (their out-of-fold errors are not significantly larger): the generating form is among ``form`` and these.
    tied: tuple = ()


def gate_mask(forms: "list[UnaryForm]") -> np.ndarray:
    """Which candidates count for the significance of a credit: the base forms (a scaled variant only competes for the name afterwards), or all of them when nothing but variants is left (the replication on held-out rows)."""
    gate = np.array([not f.variant for f in forms])
    return gate if gate.any() else np.ones(len(forms), dtype=bool)


def _partialled(forms: "list[UnaryForm]", x: np.ndarray, q: np.ndarray) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """``(candidates, norms, ok)``: the identity and the forms of ``x``, centred and partialled on the design ``q``; their squared norms and whether each one kept a non-negligible part."""
    cand = np.column_stack([np.asarray(x, dtype=np.float64)] + [apply_form(f, x) for f in forms])
    cand = cand - cand.mean(axis=0)
    raw_norms = np.einsum("ij,ij->j", cand, cand)
    cand = cand - q @ (q.T @ cand)
    norms = np.einsum("ij,ij->j", cand, cand)
    return cand, norms, norms > 1e-10 * np.maximum(raw_norms, 1e-300)


def _scores(forms: "list[UnaryForm]", x: np.ndarray, q: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """``s`` of every candidate against every residual vector (columns of ``residuals``: the observed one first, then the permuted ones): (1 + n_candidates, n_vectors), the identity in row 0."""
    cand, norms, ok = _partialled(forms, x, q)
    res_norm = np.einsum("ij,ij->j", residuals, residuals)
    cross = cand.T @ residuals
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
    gate = gate_mask(forms)
    curvature = curvature_basis(x, q)
    r_norm = max(float(r @ r), 1e-300)
    omnibus_observed = float(((curvature.T @ r) ** 2).sum() / r_norm) if curvature.shape[1] else 0.0

    def draw_stats(k: int) -> "tuple[np.ndarray, np.ndarray]":
        """``k`` null values of ``max_f (s_f - s_id)`` and of the omnibus statistic (the residual permuted and re-projected on the complement of the design)."""
        block = null_residuals(r, k, rng)
        block -= q @ (q.T @ block)
        s = _scores(forms, x, q, block)
        null_gains = s[1:] - s[0][None, :]
        omnibus = ((curvature.T @ block) ** 2).sum(axis=0) / np.maximum(np.einsum("ij,ij->j", block, block), 1e-300) if curvature.shape[1] else np.zeros(k)
        return null_gains[gate].max(axis=0), omnibus

    # two tests of the same hypothesis ("the column itself leaves curvature"), each at half the level: the max over the candidate forms (best when one form is exact and strong), and an omnibus on the second and
    # third powers (best when the curvature is faint, as the cubic term of a sine on [-1, 1]: credited in 8 of 12 seeds at 3600 rows against 3 of 12 by the candidates alone)
    p_forms, p_omnibus = adaptive_two_pvalues((float(gains[gate].max()), omnibus_observed), draw_stats, level / 2.0)
    p = min(1.0, 2.0 * min(p_forms, p_omnibus))
    if p > level:
        return int(np.argmax(gains)), float(gains.max()), p, float(s_obs[0]), float(s_obs[1 + int(np.argmax(gains))]), float(r @ r) / max(float(((yc - yc.mean()) ** 2).sum()), 1e-300)
    # only a form that also wins as a REPLACEMENT out of fold may be credited (the added-variable score keeps the raw column in the design and cannot see what a replacement throws away)
    fold_gains = np.array([oof_fold_gains(yc, others, x, f) for f in forms])
    usable = fold_gains.mean(axis=1) > 0.0
    masked = np.where(usable, gains, -np.inf)
    if not np.isfinite(masked).any() or float(masked.max()) <= 0.0:
        return int(np.argmax(gains)), float(gains.max()), 1.0, float(s_obs[0]), float(s_obs[1 + int(np.argmax(gains))]), float(r @ r) / max(float(((yc - yc.mean()) ** 2).sum()), 1e-300)
    # the form that PREDICTS best out of fold (squared error, as a replacement) is the reference; every form the paired test of the pointwise errors cannot tell from it is tied with it and the simplest closed form wins
    errors = np.column_stack([oof_squared_errors(yc, others, x, f) if usable[i] else np.full(yc.size, np.inf) for i, f in enumerate(forms)])
    top = int(np.argmin(np.where(usable, errors.mean(axis=0), np.inf)))
    best = simplest_tied(np.where(indistinguishable(errors, top, usable), gains, -np.inf), [f.name for f in forms], float(gains.max()))
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


def fold_mae(y: np.ndarray, design: np.ndarray, folds: int = OOF_FOLDS) -> np.ndarray:
    """Out-of-fold MAE of a least-squares fit on ``design``, one value per fold (the rows ``i`` with ``i % folds == k`` are the test fold ``k``)."""
    fold = np.arange(y.size) % folds
    out = np.empty(folds)
    for k in range(folds):
        test = fold == k
        coef = np.linalg.lstsq(design[~test], y[~test], rcond=None)[0]
        out[k] = float(np.abs(y[test] - design[test] @ coef).mean())
    return out


def oof_fold_gains(y: np.ndarray, others: "list[np.ndarray]", x: np.ndarray, form: UnaryForm, folds: int = OOF_FOLDS) -> np.ndarray:
    """Out-of-fold relative MAE reduction of ``[1, others, f(x)]`` against ``[1, others, x]``, one value per fold.

    The added-variable score keeps the column in the design and so cannot see what a REPLACEMENT loses: ``cbrt`` of a column that is a sine on [-1, 1] explains some of the leftover curvature and throws the
    linear signal away (held-out MAE worse by 8 percent), ``sin`` keeps both. A form with no gain here is not a candidate."""
    y = np.asarray(y, dtype=np.float64)
    base = np.column_stack([np.ones(y.size), *[np.asarray(o, dtype=np.float64) for o in others]])
    mae_id = fold_mae(y, np.column_stack([base, np.asarray(x, dtype=np.float64)]), folds)
    return np.asarray((mae_id - fold_mae(y, np.column_stack([base, apply_form(form, x)]), folds)) / np.maximum(mae_id, 1e-300))


def oof_replacement_gain(y: np.ndarray, others: "list[np.ndarray]", x: np.ndarray, form: UnaryForm, folds: int = OOF_FOLDS) -> float:
    """Mean over the folds of :func:`oof_fold_gains`."""
    return float(oof_fold_gains(y, others, x, form, folds).mean())


def oof_squared_errors(y: np.ndarray, others: "list[np.ndarray]", x: np.ndarray, form: UnaryForm, folds: int = OOF_FOLDS) -> np.ndarray:
    """Pointwise out-of-fold squared errors (one per row) of the replacement model ``[1, others, f(x)]``: the input of the paired test that tells forms apart by how well they PREDICT."""
    y = np.asarray(y, dtype=np.float64)
    design = np.column_stack([np.ones(y.size), *[np.asarray(o, dtype=np.float64) for o in others], apply_form(form, x)])
    fold = np.arange(y.size) % folds
    err = np.empty(y.size)
    for k in range(folds):
        test = fold == k
        coef = np.linalg.lstsq(design[~test], y[~test], rcond=None)[0]
        err[test] = (y[test] - design[test] @ coef) ** 2
    return err


def tied_form_names(y: np.ndarray, others: "list[np.ndarray]", x: np.ndarray, forms: "list[UnaryForm]", chosen: int) -> "tuple[str, ...]":
    """Names of the forms (other than the chosen one's) that predict out of fold as well as the best predicting form, up to the noise of the paired test: the alternatives the data cannot rule out."""
    usable = np.array([oof_replacement_gain(y, others, x, f) > 0.0 for f in forms])
    if not usable.any():
        return ()
    errors = np.column_stack([oof_squared_errors(y, others, x, f) if usable[i] else np.full(np.asarray(y).size, np.inf) for i, f in enumerate(forms)])
    top = int(np.argmin(np.where(usable, errors.mean(axis=0), np.inf)))
    mask = indistinguishable(errors, top, usable)
    return tuple(sorted({forms[i].name for i in np.flatnonzero(mask)} - {forms[chosen].name}))


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
    finite = [bool(np.isfinite(X[:, j]).all()) for j in range(m)]
    for _ in range(max(1, passes)):
        for j in range(m):
            raw = X[:, j]
            if not finite[j]:
                continue
            forms = candidate_forms(raw[disc], candidates)
            if not forms:
                continue
            # a column with a non-finite value takes no part in the design of the others either
            others_disc = [current[k][disc] for k in range(m) if k != j and finite[k]]
            best, gain, p_raw, s_id, s_best, _ = _test_column(raw[disc], y[disc], others_disc, forms, alpha / m, rng)
            p_family = min(1.0, p_raw * m)
            if p_family > alpha or gain <= 0.0:
                current[j], verdicts[j] = raw.copy(), none
                continue
            best_form = forms[best]
            others_held = [current[k][held] for k in range(m) if k != j and finite[k]]
            _, gain_h, p_h, _, _, _ = _test_column(raw[held], y[held], others_held, [best_form], alpha, rng)
            effect = heldout_mae_gain(y[disc], others_disc, raw[disc], y[held], others_held, raw[held], best_form)
            replicated = p_h <= alpha and gain_h > 0.0 and effect > 0.0
            if not replicated:
                current[j], verdicts[j] = raw.copy(), none
                continue
            nats = 0.5 * float(np.log((1.0 - min(s_id, 0.999999)) / (1.0 - min(s_best, 0.999999))))
            verdicts[j] = ColumnVerdict(best_form, nats, p_family, True, effect, s_id, s_best, tied_form_names(y[disc], others_disc, raw[disc], forms, best))
            current[j] = apply_form(best_form, raw)
    return verdicts
