"""Credit of a unary form for a column whose effect on the target runs through PRODUCTS with other columns (``y = u1(x1) * u2(x2)``, ``y = sum_{i<j} u_i(x_i) u_j(x_j)``).

The added-variable test of ``_unary_credit_stats`` asks whether ``f(x_i)`` explains what the identity leaves of ``y`` linearly. When the column acts only through a product its linear effect is zero (``sin(x1) * log(x2)``
has no main effect in either factor), so that test is blind. Conditional on the partner ``x_j`` the target is linear in ``u_i(x_i)`` with a coefficient that is a function of ``x_j``; the test here lets the
coefficient be flexible: the candidate block is ``[f(x_i), f(x_i) * h_1(x_j), ..., f(x_i) * h_K(x_j)]`` with ``h_k`` the rank-hat basis of the partner (any smooth shape of ``x_j``), the null design holds the same
block of the IDENTITY, so a form is scored by what it explains of the leftover beyond the identity (a multi-degree-of-freedom added-variable statistic, ``||Q_f' r||^2 / ||r||^2``). The maximum over the candidate
forms and over the partners is calibrated by one sign-flip null of the residuals (a wild bootstrap: heteroscedastic residuals keep their variance), adaptive and replicated on a held-out half exactly like the single-column test; among forms the null spread cannot tell
apart the simplest closed form is named, and a form must also win as a REPLACEMENT out of fold (the same filter as the single-column test).

Where the columns come from is the caller's business (``_fe_unary_credit``): the operands of the accepted pair features, whose partners are the other operands. The pass is purely additive."""

from __future__ import annotations

import numpy as np

from ._fe_credit_parsimony import rank_hats
from ._unary_credit_stats import (
    ALPHA,
    CANDIDATES,
    MAX_ROWS,
    OOF_FOLDS,
    ColumnVerdict,
    UnaryForm,
    _orthonormal_design,
    _split,
    adaptive_max_pvalue,
    apply_form,
    candidate_forms,
    gate_mask,
    fold_mae,
    indistinguishable,
    simplest_tied,
)

#: Hats of the partner's rank (the coefficient of the column is any smooth function of the partner on this many knots).
HATS = 8
#: The partners of one column (the first ones of the list the caller passes).
MAX_PARTNERS = 12
#: The interaction tests run at this share of the family-wise level (they come on top of the single-column tests).
LEVEL_SHARE = 0.5


def partner_basis(raw: np.ndarray, current: np.ndarray, credited: bool = False) -> np.ndarray:
    """``(n, 1 + HATS)`` basis of a partner: its current (credited) form, robustly standardised, and the rank hats of the raw column (any smooth shape of it); the form alone (``(n, 1)``) once the partner is credited.

    The hats cover a partner nobody has credited yet; the standardised form is the exact coefficient once the partner's own form is found (a long-tailed form is not a piecewise-linear function of rank), and then it is
    the only degree of freedom the test needs: nine degrees of freedom less, the power of a faint product in a heavy-tailed column is much higher."""
    current = np.asarray(current, dtype=np.float64)
    spread = float(np.subtract(*np.percentile(current, [75, 25]))) / 1.349
    z = np.clip((current - np.median(current)) / (spread if spread > 1e-12 else max(float(current.std()), 1e-12)), -5.0, 5.0)
    return z[:, None] if credited else np.column_stack([z, rank_hats(np.asarray(raw, dtype=np.float64), HATS)])


def _block(values: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """``[v, v * b_1, ..., v * b_K]``: the column and its product with every function of the partner."""
    values = np.asarray(values, dtype=np.float64)
    return np.column_stack([values, values[:, None] * basis])


class _PartnerDesign:
    """The null design of one partner (intercept, the other columns, the partner's basis as a main effect, the identity block of the column), the residual of ``y`` on it and the orthonormal candidate blocks of every form partialled on it. Built from the column ``x``, the target ``y``, the other columns ``mains``, the partner's ``basis`` and the candidate ``forms``."""

    def __init__(self, x: np.ndarray, y: np.ndarray, mains: "list[np.ndarray]", basis: np.ndarray, forms: "list[UnaryForm]") -> None:
        n = x.shape[0]
        self.q0 = _orthonormal_design([*mains, *basis.T, *_block(x, basis).T], n)
        self.residual = y - self.q0 @ (self.q0.T @ y)
        self.blocks = []
        for form in forms:
            block = _block(apply_form(form, x), basis)
            block = block - block.mean(axis=0)
            block = block - self.q0 @ (self.q0.T @ block)
            q, r = np.linalg.qr(block)
            self.blocks.append(q[:, np.abs(np.diag(r)) > 1e-7 * max(1.0, float(np.abs(r).max()))])

    def scores(self, residuals: np.ndarray) -> np.ndarray:
        """Share of every residual vector (columns) that each form's block explains: (forms, vectors)."""
        norms = np.maximum(np.einsum("ij,ij->j", residuals, residuals), 1e-300)
        return np.array([((q.T @ residuals) ** 2).sum(axis=0) / norms for q in self.blocks])

    def null_scores(self, signs: np.ndarray) -> np.ndarray:
        """Scores of the residual under the random sign vectors in the columns of ``signs`` (every residual keeps its magnitude: the null variance is the heteroscedastic one), re-projected on the complement of the design."""
        draws = self.residual[:, None] * signs
        draws = draws - self.q0 @ (self.q0.T @ draws)
        return self.scores(draws)


def oof_fold_gains_interaction(y: np.ndarray, mains: "list[np.ndarray]", x: np.ndarray, basis: np.ndarray, form: UnaryForm) -> np.ndarray:
    """Out-of-fold relative MAE reduction per fold of ``[1, mains, basis, f(x) block]`` against ``[1, mains, basis, x block]``: a form that only explains part of the leftover and throws the identity's effect away is no candidate."""
    n = y.size
    base = np.column_stack([np.ones(n), *[np.asarray(m, dtype=np.float64) for m in mains], basis])
    mae_id = fold_mae(y, np.column_stack([base, _block(x, basis)]))
    return np.asarray((mae_id - fold_mae(y, np.column_stack([base, _block(apply_form(form, x), basis)]))) / np.maximum(mae_id, 1e-300))


def oof_replacement_gain_interaction(y: np.ndarray, mains: "list[np.ndarray]", x: np.ndarray, basis: np.ndarray, form: UnaryForm) -> float:
    """Mean over the folds of :func:`oof_fold_gains_interaction`."""
    return float(oof_fold_gains_interaction(y, mains, x, basis, form).mean())


def oof_squared_errors_interaction(y: np.ndarray, mains: "list[np.ndarray]", x: np.ndarray, basis: np.ndarray, form: UnaryForm, folds: int = OOF_FOLDS) -> np.ndarray:
    """Pointwise out-of-fold squared errors of ``[1, mains, basis, f(x) block]``."""
    design = np.column_stack([np.ones(y.size), *[np.asarray(m, dtype=np.float64) for m in mains], basis, _block(apply_form(form, x), basis)])
    fold = np.arange(y.size) % folds
    err = np.empty(y.size)
    for k in range(folds):
        test = fold == k
        coef = np.linalg.lstsq(design[~test], y[~test], rcond=None)[0]
        err[test] = (y[test] - design[test] @ coef) ** 2
    return err


def _test_column(
    x: np.ndarray, y: np.ndarray, mains: "list[np.ndarray]", bases: "list[np.ndarray]", forms: "list[UnaryForm]", level: float, rng: np.random.Generator
) -> "tuple[int, float, float, int]":
    """``(best form index, explained share of the leftover, p, best partner index)`` of one column: the maximum over forms and partners, decided at ``level`` with as few permutation draws as needed."""
    n = x.shape[0]
    x = np.asarray(x, dtype=np.float64)
    designs = [_PartnerDesign(x, y, mains, basis, forms) for basis in bases]
    observed = np.array([d.scores(d.residual[:, None])[:, 0] for d in designs])  # (partners, forms)
    per_form = observed.max(axis=0)
    gate = gate_mask(forms)

    def draw_maxima(k: int) -> np.ndarray:
        """``k`` null values of the maximum over forms and partners."""
        signs = rng.choice([-1.0, 1.0], size=(n, k))
        null = np.array([d.null_scores(signs) for d in designs])  # (partners, forms, k)
        return np.asarray(null[:, gate].max(axis=(0, 1)))

    p = adaptive_max_pvalue(float(observed[:, gate].max()), draw_maxima, level)
    if p > level:
        best = int(np.argmax(per_form))
        return best, float(per_form[best]), p, int(np.argmax(observed[:, best]))
    fold_gains = np.array([oof_fold_gains_interaction(y, mains, x, bases[int(np.argmax(observed[:, f]))], forms[f]) for f in range(len(forms))])
    usable = fold_gains.mean(axis=1) > 0.0
    if not usable.any():
        best = int(np.argmax(per_form))
        return best, float(per_form[best]), 1.0, int(np.argmax(observed[:, best]))
    masked = np.where(usable, per_form, -np.inf)
    which = int(np.argmax(observed[:, int(np.argmax(masked))]))
    errors = np.column_stack([oof_squared_errors_interaction(y, mains, x, bases[which], f) if usable[i] else np.full(y.size, np.inf) for i, f in enumerate(forms)])
    top = int(np.argmin(np.where(usable, errors.mean(axis=0), np.inf)))
    best = simplest_tied(np.where(indistinguishable(errors, top, usable), per_form, -np.inf), [f.name for f in forms], float(per_form.max()))
    return best, float(per_form[best]), p, int(np.argmax(observed[:, best]))


def tied_form_names_interaction(
    y: np.ndarray, mains: "list[np.ndarray]", x: np.ndarray, basis: np.ndarray, forms: "list[UnaryForm]", chosen: int
) -> "tuple[str, ...]":
    """Names of the forms (other than the chosen one's) the out-of-fold errors of the block models cannot tell from the best predicting form."""
    usable = np.array([oof_replacement_gain_interaction(y, mains, x, basis, f) > 0.0 for f in forms])
    if not usable.any():
        return ()
    errors = np.column_stack([oof_squared_errors_interaction(y, mains, x, basis, f) if usable[i] else np.full(y.size, np.inf) for i, f in enumerate(forms)])
    top = int(np.argmin(np.where(usable, errors.mean(axis=0), np.inf)))
    return tuple(sorted({forms[i].name for i in np.flatnonzero(indistinguishable(errors, top, usable))} - {forms[chosen].name}))


def heldout_mae_gain_interaction(
    y_fit: np.ndarray,
    mains_fit: "list[np.ndarray]",
    x_fit: np.ndarray,
    basis_fit: np.ndarray,
    y_test: np.ndarray,
    mains_test: "list[np.ndarray]",
    x_test: np.ndarray,
    basis_test: np.ndarray,
    form: UnaryForm,
) -> float:
    """Relative reduction of the MAE on the test rows of the model with the form's block against the one with the identity's block, both fitted on the fit rows."""

    def mae(v_fit: np.ndarray, v_test: np.ndarray) -> float:
        """MAE of the model with the block of ``v`` on the test rows."""
        design_fit = np.column_stack([np.ones(y_fit.size), *[np.asarray(m, dtype=np.float64) for m in mains_fit], basis_fit, _block(v_fit, basis_fit)])
        design_test = np.column_stack([np.ones(y_test.size), *[np.asarray(m, dtype=np.float64) for m in mains_test], basis_test, _block(v_test, basis_test)])
        coef = np.linalg.lstsq(design_fit, y_fit, rcond=None)[0]
        return float(np.abs(y_test - design_test @ coef).mean())

    mae_id = mae(np.asarray(x_fit, dtype=np.float64), np.asarray(x_test, dtype=np.float64))
    return (mae_id - mae(apply_form(form, x_fit), apply_form(form, x_test))) / max(mae_id, 1e-300)


def credit_interactions(
    X: np.ndarray,
    y: np.ndarray,
    *,
    start: "list[UnaryForm | None] | None" = None,
    skip: "tuple[int, ...]" = (),
    candidates: "tuple[str, ...]" = CANDIDATES,
    alpha: float = ALPHA,
    seed: int = 0,
    passes: int = 2,
) -> "tuple[list[ColumnVerdict], list[int | None]]":
    """``(verdicts, partners)`` of the columns of ``X``: the unary form each column's PRODUCT effect with the other columns calls for, and the index of the partner it showed in.

    ``start`` holds the forms already credited to a column by the single-column test (they enter as the partners' current forms); ``skip`` lists the columns not to test. A column is tested against at most
    ``MAX_PARTNERS`` others, in the order of ``X`` (the caller puts the likeliest partners first). The decision of the last pass stands; the family-wise level is ``LEVEL_SHARE * alpha`` over the columns tested."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, m = X.shape
    rng = np.random.default_rng(seed)
    if n > MAX_ROWS:
        keep = np.sort(rng.choice(n, MAX_ROWS, replace=False))
        X, y = X[keep], y[keep]
        n = MAX_ROWS
    none = ColumnVerdict(None, 0.0, 1.0, False, 0.0, 0.0, 0.0)
    verdicts = [none] * m
    partner_of: "list[int | None]" = [None] * m
    if n < 400 or m < 2 or not np.isfinite(y).all() or float(y.std()) <= 0.0:
        return verdicts, partner_of
    disc, held = _split(n, rng)
    def initial(j: int) -> np.ndarray:
        """The column ``j`` as the single-column test left it: its credited form, or the raw column."""
        form = None if start is None else start[j]
        return X[:, j].copy() if form is None else np.asarray(apply_form(form, X[:, j]))

    current = [initial(j) for j in range(m)]
    tested = [j for j in range(m) if j not in skip and np.isfinite(X[:, j]).all()]
    level = LEVEL_SHARE * alpha / max(len(tested), 1)
    for _ in range(max(1, passes)):
        for j in tested:
            raw = X[:, j]
            forms = candidate_forms(raw[disc], candidates)
            partners = [k for k in range(m) if k != j][:MAX_PARTNERS]
            if not forms or not partners:
                continue
            # the products of the pairs already credited sit in the null design: the leftover of ``y`` would otherwise hold their (large) signal, whose variance depends on the partners and makes the permutation null liberal
            nuisance = [current[i] * current[k] for i, k in enumerate(partner_of) if k is not None and i != j and k != j]
            mains_disc = [current[k][disc] for k in range(m) if k != j] + [c[disc] for c in nuisance]
            bases_disc = [partner_basis(X[disc, k], current[k][disc]) for k in partners]
            best, share, p_raw, which = _test_column(raw[disc], y[disc], mains_disc, bases_disc, forms, level, rng)
            p_family = min(1.0, p_raw * len(tested))
            if p_family > LEVEL_SHARE * alpha or share <= 0.0:
                current[j] = initial(j)
                verdicts[j], partner_of[j] = none, None
                continue
            form = forms[best]
            mains_held = [current[k][held] for k in range(m) if k != j] + [c[held] for c in nuisance]
            bases_held = [partner_basis(X[held, k], current[k][held]) for k in partners]
            _, share_h, p_h, _ = _test_column(raw[held], y[held], mains_held, bases_held, [form], alpha, rng)
            effect = heldout_mae_gain_interaction(y[disc], mains_disc, raw[disc], bases_disc[which], y[held], mains_held, raw[held], bases_held[which], form)
            if not (p_h <= alpha and share_h > 0.0 and effect > 0.0):
                current[j] = initial(j)
                verdicts[j], partner_of[j] = none, None
                continue
            nats = -0.5 * float(np.log(max(1.0 - min(share, 0.999999), 1e-12)))
            verdicts[j] = ColumnVerdict(form, nats, p_family, True, effect, 0.0, share, tied_form_names_interaction(y[disc], mains_disc, raw[disc], bases_disc[which], forms, best))
            partner_of[j] = partners[which]
            current[j] = apply_form(form, raw)
    return verdicts, partner_of


def _product_stat(x: np.ndarray, y: np.ndarray, design: np.ndarray, rng: np.random.Generator, level: "float | None") -> "tuple[float, float]":
    """``(share of the leftover the product column explains, p)``: the added-variable test of one column ``x`` on the intercept-and-``design`` model, against the sign-flip null decided at ``level`` (``None``: the share only, p is 1)."""
    n = y.shape[0]
    q = _orthonormal_design([c for c in design.T], n)
    r = y - q @ (q.T @ y)
    c = np.asarray(x, dtype=np.float64) - np.asarray(x, dtype=np.float64).mean()
    c = c - q @ (q.T @ c)
    norm = float(c @ c)
    if norm <= 1e-10 * max(float(np.asarray(x, dtype=np.float64).var() * n), 1e-300):
        return 0.0, 1.0
    c = c / np.sqrt(norm)
    r_norm = max(float(r @ r), 1e-300)
    observed = float((c @ r) ** 2 / r_norm)
    if level is None:
        return observed, 1.0

    def draw_maxima(k: int) -> np.ndarray:
        """``k`` null values of the share (the residual keeps its magnitudes, random signs)."""
        draws = r[:, None] * rng.choice([-1.0, 1.0], size=(n, k))
        return np.asarray((c @ draws) ** 2 / np.maximum(np.einsum("ij,ij->j", draws, draws), 1e-300))

    return observed, adaptive_max_pvalue(observed, draw_maxima, level)


def select_products(
    X: np.ndarray, y: np.ndarray, forms: "list[UnaryForm | None]", pairs: "list[tuple[int, int]]", *, alpha: float = ALPHA, seed: int = 0
) -> "list[tuple[int, int]]":
    """The pairs ``(i, j)`` of columns whose product ``f_i(x_i) * f_j(x_j)`` (the credited forms, the identity where ``None``) the target needs beyond the columns in their credited forms and the products accepted before.

    Forward selection on a discovery half: the best remaining product is tested (added variable, sign-flip null, Bonferroni over the pairs) and accepted only if it replicates on the held-out half (own test at ``alpha``
    and a lower held-out MAE of the least-squares fit); the first rejection ends the search. A product of two columns that only act additively is not accepted, however well each form fits."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, m = X.shape
    rng = np.random.default_rng(seed)
    if n > MAX_ROWS:
        keep = np.sort(rng.choice(n, MAX_ROWS, replace=False))
        X, y = X[keep], y[keep]
        n = MAX_ROWS
    if n < 400 or not pairs or not np.isfinite(y).all() or float(y.std()) <= 0.0:
        return []
    z = np.column_stack([X[:, k] if f is None else apply_form(f, X[:, k]) for k in range(m) for f in [forms[k]]])
    disc, held = _split(n, rng)
    level = alpha / len(pairs)
    accepted: "list[tuple[int, int]]" = []
    remaining = list(pairs)
    while remaining:
        design_disc = np.column_stack([z[disc]] + [z[disc, i] * z[disc, j] for i, j in accepted]) if accepted else z[disc]
        stats = [_product_stat(z[disc, i] * z[disc, j], y[disc], design_disc, rng, None)[0] for i, j in remaining]
        i, j = remaining[int(np.argmax(stats))]
        share, p = _product_stat(z[disc, i] * z[disc, j], y[disc], design_disc, rng, level)
        if p > level or share <= 0.0:
            break
        design_held = np.column_stack([z[held]] + [z[held, a] * z[held, b] for a, b in accepted]) if accepted else z[held]
        share_h, p_h = _product_stat(z[held, i] * z[held, j], y[held], design_held, rng, alpha)
        base = np.column_stack([np.ones(disc.size), design_disc])
        extended = np.column_stack([base, z[disc, i] * z[disc, j]])
        coef0 = np.linalg.lstsq(base, y[disc], rcond=None)[0]
        coef1 = np.linalg.lstsq(extended, y[disc], rcond=None)[0]
        base_h = np.column_stack([np.ones(held.size), design_held])
        mae0 = float(np.abs(y[held] - base_h @ coef0).mean())
        mae1 = float(np.abs(y[held] - np.column_stack([base_h, z[held, i] * z[held, j]]) @ coef1).mean())
        if not (p_h <= alpha and share_h > 0.0 and mae1 < mae0):
            break
        accepted.append((i, j))
        remaining.remove((i, j))
    return accepted
