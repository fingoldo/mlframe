"""Usability-aware retention of PURE single-pair engineered forms (default post-FE pass).

The post-FE greedy ranks engineered candidates by conditional MI. On an ADDITIVE target whose terms
share operands -- e.g. F2 ``y = a**2/b + log(c)*sin(d)`` -- a single CROSS-MIX feature built from one
operand of EACH term (``sub(invcbrt(add(reciproc(c),invsquared(d))), log(div(sqr(a),neg(b))))``) has
higher marginal/joint MI than either pure pair form (it informs about BOTH additive components), so the
greedy keeps the cross-mix and drops the pure ``a**2/b`` and ``log(c)*sin(d)`` forms as conditionally
redundant. That is the rational MAX-MI pick for a TREE model -- but a LINEAR model cannot use the lossy
cross-mix the way it can use the clean pure forms (the cross-mix is a monotone-warped blend, not the
additive term the linear objective needs). The pure form is then absent from ``get_feature_names_out``
even though it is the genuinely useful feature for the deployed linear/additive model.

This pass RE-ATTACHES a pure single-pair engineered form whenever a CROSS-VALIDATED LINEAR wrapper
(``usability_greedy``) confirms it lowers the linear CV-MAE on top of the current selection AND the
pair is NOT already represented by a pure (<=2-operand) selected feature. The recovered form is a
replayable :class:`EngineeredRecipe` appended to ``_engineered_recipes_`` -- it rides the existing
engineered-output path (``transform`` replays it; ``get_feature_names_out`` lists it) and never touches
``support_`` or the screening matrix. It is purely ADDITIVE: nothing the MI greedy chose is removed, so
a tree pipeline keeps its features and only gains the pure forms a linear model needs.

The CV-MAE gate is the firing discriminator: on a dataset where the pure form adds no linear value
(its operands carry no joint interaction, or the MI selection already holds the pure form) the greedy
does not pick it and the pass is a no-op -- so the default selection stays byte-identical there.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def retain_usable_pure_forms(
    mrmr: Any,
    X: Any,
    y_cont: "np.ndarray | None",
    *,
    w: float = 0.7,
    K: int = 6,
    seed: int = 0,
    max_added: int = 4,
    max_base_features: int = 14,
    max_rows: int = 3000,
    verbose: int = 0,
):
    """Return ``[(recipe, name), ...]`` of PURE single-pair engineered forms to ADD to
    ``mrmr._engineered_recipes_`` so a linearly-usable pair interaction the MI greedy left trapped
    inside a cross-mix (or only as separate raw operands) is recovered.

    Gated and best-effort: any failure (no continuous target, degenerate pool, non-numeric frame)
    returns ``[]`` and never disturbs the fit. ``max_added`` caps the blast radius; ``w`` leans the
    usability pre-rank toward linear usability; the COMMIT decision is always the CV-MAE improvement
    inside :func:`usability_greedy`."""
    if y_cont is None:
        return []
    try:
        import pandas as pd

        if not isinstance(X, pd.DataFrame):
            return []
        y_cont = np.asarray(y_cont, dtype=np.float64).ravel()
        if y_cont.shape[0] != len(X) or not np.isfinite(y_cont).any():
            return []

        # Operand-pairs already covered by a PURE (<=2-operand) selected engineered form: those are
        # not trapped, so do not duplicate them. Cross-mix forms (>2 operands) do NOT count -- they
        # are exactly what traps the pair.
        existing = getattr(mrmr, "_engineered_recipes_", None) or []
        covered_pairs = set()
        for r in existing:
            src = tuple(getattr(r, "src_names", ()) or ())
            uniq = frozenset(src)
            if 1 <= len(uniq) <= 2:
                covered_pairs.add(uniq)

        # Base operands: numeric raw columns only (the usability pool builds pair forms over these).
        base_names = []
        for nm in list(getattr(mrmr, "feature_names_in_", []) or []):
            if nm in X.columns and pd.api.types.is_numeric_dtype(X[nm].dtype):
                base_names.append(nm)
        if len(base_names) < 2:
            return []
        # Scope wide frames: keep the highest-variance base operands so the O(pairs) pool stays bounded
        # (the usability pool itself also caps pairs, but trimming here bounds its input).
        if len(base_names) > max_base_features:
            stds = {nm: float(np.nanstd(X[nm].to_numpy())) for nm in base_names}
            base_names = sorted(base_names, key=lambda nm: -stds.get(nm, 0.0))[:max_base_features]

        # ROW SUBSAMPLE for the pool build + CV greedy: recovering WHICH pure pair forms are linearly
        # usable is a representative-sample question, and the chosen form is a replayable recipe whose
        # quantile edges from a 5k subsample track the full-column edges closely. Bounds the O(pool*folds)
        # CV cost so the retention does not blow the per-fit budget at large n (the nsweep's n=50k cells).
        X_fit, y_fit = X, y_cont
        n_rows = len(X)
        if n_rows > max_rows:
            _rng = np.random.default_rng(int(seed))
            _idx = np.sort(_rng.choice(n_rows, size=max_rows, replace=False))
            X_fit = X.iloc[_idx]
            y_fit = y_cont[_idx]

        from ._usability_aware_selection import (
            build_usability_candidate_pool, usability_greedy, _f64, _scrub,
        )

        _yv = _scrub(np.asarray(y_fit, dtype=np.float64))
        _nrows = _yv.shape[0]
        # NONLINEAR-RESIDUAL-VS-Y GATE. The CV-MAE greedy alone over-fires (adversarial-found 2026-06-17):
        # it admits a linear SUM like add(x1,x2) -- a linear model ALREADY builds it from the raw operands --
        # and CROSS-PAIR / noise-operand forms that lower MAE on a fold by chance. The retention's whole
        # point is the NONLINEAR interaction a linear model cannot build from the raws (a**2/b,
        # log(c)*sin(d)). Discriminator that avoids the other-additive-term variance confound (a global
        # CV-MAE-over-raws comparison is dominated by the UNEXPLAINED term's variance, so a genuine form's
        # relative gain can fall below any fixed threshold): regress the FORM on its two raw operands, take
        # the RESIDUAL (the part the raws cannot linearly reproduce), and require BOTH (a) the residual is
        # non-negligible (the form is genuinely nonlinear in its raws -- rejects add(x1,x2), whose residual
        # ~0) AND (b) that residual correlates with y (the nonlinearity is RELEVANT -- rejects a cross-pair
        # / noise form whose nonlinear part is unrelated to the target). a**2/b: large residual, correlated
        # with y -> kept; both holes -> rejected.
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        def _abscorr(u, v):
            u = u - u.mean(); v = v - v.mean()
            du, dv = float(np.sqrt((u * u).sum())), float(np.sqrt((v * v).sum()))
            if du <= 1e-12 or dv <= 1e-12:
                return 0.0
            return abs(float((u * v).sum()) / (du * dv))

        def _single_operand_basis(x):
            # a modest additive single-operand basis: any SEPARABLE function f(a)+g(b) lives in the span
            # of [basis(a)] + [basis(b)], so a cross-pair form that merely sums two single-operand
            # nonlinearities (a**3 + sqrt(d)) is reconstructed here with ~0 residual, while a genuine
            # NON-separable joint form (a**2/b is a ratio, log(c)*sin(d) a product) is not.
            xs = (x - x.mean()) / (x.std() + 1e-12)
            cols = [xs, xs * xs, xs * xs * xs, np.sign(xs) * np.sqrt(np.abs(xs)),
                    np.sign(xs) * np.log1p(np.abs(xs)), 1.0 / (np.abs(xs) + 1.0)]
            return cols

        def _adds_nonlinear_value(form_vals, nm_a, nm_b, min_resid_frac=0.10, min_resid_corr=0.08):
            try:
                xa = _f64(_scrub(X_fit[nm_a].to_numpy()))
                xb = _f64(_scrub(X_fit[nm_b].to_numpy()))
                fv = _f64(_scrub(np.asarray(form_vals)))
                if xa.shape[0] != _nrows or fv.shape[0] != _nrows:
                    return False
                f_std = float(np.std(fv))
                if f_std <= 1e-12:
                    return False
                # residual of the form after the ADDITIVE single-operand basis of BOTH operands: the part
                # that is genuinely a JOINT (non-separable) interaction of the two operands.
                Xr = np.column_stack(_single_operand_basis(xa) + _single_operand_basis(xb))
                lr = make_pipeline(StandardScaler(), LinearRegression()).fit(Xr, fv)
                resid = fv - lr.predict(Xr)
                # (a) the form must be genuinely NON-separable in its operands (rejects a linear sum AND a
                #     separable cross-pair sum-of-single-operand-nonlinearities).
                if float(np.std(resid)) < min_resid_frac * f_std:
                    return False
                # (b) that joint structure must be RELEVANT to y (rejects a non-separable but useless form).
                return _abscorr(resid, _yv) >= min_resid_corr
            except Exception:
                return False

        # Build the candidate pool, then FILTER pair forms by the non-separability gate BEFORE the greedy.
        # The CV-MAE greedy, given the raw pool, prefers a SEPARABLE cross-pair form (it absorbs the CV
        # gain first), so a post-greedy gate would reject the cross-pair and leave nothing -- the genuine
        # JOINT form was never selected. Filtering the pool to non-separable joint forms first makes the
        # greedy optimise CV-MAE over GENUINE interactions only, so it picks a**2/b / g/k / log(c)*sin(d).
        pool = build_usability_candidate_pool(
            X_fit, _yv, base_names, max_pairs=25, max_per_pair=8,
        )
        filtered = []
        for cand in pool:
            recipe = getattr(cand, "recipe", None)
            if recipe is None:
                filtered.append(cand)  # raw passthrough: the linear baseline the greedy builds on
                continue
            src = tuple(getattr(cand, "src", ()) or ())
            if len(set(src)) != 2:
                continue
            if _adds_nonlinear_value(getattr(cand, "values", None), src[0], src[1]):
                filtered.append(cand)
        usable = usability_greedy(filtered, _yv, w=w, K=K, seed=int(seed), n_folds=3, shortlist=15)

        existing_names = set(getattr(mrmr, "_engineered_features_", []) or [])
        out = []
        for cand in usable:
            recipe = getattr(cand, "recipe", None)
            if recipe is None:
                continue  # raw passthrough -- raw retention handles those
            src = tuple(getattr(cand, "src", ()) or ())
            pair = frozenset(src)
            if len(pair) != 2:
                continue  # only genuine single-PAIR forms
            if pair in covered_pairs:
                continue  # a pure form for this pair already survives
            name = getattr(cand, "name", None) or getattr(recipe, "name", None)
            if name is None or name in existing_names:
                continue
            out.append((recipe, name))
            covered_pairs.add(pair)
            existing_names.add(name)
            if len(out) >= max_added:
                break
        return out
    except Exception:
        return []


__all__ = ["retain_usable_pure_forms"]
