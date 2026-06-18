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
    min_resid_frac: float = 0.10,
    min_resid_corr: float = 0.08,
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

        # CHEAP TRAP PRE-CHECK (2026-06-18): the whole point of this pass is to rescue a pure single-pair
        # form the MI greedy TRAPPED -- either inside a high-MI CROSS-MIX (>2 distinct operands) or in
        # separate raw operands on a single-step fit that never produced the pure pair form. If NEITHER
        # trap is plausibly present, no pure form can be trapped and the (expensive) pool build + CV greedy
        # cannot recover anything, so return [] BEFORE paying that cost. This makes the pass zero-cost and
        # side-effect-free on the vast majority of fits (no cross-mix and a pure pair already engineered),
        # which is what was shifting survivor sets / adding per-fit cost on unrelated fits.
        #   (a) a CROSS-MIX recipe exists: an engineered entry whose src_names has >2 DISTINCT names -- the
        #       blend that traps pure pairs; OR
        #   (b) NO pure (<=2-operand) pair engineered form already survives (with >=2 raw numeric bases) --
        #       the greedy either picked raw operands instead of the pure pair form (MS_three_tier
        #       ``y = 5*a*b``) or a multi-step composite subsumed and dropped it (the user's F2
        #       ``a**2/b + log(c)*sin(d)``).
        # Count DISTINCT RAW OPERANDS, not src_names ENTRIES: on a MULTI-step fit a recipe's ``src_names``
        # are themselves COMPOSITE expression strings (e.g. ``mul(exp(a),sqr(c))``), so a true 4-raw
        # cross-mix ``div(sqr(add(invcbrt(b),prewarp(d))),invsqrt(mul(exp(a),sqr(c))))`` has only TWO
        # src_names entries and would be mis-counted as a pure PAIR. Resolve each src token down to the
        # raw bases it references (intersect its name tokens with the input feature set) so the
        # cross-mix-vs-pure-pair classification is over genuine raw-operand arity.
        import re as _re_pre
        _raw_base_set = set(base_names)

        def _raw_operands(recipe) -> set:
            toks = set()
            for s in (getattr(recipe, "src_names", ()) or ()):
                for t in _re_pre.split(r"[^0-9A-Za-z_]+", str(s)):
                    if t in _raw_base_set:
                        toks.add(t)
            return toks

        has_cross_mix = False
        has_pure_pair = False
        for r in existing:
            uniq = _raw_operands(r)
            if len(uniq) > 2:
                has_cross_mix = True
            elif len(uniq) == 2:
                has_pure_pair = True
        # (b) NO pure (<=2-operand) pair engineered form already survives, yet >=2 raw numeric bases exist.
        # The MI greedy then either picked raw operands instead of the pure pair form (single-step
        # ``y = 5*a*b`` MS_three_tier case) OR -- on a MULTI-step fit -- built a higher-order composite that
        # subsumed the pure pair and dropped it (the user's F2 ``a**2/b + log(c)*sin(d)``: the 2nd FE step
        # fuses a cross-mix and the post-FE greedy drops the pure ``a**2/b`` parent). Either way the pure
        # pair is trapped and recoverable, so the (multi-step) restriction the earlier draft placed on
        # fe_max_steps==1 was too narrow -- it suppressed the F2 recovery this pass exists for. Gate only on
        # "a pure pair form is not already present", independent of step count.
        trap_b = len(base_names) >= 2 and not has_pure_pair
        if not (has_cross_mix or trap_b):
            return []

        # CHEAP LINEAR-FIT GATE (2026-06-18): the trap conditions above are NECESSARY but not sufficient --
        # ``trap_b`` (no pure pair form present) is true on almost every regression fit, including ones whose
        # y is simply ADDITIVE-LINEAR in the raws (no interaction to recover at all). The expensive pool build
        # (O(pairs x unary^2 x binary) MI evals -- ~88% of fit time when it fires) must therefore ALSO be
        # gated on evidence that a nonlinear pair interaction even EXISTS: fit a plain linear model on the raw
        # bases and skip when it already explains y well (high R^2 -> the linear downstream is already served
        # by the raws, no pure form can help). a**2/b / log(c)*sin(d) / a*b leave large linear residual (low
        # R^2) -> proceed; y = sum(raws) + noise fits ~0.98 -> skip. One cheap fit vs a 200s pool build.
        try:
            from sklearn.linear_model import LinearRegression as _LR
            from sklearn.preprocessing import StandardScaler as _SS
            from sklearn.pipeline import make_pipeline as _mp

            _ng = len(y_cont)
            if _ng > 2000:
                _rg = np.random.default_rng(int(seed) + 5)
                _gi = np.sort(_rg.choice(_ng, size=2000, replace=False))
                _Xg = X.iloc[_gi][base_names].to_numpy(dtype=np.float64, copy=False)
                _yg = y_cont[_gi]
            else:
                _Xg = X[base_names].to_numpy(dtype=np.float64, copy=False)
                _yg = y_cont
            _Xg = np.nan_to_num(_Xg, nan=0.0, posinf=0.0, neginf=0.0)
            _r2 = float(_mp(_SS(), _LR()).fit(_Xg, _yg).score(_Xg, _yg))
            if _r2 >= 0.92:
                return []  # raws already fit y linearly -- no trapped nonlinear interaction to recover
        except Exception:
            pass  # gate is an optimisation; on any failure fall through to the (correct) full path

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

        # min_resid_frac / min_resid_corr are exposed as tunable kwargs (default 0.10 / 0.08). bench-attempt-rejected (qual-23, 2026-06-18): lowering
        # min_resid_corr 0.08 -> 0.05 to recover weak-but-relevant joint forms is a DEAD KNOB at tractable scale -- on every synthetic tried (weak-joint
        # additive + control + a strongly-nonlinear a**2/b + log(c)*sin(d) target with raw-linear R2=0.10) the MI greedy ALREADY selected the pure pair
        # forms, so the trap pre-check (has_cross_mix OR no-pure-pair) returns [] BEFORE this gate runs and the corr floor never fires (+0 retained at corr in
        # {0.08, 0.05, 0.0}). Bench: _benchmarks/fs_quality/qual23_pure_form_resid_corr.py. Not flipped; re-test on a dataset where the greedy traps a pure pair.
        def _adds_nonlinear_value(form_vals, nm_a, nm_b, min_resid_frac=min_resid_frac, min_resid_corr=min_resid_corr):
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
        # max_per_pair=3 (not 8): the pool replay-VERIFIES a recipe for every kept form (apply_recipe ~0.35s
        # each -- the dominant retention cost), but the non-separability filter below discards most of them,
        # so building+verifying 8/pair was wasted work. The genuine joint form is the top-joint-MI form for
        # its pair, so top-3 reliably keeps it (verified: F2 / I5 / MS still recover). max_pairs stays high --
        # a pure-synergy pair (c,d) has ~0 marginal and ranks LOW, so trimming pairs would drop it.
        # SMART-SEARCH: rank pairs by MM-corrected JOINT MI and enumerate only the top max_pairs (=10).
        # The per-pair unary^2*binary enumeration is the ~100s core; joint-MI ranking surfaces the genuine
        # synergy pairs (a,b)/(c,d) -- highest joint MI -- into the top few while the noise pairs (joint MI
        # ~0) drop out, so a SMALL max_pairs runs fast without losing a genuine pair (top-K is robust where
        # an absolute joint-MI floor mis-prunes a genuine pair whose MM-debited value is small). Measured:
        # structured n=10000 fit 193s -> ~71s, recovery intact.
        pool = build_usability_candidate_pool(
            X_fit, _yv, base_names, max_pairs=10, max_per_pair=3, rank_pairs_by_joint_mi=True,
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


def retain_usable_raw_columns(
    mrmr: Any,
    X: Any,
    y_cont: "np.ndarray | None",
    *,
    w: float = 0.7,
    K: int = 8,
    seed: int = 0,
    max_added: int = 4,
    max_base_features: int = 14,
    max_rows: int = 3000,
    verbose: int = 0,
):
    """Return ``[raw_name, ...]`` of RAW columns the MI greedy dropped from ``support_`` even though a
    CROSS-VALIDATED LINEAR wrapper confirms they carry genuine linear-usable signal toward y.

    WHY THIS EXISTS (companion to :func:`retain_usable_pure_forms`)
    --------------------------------------------------------------
    MRMR's Fleuret objective ranks raws by BINNED MI, which systematically UNDER-values a raw that is
    linearly useful but whose marginal-MI estimate is small -- e.g. the operands ``g``/``k`` of a WEAK
    additive ratio term ``+ g/k`` in ``y = w*a**2/b + g/k + log(c)*sin(d)``: their binned MI is ~0.01-0.02
    (below the relevance floor) yet their linear correlation with y is ~0.15-0.24 and a tree recovers the
    ratio from them. The MI greedy drops both, the pure-form retention cannot rescue the pair (the clean
    ``g/k`` engineered form is a pool-generation lottery -- on some subsamples only monotone-warped variants
    survive, whose joint residual is no longer linearly aligned with y), and the existing marginal-MI raw
    re-attach also skips them (their MI is below the floor AND they are not operands of a surviving recipe).
    The FE feature space then LOSES the g/k signal -> a downstream model scores BELOW raw-only (BUG3's
    "FE harmful" mode; the I5 ratio_plus_trig case).

    The discriminator is the SAME CV-MAE forward-selection wrapper :func:`usability_greedy` the module
    already trusts, run over the RAW PASSTHROUGH candidates only: a raw with genuine linear signal lowers
    the K-fold CV-MAE on a MAJORITY of folds, while a pure-noise raw (``e``) does not improve the average
    CV-MAE and is rejected. Measured 2026-06-18: on the I5 case the greedy picks ``[a,c,b,g,k]`` (recovers
    g,k, drops noise e and the interaction-only d); on the adversarial datasets it picks only the genuine
    signal raws and rejects every noise operand (additive-linear: x3 dropped; case2: e dropped; single-pair:
    c,d dropped). Returns ONLY raws NOT already in ``support_``; never touches engineered recipes.
    Best-effort: any failure returns ``[]`` and never disturbs the fit.
    """
    if y_cont is None:
        return []
    try:
        import pandas as pd

        if not isinstance(X, pd.DataFrame):
            return []
        y_cont = np.asarray(y_cont, dtype=np.float64).ravel()
        if y_cont.shape[0] != len(X) or not np.isfinite(y_cont).any():
            return []

        # raws already in support_ are kept anyway -- only propose ADDITIONS.
        feat_in = list(getattr(mrmr, "feature_names_in_", []) or [])
        _sup_raw = getattr(mrmr, "support_", None)
        support = np.asarray(_sup_raw, dtype=np.int64).ravel() if _sup_raw is not None else np.array([], dtype=np.int64)
        already = {feat_in[int(i)] for i in support if 0 <= int(i) < len(feat_in)}

        base_names = []
        for nm in feat_in:
            if nm in X.columns and pd.api.types.is_numeric_dtype(X[nm].dtype):
                base_names.append(nm)
        if len(base_names) < 2:
            return []
        if len(base_names) > max_base_features:
            stds = {nm: float(np.nanstd(X[nm].to_numpy())) for nm in base_names}
            base_names = sorted(base_names, key=lambda nm: -stds.get(nm, 0.0))[:max_base_features]

        X_fit, y_fit = X, y_cont
        n_rows = len(X)
        if n_rows > max_rows:
            _rng = np.random.default_rng(int(seed))
            _idx = np.sort(_rng.choice(n_rows, size=max_rows, replace=False))
            X_fit = X.iloc[_idx]
            y_fit = y_cont[_idx]

        from ._usability_aware_selection import (
            build_usability_candidate_pool, usability_greedy, _scrub,
        )

        _yv = _scrub(np.asarray(y_fit, dtype=np.float64))
        # max_pairs=0 -> the pool builds ONLY the raw passthrough candidates and skips the expensive
        # unary x unary x binary pair-form enumeration entirely. This pass uses raws only (below), so
        # building pair forms just to discard them was pure waste -- it ran the full FE pool on every
        # continuous-y fit, blowing per-fit budgets. Raws-only keeps the linear-usability probe cheap.
        pool = build_usability_candidate_pool(
            X_fit, _yv, base_names, max_pairs=0, max_per_pair=8,
        )
        # RAW PASSTHROUGHS ONLY: the CV-MAE greedy over the raw columns is the linear-usability probe that
        # MI is blind to. Engineered pair forms are handled by retain_usable_pure_forms; mixing them in
        # here would let the greedy spend its budget on a composite instead of surfacing the under-ranked raw.
        raws = [c for c in pool if getattr(c, "recipe", None) is None]
        if len(raws) < 2:
            return []
        usable = usability_greedy(raws, _yv, w=w, K=K, seed=int(seed), n_folds=3, shortlist=15)

        out = []
        for cand in usable:
            nm = getattr(cand, "name", None)
            if nm is None or nm in already or nm not in X.columns:
                continue
            out.append(nm)
            if len(out) >= max_added:
                break
        return out
    except Exception:
        return []


__all__ = ["retain_usable_pure_forms", "retain_usable_raw_columns"]
