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

        from ._usability_aware_selection import select_usability_aware_features

        # Cheap config: the genuine pure pairs have the highest marginal-MI sum so they survive a small
        # pool, and a 3-fold shortlisted greedy is enough to confirm linear usability. Keeps the per-fit
        # cost bounded so the pass is affordable as a DEFAULT (it runs on every continuous-y FE fit).
        usable = select_usability_aware_features(
            X_fit, y_fit, base_names, w=w, K=K, seed=int(seed),
            pool_kwargs=dict(max_pairs=15, max_per_pair=6),
            greedy_kwargs=dict(n_folds=3, shortlist=15),
        )
        existing_names = set(getattr(mrmr, "_engineered_features_", []) or [])
        out = []
        for cand in usable:
            recipe = getattr(cand, "recipe", None)
            if recipe is None:
                continue  # raw passthrough -- raw retention handles those
            pair = frozenset(getattr(cand, "src", ()) or ())
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
