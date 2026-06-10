"""Numeric pair-FE (``unary_binary``) recipe builder + replay.

``_apply_unary_binary`` reconstructs ``binary(unary_a(X[a]), unary_b(X[b]))`` and
optionally discretizes it, supporting the learned ``prewarp`` / ``gate_med``
pseudo-unaries and recursive replay of nested-engineered operands. It lazy-imports
``feature_engineering`` / ``discretization`` / ``hermite_fe`` /
``_feature_engineering_pairs`` in-body to break the import cycle with those
siblings, and lazy-imports ``apply_recipe`` from ``_recipe_dispatch`` for the
nested-parent recursion (same reason).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._recipe_core import EngineeredRecipe
from ._recipe_extract import _extract_column


def _apply_unary_binary(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    if len(recipe.src_names) != 2 or len(recipe.unary_names) != 2:
        raise ValueError(
            f"unary_binary recipe '{recipe.name}' must have exactly 2 src_names "
            f"and 2 unary_names; got {len(recipe.src_names)} / {len(recipe.unary_names)}"
        )
    # Lazy import to avoid circular dependency (feature_engineering -> mrmr via _internals).
    from ..feature_engineering import create_unary_transformations, create_binary_transformations
    from ..discretization import discretize_array

    unary_funcs = create_unary_transformations(preset=recipe.unary_preset)
    binary_funcs = create_binary_transformations(preset=recipe.binary_preset)

    name_a, name_b = recipe.src_names
    u_a, u_b = recipe.unary_names

    # The learned ``prewarp`` (2026-06-02) and ``gate_med`` (2026-06-04)
    # pseudo-unaries are NOT members of any preset registry; their closed-form
    # replay reads the fit-time state stored in ``recipe.extra`` (prewarp: poly
    # coeffs; gate_med: the TRAIN median). Validate only the REAL unary names
    # against the preset.
    _PREWARP = "prewarp"
    _GATE_MED = "gate_med"
    _PSEUDO = (_PREWARP, _GATE_MED)
    if u_a not in _PSEUDO and u_a not in unary_funcs:
        raise KeyError(
            f"Unary function '{u_a}' not in '{recipe.unary_preset}' preset. "
            f"Replay requires the same preset that was active at fit time."
        )
    if u_b not in _PSEUDO and u_b not in unary_funcs:
        raise KeyError(
            f"Unary function '{u_b}' not in '{recipe.unary_preset}' preset."
        )
    if recipe.binary_name not in binary_funcs:
        raise KeyError(
            f"Binary function '{recipe.binary_name}' not in "
            f"'{recipe.binary_preset}' preset."
        )

    # NESTED-ENGINEERED PARENTS (2026-06-08): when an operand is itself an engineered
    # column (a higher-order composite), its values are NOT in ``X`` at transform time
    # (X carries only raw columns). Recompute it by recursively replaying the stored
    # parent recipe, forced to its CONTINUOUS output (quantization stripped) so the
    # composite is built on continuous values exactly as at fit time -- the fit-time
    # operand was the parent's continuous engineered value, not its bin codes.
    def _nested_continuous(parent: "EngineeredRecipe") -> np.ndarray:
        from ._recipe_dispatch import apply_recipe
        if parent.quantization is not None:
            # Replay the parent WITHOUT quantization (continuous output).
            import dataclasses as _dc
            parent = _dc.replace(parent, quantization=None)
        return np.asarray(apply_recipe(parent, X), dtype=np.float64)

    _np_a = recipe.extra.get("nested_parent_a")
    _np_b = recipe.extra.get("nested_parent_b")
    vals_a = _nested_continuous(_np_a) if _np_a is not None else _extract_column(X, name_a)
    vals_b = _nested_continuous(_np_b) if _np_b is not None else _extract_column(X, name_b)

    def _apply_side(side: str, uname: str, vals):
        if uname == _GATE_MED:
            # Median-gate pseudo-unary: replay closed-form ``(x > train_median)``
            # from the single fit-time median float stored in ``extra`` (no y, no
            # test-time recompute -> leak-safe and bit-identical to fit).
            _mkey = f"gate_med_{side}_median"
            if _mkey not in recipe.extra:
                raise KeyError(
                    f"unary_binary recipe '{recipe.name}' uses the 'gate_med' "
                    f"pseudo-unary on side {side!r} but '{_mkey}' is missing from "
                    f"extra. Re-fit MRMR to regenerate the recipe."
                )
            from .._feature_engineering_pairs import _gate_med_apply
            return _gate_med_apply(vals, float(recipe.extra[_mkey]))
        if uname != _PREWARP:
            return unary_funcs[uname](vals)
        # Reconstruct the pre-warp spec from the flat ``extra`` fields and replay
        # it closed-form (no y) so the warped operand is bit-identical to fit.
        import orjson as _orjson
        from ..hermite_fe import apply_operand_prewarp
        for _k in (f"prewarp_{side}_coef", f"prewarp_{side}_basis"):
            if _k not in recipe.extra:
                raise KeyError(
                    f"unary_binary recipe '{recipe.name}' uses the 'prewarp' "
                    f"pseudo-unary on side {side!r} but '{_k}' is missing from "
                    f"extra. Re-fit MRMR to regenerate the recipe."
                )
        _spec = {
            "basis": str(recipe.extra[f"prewarp_{side}_basis"]),
            "degree": int(recipe.extra[f"prewarp_{side}_degree"]),
            "coef": np.asarray(recipe.extra[f"prewarp_{side}_coef"], dtype=np.float64),
            "preprocess": _orjson.loads(recipe.extra[f"prewarp_{side}_preprocess"]),
        }
        return apply_operand_prewarp(vals, _spec)

    transformed_a = _apply_side("a", u_a, vals_a)
    transformed_b = _apply_side("b", u_b, vals_b)
    out = binary_funcs[recipe.binary_name](transformed_a, transformed_b)

    # Match fit-time NaN/Inf scrubbing in ``feature_engineering.check_prospective_fe_pairs``.
    out = np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if recipe.quantization is not None:
        q = recipe.quantization
        # 2026-05-30 Wave 9.1 fix (loop iter 28): use fit-time edges when
        # the recipe stored them. Pre-fix ``discretize_array`` recomputed
        # ``np.nanpercentile`` from TEST data each replay, so the same
        # physical row mapped to DIFFERENT bin codes between fit and
        # transform under any distribution drift (58.8% disagreement
        # observed in synthetic shift demo). That's a textbook train/test
        # leak: the model trained on stale bin codes and got fresh
        # rebinned codes at inference.
        if q.get("edges") is not None:
            edges = np.asarray(q["edges"], dtype=np.float64)
            out = np.searchsorted(
                edges[1:-1] if edges.size >= 2 else edges,
                out, side="right",
            ).astype(np.dtype(q["dtype"]))
        else:
            # Back-compat: pre-iter-28 recipes (pickled before edges were
            # persisted) fall back to the leaky path. Warn so maintainers
            # see the smell. New recipes always carry edges.
            import warnings as _w_iter28
            _w_iter28.warn(
                f"unary_binary recipe '{recipe.name}' has no fit-time "
                f"quantile edges; replay will re-quantile on test data "
                f"and produce shifted codes under distribution drift. "
                f"Refit the MRMR estimator to regenerate the recipe with "
                f"persisted edges.",
                UserWarning, stacklevel=2,
            )
            out = discretize_array(
                arr=out,
                n_bins=q["nbins"],
                method=q["method"],
                dtype=np.dtype(q["dtype"]),
            )
    return out


def build_unary_binary_recipe(
    *,
    name: str,
    src_a_name: str,
    src_b_name: str,
    unary_a_name: str,
    unary_b_name: str,
    binary_name: str,
    unary_preset: str,
    binary_preset: str,
    quantization_nbins: int | None,
    quantization_method: str | None,
    quantization_dtype: Any,
    fit_values_for_edges: np.ndarray | None = None,
    prewarp_a: dict | None = None,
    prewarp_b: dict | None = None,
    gate_med_a: float | None = None,
    gate_med_b: float | None = None,
    nested_parent_a: "EngineeredRecipe | None" = None,
    nested_parent_b: "EngineeredRecipe | None" = None,
) -> EngineeredRecipe:
    """Build an ``EngineeredRecipe`` of kind ``"unary_binary"``. ``quantization`` is ``None`` if no discretization, else a dict carrying the binning
    parameters AND, when ``fit_values_for_edges`` is provided, the fit-time bin edges so transform-time replay maps each row to the SAME bin
    code regardless of test-data distribution. Dtype is stringified so the recipe is JSON-friendly and pickle-safe across numpy versions.

    2026-05-30 Wave 9.1 iter 28: ``fit_values_for_edges`` lets the caller pin the quantile boundaries. Without it (legacy code paths) the
    recipe emits a UserWarning at replay time about the train/test leakage risk.

    NESTED-ENGINEERED PARENTS (2026-06-08): ``nested_parent_a`` / ``nested_parent_b`` are
    the ``EngineeredRecipe`` objects for operands that are THEMSELVES engineered columns
    (a step-k>1 composite of two prior engineered features, e.g.
    ``add(div(sqr(a),abs(b)), mul(log(c),sin(d)))``). When supplied, the parent recipe is
    stored in ``extra["nested_parent_<side>"]`` and ``_apply_unary_binary`` recomputes that
    operand at transform time by recursively replaying the parent (forced to its CONTINUOUS
    output -- quantization stripped -- so the composite is built on continuous values exactly
    as at fit time) rather than reading ``src_<side>_name`` from ``X`` (which only carries
    raw columns at transform time). This makes higher-order composites fully replayable."""
    # Pre-warp recipes emit a CONTINUOUS learned-polynomial product (the same
    # nature as the orthogonal-poly ``hermite_pair`` recipe, which is NOT
    # quantised at replay). Quantile-binning the heavy-tailed product to integer
    # codes throws away most of its correlation with the target (measured: raw
    # mul-product corr 0.88 vs binned 0.62 on F-POLY). Mirror the hermite_pair
    # sibling and skip quantization so ``transform()`` returns the raw feature;
    # the downstream MRMR fit still discretises the fit-time column for its own
    # MI matrix via ``_mrmr_fe_step`` (unaffected by this replay-only choice).
    _is_prewarp = prewarp_a is not None or prewarp_b is not None
    if quantization_nbins is None or _is_prewarp:
        quantization = None
    else:
        quantization = {
            "nbins": int(quantization_nbins),
            "method": str(quantization_method) if quantization_method else "uniform",
            "dtype": np.dtype(quantization_dtype).str,
        }
        # Persist fit-time edges so replay never re-quantiles on test data.
        if fit_values_for_edges is not None:
            _arr = np.asarray(fit_values_for_edges, dtype=np.float64).ravel()
            if quantization["method"] == "quantile":
                _q = np.linspace(0.0, 100.0, int(quantization_nbins) + 1)
                _edges = np.nanpercentile(_arr, _q)
            else:  # uniform
                _finite = _arr[np.isfinite(_arr)]
                if _finite.size:
                    _lo = float(_finite.min())
                    _hi = float(_finite.max())
                    _edges = np.linspace(_lo, _hi, int(quantization_nbins) + 1)
                else:
                    _edges = np.linspace(0.0, 0.0, int(quantization_nbins) + 1)
            quantization["edges"] = _edges.tolist()

    # Per-operand pre-warp (2026-06-02): when the unary name on a side is the
    # learned ``prewarp`` pseudo-unary, persist its fitted spec (basis, degree,
    # coeffs, basis-preprocess params) FLAT in ``extra`` so replay reproduces
    # the closed-form 1-D warp from x alone (leak-safe; no y). ``extra`` values
    # must stay flat ndarray/scalar/str for ``_extra_equal`` -- the nested
    # ``preprocess`` dict is JSON-stringified (sorted keys, deterministic).
    extra: dict = {}
    for _side, _spec in (("a", prewarp_a), ("b", prewarp_b)):
        if _spec is None:
            continue
        extra[f"prewarp_{_side}_basis"] = str(_spec["basis"])
        extra[f"prewarp_{_side}_degree"] = int(_spec["degree"])
        extra[f"prewarp_{_side}_coef"] = np.asarray(_spec["coef"], dtype=np.float64).copy()
        extra[f"prewarp_{_side}_preprocess"] = _orjson_pp(_spec["preprocess"])
        # ROBUST WARP FIT (backlog #17): when the warp was fit with the Huber-IRLS
        # heavy-tail path, persist the robust flag + the MAD-anchored winsor bounds
        # used at fit time. Replay is closed-form on ``coef`` (no y, leak-safe) so
        # these are provenance / auditability, not required for byte-identical
        # transform; stored FLAT (scalars) for ``_extra_equal`` comparability.
        if _spec.get("robust_fit"):
            extra[f"prewarp_{_side}_robust_fit"] = True
            extra[f"prewarp_{_side}_winsor_lo"] = float(_spec["winsor_lo"])
            extra[f"prewarp_{_side}_winsor_hi"] = float(_spec["winsor_hi"])
    # Per-operand median gate (2026-06-04): when a side used the ``gate_med``
    # pseudo-unary, persist its single TRAIN-median float FLAT in ``extra`` so
    # replay reproduces ``(x > median)`` from x alone (leak-safe; no y). A plain
    # float is JSON-friendly and ``_extra_equal``-comparable as a scalar.
    for _side, _med in (("a", gate_med_a), ("b", gate_med_b)):
        if _med is None:
            continue
        extra[f"gate_med_{_side}_median"] = float(_med)
    # Nested-engineered parents (2026-06-08): store the parent recipe per side so
    # replay recomputes the engineered operand recursively. ``_extra_equal`` compares
    # them via ``EngineeredRecipe.__eq__`` (the else branch), and ``__post_init__``
    # deep-copies extra so the stored parents are owned by this recipe.
    for _side, _parent in (("a", nested_parent_a), ("b", nested_parent_b)):
        if _parent is None:
            continue
        extra[f"nested_parent_{_side}"] = _parent
    return EngineeredRecipe(
        name=name,
        kind="unary_binary",
        src_names=(src_a_name, src_b_name),
        unary_names=(unary_a_name, unary_b_name),
        binary_name=binary_name,
        unary_preset=unary_preset,
        binary_preset=binary_preset,
        quantization=quantization,
        extra=extra,
    )


def _orjson_pp(d: dict) -> str:
    """Serialise a flat preprocess-params dict to a deterministic JSON string
    (sorted keys) for storage in ``EngineeredRecipe.extra``. orjson keeps the
    recipe pickle-light and the OPT_SORT_KEYS flag makes the stored bytes a
    stable function of the params (hash-friendly)."""
    import orjson
    return orjson.dumps(d, option=orjson.OPT_SORT_KEYS).decode("ascii")
