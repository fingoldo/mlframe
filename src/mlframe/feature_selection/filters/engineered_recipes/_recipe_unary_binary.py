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

import functools
import logging
from typing import Any, Callable, cast

import numpy as np

from ._recipe_core import EngineeredRecipe
from ._recipe_extract import _extract_column

logger = logging.getLogger(__name__)


@functools.cache
def _cached_unary_transformations(preset: str) -> "dict[str, Callable]":
    """Memoised-by-preset ``create_unary_transformations`` for recipe replay. Hot path in ``transform()``
    (called once per ``unary_binary`` recipe); the preset registry only depends on ``preset`` (3-4 distinct
    values total), and its callables are looked up read-only by ``_apply_unary_binary`` (never mutated), so
    sharing the SAME dict across every replay call in the process is safe."""
    from ..feature_engineering import create_unary_transformations
    return cast("dict[str, Callable]", create_unary_transformations(preset=preset))


@functools.cache
def _cached_binary_transformations(preset: str) -> "dict[str, Callable]":
    """Memoised-by-preset ``create_binary_transformations``; see ``_cached_unary_transformations``."""
    from ..feature_engineering import create_binary_transformations
    return cast("dict[str, Callable]", create_binary_transformations(preset=preset))


def _apply_unary_binary(recipe: EngineeredRecipe, X: Any, col_cache: "dict[str, np.ndarray] | None" = None) -> np.ndarray:
    """Replay a ``unary_binary`` recipe: reconstruct ``binary(unary_a(X[a]), unary_b(X[b]))`` (including the
    ``prewarp`` / ``gate_med`` / ``poly_`` pseudo-unaries and nested-engineered operands) and return the
    CONTINUOUS engineered column. Tries the GPU-resident replay path first when opted in; falls back to numpy
    on any GPU ineligibility or failure.

    ``col_cache``: optional dict shared across every recipe replayed in ONE ``transform()`` call (see
    ``apply_recipe``); forwarded to ``_extract_column`` and to the recursive ``apply_recipe`` call for a
    nested-engineered operand so a hub raw column is pulled from ``X`` at most once per call."""
    # GPU-RESIDENT REPLAY (2026-06-28): the elementwise ``binary(unary_a(X[a]),
    # unary_b(X[b]))`` materialisation on full-n (300k-1M) operands is
    # embarrassingly parallel and was the dominant FE-replay cost (~3.4s) on the
    # F2 STRICT path. Under the resident opt-in we apply the operator chain on
    # device (one H2D up, one D2H back), selection-equivalent to the numpy path.
    # GPU-ineligible recipes (nested parents / prewarp / gate_med / unmapped ops)
    # return None -> fall through to numpy; any cupy failure falls back too. The
    # numpy path below stays the DEFAULT and the rare-failure fallback.
    try:
        from .._gpu_strict_fe import fe_gpu_strict_resident_enabled
        if fe_gpu_strict_resident_enabled():
            from ._recipe_unary_binary_gpu import apply_unary_binary_gpu
            _gpu_out = apply_unary_binary_gpu(recipe, X, col_cache=col_cache)
            if _gpu_out is not None:
                return _gpu_out
    except Exception as _e:  # pragma: no cover - rare GPU runtime failure
        logger.debug("GPU unary_binary replay fell back to numpy: %r", _e)

    if len(recipe.src_names) != 2 or len(recipe.unary_names) != 2:
        raise ValueError(
            f"unary_binary recipe '{recipe.name}' must have exactly 2 src_names " f"and 2 unary_names; got {len(recipe.src_names)} / {len(recipe.unary_names)}"
        )
    unary_funcs = _cached_unary_transformations(recipe.unary_preset)
    binary_funcs = _cached_binary_transformations(recipe.binary_preset)

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

    def _is_pseudo(_u: str) -> bool:
        """Whether ``_u`` is a state-carrying pseudo-unary (not a preset registry lookup)."""
        # prewarp / gate_med / poly_<coef> are STATE-CARRYING pseudo-unaries -- not members of any preset registry;
        # they replay closed-form from state stored in recipe.extra (poly: hermite coeffs). Skip the preset lookup.
        return _u in _PSEUDO or _u.startswith("poly_")

    if not _is_pseudo(u_a) and u_a not in unary_funcs:
        raise KeyError(f"Unary function '{u_a}' not in '{recipe.unary_preset}' preset. " f"Replay requires the same preset that was active at fit time.")
    if not _is_pseudo(u_b) and u_b not in unary_funcs:
        raise KeyError(f"Unary function '{u_b}' not in '{recipe.unary_preset}' preset.")
    if recipe.binary_name not in binary_funcs:
        raise KeyError(f"Binary function '{recipe.binary_name}' not in " f"'{recipe.binary_preset}' preset.")

    # NESTED-ENGINEERED PARENTS (2026-06-08): when an operand is itself an engineered
    # column (a higher-order composite), its values are NOT in ``X`` at transform time
    # (X carries only raw columns). Recompute it by recursively replaying the stored
    # parent recipe, forced to its CONTINUOUS output (quantization stripped) so the
    # composite is built on continuous values exactly as at fit time -- the fit-time
    # operand was the parent's continuous engineered value, not its bin codes.
    def _nested_continuous(parent: "EngineeredRecipe") -> np.ndarray:
        """Recursively replay ``parent`` (a step-k>1 composite operand), forcing its output continuous (quantization stripped) so it matches the fit-time operand."""
        from ._recipe_dispatch import apply_recipe
        if parent.quantization is not None:
            # Replay the parent WITHOUT quantization (continuous output).
            import dataclasses as _dc
            parent = _dc.replace(parent, quantization=None)
        return np.asarray(apply_recipe(parent, X, col_cache=col_cache), dtype=np.float64)

    _np_a = recipe.extra.get("nested_parent_a")
    _np_b = recipe.extra.get("nested_parent_b")
    vals_a = _nested_continuous(_np_a) if _np_a is not None else _extract_column(X, name_a, col_cache=col_cache)
    vals_b = _nested_continuous(_np_b) if _np_b is not None else _extract_column(X, name_b, col_cache=col_cache)

    def _apply_side(side: str, uname: str, vals):
        """Apply the ``uname`` unary to one operand side, dispatching between the closed-form pseudo-unaries
        (``gate_med``, ``poly_<coef>``, ``prewarp``), the frozen-anchor ``log`` replay, and the plain preset registry lookup."""
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
            if uname.startswith("poly_"):
                # ``poly_<coef>`` pseudo-unary: at fit the key mapped to a hermite coefficient ARRAY (not a callable),
                # applied via ``hermval(vals, coef)`` (feature_engineering). The recipe stores that coef in extra so
                # replay reconstructs it -- previously the coef lived only in the per-fit unary_transformations dict,
                # so recipe replay raised KeyError looking ``poly_<coef>`` up in the static preset (critique ND-1).
                from numpy.polynomial.hermite import hermval
                _ckey = f"poly_{side}_coef"
                if _ckey not in recipe.extra:
                    raise KeyError(
                        f"unary_binary recipe '{recipe.name}' uses a 'poly_' pseudo-unary on side {side!r} but "
                        f"'{_ckey}' is missing from extra. Re-fit MRMR to regenerate the recipe."
                    )
                return hermval(np.asarray(vals, dtype=np.float64), np.asarray(recipe.extra[_ckey], dtype=np.float64))
            # BUG2 FIX (2026-06-12): ``smart_log`` (the registry ``log``) additively
            # shifts non-positive inputs by ``(1e-5 - nanmin(vals))`` -- a
            # DATA-DEPENDENT anchor recomputed from ``vals`` each call. On a row-slice
            # replay ``nanmin`` differs from the full-frame fit, shifting every
            # ``log`` output and (after the downstream quantiser) drifting the bin code
            # by up to several bins on a nested ``log(...)`` operand. When the recipe
            # froze the fit-time shift anchor, replay the log from THAT frozen anchor so
            # the operand is byte-exact regardless of the slice. Non-log unaries are
            # closed-form and stay on the registry path.
            _skey = f"log_shift_{side}"
            if uname == "log" and _skey in recipe.extra:
                _shift = float(recipe.extra[_skey])
                return np.log(np.asarray(vals, dtype=np.float64) + _shift) if _shift != 0.0 else np.log(np.asarray(vals, dtype=np.float64))
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

    # Any unary/binary in the registry can overflow or hit an invalid op on an extreme replay input
    # (e.g. an unguarded power, or a reciprocal-power's pre-eps-floor era); suppress the resulting numpy
    # RuntimeWarnings for the whole operator chain (previously only the frozen-log-shift branch was
    # wrapped) since the nan_to_num scrub right below already sanitises the final output regardless.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        transformed_a = _apply_side("a", u_a, vals_a)
        transformed_b = _apply_side("b", u_b, vals_b)
        out = binary_funcs[recipe.binary_name](transformed_a, transformed_b)

    # Match fit-time NaN/Inf scrubbing in ``feature_engineering.check_prospective_fe_pairs``.
    out = np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # CONTINUOUS TRANSFORM OUTPUT (2026-06-12): ``transform()`` delivers the
    # numeric pair-FE column as its CONTINUOUS value, never as the internal
    # MI quantile code. Quantile-binning a heavy-tailed product/ratio to integer
    # codes preserves only RANK and discards MAGNITUDE -- which every non-tree
    # downstream model needs. Measured on ``y = 0.2*a**2/b``: the 10-bin code of
    # ``div(sqr(a),abs(b))`` had Pearson 0.03 with the true ``a**2/b`` (rank-corr
    # 0.99), and a linear model on the code scored test-R2 ~0.002, versus >=0.99
    # on the continuous feature. The ``prewarp`` / ``hermite_pair`` siblings
    # already skip quantization for exactly this reason (see
    # ``build_unary_binary_recipe`` -- prewarp sets ``quantization=None``); this
    # generalises it to EVERY unary_binary recipe at replay. ``recipe.quantization``
    # is left populated for provenance/audit only; it is replay-irrelevant because
    # the downstream MRMR fit discretises the fit-time column for its OWN MI matrix
    # via ``_mrmr_fe_step`` (a separate code path, unaffected by this choice).
    return np.asarray(out)


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
    poly_a_coef: np.ndarray | None = None,
    poly_b_coef: np.ndarray | None = None,
    prewarp_a: dict | None = None,
    prewarp_b: dict | None = None,
    gate_med_a: float | None = None,
    gate_med_b: float | None = None,
    nested_parent_a: "EngineeredRecipe | None" = None,
    nested_parent_b: "EngineeredRecipe | None" = None,
    log_shift_a: float | None = None,
    log_shift_b: float | None = None,
) -> EngineeredRecipe:
    """Build an ``EngineeredRecipe`` of kind ``"unary_binary"``. ``quantization`` is ``None`` if no discretization, else a dict carrying the binning
    parameters AND, when ``fit_values_for_edges`` is provided, the fit-time bin edges. Dtype is stringified so the recipe is JSON-friendly and
    pickle-safe across numpy versions.

    The stored ``edges`` are PROVENANCE/AUDIT only: ``_apply_unary_binary`` replays the CONTINUOUS engineered value and never re-quantises, so it
    never consults ``edges`` (the magnitude-preserving, drift-invariant contract -- see the apply-side rationale). ``fit_values_for_edges`` still
    lets the caller record the fit-time quantile boundaries on the recipe for inspection; passing ``None`` simply omits them. No replay-time
    leakage warning is emitted either way, because there is no replay-time quantiser left to leak.

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
    # ND-1: persist the hermite coefficient array of a ``poly_<coef>`` unary so recipe replay can reconstruct it
    # (hermval) instead of failing to find the per-fit ``poly_<coef>`` key in the static preset. Stored as a plain
    # list for JSON/pickle friendliness (mirrors prewarp_<side>_coef below).
    for _side, _coef in (("a", poly_a_coef), ("b", poly_b_coef)):
        if _coef is not None:
            extra[f"poly_{_side}_coef"] = np.asarray(_coef, dtype=np.float64).tolist()
    for _side, _spec in (("a", prewarp_a), ("b", prewarp_b)):
        if _spec is None:
            continue
        extra[f"prewarp_{_side}_basis"] = str(_spec["basis"])
        extra[f"prewarp_{_side}_degree"] = int(_spec["degree"])
        extra[f"prewarp_{_side}_coef"] = np.asarray(_spec["coef"], dtype=np.float64).copy()
        extra[f"prewarp_{_side}_preprocess"] = _orjson_pp(_spec["preprocess"])
        # ROBUST WARP FIT: when the warp was fit with the Huber-IRLS
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
    # BUG2 FIX (2026-06-12): per-operand FROZEN ``smart_log`` shift anchor. The
    # registry ``log`` (``smart_log``) shifts non-positive inputs by
    # ``(1e-5 - nanmin(operand))`` -- a data-dependent anchor that, recomputed from a
    # transform row-slice, drifts every log output (and the downstream bin code). When
    # a side's unary is ``log`` and the caller computed the fit-time anchor, persist it
    # FLAT so replay reproduces the exact fit-time shift. Stored only for ``log`` sides
    # so non-log recipes stay byte-equal with legacy pickles.
    for _side, _uname, _shift in (("a", unary_a_name, log_shift_a), ("b", unary_b_name, log_shift_b)):
        if _uname == "log" and _shift is not None:
            extra[f"log_shift_{_side}"] = float(_shift)
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
