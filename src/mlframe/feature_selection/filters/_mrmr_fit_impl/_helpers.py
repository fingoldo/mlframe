"""``MRMR._fit_impl`` main fit body for ``mlframe.feature_selection.filters.mrmr``.

Split out of ``mrmr.py`` to keep the parent below the 1k-line monolith
threshold. ``_fit_impl`` is bound back onto the ``MRMR`` class at the
parent's module bottom, so call sites that invoke ``self._fit_impl(...)``
continue to work unchanged.

Heavy lifting: signature/cache key build, content-hash short-circuit,
sub-sample loop, FE-step orchestration, MI ranking and the per-fold
selection. Many helpers (logger, signature hashing, target coercion)
live in the parent and are imported lazily inside this body to avoid the
``mrmr -> _mrmr_fit_impl -> mrmr`` import cycle.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

"""Small top-level helpers for MRMR._fit_impl.

Carved verbatim out of the former flat _mrmr_fit_impl.py when it was
promoted to a subpackage. The single giant _fit_impl body lives in the
sibling _fit_impl_core.py; these are the few small free functions it and
external callers (LRU byte-cap gate, orth-FE scorer dispatch) rely on.
"""


# 2026-06-01 Layer 85 — default-scorer dispatcher for the Layer 21 hybrid
# orth-poly univariate basis-selection stage. Imports the alternate scorer
# module lazily so callers paying for "plug_in" (the default) do not pay
# the import cost. Every alternate scorer's ``*_with_recipes`` returns the
# same ``(X_aug, scores_df, recipes_list)`` 3-tuple as Layer 21's plain
# univariate path -- so the caller plumbing in ``_fit_impl`` is unchanged.
def _orth_fe_numeric_cols(X, cols):
    """Keep only numeric (incl. bool) scalar columns from ``cols`` for the orthogonal / polynomial hybrid-FE family,
    which converts operands to float. Raw categorical / string columns (e.g. a string-coded cat 'B') would otherwise
    raise ``ValueError: could not convert string to float`` and the whole FE pass would be silently dropped. Duplicated
    column names (``X[c]`` -> DataFrame, ndim 2) are skipped as ambiguous. Format-agnostic (pandas / polars)."""
    from .._fe_frame_ops import fe_is_numeric_col
    return [c for c in cols if fe_is_numeric_col(X, c)]


def _dispatch_default_scorer(
    scorer: str,
    *,
    X: pd.DataFrame,
    y: np.ndarray,
    cols,
    degrees,
    basis: str,
    top_k: int,
):
    """Route a non-``plug_in`` ``fe_hybrid_orth_default_scorer`` value to
    the matching ``*_with_recipes`` univariate-stage builder.

    Returns the same ``(X_aug, scores_df, recipes_list)`` tuple every
    alternate scorer's ``with_recipes`` variant exposes. Raises
    ``ValueError`` on an unrecognised scorer string (defensive: the public
    validation entry point catches this in ``_validate_string_params``).
    """
    if scorer == "cmim":
        from .._orthogonal_cmim_fe import (
            hybrid_orth_mi_cmim_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "jmim":
        from .._orthogonal_jmim_fe import (
            hybrid_orth_mi_jmim_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "tc":
        from .._orthogonal_total_correlation_fe import (
            hybrid_orth_mi_tc_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "ksg":
        from .._orthogonal_ksg_mi_fe import (
            hybrid_orth_mi_ksg_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "copula":
        from .._orthogonal_copula_mi_fe import (
            hybrid_orth_mi_copula_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "dcor":
        from .._orthogonal_dcor_fe import (
            hybrid_orth_mi_dcor_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "hsic":
        from .._orthogonal_hsic_fe import (
            hybrid_orth_mi_hsic_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "lasso":
        from .._orthogonal_lasso_fe import (
            hybrid_orth_mi_lasso_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "elasticnet":
        from .._orthogonal_elasticnet_fe import (
            hybrid_orth_mi_elasticnet_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "auto":
        from .._orthogonal_scorer_auto_fe import (
            hybrid_orth_mi_auto_scorer_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "ensemble":
        from .._orthogonal_scorer_auto_fe import (
            hybrid_orth_mi_ensemble_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "meta":
        from .._orthogonal_meta_scorer_fe import (
            hybrid_orth_mi_meta_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "auto_oracle":
        # 2026-06-01 Layer 100 — UNIFIED scorer-selection. The
        # OracleScorerSelector resolves a concrete scorer for this
        # dataset's fingerprint (learned-best when the oracle has confident
        # history, else the L76 cold-start cascade), then we delegate to
        # THAT scorer's univariate builder. The bake-off that populates the
        # oracle (``benchmark_all_scorers``) runs out-of-band, so fit-time
        # cost is one recommend + one scorer run, not the full L68 sweep.
        from .._oracle_scorer_select import OracleScorerSelector
        _selector = OracleScorerSelector()
        _resolved = _selector.recommend_scorer(X, y)
        # ``auto_oracle`` never resolves to itself / "plug_in"-by-default is
        # routed through the plug_in builders below; any other resolved
        # scorer recurses into its own branch of this dispatcher.
        if _resolved == "plug_in":
            from .._orthogonal_univariate_fe import (
                hybrid_orth_mi_fe_with_recipes as _fn,
            )
            return _fn(
                X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k,
            )
        return _dispatch_default_scorer(
            _resolved, X=X, y=y, cols=cols, degrees=degrees,
            basis=basis, top_k=top_k,
        )
    raise ValueError(
        f"_dispatch_default_scorer: unrecognised scorer={scorer!r}. "
        f"This is a defensive bug -- ``_validate_string_params`` should "
        f"have caught the bad value before fit reached the dispatcher."
    )


def _mrmr_instance_state_size_bytes(instance: Any) -> int:
    """Best-effort byte estimate for a single fitted MRMR instance's selector + engineered-features state.

    Used by the LRU eviction byte gate. Walks the small set of large state attributes (``mi_scores_``, ``_selectors_``, ``_engineered_features_``, ``_y_full_hash`` retained y) so the estimate reflects the dominant footprint without paying ``pickle.dumps`` cost on every eviction probe.
    """
    total = 0
    for _attr in ("mi_scores_", "_selectors_", "_engineered_features_", "ranking_", "support_", "selected_features_"):
        try:
            _v = getattr(instance, _attr, None)
            if _v is None:
                continue
            _nb = getattr(_v, "nbytes", None)
            if isinstance(_nb, int):
                total += _nb
                continue
            if isinstance(_v, dict):
                for _vv in _v.values():
                    _vvnb = getattr(_vv, "nbytes", None)
                    if isinstance(_vvnb, int):
                        total += _vvnb
                    else:
                        try:
                            total += int(np.asarray(_vv).nbytes)
                        except Exception:
                            pass
            elif isinstance(_v, (list, tuple)):
                for _item in _v:
                    _inb = getattr(_item, "nbytes", None)
                    if isinstance(_inb, int):
                        total += _inb
        except Exception:
            continue
    return total


def _mrmr_cache_bytes_total() -> int:
    """Sum of state bytes across every cached MRMR instance in MRMR._FIT_CACHE."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    return sum(_mrmr_instance_state_size_bytes(_v) for _v in MRMR._FIT_CACHE.values())


# Cap the stored screening-matrix footprint (rows). The replay state is a per-fit
# diagnostic substrate; on the 8GB-shared box a multi-million-row binned matrix
# would be wasteful. A few thousand rows give a stable bootstrap frequency.
MAX_STABILITY_REPLAY_ROWS = 4000


def _build_stability_replay_state(
    self, *, data, cols, nbins, target_indices, selected_vars, engineered_recipes,
) -> None:
    """Persist a compact REPLAY substrate for ``MRMR.selection_stability_report``.

    Stores (a) the already-discretised candidate bin codes (target column excluded)
    row-subsampled to ``MAX_STABILITY_REPLAY_ROWS``, (b) the target codes, (c) the
    per-candidate selection outcome mask, and (d) for each ``unary_binary``
    engineered recipe the frozen engineered + source-operand bin codes so recipe
    survival can be replayed with the #15 held-out uplift-gate statistic. No copy
    of the raw frame, no recipe re-application: bootstrap resampling reads these
    frozen bins only -- the #15 "replay not refit" contract.
    """
    data = np.asarray(data)
    n_rows = int(data.shape[0])
    n_cols = int(data.shape[1])
    if n_rows < 2 or n_cols < 2:
        self._stability_replay_state_ = None
        return

    t_idx = int(np.asarray(target_indices).ravel()[0])
    cols = list(cols)
    sel_set = set(int(v) for v in np.asarray(selected_vars, dtype=np.intp).ravel())

    cand_cols = [c for c in range(n_cols) if c != t_idx]
    cand_names = [str(cols[c]) for c in cand_cols]
    selected_mask = np.array([c in sel_set for c in cand_cols], dtype=bool)

    rng = np.random.default_rng(int(getattr(self, "random_seed", 0) or 0))
    if n_rows > MAX_STABILITY_REPLAY_ROWS:
        row_idx = np.sort(rng.choice(n_rows, size=MAX_STABILITY_REPLAY_ROWS, replace=False))
    else:
        row_idx = np.arange(n_rows)

    cand_codes = np.ascontiguousarray(
        data[np.ix_(row_idx, cand_cols)], dtype=np.int32
    )
    y_codes = np.ascontiguousarray(data[row_idx, t_idx], dtype=np.int32)

    # Per-recipe frozen bin codes for the unary_binary survival replay. ``data``
    # already carries every engineered column the screen scored, keyed by name in
    # ``cols``; the recipe's source operands are likewise present (raw or nested).
    name_to_col = {str(nm): c for c, nm in enumerate(cols)}
    _recipes = engineered_recipes or {}
    if isinstance(_recipes, (list, tuple)):
        _recipes = {getattr(r, "name", str(i)): r for i, r in enumerate(_recipes)}
    recipe_replay: dict = {}
    for nm, r in _recipes.items():
        if getattr(r, "kind", None) != "unary_binary":
            continue
        eng_c = name_to_col.get(str(nm))
        if eng_c is None:
            continue
        src = tuple(getattr(r, "src_names", ()) or ())
        a_c = name_to_col.get(str(src[0])) if len(src) >= 1 else None
        b_c = name_to_col.get(str(src[1])) if len(src) >= 2 else None
        _rx = getattr(r, "extra", None) or {}
        alt = any(
            _k in _rx for _k in (
                "prewarp_a_coef", "prewarp_b_coef",
                "gate_med_a_median", "gate_med_b_median",
            )
        )
        recipe_replay[str(nm)] = {
            "eng_codes": np.ascontiguousarray(data[row_idx, eng_c], dtype=np.int32),
            "a_codes": None if a_c is None else np.ascontiguousarray(data[row_idx, a_c], dtype=np.int32),
            "b_codes": None if b_c is None else np.ascontiguousarray(data[row_idx, b_c], dtype=np.int32),
            "alt": bool(alt),
        }

    self._stability_replay_state_ = {
        "cand_codes": cand_codes,
        "cand_names": cand_names,
        "selected_mask": selected_mask,
        "y_codes": y_codes,
        "recipe_replay": recipe_replay,
    }

def fe_decide_on_subsample(
    fit_with_recipes_fn,
    X,
    y,
    *,
    subsample_n: int = 0,
    subsample_seed: int = 42,
    shared_subsample_idx=None,
    **kwargs,
):
    """Run an ``*_with_recipes`` FE family on a row-SUBSAMPLE for its DECISION, then
    rebuild the chosen columns at FULL n by replaying the returned recipes.

    CORRECTNESS BOUNDARY -- CLOSED-FORM families ONLY. This wrapper rebuilds the output
    by REPLAYING each recipe (``apply_recipe`` = the transform-time path). That equals
    the fit-time column ONLY when the engineered column is a PURE FUNCTION of x (the
    orthogonal-polynomial / Fourier / spline basis families: orth univariate / pair /
    triplet / quadruplet / extra-basis / the alternate-scorer variants, whose recipe is
    a closed-form basis eval). Do NOT use it for families whose fit-time output is an
    OUT-OF-FOLD / data-dependent encoding -- e.g. ``binned_numeric_agg`` (k-fold OOF
    target-mean/stat encoding): replaying its recipe uses full-train cell stats, which
    is the LEAKY transform-time value, not the OOF fit-time column. Such families must
    subsample only their pair/edge DECISION while keeping the OOF stat computation at
    full n (a per-family change, not this wrapper). MDLP / discretization edge selection
    is likewise a separate (edges-then-searchsorted) integration.

    The MRMR FE families historically ranked/selected on the FULL training frame, while
    the pair-search (``check_prospective_fe_pairs``) decides on a ~30k row subsample and
    replays winners at full n. This wrapper gives every CLOSED-FORM family the SAME
    treatment WITHOUT editing each function: it calls
    ``fit_with_recipes_fn`` on the seeded subsample (so the CPU-heavy MI / detection
    sweep sees ~subsample_n rows, not n), then replays each returned
    ``EngineeredRecipe`` on the full X via :func:`apply_recipe`. The recipes are
    closed-form, so the appended columns equal a full-data fit GIVEN the same winners
    -- selection-equivalence is validated by the FE pins.

    Output-safety fallbacks (return the FULL-data call, never lose columns):
      * subsample disabled / frame already small (``subsample_n`` <= 0 or >= n);
      * the family returned NO recipes (nothing replayable);
      * a winner column has no replayable recipe (partial coverage) -- rather than
        silently drop it, fall back to the full-data decision for the whole family.
    Any per-recipe replay error also triggers the full-data fallback.
    """
    n = len(X)
    from .._fe_frame_ops import fe_to_pandas
    # Prefer the fit's ONE shared row-index draw when supplied (so this closed-form family scores the
    # SAME rows as the pair-search / polynom / sufficiency floor); else fall back to the legacy draw.
    _shared = None
    if shared_subsample_idx is not None:
        try:
            _s = np.asarray(shared_subsample_idx)
            if _s.ndim == 1 and 0 < _s.shape[0] < n and int(_s.max()) < n:
                _shared = _s.astype(np.int64, copy=False)
        except Exception:
            _shared = None
    if _shared is None and not (isinstance(subsample_n, int) and 0 < subsample_n < n):
        return fit_with_recipes_fn(fe_to_pandas(X), y, **kwargs)
    if _shared is not None:
        _idx = _shared
    else:
        _idx = np.sort(
            np.random.default_rng(int(subsample_seed)).choice(
                n, size=int(subsample_n), replace=False,
            )
        )
    from .._fe_frame_ops import fe_subsample_to_pandas
    # Format-agnostic subsample: pandas -> .iloc, polars -> native gather + to_pandas on the SMALL subsample only (the
    # family decision bodies are pandas-native). Materialises ~len(_idx) rows, never the full frame -- so a 100+ GB polars
    # frame is not duplicated, and every FE family now runs on polars input (previously skipped by isinstance guards).
    _X_sub = fe_subsample_to_pandas(X, _idx)
    _y_sub = np.asarray(y)[_idx]
    # Arity-agnostic: every ``*_with_recipes`` family returns the augmented frame FIRST
    # and the recipe list LAST (3-tuple (X, scores, recipes) for univariate / extra /
    # triplet / scorers; 4-tuple (X, uni_sc, cross_sc, recipes) for the pair family).
    # Preserve the middle elements verbatim.
    _ret = fit_with_recipes_fn(_X_sub, _y_sub, **kwargs)
    if not (isinstance(_ret, tuple) and len(_ret) >= 2):
        logger.warning(
            "fe_decide_on_subsample: %s returned an unexpected shape on the %d-row subsample; re-running the DECISION "
            "on the FULL %d rows (this is the costly full-n path the subsample was meant to avoid).",
            getattr(fit_with_recipes_fn, "__name__", "FE family"), len(_idx), n,
        )
        return fit_with_recipes_fn(fe_to_pandas(X), y, **kwargs)  # unexpected shape -> full call
    X_aug_sub, recipes, _middle = _ret[0], _ret[-1], _ret[1:-1]
    _appended_sub = [c for c in X_aug_sub.columns if c not in _X_sub.columns]
    # EMPTY RESULT is a VALID decision, not a coverage failure: when the subsample DECISION engineered no columns, the
    # family simply found nothing worth adding for this target -- re-running the whole (expensive) decision on the full n
    # would just re-discover "nothing" at full cost (this was the 3x full-n FE re-run wasting minutes on TVT, where the
    # orth pair/triplet/quadruplet families legitimately find no uplift). Return X unchanged (no columns to replay) with
    # the subsample's middle payload + empty recipes; DO NOT pay the full-n path for an empty set.
    if not _appended_sub:
        return (X, *_middle, recipes)
    # Partial recipe coverage (SOME columns appended but not all have a replayable recipe) -> full-data decision
    # (correctness over speed). Surfaced (not silent): worth fixing at the family (incomplete recipe emission), not
    # paying the full-n cost each fit.
    if not recipes or len({r.name for r in recipes}) < len(_appended_sub):
        logger.warning(
            "fe_decide_on_subsample: %s produced %d engineered column(s) on the %d-row subsample but only %d replayable "
            "recipe(s); re-running the DECISION on the FULL %d rows (costly full-n path -- the subsample bypass is lost).",
            getattr(fit_with_recipes_fn, "__name__", "FE family"),
            len(_appended_sub), len(_idx), len({r.name for r in (recipes or [])}), n,
        )
        return fit_with_recipes_fn(fe_to_pandas(X), y, **kwargs)
    from ..engineered_recipes import apply_recipe
    from .._fe_frame_ops import fe_append_columns
    _full_cols: dict = {}
    try:
        # apply_recipe is format-agnostic (per-column _extract_column), so full-n replay reads polars/pandas/ndarray
        # operands as views -- no whole-matrix copy.
        for r in recipes:
            _full_cols[r.name] = np.asarray(apply_recipe(r, X))
    except Exception:
        logger.warning(
            "fe_decide_on_subsample: full-n replay failed for %r; "
            "falling back to full-data decision.",
            getattr(r, "name", "?"),
        )
        return fit_with_recipes_fn(fe_to_pandas(X), y, **kwargs)
    # Append the engineered columns in X's OWN framework (polars with_columns / pandas concat) -- only the new columns
    # are materialised, never a copy of X.
    X_aug = fe_append_columns(X, _full_cols)
    return (X_aug, *_middle, recipes)
