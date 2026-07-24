"""MLP-specific helpers carved from ``_phase_train_one_target_body``.

Module-level, closure-free helpers used by the per-target training body for
the MLP extreme-AR + group-aware protections. Lifted verbatim so behavioural
equivalence is preserved by construction; the parent re-imports from this
sibling at module-load time.

Contents:

- ``_get_per_group_pattern`` / ``_identify_per_group_columns`` -- compiled-
  regex cache + matcher used to find ``group_*_(mean|std|min|max)`` columns
  that the MLP must drop on extreme-AR + group-aware splits.
- ``_drop_columns_for_mlp`` -- polars / pandas-agnostic column drop returning
  a new frame; safe on ``None`` and on missing-column requests.
- ``_apply_mlp_extreme_ar_weight_decay_bump`` -- walks the MLP estimator
  nesting and bumps ``optimizer_kwargs['weight_decay']``; swaps Adam -> AdamW
  when prior optimizer ignored weight_decay.
- ``_apply_mlp_extreme_ar_output_activation`` -- sets
  ``network_params['output_activation']='tanh_train_range'`` so the regression
  head is bounded around the y-train midpoint.
"""
from __future__ import annotations

import logging
import re

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]


logger = logging.getLogger("mlframe.training.core._phase_train_one_target")


_PER_GROUP_PATTERN_CACHE: dict[str, "re.Pattern[str]"] = {}


def _get_per_group_pattern(pattern: str) -> "re.Pattern[str]":
    """Case-insensitive compiled regex for ``pattern``, memoized per pattern string so repeated calls avoid recompiling."""
    cached = _PER_GROUP_PATTERN_CACHE.get(pattern)
    if cached is None:
        cached = re.compile(pattern, re.IGNORECASE)
        _PER_GROUP_PATTERN_CACHE[pattern] = cached
    return cached


def _identify_per_group_columns(columns, pattern: str) -> list[str]:
    """Return the subset of ``columns`` matching the per-group-aggregate regex. Used by the MLP extreme-AR protection to strip group-level constants that extrapolate on unseen groups."""
    if not columns or not pattern:
        return []
    rx = _get_per_group_pattern(pattern)
    return [c for c in columns if rx.match(str(c))]


def _drop_columns_for_mlp(df, cols_to_drop):
    """Polars / pandas-agnostic helper that returns a NEW frame with the given columns dropped. ``None`` -> ``None``. Missing columns are silently ignored (safe in cross-tier reuse)."""
    if df is None or not cols_to_drop:
        return df
    # Avoid ``if columns:`` -- pandas Index raises on bool(). Iterate directly: list(Index) yields the column names regardless of type.
    _cols_attr = getattr(df, "columns", None)
    if _cols_attr is None:
        return df
    _have = set(list(_cols_attr))
    _drop = [c for c in cols_to_drop if c in _have]
    if not _drop:
        return df
    try:
        if pl is not None and isinstance(df, pl.DataFrame):
            return df.drop(_drop)
        return df.drop(columns=_drop)
    except Exception as exc:
        logger.debug("column-drop failed, returning frame unmodified: %s", exc)
        return df


def _apply_mlp_extreme_ar_weight_decay_bump(
    model, factor: float, base_weight_decay: float,
) -> bool:
    """Walk the MLP estimator nesting (TTR / pipeline / metamodel) to find ``model_params`` on the inner ``PytorchLightningEstimator`` and bump ``optimizer_kwargs["weight_decay"]`` by ``factor`` (against ``base_weight_decay`` when no prior decay was set). Forces AdamW when the prior optimizer was plain Adam (Adam ignores weight_decay).

    Returns True if the bump was applied (i.e. an inner Lightning estimator was found and its model_params mutated), False otherwise.
    """
    candidates = []
    visited = set()

    def _enqueue(obj):
        """Add ``obj`` to the BFS worklist unless it is ``None`` or already visited (guards against cycles in the nested-estimator graph)."""
        if obj is None or id(obj) in visited:
            return
        visited.add(id(obj))
        candidates.append(obj)

    _enqueue(model)
    found_inner = None
    while candidates:
        cur = candidates.pop(0)
        if hasattr(cur, "model_params") and hasattr(cur, "network_params"):
            found_inner = cur
            break
        if hasattr(cur, "regressor"):
            _enqueue(getattr(cur, "regressor"))
        if hasattr(cur, "named_steps"):
            for step in cur.named_steps.values():
                _enqueue(step)
        for attr in ("estimator", "base_estimator", "estimator_", "regressor_"):
            if hasattr(cur, attr):
                _enqueue(getattr(cur, attr))

    if found_inner is None:
        return False

    # Mutate in-place. The caller (per-weight loop) holds a freshly cloned model so this does NOT leak into other weight schemas.
    try:
        import torch as _torch
        mp = dict(found_inner.model_params)
        ok = dict(mp.get("optimizer_kwargs", {}) or {})
        prior_decay = float(ok.get("weight_decay", 0.0) or 0.0)
        # Bump: if user had no prior decay, multiply BASE instead of 0.
        new_decay = (prior_decay if prior_decay > 0 else base_weight_decay) * factor
        ok["weight_decay"] = new_decay
        mp["optimizer_kwargs"] = ok
        # Adam ignores weight_decay -> swap to AdamW.
        cur_opt = mp.get("optimizer", None)
        if cur_opt is _torch.optim.Adam:
            mp["optimizer"] = _torch.optim.AdamW
        found_inner.model_params = mp
        logger.info(
            "MLP extreme-AR + group-aware: weight_decay bumped %g -> %g "
            "(factor=%g, base=%g); optimizer=%s.",
            prior_decay, new_decay, factor, base_weight_decay,
            getattr(mp.get("optimizer"), "__name__", str(mp.get("optimizer"))),
        )
        return True
    except Exception as _bump_err:
        logger.warning(
            "MLP extreme-AR weight_decay bump failed (%s); leaving " "optimizer kwargs unchanged.",
            _bump_err,
        )
        return False


def _apply_mlp_extreme_ar_output_activation(model) -> bool:
    """Walk the MLP estimator nesting and set ``network_params['output_activation']='tanh_train_range'``. Bounds the regression head to ~6 sigma around the y-train midpoint.

    Returns True if applied, False otherwise.
    """
    candidates = []
    visited = set()

    def _enqueue(obj):
        """Add ``obj`` to the BFS worklist unless it is ``None`` or already visited (guards against cycles in the nested-estimator graph)."""
        if obj is None or id(obj) in visited:
            return
        visited.add(id(obj))
        candidates.append(obj)

    _enqueue(model)
    found_inner = None
    while candidates:
        cur = candidates.pop(0)
        if hasattr(cur, "network_params") and isinstance(getattr(cur, "network_params"), dict):
            found_inner = cur
            break
        if hasattr(cur, "regressor"):
            _enqueue(getattr(cur, "regressor"))
        if hasattr(cur, "named_steps"):
            for step in cur.named_steps.values():
                _enqueue(step)
        for attr in ("estimator", "base_estimator", "estimator_", "regressor_"):
            if hasattr(cur, attr):
                _enqueue(getattr(cur, attr))

    if found_inner is None:
        return False
    try:
        np_dict = dict(found_inner.network_params)
        # Respect explicit user override.
        if np_dict.get("output_activation") not in (None, "linear"):
            return False
        np_dict["output_activation"] = "tanh_train_range"
        found_inner.network_params = np_dict
        logger.info(
            "MLP extreme-AR + group-aware: output_activation='tanh_train_range' " "enabled; scale/center auto-derived from y_train at fit time.",
        )
        return True
    except Exception as _act_err:
        logger.warning(
            "MLP extreme-AR output_activation set failed (%s); leaving unchanged.",
            _act_err,
        )
        return False


def extreme_ar_skip_decision(
    model_name: str,
    target_name: str,
    *,
    skip_models,
    skip_enabled: bool,
    lag1_autocorr_per_group,
    group_aware: bool,
    threshold: float = 0.99,
) -> "tuple[bool, bool]":
    """Decide whether to SKIP fitting ``model_name`` on ``target_name``.

    Pure / side-effect-free so it is unit-testable without the suite.

    Returns ``(skip, extreme_ar_fired)``:
      * ``extreme_ar_fired`` -- the RAW-target extreme-AR + group-aware
        signal (group_aware AND lag1 >= threshold). The MLP uses this for
        its weight_decay / output-activation protections even when the hard
        skip is off. Always ``False`` for a composite target (the stored
        distribution report is the RAW target's; a composite bounds the
        variance the signal warns about, so it does not apply).
      * ``skip`` -- True iff the fit should be skipped: the signal fired
        AND skipping is enabled AND ``model_name`` is in ``skip_models``.

    Composite targets are NEVER skipped: residual/diff/linres targets bound
    the variance and are exactly where neural nets belong.
    """
    from ..composite.transforms import is_composite_target_name

    if is_composite_target_name(target_name):
        return False, False
    fired = bool(group_aware and lag1_autocorr_per_group is not None and float(lag1_autocorr_per_group) >= float(threshold))
    skip = bool(fired and skip_enabled and model_name in tuple(skip_models))
    return skip, fired
