"""Groups 1, 3 and 4 of the hybrid_orth variants run together; group 2 runs after group 1 as before.

Every variant takes its sources as the numeric columns that are not hybrid_orth outputs (``hybrid_orth_features_`` is re-read before each one), so a group
reads the same sources whether or not the groups before it ran. The exception is the three-gate variant in group 2, whose support set is the hybrid
columns appended before it; group 2 therefore runs after group 1, on the frame with group 1's columns. The groups run on private copies of the estimator
and of the recipe dict, and everything is folded back in the sequential order 1, 2, 3, 4 with the first owner of a name kept, as the sequential run keeps
it (a later variant skips a name already in the frame). ``MLFRAME_FE_HYBRID_VARIANT_CONCURRENCY=0`` runs the groups one by one.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: Walls of groups 1, 3 and 4 in a warm 100k x 100 fit (6.6, 2.1, 6.2 s): they place the groups on the device slots.
_GROUP_WEIGHTS = (6.6, 2.1, 6.2)


def hybrid_variant_concurrency_enabled() -> bool:
    """Whether groups 1, 3 and 4 run together (default on); ``MLFRAME_FE_HYBRID_VARIANT_CONCURRENCY=0`` keeps the sequential order, and so does
    ``MLFRAME_FE_STAGE_CONCURRENCY=0`` (the switch that runs the whole FE one cascade at a time)."""
    from .._fe_stage_concurrency import concurrency_enabled

    return concurrency_enabled() and os.environ.get("MLFRAME_FE_HYBRID_VARIANT_CONCURRENCY", "1").strip() != "0"


def _new_names(out: Any, base_cols: "set[str]") -> "list[str]":
    """The column names ``out`` adds over the frame the groups started from."""
    return [str(c) for c in out.columns if str(c) not in base_cols]


def run_groups_concurrently(
    self: Any, X: Any, kwargs: dict, groups: "tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any], Callable[..., Any]]",
) -> Optional[Any]:
    """The frame after all four groups, or ``None`` when no device plan exists for the concurrent groups (the caller then runs them in order)."""
    from .._fe_stage_concurrency import merge_stage_state, private_copy, run_stages_concurrently, snapshot
    from .._fe_stage_merge import _fe_merge_new_columns

    g1, g2, g3, g4 = groups
    recipes = kwargs["_hybrid_orth_pre_recipes"]
    base_recipes = dict(recipes)
    base_state = snapshot(self)
    base_cols = set(map(str, X.columns))
    private_recipes = [dict(base_recipes) for _ in range(3)]
    copies: "list[Any]" = []

    def stage(fn: Callable[..., Any], slot: int) -> Callable[[Any], Any]:
        """The group on the estimator copy it is handed, with its own recipe dict."""

        def run(_holder: Any) -> Any:
            """One group on a private copy of the estimator; the copy is kept for the ordered fold."""
            est = private_copy(self)
            copies.append((slot, est))
            kw = dict(kwargs, self=est, _hybrid_orth_pre_recipes=private_recipes[slot])
            return fn(X=X, **kw)

        return run

    outs = run_stages_concurrently(_Holder(), [stage(g1, 0), stage(g3, 1), stage(g4, 2)], _GROUP_WEIGHTS, vram_weights=(1.0, 1.0, 1.0))
    if outs is None:
        return None
    by_slot = {slot: est for slot, est in copies}
    out1, out3, out4 = outs

    # group 1 first, onto the real estimator, then group 2 on the frame and state group 1 left, exactly as in the sequential run
    merge_stage_state(self, base_state, [by_slot[0]])
    _fold_recipes(recipes, base_recipes, private_recipes[0])
    acc = _fe_merge_new_columns(X, out1, X)
    acc = g2(X=acc, **kwargs)
    # groups 3 and 4 keep only the names no earlier group owns
    for slot, out in ((1, out3), (2, out4)):
        owned = set(map(str, acc.columns))
        dup = [c for c in _new_names(out, base_cols) if c in owned]
        merge_stage_state(self, base_state, [by_slot[slot]])
        _drop_repeated_features(self)
        _fold_recipes(recipes, base_recipes, private_recipes[slot], skip=set(dup))
        acc = _fe_merge_new_columns(acc, out, X)
    return acc


class _Holder:
    """A stand-in estimator for ``run_stages_concurrently``: each group makes its own ``private_copy`` of the real one, and the copies are folded here in order."""


def _fold_recipes(target: dict, base: dict, private: dict, skip: "set[str] | None" = None) -> None:
    """Add the recipes ``private`` gained over ``base`` to ``target``, keeping a name ``target`` already holds (its first owner) and the names in ``skip``."""
    for name, recipe in private.items():
        if name in base or name in target or (skip and str(name) in skip):
            continue
        target[name] = recipe


def _drop_repeated_features(est: Any) -> None:
    """Keep the first occurrence of every name in ``hybrid_orth_features_`` after a fold appended a later group's names."""
    feats = getattr(est, "hybrid_orth_features_", None)
    if not isinstance(feats, list):
        return
    seen: "set[str]" = set()
    kept = []
    for f in feats:
        if str(f) in seen:
            continue
        seen.add(str(f))
        kept.append(f)
    if len(kept) != len(feats):
        est.hybrid_orth_features_ = kept
