"""Groups 1, 3 and 4 of the hybrid_orth variants run together; group 2 runs after group 1 as before, except for its bootstrap and KSG stages, which read only the step input and run beside group 1.

Every variant takes its sources as the numeric columns that are not hybrid_orth outputs (``hybrid_orth_features_`` is re-read before each one), so a group
reads the same sources whether or not the groups before it ran. The exception is the three-gate variant in group 2, whose support set is the hybrid
columns appended before it; group 2 therefore runs after group 1, on the frame with group 1's columns. The groups run on private copies of the estimator
and of the recipe dict, and everything is folded back in the sequential order 1, 2, 3, 4 with the first owner of a name kept, as the sequential run keeps
it (a later variant skips a name already in the frame). ``MLFRAME_FE_HYBRID_VARIANT_CONCURRENCY=0`` runs the groups one by one.
"""

from __future__ import annotations

import logging
import os
import functools
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: Walls of groups 1, 3 and 4 in a warm 100k x 100 fit (6.6, 2.1, 6.2 s): they place the groups on the device slots.
_GROUP_WEIGHTS = (8.2, 2.1, 6.2)
#: Device-pool peaks of the three groups (MB at 50000 rows): the 486 MB peak of the whole early_a cascade split by the walls above. They split a device's memory among the groups and, scaled
#: linearly with the rows, decide whether the groups fit the device together (they run one after another when they do not, as the FE cascades do).
_GROUP_VRAM_WEIGHTS = (216.0, 69.0, 203.0)


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
    from ._group1 import _hybrid_orth_family_variants_group1_tail
    from ._group2 import _stage_bootstrap, _stage_ksg

    g1_head = functools.partial(g1, tail=False)
    recipes = kwargs["_hybrid_orth_pre_recipes"]
    base_recipes = dict(recipes)
    base_state = snapshot(self)
    base_cols = set(map(str, X.columns))
    private_recipes = [dict(base_recipes) for _ in range(6)]  # slots 0, 1, 2: groups 1 (head), 3, 4; slot 3: the tail of group 1; slots 4, 5: the bootstrap and ksg stages of group 2
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

    def lane_with_tail(holder: Any) -> Any:
        """Group 3, then the tail of group 1 (adaptive_arity, adaptive_degree, conditional_routing) in the same device slot: group 3 is the shortest group and the tail is independent of the head."""
        return stage(g3, 1)(holder), stage(_hybrid_orth_family_variants_group1_tail, 3)(holder)

    def bootstrap_stage(*, X, self, _fe_family_on, _y_np, _hybrid_orth_pre_recipes, verbose, **_rest):
        """The bootstrap stage of group 2 on the step input."""
        return _stage_bootstrap(self, _fe_family_on, _y_np, X, _hybrid_orth_pre_recipes, verbose)

    def ksg_stage(*, X, self, _fe_family_on, _y_np, _hybrid_orth_pre_recipes, verbose, **_rest):
        """The KSG stage of group 2 on the step input."""
        return _stage_ksg(self, _fe_family_on, _y_np, X, _hybrid_orth_pre_recipes, verbose)

    def lane_head_with_group2_stages(holder: Any) -> Any:
        """Group 1's head, then the bootstrap and KSG stages of group 2 in the same device slot: group 1's head is the shortest lane, and these two read the step input (their sources are the raw numeric
        columns, as in every other variant); only the three-gate stage needs group 1's columns and stays after the groups."""
        return stage(g1_head, 0)(holder), stage(bootstrap_stage, 4)(holder), stage(ksg_stage, 5)(holder)

    outs = run_stages_concurrently(_Holder(), [lane_head_with_group2_stages, lane_with_tail, stage(g4, 2)], _GROUP_WEIGHTS, vram_weights=_GROUP_VRAM_WEIGHTS, n_rows=len(X))
    if outs is None:
        return None
    by_slot = {slot: est for slot, est in copies}
    (out1, out_boot, out_ksg), (out3, out_tail), out4 = outs

    # group 1 first, onto the real estimator, then group 2 on the frame and state group 1 left, exactly as in the sequential run
    merge_stage_state(self, base_state, [by_slot[0]])
    _fold_recipes(recipes, base_recipes, private_recipes[0])
    acc = _fe_merge_new_columns(X, out1, X)
    # then the tail of group 1 (the last families of its sequential order), then group 2 on the frame both left
    tail_dup = [c for c in _new_names(out_tail, base_cols) if c in set(map(str, acc.columns))]
    merge_stage_state(self, base_state, [by_slot[3]])
    _drop_repeated_features(self)
    _fold_recipes(recipes, base_recipes, private_recipes[3], skip=set(tail_dup))
    acc = _fe_merge_new_columns(acc, out_tail, X)
    # group 2 in its sequential order: diff_basis and cluster_basis, then the bootstrap result built beside group 1, then three_gate (whose support set holds the bootstrap columns, as before), then the
    # KSG result; a name an earlier owner holds stays with it
    acc = g2(X=acc, phase="pre", **kwargs)
    acc = _fold_group2_stage(self, acc, X, base_state, base_cols, by_slot[4], private_recipes[4], out_boot, recipes, base_recipes)
    acc = g2(X=acc, phase="post", skip=("bootstrap", "ksg"), **kwargs)
    acc = _fold_group2_stage(self, acc, X, base_state, base_cols, by_slot[5], private_recipes[5], out_ksg, recipes, base_recipes)
    # groups 3 and 4 keep only the names no earlier group owns
    for slot, out in ((1, out3), (2, out4)):
        owned = set(map(str, acc.columns))
        dup = [c for c in _new_names(out, base_cols) if c in owned]
        merge_stage_state(self, base_state, [by_slot[slot]])
        _drop_repeated_features(self)
        _fold_recipes(recipes, base_recipes, private_recipes[slot], skip=set(dup))
        acc = _fe_merge_new_columns(acc, out, X)
    return acc


def _fold_group2_stage(self: Any, acc: Any, X: Any, base_state: Any, base_cols: "set[str]", est: Any, private: dict, out: Any, recipes: dict, base_recipes: dict) -> Any:
    """Fold one group-2 stage computed beside group 1 onto ``acc``: its estimator state, the recipes it gained and the columns no earlier owner holds."""
    from .._fe_stage_concurrency import merge_stage_state
    from .._fe_stage_merge import _fe_merge_new_columns

    owned = set(map(str, acc.columns))
    dup = [c for c in _new_names(out, base_cols) if c in owned]
    merge_stage_state(self, base_state, [est])
    _drop_repeated_features(self)
    _fold_recipes(recipes, base_recipes, private, skip=set(dup))
    return _fe_merge_new_columns(acc, out, X)


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
