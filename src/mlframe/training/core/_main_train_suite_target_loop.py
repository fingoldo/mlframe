"""The per-target training loop of ``train_mlframe_models_suite``.

Lifted out of the suite body (at its length ceiling) so the loop can grow the per-target row handling that targets with
missing labels need without touching it again.

The unsupervised pre-screen runs here, once, before any target. It used to run inside the first target's
``_train_one_target``, which sits inside ``target_scoped_frames``: on a suite with per-target supervised columns the
scope restored the pre-screen frames on exit while ``_pre_screen_done`` stayed latched, so every later target trained
on the columns the screen had dropped.

A target with missing labels (``target_null_policy="drop_rows"``) trains inside ``target_row_scope``: the context is
narrowed to its labelled rows and restored afterwards. A fully labelled target runs exactly as before.
"""

from __future__ import annotations

from typing import Any

from pyutilz.strings import slugify
from pyutilz.system import tqdmu_lazy_start

from mlframe.training.pipeline.shared import target_scoped_frames

from ._main_train_suite_encoding import _encode_string_multiclass_target
from ._phase_train_one_target_pre_screen import _maybe_run_unsupervised_pre_screen
from ._target_row_decisions import rows_for_target
from ._target_row_scope import target_row_scope


def _training_order(ctx: Any, target_by_type: dict, metadata: dict) -> "tuple[list, dict]":
    """The targets to train as ``[(rows, [(type, name, values, working), ...]), ...]``, and the original key order.

    Fully labelled targets come first, in their original order, under ``rows=None``. Targets with missing labels follow,
    one group per distinct set of rows (largest labelled train first, original order inside), so a group's frames are
    narrowed once for all of its targets. Targets the per-split thresholds skip are left out.
    """
    plain: list = []
    groups: dict = {}
    original_order: dict = {}
    rows_by_signature: dict = {}
    for target_type, targets in target_by_type.items():
        # Written directly onto ctx so _finalize_and_save_metadata's `if ctx.slug_to_original_target_type:` guard sees
        # it -- mirrors how ctx.slug_to_original_target_name is populated in _phase_train_one_target_model_setup.py.
        ctx.slug_to_original_target_type[slugify(str(target_type).lower())] = target_type
        original_order[target_type] = list(targets)
        for cur_target_name, cur_target_values in list(targets.items()):
            cur_target_values = _encode_string_multiclass_target(target_type, cur_target_name, cur_target_values, metadata)
            targets[cur_target_name] = cur_target_values
            rows, working, train_it = rows_for_target(ctx, target_type, cur_target_name, cur_target_values, metadata, rows_by_signature)
            if not train_it:
                continue
            item = (target_type, cur_target_name, cur_target_values, working)
            if rows is None:
                plain.append(item)
            else:
                groups.setdefault(rows.signature, (rows, []))[1].append(item)
    ordered = sorted(groups.values(), key=lambda group: -group[0].n_labelled.get("filtered_train_idx", group[0].n_labelled.get("train_idx", 0)))
    return ([(None, plain)] if plain else []) + ordered, original_order


def _restore_key_order(per_type: dict, original_order: dict) -> None:
    """Put each target type's entries back in the order the targets were given (training order is group-major)."""
    for target_type, names in original_order.items():
        by_name = per_type.get(target_type)
        if not isinstance(by_name, dict):
            continue
        ordered = {name: by_name[name] for name in names if name in by_name}
        ordered.update({name: value for name, value in by_name.items() if name not in ordered})  # composite targets etc.
        by_name.clear()
        by_name.update(ordered)


def train_every_target(ctx: Any, target_by_type: dict, metadata: dict, pr: Any) -> None:
    """Train every (target type, target) pair in ``target_by_type`` through ``pr._train_one_target``.

    ``pr`` is the ``_phase_runners`` module, passed in rather than imported so the tests that patch
    ``pr._train_one_target`` keep patching the call this loop makes.
    """
    _maybe_run_unsupervised_pre_screen(ctx, None)
    order, original_order = _training_order(ctx, target_by_type, metadata)
    for rows, items in tqdmu_lazy_start(order, desc="target group"):
        with target_row_scope(ctx, rows):
            for target_type, cur_target_name, cur_target_values, working in items:
                targets = target_by_type[target_type]
                # The models see the working target (integer-filled for classification); the suite keeps the one with gaps.
                targets[cur_target_name] = working
                try:
                    # Other targets' label-supervised composite columns are hidden from this target's models.
                    with target_scoped_frames(ctx, target_type, cur_target_name):
                        pr._train_one_target(ctx, target_type, targets, cur_target_name, working)
                finally:
                    targets[cur_target_name] = cur_target_values
    for per_type in (getattr(ctx, "models", None), getattr(ctx, "ensembles", None)):
        if isinstance(per_type, dict):
            _restore_key_order(per_type, original_order)
