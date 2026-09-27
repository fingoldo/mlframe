"""The per-target training loop of ``train_mlframe_models_suite``.

Lifted out of the suite body (at its length ceiling) so the loop can grow the per-target row handling that targets with
missing labels need without touching it again.

The unsupervised pre-screen runs here, once, before any target. It used to run inside the first target's
``_train_one_target``, which sits inside ``target_scoped_frames``: on a suite with per-target supervised columns the
scope restored the pre-screen frames on exit while ``_pre_screen_done`` stayed latched, so every later target trained
on the columns the screen had dropped.
"""

from __future__ import annotations

from typing import Any

from pyutilz.strings import slugify
from pyutilz.system import tqdmu_lazy_start

from mlframe.training.pipeline.shared import target_scoped_frames

from ._main_train_suite_encoding import _encode_string_multiclass_target
from ._phase_train_one_target_pre_screen import _maybe_run_unsupervised_pre_screen


def train_every_target(ctx: Any, target_by_type: dict, metadata: dict, pr: Any) -> None:
    """Train every (target type, target) pair in ``target_by_type`` through ``pr._train_one_target``.

    ``pr`` is the ``_phase_runners`` module, passed in rather than imported so the tests that patch
    ``pr._train_one_target`` keep patching the call this loop makes.
    """
    _maybe_run_unsupervised_pre_screen(ctx, None)
    for target_type, targets in tqdmu_lazy_start(target_by_type.items(), desc="target type"):
        # Written directly onto ctx so _finalize_and_save_metadata's `if ctx.slug_to_original_target_type:` guard sees
        # it -- mirrors how ctx.slug_to_original_target_name is populated in _phase_train_one_target_model_setup.py.
        ctx.slug_to_original_target_type[slugify(str(target_type).lower())] = target_type
        for cur_target_name, cur_target_values in tqdmu_lazy_start(targets.items(), desc="target"):
            cur_target_values = _encode_string_multiclass_target(target_type, cur_target_name, cur_target_values, metadata)
            targets[cur_target_name] = cur_target_values
            # Other targets' label-supervised composite columns are hidden from this target's models.
            with target_scoped_frames(ctx, target_type, cur_target_name):
                pr._train_one_target(ctx, target_type, targets, cur_target_name, cur_target_values)
