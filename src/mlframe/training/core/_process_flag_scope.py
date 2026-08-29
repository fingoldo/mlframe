"""Restore of the process-wide reporting/evaluation overrides a training suite flips for its duration.

``setup_configuration`` flips three thread/process-wide flags (residual-audit reporting, inline plot display, and
per-format plot subfolders) so that code far below the suite -- inside ``render_and_save``, inside the evaluation
helpers -- sees the suite's configuration without every call site threading it through. The prior values are
snapshotted into ``ctx.artifacts`` and restored here.

The restore has to be reachable from BOTH the normal finalize step and a ``finally`` at the suite boundary: a suite
that raises never reaches finalize, and the flag would then stay flipped for every later caller ON THAT THREAD --
in a test process, for every later test. That is the same defect class the FE deadline had (its documented
``clear_fe_deadline`` had no call site at all), and it surfaced the same way: unrelated later tests failing on a
setting nobody in them chose. Popping the keys makes the call idempotent, so running it twice is harmless.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


SNAPSHOT_PREFIX = "_process_flag_prior_"


def capture_process_flag_snapshot(ctx: Any) -> Dict[str, Any]:
    """Lift the restore snapshot OUT of ``ctx.artifacts`` and into a dict the caller owns.

    ``setup_configuration`` stashes the prior flag values in ``ctx.artifacts``, but a later phase rebuilds that
    dict wholesale (``_phase_helpers.py`` assigns ``ctx.artifacts = artifacts`` from its own local), which threw
    the snapshot away -- so ``finalize_suite`` found nothing to restore and every suite run left the three
    process-wide flags flipped on its thread for good. Traced by printing the keys the restore actually received:
    ``restore called, keys=[]`` on a run whose setup had just reported ``prior=None set=True``.

    Taking a copy at the suite boundary makes the restore independent of what any phase does to that bag. The
    keys are left in place as well, so ``finalize_suite``'s own restore still works on the runs that reach it.
    """
    artifacts = getattr(ctx, "artifacts", None) or {}
    return {k: v for k, v in artifacts.items() if k.startswith(SNAPSHOT_PREFIX)}


def restore_process_flags(artifacts: Optional[Dict[str, Any]]) -> None:
    """Restore every snapshotted process-wide override found in ``artifacts``; no-op for the ones not flipped."""
    if not artifacts:
        return

    residual_audit_prior = artifacts.pop("_process_flag_prior_residual_audit", None)
    if residual_audit_prior is not None:
        try:
            from mlframe.training.evaluation import _set_residual_audit_enabled

            _set_residual_audit_enabled(residual_audit_prior)
        except (ImportError, AttributeError) as err:
            logger.debug("residual_audit flag restore failed: %s: %s", type(err).__name__, err)

    if "_process_flag_prior_inline_display" in artifacts:
        prior = artifacts.pop("_process_flag_prior_inline_display")
        try:
            from mlframe.reporting.renderers.save import set_inline_display_mode

            set_inline_display_mode(prior)
        except (ImportError, AttributeError) as err:
            logger.debug("inline_display flag restore failed: %s: %s", type(err).__name__, err)

    if "_process_flag_prior_format_subfolders" in artifacts:
        prior = artifacts.pop("_process_flag_prior_format_subfolders")
        try:
            from mlframe.reporting.renderers.save import set_format_subfolders

            set_format_subfolders(prior)
        except (ImportError, AttributeError) as err:
            logger.debug("format_subfolders flag restore failed: %s: %s", type(err).__name__, err)


__all__ = ["SNAPSHOT_PREFIX", "capture_process_flag_snapshot", "restore_process_flags"]
