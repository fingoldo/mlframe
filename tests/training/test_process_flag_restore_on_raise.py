"""A suite that RAISES must not leave its process-wide overrides flipped on the thread.

``setup_configuration`` flips three thread-wide flags (residual-audit reporting, inline plot display, per-format
plot subfolders) so code far below the suite sees its configuration. ``finalize_suite`` restores them -- but only on
the happy path, so a suite that raised left the flag governing every later caller on that thread. In a test process
that means unrelated later tests: the per-format subfolder flag leaking this way sent three chart files into a
``png/`` subdirectory in tests that had asked for neither.
"""

from __future__ import annotations

import pytest

from mlframe.reporting.renderers.save import get_format_subfolders, set_format_subfolders
from mlframe.training.core._process_flag_scope import restore_process_flags


@pytest.fixture(autouse=True)
def _no_leak_from_this_file():
    """This file flips the flag on purpose; leave the thread exactly as it was found."""
    prior = get_format_subfolders()
    yield
    set_format_subfolders(prior)


def test_restore_is_idempotent():
    """Both the finalize step and the suite's finally call it; whichever runs second must be a no-op, not a re-flip."""
    set_format_subfolders(None)
    artifacts = {"_process_flag_prior_format_subfolders": True}
    restore_process_flags(artifacts)
    assert get_format_subfolders() is True
    set_format_subfolders(False)
    restore_process_flags(artifacts)  # keys were popped by the first call
    assert get_format_subfolders() is False


def test_restore_handles_no_snapshot():
    """A suite whose config flipped nothing snapshots nothing; the restore must not invent a value."""
    set_format_subfolders(False)
    restore_process_flags({})
    restore_process_flags(None)
    assert get_format_subfolders() is False


def test_suite_finally_restores_when_the_body_raises():
    """The real defect: the flag survived a raising suite. Exercised through the same try/finally the suite uses."""
    set_format_subfolders(None)
    artifacts = {"_process_flag_prior_format_subfolders": None}
    set_format_subfolders(True)  # what setup_configuration does for a suite whose config asks for subfolders
    with pytest.raises(RuntimeError):
        try:
            raise RuntimeError("suite blew up before finalize")
        finally:
            restore_process_flags(artifacts)
    assert get_format_subfolders() is None, "the raising suite left the thread-wide override flipped"


class TestSnapshotSurvivesArtifactReplacement:
    """The defect that made the earlier fix insufficient: a phase REPLACES ``ctx.artifacts`` mid-run.

    ``setup_configuration`` stashes the prior flag values there, but ``_phase_helpers`` later assigns
    ``ctx.artifacts = artifacts`` from its own local dict, discarding them -- so ``finalize_suite`` found nothing to
    restore and a clean suite run still left the flags flipped. Traced by printing what the restore received:
    ``restore called, keys=[]`` right after a setup that reported ``prior=None set=True``.
    """

    class _Ctx:
        """Minimal stand-in for the training context: one mutable ``artifacts`` slot."""

        def __init__(self, artifacts):
            self.artifacts = artifacts

    def test_snapshot_is_copied_out_of_the_bag(self):
        """Capture takes a copy, so replacing the dict afterwards cannot lose it."""
        from mlframe.training.core._process_flag_scope import capture_process_flag_snapshot

        ctx = self._Ctx({"_process_flag_prior_format_subfolders": None, "unrelated": 1})
        snap = capture_process_flag_snapshot(ctx)
        ctx.artifacts = {"rebuilt": True}  # what _phase_helpers does
        assert snap == {"_process_flag_prior_format_subfolders": None}

    def test_restore_still_fires_after_the_bag_is_replaced(self):
        """End to end over the boundary: the flag comes back even though the snapshot's original home is gone."""
        from mlframe.training.core._process_flag_scope import capture_process_flag_snapshot, restore_process_flags

        set_format_subfolders(None)
        ctx = self._Ctx({})
        prior = get_format_subfolders()
        ctx.artifacts["_process_flag_prior_format_subfolders"] = prior
        snapshot = capture_process_flag_snapshot(ctx)
        set_format_subfolders(True)  # what setup_configuration does for a suite that asks for subfolders

        ctx.artifacts = {"rebuilt_by_a_later_phase": True}
        restore_process_flags(getattr(ctx, "artifacts", None))  # finalize's path: finds nothing
        assert get_format_subfolders() is True, "precondition: the old path could not restore"
        restore_process_flags(snapshot)
        assert get_format_subfolders() is None, "the boundary snapshot must restore what the bag lost"

    def test_capture_tolerates_a_context_without_artifacts(self):
        """A caller that never reached setup_configuration must not blow up at the boundary."""
        from mlframe.training.core._process_flag_scope import capture_process_flag_snapshot

        assert capture_process_flag_snapshot(self._Ctx(None)) == {}
        assert capture_process_flag_snapshot(object()) == {}
