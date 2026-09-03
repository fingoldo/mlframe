"""The phase table was readable in two wrong ways at once, and a production log showed both.

- the wall-share column divided each phase by the LARGEST phase, so the biggest one was 100% by construction
  and the column summed past 250%. It now divides by the suite clock.
- nothing said the phases NEST, so adding process_model + compute_split_metrics + model.fit gives more than
  the run took, which reads as broken timings rather than as a parent containing its children.
"""

from __future__ import annotations

import time

from mlframe.training.phases import format_phase_summary, phase, registry_elapsed, reset_phase_registry


class TestTheNestingNote:
    """A reader must not be invited to add the column up."""

    def test_summary_says_the_column_does_not_sum(self):
        """The footnote is the whole fix for a table that invites being added up."""
        reset_phase_registry()
        with phase("outer"):
            with phase("inner"):
                time.sleep(0.01)
        text = format_phase_summary()
        assert "nest" in text
        assert "does not sum" in text

    def test_the_nested_child_is_really_double_counted(self):
        """The note is only worth printing because the arithmetic really is what it warns about."""
        reset_phase_registry()
        with phase("outer"):
            with phase("inner"):
                time.sleep(0.05)
        text = format_phase_summary()
        outer = float(next(l for l in text.splitlines() if l.startswith("outer")).split()[1].rstrip("s"))
        inner = float(next(l for l in text.splitlines() if l.startswith("inner")).split()[1].rstrip("s"))
        assert outer >= inner > 0

    def test_no_note_when_there_is_nothing_to_report(self):
        """An empty registry returns the placeholder, not a table with a footnote under nothing."""
        reset_phase_registry()
        assert "nest" not in format_phase_summary()


class TestTheSuiteClock:
    """Wall-share needs a denominator that is the run, not the biggest phase in it."""

    def test_elapsed_is_measured_from_the_reset(self):
        """The suite clock is the wall-share denominator, so it starts when the suite does."""
        reset_phase_registry()
        time.sleep(0.02)
        assert registry_elapsed() >= 0.02

    def test_elapsed_covers_time_spent_outside_any_phase(self):
        """Setup and teardown between phases belong to the run, which is exactly why the largest phase is wrong."""
        reset_phase_registry()
        time.sleep(0.03)
        with phase("brief"):
            pass
        assert registry_elapsed() > 0.03

    def test_a_later_reset_restarts_the_clock(self):
        """Each suite gets its own denominator; a leftover clock would understate every share."""
        reset_phase_registry()
        time.sleep(0.03)
        reset_phase_registry()
        assert registry_elapsed() < 0.03
