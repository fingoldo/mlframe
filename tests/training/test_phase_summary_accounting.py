"""The phase table was readable in two wrong ways at once, and a production log showed both.

- the wall-share column divided each phase by the LARGEST phase, so the biggest one was 100% by construction
  and the column summed past 250%. It now divides by the suite clock.
- nothing said the phases NEST, so adding process_model + compute_split_metrics + model.fit gives more than
  the run took, which reads as broken timings rather than as a parent containing its children.
"""

from __future__ import annotations

import pytest

from mlframe.training import phases
from mlframe.training.phases import format_phase_summary, phase, registry_elapsed, reset_phase_registry


class _Clock:
    """A hand-advanced stand-in for ``time.perf_counter``."""

    def __init__(self):
        """Init."""
        self.now = 0.0

    def __call__(self):
        """Call."""
        return self.now

    def advance(self, seconds):
        """Advance."""
        self.now += seconds


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    """The phase registry reads this clock, so a test advances time instead of waiting for it."""
    c = _Clock()
    monkeypatch.setattr(phases, "_timer", c)
    return c


class TestTheNestingNote:
    """A reader must not be invited to add the column up."""

    def test_summary_says_the_column_does_not_sum(self, clock):
        """The footnote is the whole fix for a table that invites being added up."""
        reset_phase_registry()
        with phase("outer"):
            with phase("inner"):
                clock.advance(0.01)
        text = format_phase_summary()
        assert "nest" in text
        assert "does not sum" in text

    def test_the_nested_child_is_really_double_counted(self, clock):
        """The note is only worth printing because the arithmetic really is what it warns about."""
        reset_phase_registry()
        with phase("outer"):
            with phase("inner"):
                clock.advance(0.05)
        text = format_phase_summary()
        outer = float(next(l for l in text.splitlines() if l.startswith("outer")).split()[1].rstrip("s"))
        inner = float(next(l for l in text.splitlines() if l.startswith("inner")).split()[1].rstrip("s"))
        assert outer >= inner > 0

    def test_no_note_when_there_is_nothing_to_report(self, clock):
        """An empty registry returns the placeholder, not a table with a footnote under nothing."""
        reset_phase_registry()
        assert "nest" not in format_phase_summary()


class TestTheSuiteClock:
    """Wall-share needs a denominator that is the run, not the biggest phase in it."""

    def test_elapsed_is_measured_from_the_reset(self, clock):
        """The suite clock is the wall-share denominator, so it starts when the suite does."""
        reset_phase_registry()
        clock.advance(0.02)
        assert registry_elapsed() >= 0.02

    def test_elapsed_covers_time_spent_outside_any_phase(self, clock):
        """Setup and teardown between phases belong to the run, which is exactly why the largest phase is wrong."""
        reset_phase_registry()
        clock.advance(0.03)
        with phase("brief"):
            clock.advance(0.001)
        assert registry_elapsed() > 0.03

    def test_a_later_reset_restarts_the_clock(self, clock):
        """Each suite gets its own denominator; a leftover clock would understate every share.

        Stated as an ordering rather than as `< 0.03`: the restart property is that the clock goes
        backwards across the reset, and asserting instead that under 30ms passes between two adjacent
        statements is a claim about the scheduler. Windows' default timer granularity alone is ~15.6ms,
        and one preemption on a contended box clears 30ms.
        """
        reset_phase_registry()
        clock.advance(0.03)
        before = registry_elapsed()
        reset_phase_registry()
        after = registry_elapsed()
        assert after < before, f"the clock did not restart: {after:.4f}s after the reset against {before:.4f}s before it"
        assert before >= 0.03, f"the pre-reset clock read {before:.4f}s after a 30ms sleep, so it is not running at all"
