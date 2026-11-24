"""Cluster 28 regression: all_active_phases reports the current thread's open phase, and the phase stack unwinds cleanly (audit 2026-11-20, finding 28-F74).

The function had no test. Entering a phase in this thread shows it as "outer > inner" under the thread's name; after the block exits the
thread is no longer listed. The pop after a phase is written as an explicit if block.
"""

import threading

from mlframe.training import phases as ph


def test_active_phase_is_reported_inside_the_block():
    """A phase entered here appears under this thread's name (28-F74)."""
    with ph.phase("outer_phase_28f74"):
        active = ph.all_active_phases()
        assert any("outer_phase_28f74" in v for v in active.values())
    # After the block the phase is no longer active for this thread.
    assert not any("outer_phase_28f74" in v for v in ph.all_active_phases().values())


def test_nested_phases_join_with_arrows():
    """Nested phases are reported as 'outer > inner' (28-F74)."""
    with ph.phase("outer_28f74"):
        with ph.phase("inner_28f74"):
            values = list(ph.all_active_phases().values())
            assert any("outer_28f74 > inner_28f74" in v for v in values), values
    assert threading.current_thread().name  # the thread is still well-formed
