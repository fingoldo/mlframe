"""FamilyLapTimer charges stage wall to the family whose gate opened last, from the gate alone."""

import pytest

from mlframe.feature_selection.filters import _fe_family_timing as ft


class _Clock:
    """A hand-advanced stand-in for ``time.perf_counter``."""

    def __init__(self):
        """Init."""
        self.now = 100.0

    def __call__(self):
        """Call."""
        return self.now

    def advance(self, seconds):
        """Advance."""
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    """Clock."""
    c = _Clock()
    monkeypatch.setattr(ft, "perf_counter", c)
    return c


def test_laps_do_not_overlap_and_closed_gates_accrue_nothing(clock):
    """Two open laps are recorded separately; the closed gate between them adds no bucket."""
    ft.reset_fe_family_wall()
    lap = ft.FamilyLapTimer()
    assert lap.gate("fe_a_enable", True) is True
    clock.advance(0.03)
    assert lap.gate("fe_b_enable", False) is False
    clock.advance(0.03)  # no family open: charged to nobody
    assert lap.gate("fe_c_enable", True) is True
    clock.advance(0.03)
    lap.close()
    lap.close()  # idempotent
    wall = ft.get_fe_family_wall()
    assert set(wall) == {"gate:fe_a_enable", "gate:fe_c_enable"}
    assert wall["gate:fe_a_enable"][0] == pytest.approx(0.03)
    assert wall["gate:fe_c_enable"][0] == pytest.approx(0.03)
    assert wall["gate:fe_c_enable"][1] == 1
    ft.reset_fe_family_wall()
