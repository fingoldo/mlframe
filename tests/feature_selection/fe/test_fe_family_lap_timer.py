"""FamilyLapTimer charges stage wall to the family whose gate opened last, from the gate alone."""
import time

from mlframe.feature_selection.filters import _fe_family_timing as ft


def test_laps_do_not_overlap_and_closed_gates_accrue_nothing():
    """Two open laps are recorded separately; the closed gate between them adds no bucket."""
    ft.reset_fe_family_wall()
    lap = ft.FamilyLapTimer()
    assert lap.gate("fe_a_enable", True) is True
    time.sleep(0.03)
    assert lap.gate("fe_b_enable", False) is False
    time.sleep(0.03)  # no family open: charged to nobody
    assert lap.gate("fe_c_enable", True) is True
    time.sleep(0.03)
    lap.close()
    lap.close()  # idempotent
    wall = ft.get_fe_family_wall()
    assert set(wall) == {"gate:fe_a_enable", "gate:fe_c_enable"}
    assert 0.03 <= wall["gate:fe_a_enable"][0] < 0.058
    assert wall["gate:fe_c_enable"][1] == 1
    ft.reset_fe_family_wall()
