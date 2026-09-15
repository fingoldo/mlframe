"""FE-step helpers: dense class codes keep distinct labels and a 1-D shape; the pair-cache scan runs only when it can change the decision."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._mrmr_fe_step._step_class_codes import dense_class_codes
from mlframe.feature_selection.filters._mrmr_fe_step._step_pairmi import _use_serial_pair_sweep


def test_fractional_labels_stay_distinct():
    """An integer cast before np.unique merged 0.2 and 0.7 into one class on the escalation path."""
    codes = dense_class_codes(np.array([0.2, 0.7, 1.4, 0.2]))
    assert codes.tolist() == [0, 1, 2, 0]


def test_column_vector_target_gives_one_code_per_row():
    """An (n, 1) target must give (n,) codes, as the CMI gate's sibling path does."""
    codes = dense_class_codes(np.array([[3], [5], [3]]))
    assert codes.shape == (3,)
    assert codes.tolist() == [0, 1, 0]


def _counting_scan(result: bool):
    """A pair-cache scan stand-in that records how often it runs."""
    calls = {"n": 0}

    def scan():
        """Record the call and return the canned answer."""
        calls["n"] += 1
        return result

    return scan, calls


def test_cache_scan_skipped_when_serial_is_already_decided():
    """With one worker or a sub-floor permutation budget the O(k^2) scan cannot change the outcome, so it must not run."""
    for n_jobs, below_floor in ((1, False), (8, True)):
        scan, calls = _counting_scan(True)
        run_serial, _ = _use_serial_pair_sweep(n_jobs, 5000, below_floor, scan)
        assert run_serial
        assert calls["n"] == 0, f"pair-cache scan ran although serial was already chosen (n_jobs={n_jobs}, below_floor={below_floor})"


def test_cache_scan_decides_when_the_pool_is_otherwise_eligible():
    """Control: with a worker pool in play the scan runs once, and a fully cached pair set still selects serial."""
    scan, calls = _counting_scan(True)
    assert _use_serial_pair_sweep(8, 5000, False, scan) == (True, True)
    assert calls["n"] == 1
    scan, calls = _counting_scan(False)
    assert _use_serial_pair_sweep(8, 5000, False, scan) == (False, False)
