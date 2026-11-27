"""The pair family gate settles only the configurations the emission can reach (``decide`` / ``window_ratio``): their flags are the full test's, and the permutation loop ends sooner."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _family_maxt_gate as gate
from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_emit import _apply_pair_family_gate

N = 4000
BINS = 6


def _family(seed: int, graded: int = 12, noise: int = 400, top: float = 0.30, bottom: float = 0.05):
    """``(columns, y, observed rank)``: ``graded`` columns whose dependence on the class falls from ``top`` to ``bottom`` (a share of rows that copy the class), then ``noise`` independent columns."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, N).astype(np.int64)
    cols = []
    for strength in np.linspace(top, bottom, graded):
        copy = rng.random(N) < strength
        cols.append(np.where(copy, y * 3 + rng.integers(0, 3, N) // 3, rng.integers(0, BINS, N)).astype(np.int8).reshape(-1, 1))
    cols += [rng.integers(0, BINS, N).astype(np.int8).reshape(-1, 1) for _ in range(noise)]
    return cols, y


def _count_permutations(monkeypatch) -> "list[int]":
    """Wrap the host null kernel and collect how many permutations each call draws."""
    drawn: "list[int]" = []
    original = gate.family_null_max

    def counting(cols, n_x, y, n_y, m, rng, **kw):
        """The wrapped kernel, counted."""
        drawn.append(int(m))
        return original(cols, n_x, y, n_y, m, rng, **kw)

    monkeypatch.setattr(gate, "family_null_max", counting)
    return drawn


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_flags_inside_the_window_are_the_full_tests_and_the_loop_ends_sooner(seed, monkeypatch):
    """With ``decide`` on the leading configurations, their flags equal the full test's and fewer permutations are drawn; a batch outside the window is never reported failing unless the triage certified it."""
    cols, y = _family(seed)
    drawn = _count_permutations(monkeypatch)
    full = gate.family_pass_batches(cols, y, alpha=0.01, n_perm=1999, seed=seed)
    drawn_full = sum(drawn)
    drawn.clear()
    window = np.zeros(len(cols), dtype=bool)
    window[:4] = True  # the leaders: the strongest columns
    decided = gate.family_pass_batches(cols, y, alpha=0.01, n_perm=1999, seed=seed, decide=window)
    assert [decided[i] for i in range(4)] == [full[i] for i in range(4)]
    assert all(decided[i] or not full[i] for i in range(len(cols)) if not window[i]), "a batch outside the window turned from passing to failing"
    assert sum(drawn) <= drawn_full


def test_the_window_saves_permutations_when_only_far_columns_sit_near_the_threshold(monkeypatch):
    """Strong leaders with weaker columns near the threshold below them: the full test runs on for the weak ones, the windowed test stops once the leaders are certain."""
    cols, y = _family(7, graded=14, top=0.45, bottom=0.04)
    drawn = _count_permutations(monkeypatch)
    full = gate.family_pass_batches(cols, y, alpha=0.01, n_perm=1999, seed=7)
    drawn_full = sum(drawn)
    drawn.clear()
    window = np.zeros(len(cols), dtype=bool)
    window[:2] = True
    decided = gate.family_pass_batches(cols, y, alpha=0.01, n_perm=1999, seed=7, decide=window)
    assert decided[:2] == full[:2]
    assert drawn_full > 0, "the family has no batch the analytic triage leaves to the permutation test: the comparison would be vacuous"
    assert sum(drawn) < drawn_full


def test_the_pair_gate_zeroes_only_inside_the_window_and_keeps_the_leaders_verdict():
    """``_apply_pair_family_gate`` with a window ratio gives the leaders the verdicts of the full gate and leaves a configuration below the window unchanged."""
    cols, y = _family(11)
    perf = {("cfg", i): float(0.5 - 0.001 * i) for i in range(len(cols))}
    batches = [(("cfg", i), c) for i, c in enumerate(cols)]
    full_perf = dict(perf)
    _apply_pair_family_gate(batches, full_perf, y, BINS, alpha=0.01)
    window_perf = dict(perf)
    _apply_pair_family_gate(batches, window_perf, y, BINS, alpha=0.01, window_ratio=0.98)
    leaders = [cfg for cfg, v in perf.items() if v >= 0.98 * max(perf.values())]
    assert leaders and all(window_perf[cfg] == full_perf[cfg] for cfg in leaders)
    below = [cfg for cfg in perf if cfg not in leaders]
    assert below and all(window_perf[cfg] in (perf[cfg], 0.0) for cfg in below)
    assert below and all(window_perf[cfg] == perf[cfg] for cfg in below if full_perf[cfg] == perf[cfg]), "a passing configuration below the window was zeroed"
