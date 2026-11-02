"""The batched permutation-null admission gives every pair the verdict the per-pair path gives it."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cupy")

from mlframe.feature_selection.filters._mrmr_fe_step._pair_admission_batch import batched_perm_null_admissions
from mlframe.feature_selection.filters._mrmr_fe_step._step_pairs_rank import _pair_perm_null_admission


def _setup(n=24000, p=7, seed=3):
    """Binned columns, two of them carrying a joint signal with y, and every pair marked as failing prevalence but clearing max-T."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 10, (n, p)).astype(np.int64)
    y = np.where(rng.random(n) < 0.8, (data[:, 0] + data[:, 5]) % 2, rng.integers(0, 2, n)).astype(np.int64)  # (0, 5) carries a parity signal
    data[:, 2] = np.clip(data[:, 0] + rng.integers(-1, 2, n), 0, 9)
    mis = {(j,): float(rng.random()) * 0.01 for j in range(p)}
    mis[(0,)] = 0.05  # 0 anchors its pairs, so (0, 5) asks CMI(5; y | 0), which is large
    pairs = [(a, b) for a in range(p) for b in range(a + 1, p)]
    gate = {pk: (False, True, False, 1.05, 0.0) for pk in pairs}
    gate[(0, 1)] = (True, True, False, 1.05, 0.0)  # passes prevalence: not eligible
    gate[(3, 4)] = (False, False, False, 1.05, 0.0)  # misses max-T: not eligible
    return data, y, mis, gate


def test_batched_verdicts_equal_the_per_pair_verdicts():
    """Same admitted set as calling ``_pair_perm_null_admission`` pair by pair (analytic null regime, 24000 rows)."""
    data, y, mis, gate = _setup()
    est = SimpleNamespace(fe_pair_perm_null_admission_enable=True, random_seed=0, fe_pair_perm_null_excess_frac=0.05)
    batched = batched_perm_null_admissions(est, gate, mis, data, y)
    eligible = [pk for pk, st in gate.items() if not st[0] and st[1]]
    assert set(batched) == set(eligible)
    est._pair_perm_admit_cache = None
    for pk in eligible:
        single = _pair_perm_null_admission(est, False, True, pk, mis, data, y, False, 0, False)
        assert batched[pk] == single, pk
    assert any(batched.values()) and not all(batched.values())


def test_disabled_or_small_samples_fall_back_to_the_per_pair_path():
    """Admission off, or fewer rows than the analytic-null threshold: no batched verdicts."""
    data, y, mis, gate = _setup()
    assert batched_perm_null_admissions(SimpleNamespace(fe_pair_perm_null_admission_enable=False), gate, mis, data, y) == {}
    small, ys, mis_s, gate_s = _setup(n=3000)
    assert batched_perm_null_admissions(SimpleNamespace(fe_pair_perm_null_admission_enable=True, random_seed=0), gate_s, mis_s, small, ys) == {}


def test_batched_asymmetric_relaxation_gives_the_gate_the_same_state():
    """For every pair, the prevalence gate returns the same state whether the asymmetric-synergy relaxation was decided by the batched pre-pass or inline."""
    from mlframe.feature_selection.filters._mrmr_fe_step._pair_admission_batch import batched_asymmetric_relax_verdicts
    from mlframe.feature_selection.filters._mrmr_fe_step._step_pairs_rank import _resolve_pair_prevalence_gate

    data, y, mis, _gate = _setup()
    est = SimpleNamespace(random_seed=0)
    synergy = {5, 6}
    kwargs = dict(
        self=est, data=data, classes_y=y, _pair_resident=False, _pair_mm_bias={}, _pair_maxt_floor=0.0, _synergy_added_idx=synergy,
        fe_min_pair_mi_prevalence=1.05, _synergy_prev_resolved=1.5, _prevalence_debias_auto=False,
    )
    pairs = [(a, b) for a in range(7) for b in range(a + 1, 7)]
    relax = batched_asymmetric_relax_verdicts(est, pairs, synergy, data, y)
    asym = [pk for pk in pairs if (pk[0] in synergy) != (pk[1] in synergy)]
    assert set(relax) == set(asym)
    assert any(relax.values()) and not all(relax.values())
    for pk in pairs:
        pair_mi = 0.02
        ies = mis[(pk[0],)] + mis[(pk[1],)]
        inline = _resolve_pair_prevalence_gate(pk, pair_mi, ies, **kwargs)
        batched = _resolve_pair_prevalence_gate(pk, pair_mi, ies, **kwargs, relax_verdicts=relax)
        assert inline == batched, pk
