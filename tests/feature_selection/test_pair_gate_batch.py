"""The array pass over the pair gates returns, for every pair, the tuple of the scalar ``_resolve_pair_prevalence_gate``."""

import types

import numpy as np
import pytest

from mlframe.feature_selection.filters._mrmr_fe_step._pair_gate_batch import resolve_pair_gates_batch
from mlframe.feature_selection.filters._mrmr_fe_step._step_pairs_prevalence import _resolve_pair_prevalence_gate


class _Gate:
    """A stand-in for the family-wise partner test: a fixed verdict from the pair."""

    def clears(self, weak, strong):
        """True for an even index sum."""
        return (weak + strong) % 2 == 0


def _case(seed, n_cols=40, n_pairs=600, with_synergy=True, with_mm=True, floor=0.02, debias=False, relax_cover=0.5):
    """Random pair MIs, operand MIs, synergy set, MM bias and relaxation verdicts."""
    rng = np.random.default_rng(seed)
    singles = rng.random(n_cols) * 0.05
    singles[rng.random(n_cols) < 0.2] = 0.0
    pairs, pmis = [], []
    for _ in range(n_pairs):
        a, b = sorted(rng.choice(n_cols, 2, replace=False).tolist())
        pairs.append((a, b))
        pmis.append(float(rng.random() * 0.12))
    pairs = list(dict.fromkeys(pairs))
    pmis = pmis[: len(pairs)]
    keep = [i for i, (a, b) in enumerate(pairs) if singles[a] + singles[b] > 0]
    pairs = [pairs[i] for i in keep]
    pmis = [pmis[i] for i in keep]
    syn = set(rng.choice(n_cols, 8, replace=False).tolist()) if with_synergy else set()
    mm = {p: float(rng.random() * 0.01) for p in pairs[::3]} if with_mm else {}
    relax = {p: bool(rng.random() < 0.5) for p in pairs if (p[0] in syn) != (p[1] in syn) and rng.random() < relax_cover}
    data = rng.integers(0, 5, size=(500, n_cols)).astype(np.int8)
    classes_y = rng.integers(0, 2, size=500).astype(np.int64)
    kw = dict(
        self=types.SimpleNamespace(random_seed=0), data=data, classes_y=classes_y, _pair_resident=False, _pair_mm_bias=mm, _pair_maxt_floor=floor, _synergy_added_idx=syn,
        fe_min_pair_mi_prevalence=1.05, _synergy_prev_resolved=1.5, _prevalence_debias_auto=debias, conditional_gate=_Gate(), relax_verdicts=relax,
    )
    ies = [float(singles[a] + singles[b]) for a, b in pairs]
    s0 = [float(singles[a]) for a, b in pairs]
    s1 = [float(singles[b]) for a, b in pairs]
    return pairs, pmis, ies, (s0, s1), kw


def _reference(pairs, pmis, ies, singles, kw):
    """The scalar result of every pair."""
    cache: dict = {}
    return [_resolve_pair_prevalence_gate(p, pmis[i], ies[i], _col_codes_cache=cache, single_mis=(singles[0][i], singles[1][i]), **kw) for i, p in enumerate(pairs)]


@pytest.mark.parametrize("kwargs", [dict(), dict(floor=0.0), dict(debias=True), dict(with_synergy=False), dict(with_mm=False), dict(relax_cover=1.0), dict(seed=7, floor=0.05)])
def test_batch_equals_scalar(kwargs):
    """Every field of every pair's tuple equals the scalar function's, across synergy / MM / floor / debias settings."""
    seed = kwargs.pop("seed", 0)
    pairs, pmis, ies, singles, kw = _case(seed, **kwargs)
    ref = _reference(pairs, pmis, ies, singles, kw)
    got = resolve_pair_gates_batch(pairs, pmis, ies, singles, scalar_resolve=_resolve_pair_prevalence_gate, col_codes_cache={}, **kw)
    assert len(pairs) == len(ref) == len(got) > 100
    for p, r, g in zip(pairs, ref, got):
        assert (bool(r[0]), bool(r[1]), bool(r[2])) == (bool(g[0]), bool(g[1]), bool(g[2])), p
        assert float(r[3]) == float(g[3]) and float(r[4]) == float(g[4]), p


def test_non_float64_goes_through_the_scalar_code():
    """float32 operand MIs are not evaluated in array form (the arithmetic would round differently): the result is the scalar one."""
    pairs, pmis, ies, singles, kw = _case(3)
    s32 = (list(np.asarray(singles[0], dtype=np.float32)), list(np.asarray(singles[1], dtype=np.float32)))
    ref = _reference(pairs, pmis, ies, s32, kw)
    got = resolve_pair_gates_batch(pairs, pmis, ies, s32, scalar_resolve=_resolve_pair_prevalence_gate, col_codes_cache={}, **kw)
    assert [tuple(map(float, r[3:])) for r in ref] == [tuple(map(float, g[3:])) for g in got]
