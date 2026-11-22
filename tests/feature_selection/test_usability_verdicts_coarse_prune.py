"""The coarse prune of the usability-admission verdicts leaves every verdict as the full pass gives it, and scores only a few pairs when the pool has a clear leader."""

import types

import numpy as np
import pytest

from mlframe.feature_selection.filters._mrmr_fe_step import _step_pairs_prevalence as sp
from mlframe.feature_selection.filters._mrmr_fe_step import _step_pairs_rank as sr

P = 210
N = 30000


def _case(kind, seed):
    """Operands and a continuous target: a ratio carried by the outlier rows of the denominator (tail concentrated), a clean product, or noise."""
    rng = np.random.default_rng(seed)
    ops = rng.standard_normal((P, N)).astype(np.float32)
    a, b = 3, 7
    noise = rng.standard_normal(N)
    if kind == "tail":
        out = rng.choice(N, N // 20, replace=False)
        ops[b, out] = ops[b, out] * 0.03 + 0.02  # small divisors on 5 % of the rows
        y = ops[a].astype(np.float64) ** 2 / (np.abs(ops[b]) + 0.02) + 0.2 * noise
    elif kind == "clean":
        y = ops[a].astype(np.float64) * ops[b] + 0.3 * noise
    else:
        y = noise
    pairs = [(i, j) for i in range(P) for j in range(i + 1, P)]
    return pairs, y, ops


def _verdicts(pairs, y, ops, coarse):
    """The verdict dict of the batched function, with the coarse prune on or switched off."""
    est = types.SimpleNamespace()
    orig = sp.coarse_pair_form_keep
    if not coarse:
        sp.coarse_pair_form_keep = lambda *a, **k: None
    try:
        return sr._batch_usability_admission_verdicts(est, need_usability=pairs, y_continuous=y, cached_operand=lambda i: ops[i], cached_single_corr=lambda i: 0.0)
    finally:
        sp.coarse_pair_form_keep = orig


@pytest.mark.parametrize("kind", ["tail", "clean", "noise"])
@pytest.mark.parametrize("seed", [0, 1])
def test_verdicts_equal_the_full_pass(kind, seed):
    """Every pair has the verdict of the pass without the prune."""
    pairs, y, ops = _case(kind, seed)
    assert _verdicts(pairs, y, ops, True) == _verdicts(pairs, y, ops, False)


def test_the_prune_scores_few_pairs_when_a_pair_leads(monkeypatch):
    """With a clean product pair the kept set is a tiny share of the pool; the planted pair keeps its verdict."""
    pairs, y, ops = _case("clean", 0)
    sizes = []
    real = sp.coarse_pair_form_keep

    def spy(y_b, om, pa, pb, mc):
        """Record how many pairs stay."""
        keep = real(y_b, om, pa, pb, mc)
        sizes.append(None if keep is None else int(keep.sum()))
        return keep

    monkeypatch.setattr(sp, "coarse_pair_form_keep", spy)
    sr._batch_usability_admission_verdicts(types.SimpleNamespace(), need_usability=pairs, y_continuous=y, cached_operand=lambda i: ops[i], cached_single_corr=lambda i: 0.0)
    assert sizes and sizes[0] is not None and sizes[0] < 0.02 * len(pairs)


def test_a_tail_carried_pair_survives_the_prune_even_without_the_noise_check(monkeypatch):
    """The risk of the prune is a pair whose |corr| sits in the tail rows: with the noise check switched off, the pair of a heavy-tailed ratio is still among the kept pairs."""
    rng = np.random.default_rng(0)
    ops = rng.standard_normal((P, N)).astype(np.float32)
    ops[7] = (rng.pareto(1.5, N) + 0.2).astype(np.float32)
    y = ops[3].astype(np.float64) ** 2 / ops[7] + rng.normal(scale=0.1, size=N)
    pairs = [(i, j) for i in range(P) for j in range(i + 1, P)]
    pa = np.array([p[0] for p in pairs], dtype=np.int64)
    pb = np.array([p[1] for p in pairs], dtype=np.int64)
    monkeypatch.setattr(sp, "_COARSE_NOISE_MAX", 99.0)
    keep = sp.coarse_pair_form_keep(y.astype(np.float32), ops, pa, pb, 0.5)
    assert keep is not None and bool(keep[pairs.index((3, 7))]) and int(keep.sum()) < 0.05 * len(pairs)
