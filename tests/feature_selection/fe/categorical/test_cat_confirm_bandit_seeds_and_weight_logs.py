"""Cat-FE confirmation: bandit shuffles never repeat a permutation, and ignored sample weights / missing cupy are logged."""

from __future__ import annotations

import logging
import sys

import numpy as np

from mlframe.feature_selection.filters import _cat_confirm_bandit as B
from mlframe.feature_selection.filters import _cat_confirm_permutation as P
from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig
from mlframe.feature_selection.filters.info_theory._class_encoding import merge_vars

_C = 0x9E3779B1
_MASK = (1 << 64) - 1


def _setup():
    """Small 4-column categorical frame and the target merge."""
    rng = np.random.default_rng(3)
    n = 400
    nbins = np.array([3, 4, 4, 4], dtype=np.int64)
    fd = np.column_stack([rng.integers(0, int(b), n) for b in nbins]).astype(np.int32)
    cy, fy, _ = merge_vars(factors_data=fd, vars_indices=np.array([0], dtype=np.int64), var_is_nominal=None, factors_nbins=nbins, dtype=np.int32)
    return fd, nbins, cy, fy


def test_bandit_bulk_shuffle_streams_never_overlap(monkeypatch):
    """The kernel starts shuffle p at ``seed + p * 0x9E3779B1``; callers stepping seeds by the same constant re-ran identical permutations across pairs and bursts."""
    fd, nbins, cy, fy = _setup()
    seen: list = []
    calls = [0]

    def fake_bulk(cp, fp, c1, f1, c2, f2, y, fq, n_perms, base_seed, dtype):
        """Record the start states of the shuffles and answer with alternating outcomes so the pair stays ambiguous."""
        seen.extend(((int(base_seed) + p * _C) & _MASK) for p in range(int(n_perms)))
        out = np.zeros(int(n_perms))
        out[::2] = 1.0
        calls[0] += 1
        return out, np.zeros(int(n_perms)), np.zeros(int(n_perms))

    monkeypatch.setattr(B, "_bulk_shuffle_and_compute_three_mis", fake_bulk)
    cfg = CatFEConfig(full_npermutations=40, fwer_correction="none")
    pairs_a = np.array([1, 1], dtype=np.int64)
    pairs_b = np.array([2, 3], dtype=np.int64)
    B._confirm_pairs_bandit_ucb1(fd, pairs_a, pairs_b, np.array([0, 1]), np.array([0.5, 0.5]), nbins, cy, fy, cfg, 2, np.int32, 0)
    assert calls[0] > 3
    assert len(seen) == len(set(seen))


def test_bandit_ignored_weights_warn_even_when_not_verbose(caplog):
    """Weights are dropped by the bandit allocator; that is warned about at verbose=0 too."""
    fd, nbins, cy, fy = _setup()
    cfg = CatFEConfig(full_npermutations=10, fwer_correction="none")
    with caplog.at_level(logging.WARNING, logger=B.logger.name):
        B._confirm_pairs_bandit_ucb1(fd, np.array([1]), np.array([2]), np.array([0]), np.array([0.1]), nbins, cy, fy, cfg, 1, np.int32, 0, weights=np.ones(400))
    assert any("weights ignored" in r.getMessage() for r in caplog.records)


def test_conditional_null_ignored_weights_warn_even_when_not_verbose(caplog):
    """The conditional permutation null drops weights; warned at verbose=0 too."""
    fd, nbins, cy, fy = _setup()
    cfg = CatFEConfig(full_npermutations=2, permutation_null="conditional", fwer_correction="none")
    with caplog.at_level(logging.WARNING, logger=P.logger.name):
        P._confirm_pairs_via_permutation(fd, np.array([1]), np.array([2]), np.array([0]), np.array([0.1]), nbins, cy, fy, cfg, 1, np.int32, 0, weights=np.ones(400))
    assert any("weights ignored" in r.getMessage() for r in caplog.records)


def test_forced_gpu_without_cupy_is_logged(monkeypatch, caplog):
    """backend='gpu' with cupy missing falls to the CPU kernel and says so."""
    monkeypatch.setitem(sys.modules, "cupy", None)
    import mlframe.feature_selection.filters._gpu_policy as gp

    monkeypatch.setattr(gp, "gpu_globally_disabled", lambda: False)
    with caplog.at_level(logging.WARNING, logger=P.logger.name):
        assert P._perm_kernel_dispatch_use_gpu(1000, 10, "gpu") is False
    assert any("cupy is not importable" in r.getMessage() for r in caplog.records)
