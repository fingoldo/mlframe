"""The fused optimal_joint honest floor equals the per-permutation floor inside a real fit (same permutations, same statistic)."""

from __future__ import annotations


import numpy as np
import pytest

from mlframe.feature_selection._benchmarks.profiling import family_scale as Fm


@pytest.mark.parametrize("strategy", ["optimal_joint"])
def test_fused_floor_equals_per_permutation(monkeypatch, strategy):
    """Every floor call of a small fit: the fused floor equals the pooled per-permutation floor, and the fused path answered."""
    import sys

    import mlframe.feature_selection.filters._honest_mdlp_null as H
    import mlframe.feature_selection.filters._honest_oj_fused as FU
    from mlframe.feature_selection.filters.mrmr import MRMR

    orig = H.honest_fdr_floor
    seen = []

    def wrapped(hook, fd, fnb, yidx, pool, nperm, q, seed, eval_rows=None):
        """Both floors for the same call."""
        # the fused floor first, as production calls it (before any permutation has built the plans), then the per-permutation floor
        monkeypatch.setenv("MLFRAME_HONEST_OJ_FUSED", "1")
        rng = np.random.default_rng(0 if seed is None else int(seed))
        direct = FU.oj_fused_floor(hook, fd, fnb, yidx, [int(c) for c in pool], int(nperm), float(q), int(rng.integers(0, 2**31 - 1)), eval_rows)
        monkeypatch.setenv("MLFRAME_HONEST_OJ_FUSED", "0")
        want = orig(hook, fd, fnb, yidx, pool, nperm, q, seed, eval_rows)
        monkeypatch.setenv("MLFRAME_HONEST_OJ_FUSED", "1")
        seen.append((want, direct))
        return want

    for m in list(sys.modules.values()):
        if m is not None and getattr(m, "honest_fdr_floor", None) is orig:
            monkeypatch.setattr(m, "honest_fdr_floor", wrapped)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    kw = {f: False for f in Fm.all_families_kwargs()}
    kw["nbins_strategy"] = strategy
    MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*Fm._make(2500, 8, 0))
    assert seen, "the honest floor was not reached"
    for want, direct in seen:
        assert direct is not None and direct == want
