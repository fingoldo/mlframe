"""Rows no test fold covers are NaN in every trial of a time-ordered pool; selection must use the covered rows instead of rejecting every trial."""

from types import SimpleNamespace

import numpy as np

from mlframe.training.composite.hpo_ensembling import select_oof_pool_ensemble


def test_pool_with_a_nan_warmup_block_is_still_selectable():
    """The first 100 rows are NaN in all trials; the selection runs on the other 200 and the blend keeps NaN only on the warm-up rows."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    pool = []
    for noise in (0.1, 0.5, 1.0):
        oof = y + rng.normal(scale=noise, size=300)
        oof[:100] = np.nan
        pool.append(oof)
    out = select_oof_pool_ensemble(SimpleNamespace(trial_oof_pool=pool), y)
    assert 0 in out.kept_trial_indices
    assert np.isnan(out.combined_oof[:100]).all() and np.isfinite(out.combined_oof[100:]).all()


def test_trial_with_a_nan_inside_the_covered_rows_is_still_dropped():
    """A trial whose fold failed on covered rows is excluded (negative control)."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=200)
    good = y + rng.normal(scale=0.1, size=200)
    broken = y + rng.normal(scale=0.1, size=200)
    broken[150] = np.nan
    out = select_oof_pool_ensemble(SimpleNamespace(trial_oof_pool=[good, broken]), y)
    assert out.kept_trial_indices == [0]
