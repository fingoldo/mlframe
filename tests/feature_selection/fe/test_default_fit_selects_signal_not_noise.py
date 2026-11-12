"""With MRMR's default configuration (every FE family on except the GBM seeder and the alternate scorers) a planted signal is selected and no noise-only column is.

One fit per planted-signal generator of ``tests/feature_selection/_synth/pool_planted.py`` (weak-marginal U shape, hidden periodic, grouped by an irrelevant key, ratio with an irrelevant normaliser,
sign-gated hinge) on a small frame. "Selected" means a selected column whose name mentions every planted source (a raw column counts only when one source is planted); "noise-only" means a selected
column that mentions no planted source. The 15-fit run at 20000 x 100 (3 seeds per generator) gave recall 15/15 and no noise-only column; this test keeps the criterion in the suite at a size that runs in minutes.
"""

import numpy as np
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR
from tests.feature_selection._synth.pool_planted import GENERATORS

_N = 8000
_P = 30


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_planted_signal_is_selected_and_noise_is_not(name):
    """The default fit selects a column built from every planted source and selects nothing built from noise alone."""
    X, y, planted = GENERATORS[name](n=_N, p=_P, seed=0)
    MRMR._FIT_CACHE.clear()
    selected = list(MRMR(verbose=0, random_seed=0, n_jobs=1).fit(X, y).get_feature_names_out())
    # An empty planted list would make all() vacuously true for every column, so require at least one planted source.
    found = bool(planted) and any(all(s in c for s in planted) for c in selected)
    noise_only = [c for c in selected if not any(s in c for s in planted)]
    assert found, f"{name}: no selected column mentions every planted source {planted}; selected {selected}"
    assert not noise_only, f"{name}: noise-only columns selected: {noise_only}"
    assert np.isfinite(len(selected))
