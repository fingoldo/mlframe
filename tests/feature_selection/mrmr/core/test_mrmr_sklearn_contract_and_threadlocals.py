"""Regression sensors for the MRMR sklearn-contract + thread-local-restore fixes.

Each test pins a specific bug the pre-fix code exhibited (the pre-fix value is stated in the test body)
and asserts the post-fix contract:

* F1/F3 -- ``__init__`` no longer mutates constructor parameters before ``store_params_in_object``, so
  ``get_params`` round-trips exactly what the user passed (``random_state`` does NOT leak into
  ``random_seed``; a default-constructed estimator no longer re-emits a DeprecationWarning on every clone).
* D5 -- the sample_weight resample uses the deterministic ``int(seed or 0)`` convention, so two fits with
  the default (``None``) seed and the same non-uniform weights draw the SAME resample (pre-fix: OS entropy
  via ``default_rng(None)`` -> different draws).
* D3 -- the fit ``finally`` restores the MI thread-locals to the values held at fit ENTRY (snapshot), not to
  hardcoded literals, so an inner fit cannot clobber an outer fit's toggles.
"""

import numpy as np
import pandas as pd
import pytest
import sklearn

from mlframe.feature_selection.filters import MRMR
import mlframe.feature_selection.filters.info_theory as it
from mlframe.feature_selection.filters.info_theory import (
    use_su_normalization,
    get_bur_lambda,
    use_jmim_aggregator,
    get_relaxmrmr_alpha,
)


@pytest.fixture
def small_xy():
    """Build a small synthetic classification fixture with signal on columns f0/f1."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series((X["f0"] + X["f1"] > 0).astype(int))
    return X, y


def test_f1_random_state_does_not_leak_into_random_seed():
    """Pre-fix: ``MRMR(random_state=42).get_params()['random_seed']`` returned 42 (the ctor promoted
    ``random_state`` into the ``random_seed`` local before storing). Post-fix it stays ``None`` and
    ``random_state`` echoes 42 unchanged."""
    m = MRMR(random_state=42)
    params = m.get_params()
    assert params["random_seed"] is None
    assert params["random_state"] == 42
    # The effective seed used by fit still resolves the fallback alias.
    assert m._effective_random_seed() == 42


def test_f3_random_seed_does_not_leak_into_random_state():
    """The reverse cross-contamination: passing only ``random_seed`` must leave ``random_state`` at None."""
    m = MRMR(random_seed=7)
    params = m.get_params()
    assert params["random_seed"] == 7
    assert params["random_state"] is None


def test_f1_clone_round_trips_get_params():
    """sklearn.clone(m) must produce an estimator whose get_params equals the original's (the sklearn
    estimator contract). Pre-fix the mutated ctor locals broke this for ``random_state``."""
    m = MRMR(random_state=11, quantization_nbins=12, skip_retraining_on_same_content=False)
    assert sklearn.clone(m).get_params() == m.get_params()


def test_d5_default_seed_resample_is_reproducible():
    """Pre-fix: the sample_weight resample used ``np.random.default_rng(self.random_seed)``, so the default
    ``random_seed=None`` pulled OS entropy and two calls drew DIFFERENT resamples (non-reproducible).
    Post-fix it uses ``int(seed or 0)`` (None -> deterministic 0) so the draws are identical."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(300, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series((X["f0"] * 0.8 + X["f2"] * 0.6 + rng.normal(size=300) * 0.3 > 0).astype(int))
    sw = np.where(y.values == 1, 3.0, 1.0).astype(float)  # non-uniform -> resample path is exercised

    m = MRMR()  # default random_seed=None
    Xr1, yr1 = m._maybe_resample_for_sample_weight(X, y, sw)
    Xr2, yr2 = m._maybe_resample_for_sample_weight(X, y, sw)
    assert np.array_equal(Xr1.values, Xr2.values)
    assert np.array_equal(yr1.values, yr2.values)


def test_d3_fit_restores_mi_thread_locals_to_pre_fit_snapshot(small_xy):
    """Pre-fix: the fit ``finally`` reset the MI thread-locals to hardcoded literals (False/0.0), so a fit
    nested inside an outer fit clobbered the outer toggles. Post-fix the ``finally`` restores the values
    held at fit ENTRY, so toggles set BEFORE the fit survive it."""
    X, y = small_xy
    it.set_su_normalization(True)
    it.set_bur_lambda(0.7)
    it.set_jmim_aggregator(True)
    it.set_relaxmrmr_alpha(0.3)
    try:
        m = MRMR(fe_max_steps=0, max_runtime_mins=0.1)
        m.fit(X, y)
        assert use_su_normalization() is True
        assert get_bur_lambda() == 0.7
        assert use_jmim_aggregator() is True
        assert get_relaxmrmr_alpha() == 0.3
    finally:
        it.set_su_normalization(False)
        it.set_bur_lambda(0.0)
        it.set_jmim_aggregator(False)
        it.set_relaxmrmr_alpha(0.0)


def test_d3_default_fit_leaves_toggles_off_when_they_started_off(small_xy):
    """Complement to the snapshot test: when the toggles start OFF (the common case), a default fit must
    leave them OFF afterwards -- the snapshot-restore must not accidentally turn anything on."""
    X, y = small_xy
    m = MRMR(fe_max_steps=0, max_runtime_mins=0.1)
    m.fit(X, y)
    assert use_su_normalization() is False
    assert get_bur_lambda() == 0.0
    assert use_jmim_aggregator() is False
    assert get_relaxmrmr_alpha() == 0.0
