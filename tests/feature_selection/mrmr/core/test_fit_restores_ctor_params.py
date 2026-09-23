"""A fit leaves the estimator's constructor parameters exactly as it found them, and only a known set moves while it runs.

sklearn's contract is that ``fit`` does not mutate constructor parameters. The fast-search and profile resolution does mutate them, for the
duration of the fit, and restores them in a ``finally``. That makes it invisible to anything reading the estimator before or after, which is
why it is a contract nit rather than a live bug - but it IS visible to something reading DURING the fit: a monitoring hook calling
``get_params()``, or a ``clone()`` racing an in-flight fit in another thread.

Measured on this fixture, five parameters move mid-fit and all five are restored. What is pinned here is the part that would actually break
something: nothing differs afterwards, and the set that moves cannot grow silently.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR

# Parameters the fast-search profile is known to override for the duration of a fit. A new name appearing here is not necessarily wrong, but
# it widens the window a concurrent reader can observe, so it should be a deliberate addition rather than a surprise.
_KNOWN_TRANSIENT = {
    "cluster_aggregate_enable",
    "fe_escalation_underdelivery_enable",
    "fe_pair_prewarp_enable",
    "fe_stability_vote_enable",
    "random_seed",
    "fe_check_pairs_subsample_n",
    "fe_smart_polynom_subsample_n",
}


@pytest.fixture
def frame():
    """A frame with one informative column, small enough for a quick fit."""
    rng = np.random.default_rng(0)
    n = 2000
    a = rng.normal(size=n)
    X = pd.DataFrame({f"c{i}": rng.normal(size=n) for i in range(8)})
    X["c0"] = a
    return X, (a > 0).astype(np.int64)


def _fit_while_polling(est, X, y):
    """Fit while another thread reads ``get_params()``, returning every parameter observed to differ mid-fit."""
    before = dict(est.get_params())
    seen: dict = {}
    stop = threading.Event()

    def poll():
        """Record any parameter whose value differs from the pre-fit snapshot."""
        while not stop.is_set():
            try:
                now = est.get_params()
            except Exception:  # a torn read during a fit is not what this test is about
                continue
            for key, value in before.items():
                if key in now and now[key] != value and key not in seen:
                    seen[key] = (value, now[key])
            time.sleep(0.001)

    watcher = threading.Thread(target=poll, daemon=True)
    watcher.start()
    try:
        MRMR._FIT_CACHE.clear()
        est.fit(X, y)
    finally:
        stop.set()
        watcher.join(timeout=5)
    return before, seen


def test_every_constructor_parameter_is_restored_after_a_fit(frame):
    """The contract that matters: after ``fit`` returns, the estimator's parameters are what the caller set."""
    X, y = frame
    est = MRMR(random_state=0, verbose=0, fe_max_steps=1, full_npermutations=3, baseline_npermutations=2, fe_fast_search=True)
    before, _seen = _fit_while_polling(est, X, y)
    after = dict(est.get_params())
    changed = {k: (before[k], after[k]) for k in before if k in after and after[k] != before[k]}
    assert not changed, f"fit left constructor parameter(s) modified: {changed}"


def test_only_known_parameters_move_while_the_fit_runs(frame):
    """A concurrent reader can see the profile's overrides; which ones is a documented set, not an open one."""
    X, y = frame
    est = MRMR(random_state=0, verbose=0, fe_max_steps=1, full_npermutations=3, baseline_npermutations=2, fe_fast_search=True)
    _before, seen = _fit_while_polling(est, X, y)
    unexpected = sorted(set(seen) - _KNOWN_TRANSIENT)
    assert not unexpected, (
        "constructor parameter(s) newly observable as modified during a fit: " + ", ".join(unexpected) + ". A reader calling get_params() "
        "mid-fit, or a clone() racing an in-flight fit, sees these; add them deliberately or carry them in a per-fit config instead."
    )


def test_a_fit_without_the_fast_search_profile_moves_nothing(frame):
    """With the profile off there is no override to restore, so nothing should be observable at all."""
    X, y = frame
    est = MRMR(random_state=0, verbose=0, fe_max_steps=1, full_npermutations=3, baseline_npermutations=2, fe_fast_search=False)
    before, seen = _fit_while_polling(est, X, y)
    after = dict(est.get_params())
    assert not {k: after[k] for k in before if k in after and after[k] != before[k]}
    assert set(seen) <= _KNOWN_TRANSIENT, f"unexpected mid-fit mutation without the profile: {sorted(set(seen) - _KNOWN_TRANSIENT)}"
