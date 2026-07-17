"""The usability lists must NOT embed full-n training arrays in the fitted/pickled estimator
(audit, 2026-06-13).

`support_linear_` / `support_universal_` hold `UsableCandidate`s; the greedy needs their full-n
`values` arrays DURING selection, but `transform_usability` replays each feature from its `recipe`
(or a raw column by name) and never reads `values`. Keeping them would embed the TRAINING DATA in the
pickle (privacy leak + ~n*8 bytes per selected candidate of bloat). build_usability_lists clears them.
"""

from __future__ import annotations

import io
import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
import warnings

import numpy as np
import pandas as pd
import pytest

from tests.feature_selection.conftest import is_fast_mode


def _fit_usability(n, seed=0):
    """Fit usability."""
    from mlframe.feature_selection.filters import MRMR

    rng = np.random.default_rng(seed)
    a, b, c, d, e, f = (rng.random(n) for _ in range(6))
    y = 0.2 * a**2 / b + np.log(c * 2) * np.sin(d / 3) + f / 5
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs = MRMR(
            verbose=0,
            random_seed=seed,
            usability_aware_lists=True,
            usability_greedy_kwargs=dict(shortlist=14, n_folds=3),
            usability_pool_kwargs=dict(max_per_pair=8),
        ).fit(X=df, y=pd.Series(y, name="y"))
    return fs, df


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_usability_lists_drop_training_values_and_replay_survives_pickle():
    """Usability lists drop training values and replay survives pickle."""
    n = 6000 if is_fast_mode() else 12000
    fs, df = _fit_usability(n)
    lists = (fs.support_linear_ or []) + (fs.support_universal_ or [])
    assert lists, "usability lists not populated"
    # (1) no candidate retains a full-n values array (training-data embedding).
    for c in lists:
        v = getattr(c, "values", None)
        assert (not isinstance(v, np.ndarray)) or v.size == 0, (
            f"candidate {c.name!r} still stores a full-n values array (size {getattr(v, 'size', '?')}) -- training data would be embedded in the pickle"
        )
    # (2) transform_usability still works (replays from recipe / raw name, not from values).
    z_lin = fs.transform_usability(df, "linear")
    assert z_lin.shape[0] == n and z_lin.shape[1] == len(fs.support_linear_)
    assert np.isfinite(z_lin.to_numpy(dtype=float)).all()

    # (3) pickle round-trip: small-ish (no n-scaled training arrays) + replay identical post-unpickle.
    buf = io.BytesIO()
    pickle.dump(fs, buf)
    fs2 = pickle.loads(buf.getvalue())  # nosec B301 -- round-trip of a locally-created, trusted object
    z_lin2 = fs2.transform_usability(df, "linear")
    assert list(z_lin2.columns) == list(z_lin.columns)
    assert np.allclose(z_lin2.to_numpy(dtype=float), z_lin.to_numpy(dtype=float), atol=1e-9, equal_nan=True), (
        "transform_usability diverged after pickle round-trip"
    )
