"""Regression: a ``prewarp`` engineered column must replay at transform() time with the SAME warp it was scored/selected with.

A var's joint ALS pre-warp is fit per prospective pair; the pairs run in several chunks, each fitting its own spec for a shared
var, and the per-var spec dicts were merged with ``update``. Recipes looked specs up by var alone, so a feature got whichever
chunk's spec merged last. On california_housing ``sub(prewarp(MedInc),reciproc(AveOccup))`` was selected with fit-time values
~-1.5 on high-MedInc rows but replayed as -4.8 on the SAME rows (and -11.7 on a holdout row), dropping the showdown holdout R^2
from ~0.67 to 0.13-0.38. The pair-scoped spec key pins the warp each pair actually used.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.feature_selection.conftest import make_fast_mrmr


def _california():
    datasets = pytest.importorskip("sklearn.datasets")
    try:
        c = datasets.fetch_california_housing(as_frame=True)
    except OSError as e:  # offline CI without the cached dataset
        pytest.skip(f"california_housing unavailable: {e}")
    X = c.data.iloc[:1500].reset_index(drop=True)
    y = c.target.iloc[:1500].reset_index(drop=True)
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=0.25, random_state=0)


def test_prewarp_feature_replays_fit_time_values(monkeypatch):
    X_tr, X_te, y_tr, y_te = _california()
    import mlframe.feature_selection.filters._fe_raw_redundancy_drop as rd

    captured: dict = {}
    orig = rd.drop_redundant_raw_operands

    def _spy(*args, **kwargs):
        ec = kwargs.get("engineered_continuous") or {}
        for k, v in ec.items():
            captured[k] = np.asarray(v, dtype=np.float64).copy()
        return orig(*args, **kwargs)

    monkeypatch.setattr(rd, "drop_redundant_raw_operands", _spy)
    m = make_fast_mrmr(
        fe_hybrid_orth_enable=True,
        fe_hybrid_orth_pair_enable=False,
        fe_hybrid_orth_basis="hermite",
        fe_hybrid_orth_top_k=5,
        fe_hybrid_orth_ksg_enable=True,
    )
    m.fit(X_tr, y_tr)
    out = m.transform(X_tr)
    checked = [c for c in out.columns if "prewarp(" in c and c in captured]
    assert checked, f"scenario no longer selects a captured prewarp feature: {list(out.columns)}"
    for c in checked:
        # f32 fit-time scratch (MLFRAME_CRIT_DTYPE_RELAXED) vs f64 replay differ ~1e-4; the mismatched-spec bug differs by ~3.
        replay = np.asarray(out[c], dtype=np.float64)
        np.testing.assert_allclose(replay, captured[c], rtol=1e-3, atol=1e-3, err_msg=f"{c}: transform() replay differs from fit-time values")
