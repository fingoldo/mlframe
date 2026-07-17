"""Regression tests for the BorutaShap stability-gate fixes in ``boruta_shap/_fit_explain.py``.

Each test pins one freshly-landed bug:
  * [4]  intersection mode (stability_threshold==1.0) must NOT let ``optimistic`` re-add the sub-majority
         'tentative' bucket (that bucket is exactly the draw-level-spurious columns intersection drops).
  * [11] the >=10-row subsample floor must be capped by n, else ``rng.choice(n, size>n, replace=False)``
         raises 'Cannot take a larger sample than population' on tiny frames.
  * [12] per-row ``stratify`` must be subsampled alongside the rows, else each sub-fit gets a length-n
         stratify array against a ``size``-row subsample and train_test_split raises a length mismatch.
  * [19] the single-fit ``fit()`` path must ``return self`` (sklearn contract), so ``fit(X, y).transform(X)``
         works regardless of whether stability is enabled.

The stability-orchestration tests stub ``BorutaShap.fit`` so the per-subsample work is cheap and
deterministic while the REAL orchestrator code (size/choice, stratify slicing, keep-policy) runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_boruta(**kw):
    """Construct a BorutaShap with optional deps available (skip the whole module otherwise)."""
    pytest.importorskip("sklearn")
    pytest.importorskip("shap")
    pytest.importorskip("statsmodels")
    from mlframe.feature_selection.boruta_shap import BorutaShap

    return BorutaShap(**kw)


def _orchestrate_with_stub(self, X, y, accepted_for, base_seed=0):
    """Run the real ``_fit_with_subsample_stability`` but with each sub-fit replaced by a stub that
    sets ``accepted`` from ``accepted_for(k, sub)``. ``k`` is derived from the sub's random_state
    (the orchestrator sets it to base_seed+1+k), so the mapping is order-independent and correct even
    if the orchestrator runs the sub-fits concurrently (joblib threading backend). Returns the
    populated ``self`` plus the list of per-sub-fit observations.
    """
    from mlframe.feature_selection.boruta_shap import _fit_explain as mod
    from mlframe.feature_selection.boruta_shap import BorutaShap

    calls: list = []

    def _stub_fit(sub_self, Xk, yk):
        # k from random_state so accept-counts are independent of (parallel) call order.
        k = int(sub_self.random_state) - base_seed - 1
        # list.append is atomic under the CPython GIL, safe under the threading backend.
        calls.append({"n_rows": len(Xk), "stratify": getattr(sub_self, "stratify", None), "k": k})
        sub_self.accepted = list(accepted_for(k, sub_self))
        return sub_self

    orig_fit = BorutaShap.fit
    BorutaShap.fit = _stub_fit
    try:
        mod._fit_with_subsample_stability(self, X, y)
    finally:
        BorutaShap.fit = orig_fit
    return self, calls


def test_tiny_frame_does_not_crash_replace_false_choice():
    """[11] On a frame with fewer than 10 rows the subsample size must be capped by n, so the
    WITHOUT-replacement ``rng.choice(n, size=size, replace=False)`` inside each sub-fit does not raise."""
    sel = _make_boruta(
        model=None,
        classification=True,
        n_trials=2,
        verbose=False,
        random_state=0,
        stability_subsamples=3,
        stability_subsample_fraction=0.75,
        stability_threshold=1.0,
    )
    X = pd.DataFrame({"a": np.arange(6.0), "b": np.arange(6.0)[::-1].copy()})
    y = pd.Series([0, 1, 0, 1, 0, 1])

    # Pre-fix: size = max(10, round(0.75*6)=5) = 10 > n=6 -> ValueError in rng.choice(replace=False).
    _self, calls = _orchestrate_with_stub(sel, X, y, accepted_for=lambda k, s: [])
    assert len(calls) == 3
    # Each sub-fit got exactly ``size`` rows, and size never exceeds n.
    for c in calls:
        assert 0 < c["n_rows"] <= len(X)


def test_stratify_is_subsampled_to_match_rows():
    """[12] A per-row stratify array must be sliced to the subsample so it matches the sub-fit row count;
    otherwise train_test_split(train_or_test='test') raises 'inconsistent numbers of samples'."""
    n = 40
    sel = _make_boruta(
        model=None,
        classification=True,
        n_trials=2,
        verbose=False,
        random_state=0,
        train_or_test="test",
        stratify=np.array([0, 1] * (n // 2)),
        stability_subsamples=3,
        stability_subsample_fraction=0.5,
        stability_threshold=1.0,
    )
    X = pd.DataFrame({"a": np.arange(float(n)), "b": np.arange(float(n))[::-1].copy()})
    y = pd.Series([0, 1] * (n // 2))

    _self, calls = _orchestrate_with_stub(sel, X, y, accepted_for=lambda k, s: [])
    expected_size = min(n, max(10, round(0.5 * n)))  # = 20
    for c in calls:
        assert c["n_rows"] == expected_size
        # Pre-fix: stratify stayed the original length-n array; post-fix it is sliced to the sub-fit rows.
        assert c["stratify"] is not None
        assert len(c["stratify"]) == expected_size, "stratify was not subsampled to match the row count"


def test_intersection_mode_drops_submajority_even_when_optimistic():
    """[4] With stability_threshold==1.0 (intersection) and optimistic=True (default), a column accepted by
    SOME but not ALL subsamples lands in 'tentative' and MUST NOT be re-added to selected_features_."""
    n_sub = 4
    sel = _make_boruta(
        model=None,
        classification=True,
        n_trials=2,
        verbose=False,
        random_state=0,
        optimistic=True,
        stability_subsamples=n_sub,
        stability_subsample_fraction=0.75,
        stability_threshold=1.0,
    )
    X = pd.DataFrame({"sig": np.arange(50.0), "spurious": np.arange(50.0)[::-1].copy(), "junk": np.zeros(50)})
    y = pd.Series([0, 1] * 25)

    # 'sig' accepted by ALL subsamples; 'spurious' accepted by only some (k<2) -> tentative; 'junk' never.
    def accepted_for(k, s):
        out = ["sig"]
        if k < 2:
            out.append("spurious")
        return out

    self, _ = _orchestrate_with_stub(sel, X, y, accepted_for=accepted_for)
    assert "sig" in self.accepted
    assert "spurious" in self.tentative, "setup invariant: spurious must be sub-majority (tentative)"
    # The fix: intersection mode keeps ONLY all-accept features regardless of optimistic.
    assert set(self.selected_features_) == {"sig"}
    assert "spurious" not in self.selected_features_
    # Mask is consistent with selected_features_.
    cols = list(X.columns)
    assert list(self.support_) == [c in {"sig"} for c in cols]


def test_majority_mode_still_honors_optimistic():
    """[4] guard: the fix must only fire for intersection (threshold==1.0). With a majority threshold (<1.0)
    optimistic=True still re-adds the tentative bucket (pre-existing documented behavior preserved)."""
    n_sub = 4
    sel = _make_boruta(
        model=None,
        classification=True,
        n_trials=2,
        verbose=False,
        random_state=0,
        optimistic=True,
        stability_subsamples=n_sub,
        stability_subsample_fraction=0.75,
        stability_threshold=0.5,
    )
    X = pd.DataFrame({"sig": np.arange(50.0), "weak": np.arange(50.0)[::-1].copy()})
    y = pd.Series([0, 1] * 25)

    # need = ceil(0.5*4) = 2. 'sig' accepted 4/4 (>=need -> accepted); 'weak' accepted 1/4 (0<1<2 -> tentative).
    def accepted_for(k, s):
        out = ["sig"]
        if k == 0:
            out.append("weak")
        return out

    self, _ = _orchestrate_with_stub(sel, X, y, accepted_for=accepted_for)
    assert "sig" in self.accepted
    assert "weak" in self.tentative
    # Majority mode + optimistic re-adds tentative.
    assert set(self.selected_features_) == {"sig", "weak"}


def test_single_fit_path_returns_self_and_chains():
    """[19] The single-fit fit() path must return self (sklearn contract), so fit(X, y).transform(X) works
    without stability enabled. A small real fit is used to exercise the actual return statement."""
    pytest.importorskip("sklearn")
    pytest.importorskip("shap")
    pytest.importorskip("statsmodels")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.boruta_shap import BorutaShap

    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"f{i}": rng.standard_normal(80) for i in range(4)})
    y = pd.Series((X["f0"] + 0.5 * X["f1"] > 0).astype(int).to_numpy())
    sel = BorutaShap(
        model=RandomForestClassifier(n_estimators=20, n_jobs=1, random_state=0),
        importance_measure="gini",
        classification=True,
        n_trials=2,
        percentile=100,
        verbose=False,
        random_state=0,
    )
    returned = sel.fit(X, y)
    assert returned is sel, "single-fit fit() must return self"
    # The canonical chain used across the codebase must not raise AttributeError on None.
    out = sel.fit(X, y).transform(X)
    assert out.shape[0] == X.shape[0]
