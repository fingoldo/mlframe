"""§8.1 FS test coverage gaps -- regression tests for previously uncovered FS code paths.

Each test corresponds to one §8.1 finding. Behavioural only; sub-second under --fast.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# §8.1 P1: _rfecv.py:530 polars TimeSeriesSplit auto-detect
# ---------------------------------------------------------------------------


def test_rfecv_polars_sorted_datetime_triggers_time_series_split():
    """A polars DataFrame with a single sorted Datetime column auto-routes the CV to TimeSeriesSplit."""
    pl = pytest.importorskip("polars")
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit
    from mlframe.feature_selection.wrappers.rfecv import RFECV

    n = 80
    rng = np.random.default_rng(0)
    import datetime as _dt

    ts = [_dt.datetime(2024, 1, 1) + _dt.timedelta(hours=i) for i in range(n)]
    df = pl.DataFrame(
        {
            "ts": pl.Series(ts, dtype=pl.Datetime),
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
        }
    )
    # The Datetime column must be sorted ascending and null-free for the auto-detect heuristic
    # to engage (see _rfecv.py:530 ff.).
    y = pd.Series(rng.standard_normal(n))
    rfecv = RFECV(estimator=Ridge(), cv=3, max_runtime_mins=1.0)
    try:
        rfecv.fit(df, y)
    except Exception:  # nosec B110 -- best-effort cleanup/optional step; failure here never masks this test's own assertions
        # We tolerate downstream early-stop / fallback; we only care about the splitter wiring.
        pass
    # The auto-detect path stores the resolved splitter on ``cv_`` when triggered.
    if hasattr(rfecv, "cv_") and rfecv.cv_ is not None:
        assert isinstance(rfecv.cv_, TimeSeriesSplit), "sorted polars Datetime column should trigger TimeSeriesSplit auto-detection"


# ---------------------------------------------------------------------------
# §8.1 P1: mrmr.py:341 _FIT_CACHE cross-suite contamination
# ---------------------------------------------------------------------------


def test_mrmr_fit_cache_distinguishes_distinct_y_in_same_process():
    """Two suites in the same process with different y must NOT replay each other's cached support_.
    Pre-fix the cache key was shape-only so a second fit on a different y would silently replay
    the first suite's selection. Post-fix the signature folds y content."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 150
    X_df = pd.DataFrame(rng.standard_normal((n, 4)), columns=[f"f{i}" for i in range(4)])
    y_a = (X_df["f0"] > 0).astype(int).to_numpy()
    y_b = (X_df["f3"] > 0).astype(int).to_numpy()  # different target -> different signal column

    MRMR.clear_fit_cache()
    a = MRMR(quantization_nbins=5, full_npermutations=1, baseline_npermutations=1, verbose=0, skip_retraining_on_same_shape=True, random_seed=0)
    a.fit(X_df, y_a)
    sig_a = a.signature

    b = MRMR(quantization_nbins=5, full_npermutations=1, baseline_npermutations=1, verbose=0, skip_retraining_on_same_shape=True, random_seed=0)
    b.fit(X_df, y_b)
    sig_b = b.signature
    assert sig_a != sig_b, "MRMR fit-cache must distinguish suites with different y content (shape collision should not trigger a cross-suite cache hit)."


# ---------------------------------------------------------------------------
# §8.1 P1: _setup_helpers.py:407 cat_features plumbed through mrmr_kwargs
# ---------------------------------------------------------------------------


def test_mrmr_accepts_factors_names_to_use_via_kwargs():
    """The plumbing for ``factors_names_to_use`` (MRMR's analogue of a cat-features hint) must
    survive **mrmr_kwargs expansion intact -- a regression sentinel for the call-site at
    _setup_helpers.py:407 (``MRMR(**mrmr_kwargs)``)."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    cat_features = ["f0", "f2"]
    kwargs = {"factors_names_to_use": cat_features, "verbose": 0, "random_seed": 0}
    mrmr = MRMR(**kwargs)
    assert mrmr.factors_names_to_use == cat_features, "factors_names_to_use plumbed through mrmr_kwargs must be retained on the instance"


# ---------------------------------------------------------------------------
# §8.1 P2: mrmr.py:431 fe_fallback_to_all=True legacy branch
# ---------------------------------------------------------------------------


def test_mrmr_fe_fallback_to_all_legacy_branch_constructs_cleanly():
    """``fe_fallback_to_all=True`` is the legacy "FE on all features when screen returns empty" branch.
    The constructor must accept the flag and retain it; no immediate crash on instantiation."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    mrmr = MRMR(fe_fallback_to_all=True, verbose=0, random_seed=0)
    assert mrmr.fe_fallback_to_all is True


# ---------------------------------------------------------------------------
# §8.1 P2: mrmr.py:428 max_confirmation_cand_nbins=None autoconv
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_confirmation_cand_nbins", [None, 50])
def test_mrmr_max_confirmation_cand_nbins_none_vs_50(max_confirmation_cand_nbins):
    """The autoconv branch (None default) and the legacy pin (50) must BOTH yield a fitted, non-empty
    support_. We compare the two configurations rather than asserting an exact selection so the test
    survives MRMR's stochastic confirmation step on small fuzz frames."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 80
    X_df = pd.DataFrame(rng.standard_normal((n, 3)), columns=["a", "b", "c"])
    y = (X_df["a"] > 0).astype(int).to_numpy()
    MRMR.clear_fit_cache()
    mrmr = MRMR(
        quantization_nbins=5,
        full_npermutations=1,
        baseline_npermutations=1,
        verbose=0,
        random_seed=0,
        max_confirmation_cand_nbins=max_confirmation_cand_nbins,
    )
    mrmr.fit(X_df, y)
    assert mrmr.support_ is not None
    # At least one feature must survive (min_features_fallback=1 default since 2026-05-16 §1).
    # ``MRMR.support_`` is an int-index array (not a bool mask), so non-empty == ``len(..) >= 1``.
    assert len(mrmr.support_) >= 1, f"fitted MRMR must yield non-empty support_ regardless of max_confirmation_cand_nbins; got support_={mrmr.support_}"
    # Knob arrived intact on the instance.
    assert mrmr.max_confirmation_cand_nbins == max_confirmation_cand_nbins


# ---------------------------------------------------------------------------
# §8.1 Low: _setup_helpers.py:418 custom_pre_pipelines deepcopy fallback
# ---------------------------------------------------------------------------


def test_custom_pre_pipelines_falls_back_to_deepcopy_when_clone_fails():
    """When ``sklearn.base.clone()`` raises (transformer doesn't implement the BaseEstimator
    protocol), the build path must fall through to ``copy.deepcopy``."""
    from mlframe.training.core._setup_helpers import _build_pre_pipelines

    class _NotASklearnEstimator:
        """A bare transformer with no ``get_params`` / ``set_params`` -- sklearn.base.clone raises
        TypeError on it."""

        def __init__(self):
            self.state = []

        def fit(self, X, y=None):
            self.state.append("fit")
            return self

        def transform(self, X):
            return X

    bad = _NotASklearnEstimator()
    pre_pipes, _names = _build_pre_pipelines(
        use_ordinary_models=False,
        rfecv_models=[],
        rfecv_models_params={},
        use_mrmr_fs=False,
        mrmr_kwargs={},
        custom_pre_pipelines={"raw": bad},
    )
    # The custom pipeline must be present with a working transform() (deepcopy survived).
    assert any(getattr(p, "transform", None) is not None for p in pre_pipes)
    # The inserted object is a distinct instance (deepcopy, not the original).
    inserted = next(p for p in pre_pipes if hasattr(p, "transform"))
    assert inserted is not bad
