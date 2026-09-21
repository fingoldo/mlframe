"""Every discovery CV splits the same way: by group when groups exist, in time order under a time key, one scheme per rerank.

The tiny rerank used GroupKFold / TimeSeriesSplit, while the WAIC tie-break and the auto-chain "beats both singles" CV used
shuffled KFold on the same specs, rewarding the per-group memorisation and look-ahead the rerank was fixed to reject; and the
raw-y baseline and a spec could be scored under different fold schemes in one rerank.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.discovery import _auto_chain, _eval_waic, _splitter
from mlframe.training.composite.discovery import _tiny_rerank


def _spy_splits(monkeypatch, module):
    """Record the ``groups`` / ``time_aware`` of every ``discovery_splits`` call made through ``module``, and the folds."""
    calls = []
    real = _splitter.discovery_splits

    def _spy(n_rows, n_splits, **kw):
        folds = real(n_rows, n_splits, **kw)
        calls.append({"groups": kw.get("groups"), "time_aware": kw.get("time_aware"), "folds": folds})
        return folds

    monkeypatch.setattr(module, "discovery_splits", _spy)
    return calls


def _grouped_frame(n: int = 600, n_groups: int = 12, seed: int = 0):
    """y = base + per-group offset + noise, with a group label per row."""
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, n_groups, n)
    base = rng.uniform(1.0, 10.0, n)
    x = rng.normal(size=(n, 2))
    y = base + rng.normal(0.0, 2.0, n_groups)[groups] + 0.5 * x[:, 0] + rng.normal(0.0, 0.2, n)
    return y, base, x, groups


def _assert_group_disjoint(folds, groups):
    """No group has rows on both sides of any fold."""
    for tr, va in folds:
        assert not (set(groups[tr]) & set(groups[va])), "a group straddles a train/validation fold"


def test_the_chain_cv_splits_by_group(monkeypatch):
    """``discover_chains(groups=...)`` scores raw, singles and chains on group-disjoint folds."""
    calls = _spy_splits(monkeypatch, _auto_chain)
    y, base, x, groups = _grouped_frame()
    _auto_chain.discover_chains(y=y, base=base, x_matrix=x, cv_folds=3, n_estimators=10, compute_mi_gain=False, groups=groups)
    assert calls, "the chain CV never asked the splitter for folds"
    for c in calls:
        assert c["groups"] is not None
        _assert_group_disjoint(c["folds"], groups)


def test_the_waic_cv_splits_by_group_and_time(monkeypatch):
    """The WAIC pass takes the rerank's groups (masked with the finite rows) and time flag."""
    calls = _spy_splits(monkeypatch, _splitter)
    monkeypatch.setattr(_eval_waic, "_default_tiny_model", lambda: __import__("sklearn.linear_model", fromlist=["Ridge"]).Ridge())
    y, _base, x, groups = _grouped_frame()
    y = y.copy()
    y[:5] = np.nan  # the finite mask must be applied to the groups too
    _eval_waic.compute_transform_waic(y, x, n_folds=3, groups=groups)
    assert calls and calls[-1]["groups"] is not None and len(calls[-1]["groups"]) == len(y) - 5
    _assert_group_disjoint(calls[-1]["folds"], groups[5:])
    _eval_waic.compute_transform_waic(y[5:], x[5:], n_folds=3, time_aware=True)
    tr, va = calls[-1]["folds"][0]
    assert tr.max() < va.min(), "a time-aware WAIC fold must train on the past only"


def test_the_factory_precedence(caplog):
    """Groups over time over shuffled; too few groups falls through with a warning; contiguous gives unshuffled blocks."""
    from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit

    assert isinstance(_splitter.make_discovery_splitter(3, groups=np.arange(30) % 5, time_aware=True)[0], GroupKFold)
    assert isinstance(_splitter.make_discovery_splitter(3, groups=np.arange(30) % 2, time_aware=True)[0], TimeSeriesSplit)
    assert "group separation is NOT enforced" in caplog.text
    kf = _splitter.make_discovery_splitter(3, contiguous=True)[0]
    assert isinstance(kf, KFold) and kf.shuffle is False


def test_one_rerank_scores_raw_and_every_spec_under_one_fold_scheme(monkeypatch):
    """A monotone base on one spec and a non-monotone base on another must not split raw and specs differently."""
    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    seen: list[bool] = []
    for name in ("_tiny_cv_rmse_raw_y", "_tiny_cv_rmse_raw_y_multiseed", "_tiny_cv_rmse_y_scale", "_tiny_cv_rmse_y_scale_multiseed"):
        real = getattr(_tiny_rerank, name)

        def _spy(*a, _real=real, **kw):
            seen.append(bool(kw.get("time_aware", False)))
            return _real(*a, **kw)

        monkeypatch.setattr(_tiny_rerank, name, _spy)
    rng = np.random.default_rng(0)
    n = 800
    mono = np.sort(rng.uniform(1.0, 10.0, n))
    other = rng.uniform(1.0, 10.0, n)
    y = 0.5 * mono + 0.5 * other + rng.normal(0.0, 0.2, n)
    df = pd.DataFrame({"mono": mono, "other": other, "x": rng.normal(size=n), "y": y})
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, random_state=0, base_candidates=["mono", "other"], transforms=["linear_residual"], eps_mi_gain=-1.0,
        tiny_model_n_estimators=10, tiny_model_cv_folds=3, auto_chain_discovery_enabled=False, multi_base_enabled=False,
        interaction_base_discovery_enabled=False, auto_base_null_perms=0, honest_holdout_frac=0.0,
    )
    CompositeTargetDiscovery(cfg).fit(df, "y", ["mono", "other", "x"], np.arange(n))
    assert seen, "the tiny rerank never ran"
    assert len(set(seen)) == 1, f"raw and specs were split under different schemes in one rerank: {seen}"


@pytest.mark.parametrize("contiguous", [False, True])
def test_discovery_splits_cover_every_row_once(contiguous):
    """The folds partition the rows: every row is validated exactly once."""
    folds = _splitter.discovery_splits(50, 5, contiguous=contiguous)
    val = np.concatenate([va for _tr, va in folds])
    assert sorted(val.tolist()) == list(range(50))


def test_a_dominant_group_does_not_become_most_of_the_honest_holdout():
    """One group holding 90% of the rows cannot be the holdout: the carve stays within 25% of the configured size."""
    from mlframe.training.composite.discovery._honest_holdout import split_screening_holdout

    n = 2000
    groups = np.where(np.arange(n) < 1800, 0, 1 + np.arange(n) % 40)  # group 0 = 90% of the rows, 40 small groups
    for seed in range(5):
        screen, hold = split_screening_holdout(np.arange(n), 0.2, seed, group_ids=groups)
        assert hold is not None and 0.15 * n <= hold.size <= 0.25 * n, f"seed {seed}: holdout of {hold.size} rows for a 20% target"
        assert screen.size + hold.size == n


def test_honest_holdout_groups_are_frame_aligned_and_a_short_array_raises():
    """Group ids index the frame (like every other reader); an array too short for that raises instead of being reinterpreted."""
    from mlframe.training.composite.discovery._honest_holdout import split_screening_holdout

    rng = np.random.default_rng(0)
    frame_groups = rng.integers(0, 30, 1000)
    train_idx = rng.permutation(1000)[:800]  # a permutation, not arange
    screen, hold = split_screening_holdout(train_idx, 0.2, 0, group_ids=frame_groups)
    assert hold is not None and not (set(frame_groups[screen]) & set(frame_groups[hold])), "the holdout is not group-disjoint"
    with pytest.raises(ValueError, match="aligned to the frame"):
        split_screening_holdout(np.arange(500, 1000), 0.2, 0, group_ids=frame_groups[:500])


def test_the_stratified_sampler_skips_non_finite_targets():
    """Rows with a NaN target are not sampled: nothing downstream can use them."""
    from mlframe.training.composite.discovery.screening import _sample_indices

    rng = np.random.default_rng(0)
    y = rng.normal(size=10_000)
    y[rng.choice(10_000, 1000, replace=False)] = np.nan
    idx = _sample_indices(10_000, 2000, 0, strategy="stratified_quantile", y=y, n_strata=10)
    assert np.all(np.isfinite(y[idx])), f"{int((~np.isfinite(y[idx])).sum())} NaN-target rows sampled"
    assert idx.size == 2000


def test_a_recurrent_component_gets_contiguous_oof_folds(monkeypatch):
    """With no time or group signal, an ensemble holding a recurrent component splits its OOF into contiguous blocks."""
    from sklearn.linear_model import LinearRegression

    from mlframe.training.composite import CompositeTargetEstimator
    from mlframe.training.composite.ensemble import compute_oof_holdout_predictions

    requested = []
    real = _splitter.make_discovery_splitter

    def _spy(*a, **kw):
        requested.append(kw.get("contiguous"))
        return real(*a, **kw)

    monkeypatch.setattr(_splitter, "make_discovery_splitter", _spy)
    rng = np.random.default_rng(0)
    n = 600
    base = np.cumsum(rng.normal(size=n)) + 50.0
    X = pd.DataFrame({"base": base, "feat": rng.normal(size=n)})
    y = base + 0.5 * X["feat"].to_numpy() + rng.normal(0.0, 0.1, n)
    est = CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name="ewma_residual", base_column="base").fit(X, y)
    spec = {"name": "y-ewma-base", "transform_name": "ewma_residual", "base_column": "base", "fitted_params": dict(est.fitted_params_)}
    compute_oof_holdout_predictions(
        component_models=[est], component_names=["ewma"], component_specs=[spec], train_X=X, y_train_full=y,
        base_train_full_per_spec={"y-ewma-base": base}, holdout_frac=0.2, random_state=0, kfold=3,
    )
    assert requested == [True], f"the OOF split for a recurrent component must be contiguous; requested {requested}"
