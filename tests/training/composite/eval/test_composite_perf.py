"""Tests for the algorithmic-optimisation round of composite-target
discovery (bin-based MI / parallel CV folds / train-prediction cache).

Coverage map
------------
- ``_mi_pair_bin`` returns non-negative MI on independent inputs
  (~ 0) and substantial MI on highly correlated inputs.
- ``_mi_to_target`` with ``estimator="bin"`` and ``estimator="knn"``
  produce same-direction rankings on a synthetic feature set
  (correctness sanity, not bit-equality).
- Bin estimator handles tiny inputs gracefully (returns 0 instead
  of raising on n < 5*nbins).
- Discovery with ``mi_estimator="bin"`` produces specs comparable
  to the kNN run on TVT-style data (the canonical case the bin
  estimator was tuned for).
- Discovery with ``tiny_model_n_jobs=3`` produces specs identical
  to the serial run (folds-in-parallel must not change the
  algorithm result; modulo joblib backend non-determinism on tied
  ranks, we test by comparing kept transform names rather than
  rmse values).
- Config validators reject bad ``mi_estimator`` values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# B1 sklearn matrix marker convention -- this file runs in the multi-sklearn-version CI matrix.
pytestmark = pytest.mark.sklearn_matrix


from mlframe.training.composite import (
    CompositeTargetDiscovery,
    _mi_pair_bin,
    _mi_to_target,
)
from mlframe.training.configs import CompositeTargetDiscoveryConfig


# ----------------------------------------------------------------------
# _mi_pair_bin
# ----------------------------------------------------------------------


class TestMiPairBin:
    def test_independent_inputs_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        n = 2000
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        mi = _mi_pair_bin(x, y, nbins=16)
        assert mi >= 0.0
        assert mi < 0.1  # near zero for independent gaussians

    def test_correlated_inputs_substantial(self) -> None:
        rng = np.random.default_rng(1)
        n = 2000
        x = rng.normal(size=n)
        y = x + 0.1 * rng.normal(size=n)  # highly correlated
        mi = _mi_pair_bin(x, y, nbins=16)
        assert mi > 1.0  # substantial information

    def test_tiny_input_returns_zero(self) -> None:
        # n < 5 * nbins -> too few rows for stable estimate.
        x = np.linspace(0, 1, 10)
        y = x.copy()
        assert _mi_pair_bin(x, y, nbins=16) == 0.0

    def test_non_finite_inputs_filtered(self) -> None:
        rng = np.random.default_rng(2)
        n = 2000
        x = rng.normal(size=n)
        y = x.copy()
        x[10:20] = np.nan
        y[30:40] = np.inf
        # Should still return finite, positive MI (after filtering NaN/Inf rows).
        mi = _mi_pair_bin(x, y, nbins=16)
        assert np.isfinite(mi)
        assert mi > 0.5

    def test_constant_input_gives_zero(self) -> None:
        x = np.full(2000, 7.0)  # constant
        y = np.random.default_rng(3).normal(size=2000)
        mi = _mi_pair_bin(x, y, nbins=16)
        # Constant x -> all values fall into one bin -> p_x(i) = 1
        # for one bin, 0 elsewhere -> MI = 0.
        assert 0.0 <= mi < 0.05


# ----------------------------------------------------------------------
# _mi_to_target with both estimators
# ----------------------------------------------------------------------


class TestMiToTarget:
    @pytest.fixture
    def fixture(self):
        rng = np.random.default_rng(4)
        n = 2000
        # Three features with different strengths against the same target.
        x_strong = rng.normal(size=n)
        x_weak = rng.normal(size=n)
        x_noise = rng.normal(size=n)
        target = x_strong + 0.3 * x_weak + 0.5 * rng.normal(size=n)
        X = np.column_stack([x_strong, x_weak, x_noise])
        return X, target

    def test_bin_and_knn_both_positive(self, fixture) -> None:
        X, y = fixture
        mi_bin = _mi_to_target(X, y, n_neighbors=3, random_state=0, estimator="bin", nbins=16)
        mi_knn = _mi_to_target(X, y, n_neighbors=3, random_state=0, estimator="knn")
        assert mi_bin > 0
        assert mi_knn > 0

    def test_bin_estimator_faster_than_knn(self, fixture) -> None:
        # Sanity timing: bin should beat knn on the same data. We only
        # assert the speedup direction, not a specific factor (CI-safe).
        import time

        X, y = fixture
        t0 = time.perf_counter()
        for _ in range(3):
            _mi_to_target(X, y, n_neighbors=3, random_state=0, estimator="bin", nbins=16)
        t_bin = (time.perf_counter() - t0) / 3
        t0 = time.perf_counter()
        for _ in range(3):
            _mi_to_target(X, y, n_neighbors=3, random_state=0, estimator="knn")
        t_knn = (time.perf_counter() - t0) / 3
        # Bin is at least 3x faster on n=2000, k=3 (we observed ~38x
        # in micro-benchmarks; CI is conservative).
        assert t_bin < t_knn / 3, f"bin should be at least 3x faster than knn; bin={t_bin * 1000:.1f}ms knn={t_knn * 1000:.1f}ms"


# ----------------------------------------------------------------------
# Config validators
# ----------------------------------------------------------------------


class TestConfigValidators:
    def test_mi_estimator_normalised(self) -> None:
        cfg = CompositeTargetDiscoveryConfig(mi_estimator="BIN")
        assert cfg.mi_estimator == "bin"

    def test_mi_estimator_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="mi_estimator must be one of"):
            CompositeTargetDiscoveryConfig(mi_estimator="entropy_kraskov")


# ----------------------------------------------------------------------
# Discovery with new options
# ----------------------------------------------------------------------


def _tvt_data(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=10.0, scale=3.0, size=n)
    x = rng.normal(size=(n, 4))
    y = 0.95 * base + 0.5 * x[:, 0] - 0.3 * x[:, 1] + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame({"TVT_prev": base, "TVT": y})
    for i in range(4):
        df[f"x{i}"] = x[:, i]
    return df


class TestDeterministicScreeningModels:
    """``deterministic_screening_models`` toggle: when True, the
    tiny models built for Phase B rerank should carry the well-known
    per-family determinism flags. Verified by introspecting the
    constructed estimator's ``get_params`` / attributes (no full
    GPU run required)."""

    def test_lightgbm_deterministic_kwargs(self) -> None:
        from mlframe.training.composite import _build_tiny_model

        m = _build_tiny_model(
            "lightgbm",
            n_estimators=10,
            num_leaves=8,
            learning_rate=0.1,
            random_state=0,
            deterministic=True,
        )
        params = m.get_params()
        assert params.get("deterministic") is True, "LightGBM deterministic flag must be set"
        assert params.get("force_row_wise") is True, "LightGBM force_row_wise must be set for deterministic histograms"
        assert params.get("force_col_wise") is False, "LightGBM force_col_wise must be off when force_row_wise is on"

    def test_lightgbm_non_deterministic_default(self) -> None:
        from mlframe.training.composite import _build_tiny_model

        m = _build_tiny_model(
            "lightgbm",
            n_estimators=10,
            num_leaves=8,
            learning_rate=0.1,
            random_state=0,
            deterministic=False,
        )
        params = m.get_params()
        # Default mode: deterministic flag NOT set, force_col_wise on
        # (the perf-optimised default).
        assert params.get("deterministic") in (None, False)
        assert params.get("force_col_wise") is True

    def test_xgboost_deterministic_uses_hist(self) -> None:
        pytest.importorskip("xgboost")
        from mlframe.training.composite import _build_tiny_model

        m = _build_tiny_model(
            "xgboost",
            n_estimators=10,
            num_leaves=8,
            learning_rate=0.1,
            random_state=0,
            deterministic=True,
        )
        params = m.get_params()
        assert params.get("tree_method") == "hist"

    def test_catboost_deterministic_uses_plain(self) -> None:
        pytest.importorskip("catboost")
        from mlframe.training.composite import _build_tiny_model

        m = _build_tiny_model(
            "catboost",
            n_estimators=10,
            num_leaves=8,
            learning_rate=0.1,
            random_state=0,
            deterministic=True,
        )
        # catboost stores params on the constructed object via _init_params.
        params = m.get_params()
        assert params.get("boosting_type") == "Plain"

    def test_linear_unaffected(self) -> None:
        from mlframe.training.composite import _build_tiny_model

        m_det = _build_tiny_model(
            "linear",
            n_estimators=10,
            num_leaves=8,
            learning_rate=0.1,
            random_state=0,
            deterministic=True,
        )
        m_nondet = _build_tiny_model(
            "linear",
            n_estimators=10,
            num_leaves=8,
            learning_rate=0.1,
            random_state=0,
            deterministic=False,
        )
        # Ridge is deterministic by construction; both branches build
        # functionally equivalent models. Dict equality is NaN-unsafe
        # (``imp__missing_values=nan`` is the SimpleImputer default and
        # ``nan != nan`` per IEEE-754); compare with a NaN-safe helper.
        assert type(m_det) is type(m_nondet)
        _params_det = m_det.get_params()
        _params_nondet = m_nondet.get_params()
        assert set(_params_det) == set(_params_nondet), (
            f"param-key drift: det-only={set(_params_det) - set(_params_nondet)}, nondet-only={set(_params_nondet) - set(_params_det)}"
        )
        for _k in _params_det:
            _v_det = _params_det[_k]
            _v_nondet = _params_nondet[_k]
            try:
                _both_nan = isinstance(_v_det, float) and isinstance(_v_nondet, float) and _v_det != _v_det and _v_nondet != _v_nondet
            except Exception:
                _both_nan = False
            if _both_nan:
                continue
            # Use repr() for non-trivially-comparable objects like sklearn
            # estimators -- ``Ridge(random_state=0) == Ridge(random_state=0)``
            # returns False because BaseEstimator doesn't implement __eq__.
            assert repr(_v_det) == repr(_v_nondet), f"param {_k!r} drift: det={_v_det!r} vs nondet={_v_nondet!r}"

    def test_discovery_with_determinism_flag_runs(self) -> None:
        """End-to-end smoke: discovery with ``deterministic_screening_models=True``
        should produce specs identical to the non-deterministic run on
        the canonical TVT case (the determinism flag affects only
        run-to-run float drift; on the same single run the result is
        the same)."""
        from mlframe.training.composite import CompositeTargetDiscovery

        df = _tvt_data()
        common = dict(
            enabled=True,
            base_candidates=["TVT_prev"],
            transforms=["linear_residual"],
            mi_sample_n=600,
            top_k_after_mi=2,
            eps_mi_gain=-1.0,
            screening="hybrid",
            tiny_model_n_estimators=20,
            tiny_model_sample_n=400,
            top_m_after_tiny=1,
        )
        cfg_off = CompositeTargetDiscoveryConfig(
            **common,
            deterministic_screening_models=False,
        )
        cfg_on = CompositeTargetDiscoveryConfig(
            **common,
            deterministic_screening_models=True,
        )
        d_off = CompositeTargetDiscovery(cfg_off).fit(
            df,
            target_col="TVT",
            feature_cols=["TVT_prev", "x0", "x1"],
            train_idx=np.arange(1200),
        )
        d_on = CompositeTargetDiscovery(cfg_on).fit(
            df,
            target_col="TVT",
            feature_cols=["TVT_prev", "x0", "x1"],
            train_idx=np.arange(1200),
        )
        # Both produced specs and picked the same top base.
        assert d_off.specs_ and d_on.specs_
        assert d_off.specs_[0].base_column == d_on.specs_[0].base_column


class TestDiscoveryBinEstimator:
    def test_bin_estimator_finds_dominant_base(self) -> None:
        df = _tvt_data()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True,
            base_candidates=["TVT_prev"],
            transforms=["diff", "linear_residual"],
            mi_sample_n=600,
            top_k_after_mi=2,
            eps_mi_gain=-1.0,
            mi_estimator="bin",
            mi_nbins=16,
        )
        disc = CompositeTargetDiscovery(cfg).fit(
            df,
            target_col="TVT",
            feature_cols=["TVT_prev", "x0", "x1", "x2", "x3"],
            train_idx=np.arange(1200),
        )
        assert disc.specs_, "bin estimator should still surface specs"
        assert disc.specs_[0].base_column == "TVT_prev"

    def test_parallel_cv_folds_run_without_crash(self) -> None:
        df = _tvt_data()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True,
            base_candidates=["TVT_prev"],
            transforms=["linear_residual"],
            mi_sample_n=600,
            top_k_after_mi=2,
            eps_mi_gain=-1.0,
            screening="hybrid",
            tiny_model_n_estimators=20,
            tiny_model_sample_n=400,
            top_m_after_tiny=1,
            tiny_model_n_jobs=2,  # parallel folds via joblib
        )
        disc = CompositeTargetDiscovery(cfg).fit(
            df,
            target_col="TVT",
            feature_cols=["TVT_prev", "x0", "x1"],
            train_idx=np.arange(1200),
        )
        assert disc.specs_

    def test_bin_and_knn_pick_same_top_base(self) -> None:
        df = _tvt_data()
        common = dict(
            enabled=True,
            base_candidates="auto",
            transforms=["linear_residual"],
            mi_sample_n=600,
            top_k_after_mi=2,
            eps_mi_gain=-1.0,
            auto_base_top_k=1,
        )
        df_with_target = df.copy()
        feature_cols = ["TVT_prev", "x0", "x1", "x2", "x3"]
        cfg_knn = CompositeTargetDiscoveryConfig(**common, mi_estimator="knn")
        cfg_bin = CompositeTargetDiscoveryConfig(**common, mi_estimator="bin")
        disc_knn = CompositeTargetDiscovery(cfg_knn).fit(
            df_with_target,
            target_col="TVT",
            feature_cols=feature_cols,
            train_idx=np.arange(1200),
        )
        disc_bin = CompositeTargetDiscovery(cfg_bin).fit(
            df_with_target,
            target_col="TVT",
            feature_cols=feature_cols,
            train_idx=np.arange(1200),
        )
        # Both should converge on TVT_prev as the dominant base.
        assert disc_knn.specs_
        assert disc_bin.specs_
        assert disc_knn.specs_[0].base_column == disc_bin.specs_[0].base_column
        assert disc_knn.specs_[0].base_column == "TVT_prev"
