"""Test-gap closures for the 2026-06-10 composite audit FUTURE items.

One file, five gaps -- each previously had NO test surface, so the bug
classes they cover (stacked pass-2 rebuild hazards, docs-symbol drift,
uncovered sklearn-compliance rows, OOF-hygiene combinations, transform
round-trip parity) could regress silently:

A34   stacked pass-2 suite-TRAINING coverage. ``fit_stacked`` end-to-end on a
      small synthetic exercising the pass-2 rebuild path, plus pins on the A6
      (unrebuildable ``_oof_*`` base) and A7 (residual-vs-raw) warnings the
      pass-2 merge emits.
DX18  docs-symbol smoke. Imports every composite public symbol the README /
      composite-targets tutorial reference so a rename / move / removal fails
      fast instead of rotting the docs.
E21   compliance round-trip rows the sklearn-compliance matrix omitted -- a
      unary transform, a multi-base, a grouped, a polars-input fit+predict, a
      predict_quantile, and the domain-fallback path -- each returns all-finite.
N28   OOF-hygiene combinations the audit test omitted: group_ids AND a per-fold
      sample_weight together, the non-monotone-time kfold downgrade warning, and
      a conformal calibrate+predict_interval coverage check.
T25   adversarial forward->inverse round-trip parity sweep over the registered
      (y, base)-valid transforms; ``inverse(forward(y))`` must recover y within
      tolerance for every invertible transform (the genuinely lossy ones --
      ``y_quantile_clip`` clips, so it is non-invertible by construction -- are
      skipped with a stated reason).

All tests are fast (< 5s each): n <= 1500 synthetics, tiny inners, no GPU.
"""
from __future__ import annotations

import importlib
import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")


# ===========================================================================
# A34: stacked pass-2 suite-TRAINING coverage
# ===========================================================================


def _two_signal_frame(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """y = 1.5*x_a + 2.0*x_b + noise: a clean two-base linear DGP where pass 1
    absorbs one base via ``linear_residual`` and pass 2 can pick up the other on
    the leftover residual / augmented feature set."""
    rng = np.random.default_rng(seed)
    x_a = rng.normal(50.0, 10.0, n)
    x_b = rng.normal(0.0, 5.0, n)
    y = 1.5 * x_a + 0.5 + 2.0 * x_b + rng.normal(0.0, 1.0, n)
    return pd.DataFrame({"x_a": x_a, "x_b": x_b, "n0": rng.standard_normal(n), "y": y})


class TestA34StackedPass2Training:
    """fit_stacked drives a real pass-1 fit, OOF feature augmentation, and a
    pass-2 re-fit on the augmented frame -- the rebuild path that A6/A7 guard."""

    def _config(self):
        """Small two-transform discovery config so the pass-1 + OOF-augment + pass-2 rebuild path stays under the per-test budget."""
        from mlframe.training.configs import CompositeTargetDiscoveryConfig

        # Restrict the transform set + rerank repeats so the two-pass discovery
        # stays well under the per-test budget; the pass-2 REBUILD path (which
        # A6/A7 guard) is exercised regardless of how many transforms compete.
        return CompositeTargetDiscoveryConfig(
            enabled=True,
            mi_sample_n=400,
            composite_skip_when_raw_dominates_ratio=0.0,
            base_candidates=["x_a", "x_b"],
            transforms=["diff", "linear_residual"],
            tiny_model_n_seed_repeats=1,
            tiny_model_n_estimators=30,
            top_m_after_tiny=3,
        )

    def test_fit_stacked_end_to_end_runs_pass2(self) -> None:
        """fit_stacked's full pass-1 + OOF-augment + pass-2 rebuild path returns a coherent specs_ list on a clean two-base signal."""
        from mlframe.training.composite.discovery import CompositeTargetDiscovery

        df = _two_signal_frame()
        n = len(df)
        disc = CompositeTargetDiscovery(config=self._config()).fit_stacked(
            df=df, target_col="y",
            feature_cols=["x_a", "x_b", "n0"],
            train_idx=np.arange(int(0.8 * n)),
            n_oof_folds=2,
            max_pass1_specs_to_stack=2,
        )
        # The full pass-1 + augment + pass-2 path returns a coherent specs_ list
        # on a clean two-base signal where pass 1 finds at least one composite.
        assert hasattr(disc, "specs_")
        assert len(disc.specs_) >= 1, "stacked discovery found no specs on a clean two-base linear signal"

    def test_warn_unrebuildable_oof_specs_fires_on_oof_base(self) -> None:
        """A6: a pass-2 spec adopting an ephemeral ``_oof_*`` column as its base
        is NOT rebuildable by the suite (the column lives only in the augmented
        train frame -> all-NaN at integration). The helper must flag it by name
        so it is not trained on garbage silently."""
        from mlframe.training.composite.discovery._stacked import (
            _OOF_FEATURE_PREFIX,
            _warn_unrebuildable_oof_specs,
        )
        from mlframe.training.composite import CompositeSpec

        def _spec(name: str, base_column: str) -> CompositeSpec:
            """Minimal CompositeSpec fixture naming one base column."""
            return CompositeSpec(
                name=name, target_col="y", transform_name="linear_residual",
                base_column=base_column, fitted_params={}, mi_gain=0.1,
                mi_y=0.2, mi_t=0.3, valid_domain_frac=1.0, n_train_rows=100,
            )

        bad = _spec("y-linres-oofa", f"{_OOF_FEATURE_PREFIX}linres-x_a")
        good = _spec("y-linres-xb", "x_b")
        flagged = _warn_unrebuildable_oof_specs([bad, good], existing_names=set())
        assert bad.name in flagged, "spec whose base is an ephemeral _oof_ column must be flagged " "unrebuildable"
        assert good.name not in flagged, "spec on a real feature column must NOT be flagged"

    def test_residual_stacked_warns_on_residual_fitted_specs(self, caplog) -> None:
        """A7: pass-2 specs discovered on the RESIDUAL target carry residual-
        fitted params but the suite has no residual-aware training route, so the
        merge into specs_ must emit a loud warning (training would apply the
        residual-fitted params against raw y)."""
        from mlframe.training.composite.discovery import CompositeTargetDiscovery

        df = _two_signal_frame()
        n = len(df)
        with caplog.at_level(logging.WARNING):
            disc = CompositeTargetDiscovery(
                config=self._config(),
            ).fit_stacked_on_residual(
                df=df, target_col="y",
                feature_cols=["x_a", "x_b", "n0"],
                train_idx=np.arange(int(0.8 * n)),
                n_oof_folds=2,
                max_pass1_specs_to_aggregate=2,
            )
        # Either pass-2 found new residual specs (the warning fired) or it found
        # none (nothing to warn about). When new residual specs are merged the
        # discovered_on_residual flag must be set AND the A7 warning emitted.
        new_residual = [s for s in disc.specs_ if getattr(s, "discovered_on_residual", False)]
        if new_residual:
            assert any("discovered on the RESIDUAL target" in rec.message for rec in caplog.records), (
                "residual-fitted pass-2 specs were merged without the A7 " "residual-vs-raw warning"
            )


# ===========================================================================
# DX18: docs-symbol smoke
# ===========================================================================


class TestDX18DocsSymbols:
    """Pin every composite public symbol the README quickstart + the
    composite-targets tutorial reference. A rename / move / removal fails here
    rather than rotting the docs silently."""

    # (module, symbol). Symbols genuinely absent in some installs are guarded in
    # the test body; everything here is documented and must resolve.
    _DOCS_SYMBOLS = [
        ("mlframe.training.composite", "CompositeTargetEstimator"),
        ("mlframe.training.composite", "CompositeTargetDiscovery"),
        ("mlframe.training.composite", "CompositeTargetDiscoveryConfig"),
        ("mlframe.training.composite", "CompositeClassificationEstimator"),
        ("mlframe.training.composite", "conformal_quantile"),
        ("mlframe.training.composite", "CompositeSpec"),
        ("mlframe.training.composite", "report_to_markdown"),
    ]

    @pytest.mark.parametrize("module_path,symbol", _DOCS_SYMBOLS)
    def test_documented_symbol_importable(self, module_path: str, symbol: str) -> None:
        """Every symbol the docs reference actually exists on the module it's documented against."""
        mod = importlib.import_module(module_path)
        assert hasattr(mod, symbol), f"{module_path!r} is missing {symbol!r}"

    def test_conformal_interval_helpers_present(self) -> None:
        """The split-conformal calibrate/predict surface the conformal docs
        reference lives on the conformal submodule."""
        mod = importlib.import_module("mlframe.training.composite.conformal")
        for sym in ("conformal_quantile", "calibrate_conformal", "predict_interval"):
            assert hasattr(mod, sym), f"conformal module missing {sym!r}"


# ===========================================================================
# E21: sklearn-compliance round-trip rows the matrix omitted
# ===========================================================================


class _QuantileConstInner(BaseEstimator, RegressorMixin):
    """Inner exposing both ``predict`` and ``predict_quantile`` (the latter
    required by CompositeTargetEstimator.predict_quantile). Returns the train-T
    mean as the point estimate and mean +/- a fixed spread per quantile level so
    the wrapped y-scale quantiles are finite + ordered."""

    def fit(self, X, y, **kw):
        """Store the train-T mean/std as the point estimate + quantile spread."""
        self.n_features_in_ = np.asarray(X).shape[1] if hasattr(X, "shape") else len(X.columns)
        self._t = float(np.mean(np.asarray(y, dtype=np.float64)))
        self._s = float(np.std(np.asarray(y, dtype=np.float64))) or 1.0
        return self

    def predict(self, X):
        """Return the fitted train-T mean, broadcast to every row."""
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full(n, self._t, dtype=np.float64)

    def predict_quantile(self, X, alpha=0.5):
        """Return mean +/- a fixed spread per requested quantile level."""
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        levels = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
        # Monotone in alpha: low alpha -> below mean, high alpha -> above.
        offs = (levels - 0.5) * 2.0 * self._s
        out = self._t + offs[None, :] * np.ones((n, 1))
        if np.isscalar(alpha):
            return out.reshape(-1)
        return out


def _e21_frame(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Randomized (X, y) fixture with two bases, one plain feature, and a group column."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, size=n)
    b2 = rng.normal(0.0, 1.0, size=n)
    feat = rng.normal(0.0, 1.0, size=n)
    grp = rng.integers(0, 4, size=n).astype(str)
    y = 2.0 * base + 0.7 * b2 + 0.3 * feat + rng.normal(0.0, 0.1, size=n)
    X = pd.DataFrame({"base": base, "b2": b2, "feat": feat, "grp": grp})
    return X, y


class TestE21ComplianceUncoveredRows:
    """The sklearn-compliance matrix exercised only diff / pandas /
    LinearRegression. These are the omitted rows -- each fit+predict returns
    all-finite predictions of the right shape."""

    def test_unary_cbrt_y_fit_predict_finite(self) -> None:
        """Unary transform (no base column) fit+predict returns finite, right-shaped predictions."""
        X, y = _e21_frame()
        from mlframe.training.composite import CompositeTargetEstimator

        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="cbrt_y",  # unary: no base column required
        )
        est.fit(X[["base", "b2", "feat"]], y)
        preds = est.predict(X[["base", "b2", "feat"]])
        assert preds.shape == (len(X),)
        assert np.all(np.isfinite(preds))

    def test_multi_base_linear_residual_multi_fit_predict_finite(self) -> None:
        """Multi-base transform (2 base columns) fit+predict returns finite, right-shaped predictions."""
        X, y = _e21_frame()
        from mlframe.training.composite import CompositeTargetEstimator

        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="linear_residual_multi",
            base_columns=["base", "b2"],
        )
        est.fit(X[["base", "b2", "feat"]], y)
        preds = est.predict(X[["base", "b2", "feat"]])
        assert preds.shape == (len(X),)
        assert np.all(np.isfinite(preds))

    def test_grouped_linear_residual_grouped_fit_predict_finite(self) -> None:
        """Grouped transform fit+predict returns finite, right-shaped predictions."""
        X, y = _e21_frame()
        from mlframe.training.composite import CompositeTargetEstimator

        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="linear_residual_grouped",
            base_column="base",
            group_column="grp",
        )
        est.fit(X, y)
        preds = est.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(np.isfinite(preds))

    def test_polars_input_fit_predict_finite(self) -> None:
        """A polars-DataFrame input fit+predict returns finite, right-shaped predictions."""
        pl = pytest.importorskip("polars")
        X, y = _e21_frame()
        from mlframe.training.composite import CompositeTargetEstimator

        X_pl = pl.from_pandas(X[["base", "b2", "feat"]])
        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="diff",
            base_column="base",
        )
        est.fit(X_pl, y)
        preds = est.predict(X_pl)
        assert preds.shape == (len(X),)
        assert np.all(np.isfinite(preds))

    def test_predict_quantile_fit_predict_finite(self) -> None:
        """predict_quantile round-trips through the wrapper's y-scale inverse with finite, properly ordered quantiles."""
        X, y = _e21_frame()
        from mlframe.training.composite import CompositeTargetEstimator

        est = CompositeTargetEstimator(
            base_estimator=_QuantileConstInner(),
            transform_name="diff",
            base_column="base",
        )
        est.fit(X[["base", "b2", "feat"]], y)
        q = est.predict_quantile(X[["base", "b2", "feat"]], alpha=[0.1, 0.5, 0.9])
        assert q.shape == (len(X), 3)
        assert np.all(np.isfinite(q))
        # Quantile ordering preserved under the additive ``diff`` inverse.
        assert np.all(q[:, 0] <= q[:, 1] + 1e-9)
        assert np.all(q[:, 1] <= q[:, 2] + 1e-9)

    def test_domain_fallback_path_finite(self) -> None:
        """The domain-fallback path: a ``logratio`` predict over rows whose base
        is out of domain (base <= 0) must route to the fallback value, NEVER a
        silent NaN. Fit on strictly-positive data, predict on a frame with
        negative-base rows."""
        from mlframe.training.composite import CompositeTargetEstimator

        rng = np.random.default_rng(1)
        n = 200
        base_pos = np.abs(rng.normal(5.0, 1.0, n)) + 1.0
        feat = rng.normal(0.0, 1.0, n)
        y = base_pos * np.exp(rng.normal(0.0, 0.1, n))
        X_fit = pd.DataFrame({"base": base_pos, "feat": feat})
        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="logratio",
            base_column="base",
            fallback_predict="y_train_median",
        )
        est.fit(X_fit, y)
        # Predict frame with some out-of-domain (non-positive) base rows.
        base_pred = base_pos.copy()
        base_pred[:20] = -1.0
        X_pred = pd.DataFrame({"base": base_pred, "feat": feat})
        preds = est.predict(X_pred)
        assert preds.shape == (n,)
        assert np.all(np.isfinite(preds)), "out-of-domain base rows produced non-finite predictions instead of " "routing to the fallback value"


# ===========================================================================
# N28: OOF-hygiene combinations the audit test omitted
# ===========================================================================


class _SwRecordingRaw(BaseEstimator, RegressorMixin):
    """Raw component that records whether it received sample_weight and the
    group ids it was fit / predicted on (via a 'gid' column)."""

    saw_sample_weight: bool = False
    fit_gids: set = set()
    pred_gids: set = set()

    def fit(self, X, y, eval_set=None, sample_weight=None, **kw):
        """Record whether sample_weight was passed and which group ids were fit on."""
        if sample_weight is not None:
            type(self).saw_sample_weight = True
        type(self).fit_gids |= set(np.asarray(X["gid"]).tolist())
        self.n_features_in_ = X.shape[1]
        self._t = float(np.mean(np.asarray(y, dtype=np.float64)))
        return self

    def predict(self, X):
        """Record which group ids were predicted on, return the fitted train-T mean."""
        type(self).pred_gids |= set(np.asarray(X["gid"]).tolist())
        return np.full(X.shape[0], self._t, dtype=np.float64)


class TestN28OofHygieneGaps:
    """OOF-hygiene combinations the audit test omitted: group_ids + sample_weight together, and out-of-domain rows falling back cleanly."""

    def test_group_ids_with_per_fold_sample_weight_together(self) -> None:
        """group_ids AND a per-row sample_weight passed together: the group-aware
        single-split OOF must (a) carve whole groups so none spans refit-train and
        holdout, and (b) thread the per-row weight to the inner fit -- the audit
        test covered each alone, never the combination."""
        from mlframe.training.composite import compute_oof_holdout_predictions

        _SwRecordingRaw.saw_sample_weight = False
        _SwRecordingRaw.fit_gids = set()
        _SwRecordingRaw.pred_gids = set()
        rng = np.random.default_rng(2)
        n = 4000
        gid = np.repeat(np.arange(40), 100)
        feat = rng.normal(0.0, 1.0, size=n)
        y = gid.astype(np.float64) + rng.normal(0.0, 0.1, size=n)
        sw = np.abs(rng.normal(1.0, 0.1, size=n))
        X = pd.DataFrame({"gid": gid.astype(np.float64), "feat": feat})
        comp = _SwRecordingRaw()
        compute_oof_holdout_predictions(
            component_models=[comp],
            component_names=["raw0"],
            component_specs=[None],
            train_X=X,
            y_train_full=y,
            base_train_full_per_spec={},
            holdout_frac=0.25,
            random_state=0,
            kfold=1,
            sample_weight=sw,
            group_ids=gid,
        )
        assert _SwRecordingRaw.saw_sample_weight, "per-row sample_weight was not threaded to the inner fit"
        overlap = _SwRecordingRaw.fit_gids & _SwRecordingRaw.pred_gids
        assert not overlap, f"group(s) {sorted(overlap)[:5]} span fit + holdout " f"(group-blind split leak under combined group_ids+sample_weight)"

    def test_non_monotone_time_kfold_downgrade_warns(self, caplog) -> None:
        """kfold>1 with a NON-monotone time_ordering cannot be forward-walked
        without sorting the OOF frame, so it downgrades to a single trailing
        slice -- and MUST warn loudly so the silent loss of K-fold coverage is
        observable."""
        from mlframe.training.composite import compute_oof_holdout_predictions

        rng = np.random.default_rng(3)
        n = 600
        base = rng.normal(0.0, 1.0, size=n)
        feat = rng.normal(0.0, 1.0, size=n)
        y = base + 0.3 * feat + rng.normal(0.0, 0.1, size=n)
        X = pd.DataFrame({"base": base, "feat": feat})
        # Deliberately NON-monotone time signal (shuffled), so the forward-walk
        # downgrade path is exercised.
        time_ordering = rng.permutation(n).astype(np.float64)
        comp = LinearRegression()
        with caplog.at_level(logging.WARNING):
            compute_oof_holdout_predictions(
                component_models=[comp],
                component_names=["raw0"],
                component_specs=[None],
                train_X=X,
                y_train_full=y,
                base_train_full_per_spec={},
                holdout_frac=0.2,
                random_state=0,
                kfold=3,
                time_ordering=time_ordering,
            )
        assert any("NON-monotone" in rec.message and "Downgrading" in rec.message for rec in caplog.records), (
            "non-monotone-time kfold downgrade did not emit the expected warning; " f"got: {[r.message for r in caplog.records]}"
        )

    def test_conformal_calibrate_predict_interval_coverage(self) -> None:
        """Split-conformal: calibrate on held-out rows then predict_interval must
        deliver marginal coverage >= 1 - alpha on a fresh test set (the whole
        point of the conformal guarantee)."""
        from mlframe.training.composite import CompositeTargetEstimator

        rng = np.random.default_rng(4)
        n = 1200
        base = rng.normal(0.0, 1.0, n)
        feat = rng.normal(0.0, 1.0, n)
        noise = rng.normal(0.0, 1.0, n)
        y = 2.0 * base + 0.5 * feat + noise
        X = pd.DataFrame({"base": base, "feat": feat})
        tr, cal, te = np.arange(600), np.arange(600, 900), np.arange(900, n)
        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="diff",
            base_column="base",
        )
        est.fit(X.iloc[tr], y[tr])
        alpha = 0.1
        est.calibrate_conformal(X.iloc[cal], y[cal], alpha=alpha)
        lower, upper = est.predict_interval(X.iloc[te], alpha=alpha)
        covered = np.mean((y[te] >= lower) & (y[te] <= upper))
        # Split-conformal guarantees >= 1 - alpha marginal coverage under
        # exchangeability; allow a small finite-sample slack below the nominal.
        assert covered >= (1.0 - alpha) - 0.05, f"conformal coverage {covered:.3f} below nominal {1.0 - alpha:.2f}"
        assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))


# ===========================================================================
# T25: adversarial forward->inverse round-trip parity sweep
# ===========================================================================


# Transforms whose POINT round-trip is genuinely lossy / non-invertible -- skip
# with the reason. ``y_quantile_clip`` clips y to the train [q005, q995]
# envelope, so any value outside it is unrecoverable by construction.
_T25_NON_INVERTIBLE = {
    "y_quantile_clip": "clips y to the train quantile envelope (lossy by design)",
}

# Multi-base transforms need a K-column base matrix, not a 1-D base.
_T25_MULTI_BASE = {
    "linear_residual_multi",
    "geometric_mean_residual",
    "pairwise_interaction_residual",
}

# Transforms requiring strictly-positive y / base for their domain.
_T25_POSITIVE_DOMAIN = {
    "ratio", "logratio", "reciprocal_residual",
    "geometric_mean_residual", "rolling_quantile_ratio",
    # box_cox_y: unlike log_y (fits an offset so any y-range is in-domain),
    # box_cox_y's classical Box-Cox likelihood strictly requires y > 0 --
    # scipy.special.boxcox returns NaN on non-positive input by design
    # (domain_check gates this in production; the round-trip fixture must
    # respect the same gate).
    "box_cox_y",
}


def _t25_transform_names() -> list[str]:
    """Registered transform names eligible for the T25 (y, base) round-trip sweep -- excludes grouped and non-invertible transforms."""
    from mlframe.training.composite import list_transforms, get_transform

    names = []
    for nm in list_transforms():
        t = get_transform(nm)
        if t.requires_groups:
            # Grouped transforms need a ``groups`` kwarg threaded through
            # fit/forward/inverse; their round-trip is covered at the estimator
            # level (E21 grouped row + the dedicated grouped test module).
            continue
        if nm in _T25_NON_INVERTIBLE:
            continue
        names.append(nm)
    return names


def _t25_make_data(name: str, seed: int) -> tuple:
    """Randomized in-domain (y, base) for transform ``name``. For ``logratio``
    keep the scale tight so the MAD soft-cap (|T - median| > 10*MAD) never bites
    -- the soft-cap is a documented lossy clamp only on extreme in-domain rows,
    not a round-trip defect."""
    rng = np.random.default_rng(seed)
    n = 250
    if name == "logratio":
        y = np.exp(rng.normal(2.0, 0.3, n))
        base = np.exp(rng.normal(1.0, 0.3, n))
        return y, base
    if name in _T25_POSITIVE_DOMAIN:
        y = np.abs(rng.normal(10.0, 3.0, n)) + 1.0
        base = np.abs(rng.normal(5.0, 1.5, n)) + 1.0
        return y, base
    y = rng.normal(0.0, 5.0, n)
    base = rng.normal(0.0, 3.0, n)
    return y, base


def _t25_base_matrix(name: str, base: np.ndarray, seed: int) -> np.ndarray:
    """2-column base matrix (base, a correlated second column) for multi-base transforms in the T25 sweep."""
    rng = np.random.default_rng(seed + 1000)
    n = base.shape[0]
    b2 = base * 0.5 + rng.normal(0.0, 0.3, n)
    if name == "geometric_mean_residual":
        return np.abs(np.column_stack([base, b2])) + 1.0
    return np.column_stack([base, b2])


class TestT25RoundTripParitySweep:
    """For every registered invertible (y, base) transform, ``inverse(forward(y))``
    must recover y within tolerance on randomized in-domain data. Catches any
    forward/inverse drift (sign flip, dropped term, missing fitted param) across
    the whole transform family in one parametrized sweep."""

    @pytest.mark.parametrize("name", _t25_transform_names())
    @pytest.mark.parametrize("seed", [0, 17])
    def test_forward_inverse_recovers_y(self, name: str, seed: int) -> None:
        """inverse(forward(y)) recovers y within tolerance on randomized in-domain (y, base) data."""
        from mlframe.training.composite import get_transform

        t = get_transform(name)
        y, base = _t25_make_data(name, seed)
        if name in _T25_MULTI_BASE:
            base_arg = _t25_base_matrix(name, base, seed)
        else:
            base_arg = base
        params = t.fit(y, base_arg)
        fwd = t.forward(y, base_arg, params)
        inv = t.inverse(fwd, base_arg, params)
        assert np.all(np.isfinite(inv)), f"transform {name!r}: inverse produced non-finite values"
        rel_err = float(np.max(np.abs(inv - y))) / (float(np.std(y)) + 1e-12)
        # All invertible transforms recover y to ~machine precision on in-domain
        # data; a generous 1e-6 absorbs the rank / spline / quantile transforms'
        # interpolation rounding without masking a real forward/inverse defect.
        assert rel_err < 1e-6, f"transform {name!r}: round-trip rel-error {rel_err:.3e} exceeds tol " f"(forward/inverse parity broken)"


class TestT25DomainClassificationCoverage:
    """A registered transform whose true domain excludes some of the default
    any-real T25 fixture (e.g. box_cox_y requiring y > 0) but is missing from
    ``_T25_POSITIVE_DOMAIN`` gets silently fed out-of-domain data: forward()
    itself already produces NaN there, and ``test_forward_inverse_recovers_y``
    only reports the SYMPTOM ("inverse produced non-finite values") with no
    pointer to the actual fix (2026-07-16 incident: box_cox_y was registered
    with a real provenance formula + domain_check but never added to this
    list, and the round-trip failure gave no hint that the fixture -- not the
    transform -- was wrong).

    This test probes forward() directly on the default any-real fixture for
    every transform NOT already classified in ``_T25_POSITIVE_DOMAIN`` and
    fails with an actionable message naming the missing classification,
    BEFORE the round-trip sweep's opaque NaN assertion would fire."""

    def test_unclassified_transform_forward_is_finite_on_any_real_fixture(self) -> None:
        """Every transform not in _T25_POSITIVE_DOMAIN must tolerate the default any-real fixture -- see class docstring for the incident this guards."""
        from mlframe.training.composite import get_transform

        offenders = []
        for name in _t25_transform_names():
            if name in _T25_POSITIVE_DOMAIN:
                continue
            t = get_transform(name)
            y, base = _t25_make_data(name, seed=0)  # any-real fixture (name not in _T25_POSITIVE_DOMAIN)
            base_arg = _t25_base_matrix(name, base, seed=0) if name in _T25_MULTI_BASE else base
            params = t.fit(y, base_arg)
            fwd = np.asarray(t.forward(y, base_arg, params), dtype=np.float64)
            if not np.all(np.isfinite(fwd)):
                offenders.append(name)
        assert not offenders, (
            f"{offenders}: forward() produced non-finite output on the default any-real T25 "
            "fixture. If this transform genuinely requires a restricted domain (e.g. y > 0), add "
            "it to _T25_POSITIVE_DOMAIN (or _T25_NON_INVERTIBLE with a reason if round-tripping "
            "is inherently lossy) -- do not leave it unclassified, which silently defaults to "
            "'domain = all reals' and surfaces as an opaque NaN failure elsewhere."
        )
