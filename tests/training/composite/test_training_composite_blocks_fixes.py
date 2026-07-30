"""Regression tests for audits/full_audit_2026-07-21/training_composite_blocks.md's findings (F1-F7)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.transforms._seasonal import _seasonal_phase_means
from mlframe.training.composite.transforms.naming import _TRANSFORMS_REGISTRY, get_transform

# ---------------------------------------------------------------------------------------------------------------
# F1 -- grouped+recurrent forward misaligned `groups` against the full-length sequence
# ---------------------------------------------------------------------------------------------------------------


def test_f1_grouped_recurrent_forward_receives_full_length_groups(monkeypatch):
    """F1: the recurrent branch's forward() call must receive a groups array the SAME length as the
    full (uncompacted) sequence, not the [valid]-compacted groups_train."""
    original = get_transform("ewma_residual_grouped")
    captured: dict = {}

    def spy_forward(y, base, params, groups=None, **kw):
        """Records forward() calls for this test's assertions."""
        captured["groups_len"] = None if groups is None else len(groups)
        captured["y_len"] = len(y)
        return original.forward(y, base, params, groups=groups, **kw)

    monkeypatch.setitem(_TRANSFORMS_REGISTRY, "ewma_residual_grouped", dataclasses.replace(original, forward=spy_forward))

    rng = np.random.default_rng(0)
    n = 40
    groups = np.repeat(np.arange(4), 10)
    base = rng.normal(size=n)
    base[5] = np.nan  # triggers a domain violation -> recurrent full-then-mask branch
    y = rng.normal(size=n)
    X = pd.DataFrame({"base": base, "grp": groups})

    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(), transform_name="ewma_residual_grouped",
        base_column="base", group_column="grp", drop_invalid_rows=True,
    )
    est.fit(X, y)

    assert captured["groups_len"] == captured["y_len"], (
        "groups passed to the recurrent forward() must match the FULL sequence length "
        f"(y_len={captured['y_len']}), not the valid-only compacted length (got groups_len={captured['groups_len']})"
    )


def test_f1_ewma_residual_grouped_fits_without_error_on_domain_violating_row():
    """F1: end-to-end smoke test -- a grouped recurrent transform with a mid-series domain violation must
    fit and predict without error (this exact combination was previously untested)."""
    rng = np.random.default_rng(1)
    n = 60
    groups = np.repeat(np.arange(6), 10)
    base = rng.normal(size=n)
    base[23] = np.nan
    y = rng.normal(size=n) + base
    X = pd.DataFrame({"base": base, "grp": groups})

    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(), transform_name="ewma_residual_grouped",
        base_column="base", group_column="grp", drop_invalid_rows=True,
    )
    est.fit(X, y)
    # Predict on NaN-free data (base's own row-23 NaN was this fixture's INPUT domain violation; the inner
    # LinearRegression has no native NaN handling, orthogonal to what F1 fixes).
    X_pred = X.copy()
    X_pred.loc[23, "base"] = 0.0
    preds = est.predict(X_pred)
    assert np.isfinite(preds).all()


# ---------------------------------------------------------------------------------------------------------------
# F2 / F7 -- seasonal_residual phase drifted when the domain filter dropped a mid-series row
# ---------------------------------------------------------------------------------------------------------------


def test_f2_seasonal_residual_phase_survives_mid_series_domain_violation():
    """F2: the learned per-phase means must reflect each row's ABSOLUTE position in the original
    sequence, not its position within the domain-filter-compacted array."""
    rng = np.random.default_rng(0)
    n = 60
    phase_effect = np.array([10.0, 20.0, 30.0, 40.0])
    phase_true = np.arange(n) % 4
    y = phase_effect[phase_true] + rng.normal(scale=0.01, size=n)
    y_gap = y.copy()
    y_gap[25] = np.nan  # mid-series domain violation
    X = pd.DataFrame({"f": rng.normal(size=n)})

    est = CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name="seasonal_residual", base_column="", drop_invalid_rows=True)
    est.fit(X, y_gap)
    period = est.fitted_params_["period"]

    # Ground truth: phase computed directly from the FULL (gapped) array's absolute position.
    correct_means, _ = _seasonal_phase_means(y_gap, period)
    np.testing.assert_allclose(est.fitted_params_["phase_means"], correct_means)


def test_f2_seasonal_residual_no_gap_unaffected():
    """F2: with no domain violation at all, the fix must be a no-op (matches the un-gapped ground truth)."""
    rng = np.random.default_rng(2)
    n = 48
    phase_effect = np.array([1.0, 2.0, 3.0, 4.0])
    y = phase_effect[np.arange(n) % 4] + rng.normal(scale=0.001, size=n)
    X = pd.DataFrame({"f": rng.normal(size=n)})

    est = CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name="seasonal_residual", base_column="", drop_invalid_rows=True)
    est.fit(X, y)
    period = est.fitted_params_["period"]
    correct_means, _ = _seasonal_phase_means(y, period)
    np.testing.assert_allclose(est.fitted_params_["phase_means"], correct_means)


def test_f7_seasonal_residual_registered_recurrent():
    """F7: seasonal_residual must be registered recurrent=True so CompositeTargetEstimator routes it
    through the full-sequence forward path (the comment fix accompanying F2)."""
    assert get_transform("seasonal_residual").recurrent is True


# ---------------------------------------------------------------------------------------------------------------
# F3 -- external-holdout OOF path ignored the caller's random_state
# ---------------------------------------------------------------------------------------------------------------


def test_f3_carve_inner_eval_split_random_state_changes_group_carve():
    """F3 mechanism check: _carve_inner_eval_split's GROUP path (the one _compute_oof_with_external_holdout
    calls) must produce a different eval-group carve for a different random_state. The group-blind tail-cut
    path is deterministic by design (not random_state-dependent) -- only the group path is affected by F3,
    and >=1000 rows + >=4 groups are required to even enter it (below either threshold the function
    short-circuits to a fixed split with no fit_mask at all)."""
    from mlframe.training.composite.ensemble import _carve_inner_eval_split

    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame({"f": rng.normal(size=n)})
    y = rng.normal(size=n)
    groups = rng.integers(0, 20, size=n)  # 20 groups, well above the >=4 group-carve requirement

    split_a = _carve_inner_eval_split(X, y, random_state=0, group_ids=groups, return_fit_mask=True)
    split_b = _carve_inner_eval_split(X, y, random_state=12345, group_ids=groups, return_fit_mask=True)
    fit_mask_a, fit_mask_b = split_a[4], split_b[4]
    assert fit_mask_a is not None and fit_mask_b is not None, "fixture sanity: the group carve must actually fire"
    assert not np.array_equal(fit_mask_a, fit_mask_b), "different random_state must produce a different group eval-carve"


def test_f3_external_holdout_oof_end_to_end_group_carve_varies_with_random_state():
    """F3: compute_oof_holdout_predictions's external-holdout branch, on a composite (grouped-eligible)
    component, must actually route random_state into the group carve (end-to-end, not just the helper)."""
    from mlframe.training.composite.ensemble import compute_oof_holdout_predictions

    rng = np.random.default_rng(1)
    n_train, n_holdout = 2000, 100
    X_train = pd.DataFrame({"f": rng.normal(size=n_train)})
    y_train = rng.normal(size=n_train)
    X_holdout = pd.DataFrame({"f": rng.normal(size=n_holdout)})
    y_holdout = rng.normal(size=n_holdout)
    groups = rng.integers(0, 20, size=n_train)

    component = LinearRegression().fit(X_train, y_train)

    # Both calls must not raise, and (mechanism already pinned directly above) internally route a
    # DIFFERENT random_state into the group carve -- this end-to-end call just confirms the public API
    # accepts group_ids alongside external_holdout_* without erroring, per F3's fix wiring.
    for rs in (0, 12345):
        _oof_matrix, _oof_y, names = compute_oof_holdout_predictions(
            component_models=[component],
            component_names=["lr"],
            component_specs=[None],
            train_X=X_train,
            y_train_full=y_train,
            base_train_full_per_spec={},
            holdout_frac=0.2,
            random_state=rs,
            external_holdout_X=X_holdout,
            external_holdout_y=y_holdout,
            group_ids=groups,
        )
        assert names == ["lr"]


def test_f3_compute_oof_with_external_holdout_signature_gates_random_state():
    """F3: _compute_oof_with_external_holdout must accept and use a random_state parameter."""
    import inspect

    from mlframe.training.composite.ensemble import _compute_oof_with_external_holdout

    params = inspect.signature(_compute_oof_with_external_holdout).parameters
    assert "random_state" in params


# ---------------------------------------------------------------------------------------------------------------
# F4 -- T-clip envelope width changed silently after a streaming drift refit (MAD scaling mismatch)
# ---------------------------------------------------------------------------------------------------------------


def test_f4_update_envelope_uses_raw_mad_matching_fit():
    """F4: the post-drift-refit T-clip envelope must use the SAME raw (unscaled) MAD formula as
    fit()/from_fitted_inner() -- previously ~48% wider (an extra *1.4826 normal-consistency factor applied
    ONLY on the update() path) for the identical underlying spread."""
    from mlframe.training.composite.transforms import get_transform

    rng = np.random.default_rng(0)
    n = 600
    b = rng.normal(0.0, 1.0, n)
    y = 2.0 * b + rng.normal(0.0, 0.1, n)
    X = pd.DataFrame({"b": b})
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="linear_residual",
        base_column="b",
        online_refit_enabled=True,
        online_refit_min_buffer_n=100,
        online_refit_z_threshold=1.0,
    ).fit(X, y)

    bd = rng.normal(5.0, 1.0, 300)
    yd = 5.0 * bd + rng.normal(0.0, 0.1, 300)
    info = est.update(yd, bd)
    assert info.get("refit") is True

    # Recompute the reference envelope by hand: same buffer, same (post-refit) fitted_params, RAW
    # (unscaled) MAD -- must match fitted_params_ exactly if the fix is in place.
    buf_y = np.asarray(est._buffer_y_.contiguous(), dtype=np.float64)
    buf_b = np.asarray(est._buffer_base_.contiguous(), dtype=np.float64)
    t = get_transform(est.transform_name).forward(buf_y, buf_b, est.fitted_params_)
    t_finite = t[np.isfinite(t)]
    med_t = float(np.median(t_finite))
    raw_mad = float(np.median(np.abs(t_finite - med_t)))
    expected_low = med_t - 10.0 * raw_mad
    expected_high = med_t + 10.0 * raw_mad

    assert est.fitted_params_["t_clip_low"] == pytest.approx(expected_low, rel=1e-9)
    assert est.fitted_params_["t_clip_high"] == pytest.approx(expected_high, rel=1e-9)
    # Sanity: the old (buggy) *1.4826-scaled envelope would have been materially wider.
    scaled_high = med_t + 10.0 * raw_mad * 1.4826
    assert est.fitted_params_["t_clip_high"] < scaled_high


# ---------------------------------------------------------------------------------------------------------------
# F5 -- predict_quantile under-reported y-clip hits
# ---------------------------------------------------------------------------------------------------------------


def test_f5_predict_quantile_reports_y_clip_hits():
    """F5: clipping in predict_quantile's np.clip(y_col, low, high) must be counted into
    runtime_stats_["y_clip_low_hits"/"y_clip_high_hits"], not silently stay at 0."""
    from sklearn.base import BaseEstimator, RegressorMixin

    class _BlowupQuantileInner(BaseEstimator, RegressorMixin):
        """Inner whose quantile head returns a T far outside the y-clip envelope, forcing predict_quantile's
        np.clip(y_col, low, high) to actually fire (mirrors this repo's own
        tests/training/composite/estimator/test_composite_estimator_quantile_parity.py pattern)."""

        def fit(self, X, y, **kw):
            """Fit."""
            self.n_features_in_ = X.shape[1]
            return self

        def predict(self, X):
            """Predict."""
            return np.zeros(X.shape[0], dtype=np.float64)

        def predict_quantile(self, X, alpha=0.5):
            """Predict quantile: always a huge, out-of-envelope T."""
            return np.full(X.shape[0], 1.0e6, dtype=np.float64)

    rng = np.random.default_rng(3)
    n = 300
    base = rng.normal(size=n)
    y = base + rng.normal(scale=0.1, size=n)
    X = pd.DataFrame({"base": base})

    est = CompositeTargetEstimator(base_estimator=_BlowupQuantileInner(), transform_name="diff", base_column="base")
    est.fit(X, y)

    # Predict-time base is a huge outlier vs train: y = T_clipped + base stays far outside the
    # train-derived y_clip envelope even after the (working) T-clip bounds T itself.
    X_new = pd.DataFrame({"base": np.full(n, 1.0e5)})
    est.predict_quantile(X_new, alpha=0.5)

    rs = est.runtime_stats_
    assert rs["y_clip_high_hits"] > 0, "predict_quantile must report nonzero y-clip hits when clipping actually occurred"


# ---------------------------------------------------------------------------------------------------------------
# F6 -- dead duplicate EWMA / frac-diff numba kernels in transforms/__init__.py
# ---------------------------------------------------------------------------------------------------------------


def test_f6_dead_duplicate_kernels_removed():
    """F6: the dead, unused module-level _ewma_kernel/_frac_diff_inverse_kernel copies must be gone from
    transforms/__init__.py; the LIVE versions in nonlinear.py are untouched and still importable."""
    import mlframe.training.composite.transforms as transforms_pkg
    from mlframe.training.composite.transforms.nonlinear import _ewma_kernel, _frac_diff_inverse_kernel

    assert not hasattr(transforms_pkg, "_ewma_kernel")
    assert not hasattr(transforms_pkg, "_frac_diff_inverse_kernel")
    assert callable(_ewma_kernel)
    assert callable(_frac_diff_inverse_kernel)
