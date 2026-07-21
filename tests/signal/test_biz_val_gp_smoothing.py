"""biz_value test for ``signal.gp_smoothing.compute_gp_smoothed_features`` / ``gp_smooth_irregular_series``.

The win: objects have a smooth true underlying curve (a Gaussian bump whose peak TIME determines the
object's class) but are sparsely and irregularly sampled with noise. A naive "nearest observed raw value at
each query time" feature is badly corrupted by per-observation noise and irregular gaps. GP-smoothing (with
a FIXED length scale, matching the source's own choice -- letting sklearn auto-optimize the length scale
was measured to actively hurt, overfitting through the noise on sparse curves) should recover the true
smooth curve shape far better, giving a downstream classifier a materially better signal. Also verifies the
GP posterior std behaves as the intended "confidence from local data density" companion feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from mlframe.signal.gp_smoothing import compute_gp_smoothed_features, gp_smooth_irregular_series


def _make_sparse_lightcurve_dataset(n_objects: int, seed: int):
    """Builds seeded synthetic test data; returns ``(pd.DataFrame(rows), labels, query_times)``."""
    rng = np.random.default_rng(seed)
    query_times = np.linspace(0, 20, 8)
    rows = []
    labels = np.zeros(n_objects, dtype=int)
    for obj in range(n_objects):
        label = rng.integers(0, 2)
        labels[obj] = label
        peak_t = 8.0 if label == 0 else 12.0
        n_obs = rng.integers(3, 6)
        t_obs = np.sort(rng.uniform(0, 20, n_obs))
        y_true = np.exp(-((t_obs - peak_t) ** 2) / (2 * 2.0**2))
        y_obs = y_true + rng.normal(scale=0.3, size=n_obs)
        for t, y in zip(t_obs, y_obs):
            rows.append({"obj": obj, "t": t, "y": y})
    return pd.DataFrame(rows), labels, query_times


def _nearest_raw_features(df: pd.DataFrame, entities: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    """Returns ``out`` (after 2 setup steps)."""
    out = np.zeros((len(entities), len(query_times)))
    for i, e in enumerate(entities):
        sub = df[df["obj"] == e]
        t_obs, y_obs = sub["t"].to_numpy(), sub["y"].to_numpy()
        for j, qt in enumerate(query_times):
            out[i, j] = y_obs[np.argmin(np.abs(t_obs - qt))]
    return out


def test_biz_val_gp_smoothed_features_beats_nearest_raw_value_auc():
    """Gp smoothed features beats nearest raw value auc."""
    df, labels, query_times = _make_sparse_lightcurve_dataset(n_objects=100, seed=1)
    entities = pd.unique(df["obj"])
    y_labels = labels[entities]

    X_baseline = _nearest_raw_features(df, entities, query_times)
    auc_baseline = cross_val_score(LogisticRegression(max_iter=500), X_baseline, y_labels, cv=5, scoring="roc_auc").mean()

    gp_feats = (
        compute_gp_smoothed_features(df, "obj", "t", "y", query_times, length_scale=3.0, nu=1.5, alpha=0.05, optimize_hyperparameters=False)
        .to_pandas()
        .set_index("obj")
        .reindex(entities)
    )
    X_gp = gp_feats[[c for c in gp_feats.columns if "mean" in c]].to_numpy()
    auc_gp = cross_val_score(LogisticRegression(max_iter=500), X_gp, y_labels, cv=5, scoring="roc_auc").mean()

    error_reduction = 1.0 - (1.0 - auc_gp) / (1.0 - auc_baseline)
    assert (
        error_reduction > 0.3
    ), f"expected >30% error-rate reduction vs. nearest-raw-value features, got {error_reduction:.4f} (baseline_auc={auc_baseline:.4f}, gp_auc={auc_gp:.4f})"


def test_gp_smooth_std_is_low_near_observations_high_in_sparse_gaps():
    """Gp smooth std is low near observations high in sparse gaps."""
    t = np.array([1.0, 2.0, 3.0, 15.0])
    y = np.array([1.0, 1.1, 0.9, 2.0])
    t_query = np.array([2.0, 9.0, 15.0])  # near-observed, far-gap, near-observed
    _, std = gp_smooth_irregular_series(t, y, t_query, length_scale=2.0, alpha=0.05)
    assert std[1] > std[0] * 2
    assert std[1] > std[2] * 2


def _make_mixed_regime_dataset(n_objects: int, seed: int):
    """Pool of BOTH fast-varying (narrow bump) and slow-varying (wide bump) objects, densely sampled.

    A single fixed length scale is a documented compromise: small enough to track the fast objects
    undersmooths the slow ones (chases noise on the wide, gently-curving bump), large enough to smooth the
    slow objects oversmooths the fast ones (blurs away the narrow peak entirely). The multi-length-scale
    ensemble should let each object pick its own regime-appropriate scale.
    """
    rng = np.random.default_rng(seed)
    query_times = np.linspace(0, 20, 40)
    rows = []
    true_curves = np.zeros((n_objects, len(query_times)))
    for obj in range(n_objects):
        fast = obj % 2 == 0
        width = 0.4 if fast else 10.0
        peak_t = rng.uniform(6, 14)
        n_obs = rng.integers(12, 18)
        t_obs = np.sort(rng.uniform(0, 20, n_obs))
        y_true_obs = np.exp(-((t_obs - peak_t) ** 2) / (2 * width**2))
        y_obs = y_true_obs + rng.normal(scale=0.1, size=n_obs)
        for t, y in zip(t_obs, y_obs):
            rows.append({"obj": obj, "t": t, "y": y})
        true_curves[obj] = np.exp(-((query_times - peak_t) ** 2) / (2 * width**2))
    return pd.DataFrame(rows), true_curves, query_times


def test_biz_val_gp_smooth_irregular_series_ensemble_beats_best_single_scale_mixed_regime():
    """Gp smooth irregular series ensemble beats best single scale mixed regime."""
    df, true_curves, query_times = _make_mixed_regime_dataset(n_objects=60, seed=3)
    entities = pd.unique(df["obj"])
    grouped = {e: sub for e, sub in df.groupby("obj", sort=False)}

    candidate_scales = [0.4, 2.0, 10.0]

    def _mse_for_scale(length_scale: float) -> float:
        """Returns ``float(np.mean(sq_errors))`` (after 2 setup steps)."""
        sq_errors = []
        for i, e in enumerate(entities):
            sub = grouped[e]
            mean, _ = gp_smooth_irregular_series(sub["t"].to_numpy(), sub["y"].to_numpy(), query_times, length_scale=length_scale, nu=1.5, alpha=0.05)
            sq_errors.append(np.mean((mean - true_curves[i]) ** 2))
        return float(np.mean(sq_errors))

    single_scale_mse = {ls: _mse_for_scale(ls) for ls in candidate_scales}
    best_single_mse = min(single_scale_mse.values())

    ensemble_sq_errors = []
    for i, e in enumerate(entities):
        sub = grouped[e]
        mean, _ = gp_smooth_irregular_series(
            sub["t"].to_numpy(), sub["y"].to_numpy(), query_times, nu=1.5, alpha=0.05, length_scales=candidate_scales, ensemble_mode="cv_best"
        )
        ensemble_sq_errors.append(np.mean((mean - true_curves[i]) ** 2))
    ensemble_mse = float(np.mean(ensemble_sq_errors))

    improvement = 1.0 - ensemble_mse / best_single_mse
    assert improvement > 0.14, (
        f"expected ensemble MSE to beat the best single fixed scale by >10%, got {improvement:.4f} "
        f"(ensemble_mse={ensemble_mse:.4f}, per-scale_mse={single_scale_mse}, best_single_mse={best_single_mse:.4f})"
    )


def test_gp_smooth_irregular_series_length_scales_none_is_bit_identical_to_legacy_default():
    """Gp smooth irregular series length scales none is bit identical to legacy default."""
    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(0, 20, 8))
    y = rng.normal(size=8)
    t_query = np.linspace(0, 20, 5)
    mean_a, std_a = gp_smooth_irregular_series(t, y, t_query, length_scale=2.5, alpha=0.05)
    mean_b, std_b = gp_smooth_irregular_series(t, y, t_query, length_scale=2.5, alpha=0.05, length_scales=None)
    np.testing.assert_array_equal(mean_a, mean_b)
    np.testing.assert_array_equal(std_a, std_b)


def test_gp_smoothed_features_output_shape():
    """Gp smoothed features output shape."""
    df, _labels, query_times = _make_sparse_lightcurve_dataset(n_objects=10, seed=2)
    result = compute_gp_smoothed_features(df, "obj", "t", "y", query_times, alpha=0.05)
    assert result.shape[0] == 10
    assert result.shape[1] == 1 + 2 * len(query_times)  # entity col + mean/std per query time
