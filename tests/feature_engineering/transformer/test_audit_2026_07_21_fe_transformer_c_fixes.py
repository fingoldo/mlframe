"""Regression tests for audits/full_audit_2026-07-21/fe_transformer_c.md (F1-F15).

F11 (stale comment) and the dismissed PR5 (row_attention_stage4_adaptive_njit turned out to have a live
caller in adaptive_bandwidth.py, outside this cluster's file list -- confirmed via grep, nothing to fix)
have no dedicated test. PR1-PR4/PR6 are test-coverage/docs proposals satisfied by the tests below plus the
docstring additions made alongside each fix.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer._key_bank import _key_bank_fingerprint
from mlframe.feature_engineering.transformer._residual_oof import compute_oof_yhat_within
from mlframe.feature_engineering.transformer.anomaly_score_features import compute_anomaly_score_features
from mlframe.feature_engineering.transformer.band_conditional_anchor import compute_band_conditional_anchor_features
from mlframe.feature_engineering.transformer.baseline_surprise import compute_baseline_surprise_features
from mlframe.feature_engineering.transformer.bidir_residual_band import compute_bidir_residual_band_features
from mlframe.feature_engineering.transformer.boosting_leaf import compute_boosting_leaf_features
from mlframe.feature_engineering.transformer.conformal_coverage_failure import compute_conformal_coverage_failure_features
from mlframe.feature_engineering.transformer.decision_region_depth import compute_decision_region_depth_features
from mlframe.feature_engineering.transformer.distributional_moments import compute_distributional_moments_features
from mlframe.feature_engineering.transformer.multi_temp_band_attention import compute_multi_temp_band_attention_features
from mlframe.feature_engineering.transformer.multi_temp_cbhr import compute_multi_temp_cbhr_features
from mlframe.feature_engineering.transformer.quantile_band_attention import compute_quantile_band_attention_features
from mlframe.feature_engineering.transformer.sign_residual_baseline import compute_sign_residual_baseline_features

# ----------------------------------------------------------------------
# F1 (P0) -- KeyBank fingerprint must include projection and dtype.
# ----------------------------------------------------------------------


def test_f1_key_bank_fingerprint_changes_with_projection():
    """F1: key bank fingerprint changes with projection."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4)).astype(np.float32)
    kw = dict(X_train=X, seed=0, n_heads=2, head_dim=2, metric="cosine", standardize=True, ann_M=16, ann_ef_construction=100, dtype=np.float32)
    fp_random = _key_bank_fingerprint(**kw, projection="random")
    fp_pls = _key_bank_fingerprint(**kw, projection="pls")
    assert fp_random != fp_pls, "switching projection must invalidate the cache fingerprint"


def test_f1_key_bank_fingerprint_changes_with_dtype():
    """F1: key bank fingerprint changes with dtype."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4)).astype(np.float32)
    kw = dict(X_train=X, seed=0, n_heads=2, head_dim=2, metric="cosine", standardize=True, ann_M=16, ann_ef_construction=100, projection="random")
    fp_f32 = _key_bank_fingerprint(**kw, dtype=np.float32)
    fp_f64 = _key_bank_fingerprint(**kw, dtype=np.float64)
    assert fp_f32 != fp_f64, "switching dtype must invalidate the cache fingerprint"


# ----------------------------------------------------------------------
# F2-F5 -- empty y-band no longer creates a phantom attention target at the origin.
# ----------------------------------------------------------------------


def test_f2_quantile_band_attention_empty_band_not_phantom_center():
    """A query at the standardized-space origin (where an unmasked empty band's centroid sits) must attend
    entirely to the real (non-empty) band, not get pulled toward a phantom y_mean=0."""
    rng = np.random.default_rng(0)
    n = 200
    X_train = rng.normal(size=(n, 3)).astype(np.float32)
    y_train = np.ones(n, dtype=np.float32)  # binary, all-positive -> the "neg" band is empty
    X_query = np.zeros((5, 3), dtype=np.float32)
    df = compute_quantile_band_attention_features(X_train, y_train, X_query, seed=0, task="binary", standardize=True)
    agg_y_mean = df["qbattn_y_mean"].to_numpy()
    assert np.allclose(agg_y_mean, 1.0), f"phantom empty band pulled agg_y_mean away from 1.0: {agg_y_mean}"


def test_f3_band_conditional_anchor_empty_band_masked_and_correctly_labeled():
    """F3: band conditional anchor empty band masked and correctly labeled."""
    rng = np.random.default_rng(0)
    n = 200
    X_train = rng.normal(size=(n, 3)).astype(np.float32)
    y_train = np.ones(n, dtype=np.float32)
    X_query = np.zeros((5, 3), dtype=np.float32)
    df = compute_band_conditional_anchor_features(X_train, y_train, X_query, seed=0, task="binary", anchors_per_band=2, standardize=True)
    assert np.allclose(df["bcanc_flat_y_mean"].to_numpy(), 1.0)
    # The empty "neg" band's mass must be exactly 0 (masked out), not silently absorbing weight into band 0.
    assert np.allclose(df["bcanc_mass_neg"].to_numpy(), 0.0)


def test_f4_multi_temp_band_attention_empty_band_masked_across_all_temps():
    """F4: multi temp band attention empty band masked across all temps."""
    rng = np.random.default_rng(0)
    n = 200
    X_train = rng.normal(size=(n, 3)).astype(np.float32)
    y_train = np.ones(n, dtype=np.float32)
    X_query = np.zeros((5, 3), dtype=np.float32)
    df = compute_multi_temp_band_attention_features(X_train, y_train, X_query, seed=0, task="binary", standardize=True)
    y_mean_cols = [c for c in df.columns if c.endswith("_y_mean")]
    assert y_mean_cols, "expected at least one *_y_mean column"
    for col in y_mean_cols:
        assert np.allclose(df[col].to_numpy(), 1.0), f"{col} pulled toward a phantom empty band"


def test_f5_bidir_residual_band_empty_bands_get_zero_weight(monkeypatch):
    """Force 3 of 5 |residual|-quantile bands to be literally empty (via a monkeypatched constant-zero
    baseline, so residuals are exactly {0, 10}) and assert their attention weight is exactly 0."""
    import mlframe.feature_engineering.transformer.bidir_residual_band as brb

    def fake_predict(Xt, y_t, task, seed, n_estimators=50, max_depth=3):
        """Fake predict callable returning a fixed array."""
        return np.zeros_like(y_t)

    monkeypatch.setattr(brb, "_fit_baseline_predict", fake_predict)

    rng = np.random.default_rng(0)
    n = 100
    X_train = rng.normal(size=(n, 3)).astype(np.float32)
    y_train = np.zeros(n, dtype=np.float32)
    y_train[:20] = 10.0
    X_query = np.zeros((5, 3), dtype=np.float32)
    df = brb.compute_bidir_residual_band_features(X_train, y_train, X_query, seed=0, n_bands=5, standardize=True)
    w = df.select([f"bidrbattn_w_R{i}" for i in range(1, 6)]).to_numpy()
    # R2, R3, R4 (0-indexed bands 1, 2, 3) are empty by construction -- must get exactly zero weight.
    assert np.allclose(w[:, 1], 0.0)
    assert np.allclose(w[:, 2], 0.0)
    assert np.allclose(w[:, 3], 0.0)
    assert np.isfinite(w).all()


# ----------------------------------------------------------------------
# F6 -- multi_temp_cbhr.py: an entirely-empty side must not silently reuse a wrong-class row.
# ----------------------------------------------------------------------


def test_f6_multi_temp_cbhr_empty_side_masked_not_wrong_class_row():
    """F6: multi temp cbhr empty side masked not wrong class row."""
    rng = np.random.default_rng(0)
    n = 100
    X_train = rng.normal(size=(n, 3)).astype(np.float32)
    y_train = np.ones(n, dtype=np.float32)  # binary all-positive -> the "neg" side is empty
    X_query = rng.normal(size=(5, 3)).astype(np.float32)
    df = compute_multi_temp_cbhr_features(X_train, y_train, X_query, seed=0, task="binary", n_hard_per_side=4, standardize=True)
    neg_weight_cols = [c for c in df.columns if "_w_neg_" in c]
    assert neg_weight_cols
    for col in neg_weight_cols:
        assert np.allclose(df[col].to_numpy(), 0.0), f"{col} should be exactly 0 (masked empty side)"
    side2_cols = [c for c in df.columns if c.endswith("_y_side2")]
    for col in side2_cols:
        assert np.allclose(df[col].to_numpy(), 0.0), f"{col}: empty side's aggregate must not leak a wrong-class y"


# ----------------------------------------------------------------------
# F7 -- distributional_moments.py: quantiles length must be validated.
# ----------------------------------------------------------------------


def test_f7_distributional_moments_rejects_wrong_length_quantiles():
    """F7: distributional moments rejects wrong length quantiles."""
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(50, 3)).astype(np.float32)
    y_train = rng.normal(size=50).astype(np.float32)
    with pytest.raises(ValueError, match="7 quantiles"):
        compute_distributional_moments_features(X_train, y_train, X_train[:5], seed=0, quantiles=(0.1, 0.5, 0.9))


def test_f7_distributional_moments_accepts_default_7_quantiles():
    """F7: distributional moments accepts default 7 quantiles."""
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(80, 3)).astype(np.float32)
    y_train = rng.normal(size=80).astype(np.float32)
    df = compute_distributional_moments_features(X_train, y_train, X_train[:5], seed=0)
    assert df.shape == (5, 5)


# ----------------------------------------------------------------------
# F8 -- conformal_coverage_failure.py: tiny-fold guard (mirrors conformal_locally_adaptive.py's n<4 guard).
# ----------------------------------------------------------------------


def test_f8_conformal_coverage_failure_tiny_fold_no_crash():
    """F8: conformal coverage failure tiny fold no crash."""
    rng = np.random.default_rng(0)
    splitter = KFold(n_splits=2, shuffle=False)
    X_train = rng.normal(size=(3, 3)).astype(np.float32)  # a fold's train complement can be this small
    y_train = rng.normal(size=3).astype(np.float32)
    df = compute_conformal_coverage_failure_features(X_train, y_train, X_query=None, splitter=splitter, seed=0)
    assert df.shape[0] == 3
    assert np.isfinite(df.to_numpy()).all()


# ----------------------------------------------------------------------
# F9 -- _residual_oof.py: n<2 subset must not crash KFold(n_splits=2).
# ----------------------------------------------------------------------


def test_f9_compute_oof_yhat_within_handles_n_below_2():
    """F9: compute oof yhat within handles n below 2."""
    from sklearn.linear_model import LinearRegression

    for n in (0, 1):
        X_sub = np.zeros((n, 3), dtype=np.float32)
        y_sub = np.zeros(n, dtype=np.float32)
        out = compute_oof_yhat_within(X_sub, y_sub, task="regression", make_aux=LinearRegression, aux_n_splits=5, seed=0)
        assert out.shape == (n,)


def test_f9_compute_oof_yhat_within_still_works_for_n_ge_2():
    """F9: compute oof yhat within still works for n ge 2."""
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    X_sub = rng.normal(size=(20, 3)).astype(np.float32)
    y_sub = rng.normal(size=20).astype(np.float32)
    out = compute_oof_yhat_within(X_sub, y_sub, task="regression", make_aux=LinearRegression, aux_n_splits=5, seed=0)
    assert out.shape == (20,)
    assert np.isfinite(out).all()


# ----------------------------------------------------------------------
# F10 -- honest inner-OOF baseline (not in-sample) drives band/anchor/threshold derivation.
# ----------------------------------------------------------------------


def _spy_kfold_calls(monkeypatch) -> list:
    """Spy kfold calls."""
    calls: list = []
    orig_init = KFold.__init__

    def spy_init(self, *a, **kw):
        """Records constructor calls for this test's assertions."""
        calls.append(kw)
        return orig_init(self, *a, **kw)

    monkeypatch.setattr(KFold, "__init__", spy_init)
    return calls


def test_f10_bidir_residual_band_uses_inner_oof(monkeypatch):
    """F10: bidir residual band uses inner oof."""
    calls = _spy_kfold_calls(monkeypatch)
    rng = np.random.default_rng(0)
    n = 60
    X_train = rng.normal(size=(n, 3)).astype(np.float32)
    y_train = rng.normal(size=n).astype(np.float32)
    compute_bidir_residual_band_features(X_train, y_train, X_train[:5], seed=0, baseline_n_estimators=5, standardize=True)
    assert any(kw.get("n_splits") == 3 for kw in calls), "expected an inner KFold(n_splits=3) OOF pass"


def test_f10_multi_temp_cbhr_uses_inner_oof(monkeypatch):
    """F10: multi temp cbhr uses inner oof."""
    calls = _spy_kfold_calls(monkeypatch)
    rng = np.random.default_rng(0)
    n = 60
    X_train = rng.normal(size=(n, 3)).astype(np.float32)
    y_train = rng.normal(size=n).astype(np.float32)
    compute_multi_temp_cbhr_features(X_train, y_train, X_train[:5], seed=0, baseline_n_estimators=5, standardize=True)
    assert any(kw.get("n_splits") == 3 for kw in calls)


def test_f10_baseline_surprise_uses_inner_oof(monkeypatch):
    """F10: baseline surprise uses inner oof."""
    calls = _spy_kfold_calls(monkeypatch)
    rng = np.random.default_rng(0)
    n = 60
    X_train = rng.normal(size=(n, 3)).astype(np.float32)
    y_train = rng.normal(size=n).astype(np.float32)
    compute_baseline_surprise_features(X_train, y_train, X_train[:5], seed=0, standardize=True)
    assert any(kw.get("n_splits") == 3 for kw in calls)


def test_f10_decision_region_depth_uses_inner_oof(monkeypatch):
    """F10: decision region depth uses inner oof."""
    calls = _spy_kfold_calls(monkeypatch)
    rng = np.random.default_rng(0)
    n = 60
    X_train = rng.normal(size=(n, 3)).astype(np.float32)
    y_train = rng.normal(size=n).astype(np.float32)
    compute_decision_region_depth_features(X_train, y_train, X_train[:5], seed=0, task="regression", standardize=True)
    assert any(kw.get("n_splits") == 3 for kw in calls)


def test_f10_sign_residual_baseline_uses_inner_oof(monkeypatch):
    """F10: sign residual baseline uses inner oof."""
    calls = _spy_kfold_calls(monkeypatch)
    rng = np.random.default_rng(0)
    n = 60
    X_train = rng.normal(size=(n, 3)).astype(np.float32)
    y_train = rng.normal(size=n).astype(np.float32)
    compute_sign_residual_baseline_features(X_train, y_train, X_train[:5], seed=0, task="regression", standardize=True)
    assert any(kw.get("n_splits") == 3 for kw in calls)


def test_f10_sign_residual_baseline_score_bug_regression_still_holds():
    """The pre-existing test_sign_residual_baseline_score_bug.py fixture (a strong, non-subtle directional
    bias) must still be detected after switching mu_train to an inner-OOF estimate."""
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 4)).astype(np.float32)
    base = X[:, 0] * 2.0 + X[:, 1]
    skew = np.where(X[:, 0] > 0, 3.0, 0.0)
    y = (base + skew + rng.normal(scale=0.1, size=n)).astype(np.float32)
    feats = compute_sign_residual_baseline_features(X_train=X[:300], y_train=y[:300], X_query=X[300:], seed=42, task="regression", standardize=True)
    mu = feats["signres_mu"].to_numpy()
    baseline_score = feats["signres_baseline_score"].to_numpy()
    assert not np.allclose(mu, baseline_score)
    bias_signal = feats["signres_bias_signal"].to_numpy()
    assert (np.abs(bias_signal) > 1e-6).any()


# ----------------------------------------------------------------------
# F12 -- boosting_leaf.py: an optional splitter overrides the internal KFold.
# ----------------------------------------------------------------------


class _SpySplitter:
    """Splitter stub that records the folds it was asked to produce."""
    def __init__(self, base):
        self.base = base
        self.calls = 0

    def split(self, X, y=None, groups=None):
        """No-op / recording stub matching the splitter's split() signature."""
        self.calls += 1
        return self.base.split(X)


def test_f12_boosting_leaf_accepts_custom_splitter():
    """F12: boosting leaf accepts custom splitter."""
    rng = np.random.default_rng(0)
    n = 100
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = rng.normal(size=n).astype(np.float32)
    spy = _SpySplitter(KFold(n_splits=3, shuffle=True, random_state=1))
    df = compute_boosting_leaf_features(X, y, None, seed=0, n_estimators=5, splitter=spy)
    assert spy.calls == 1
    assert df.shape[0] == n


def test_f12_boosting_leaf_default_unchanged_without_splitter():
    """F12: boosting leaf default unchanged without splitter."""
    rng = np.random.default_rng(0)
    n = 100
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = rng.normal(size=n).astype(np.float32)
    df = compute_boosting_leaf_features(X, y, None, seed=0, n_estimators=5)
    assert df.shape[0] == n


# ----------------------------------------------------------------------
# F13 -- stacked_attention.py: gpu_stage4 is now a caller-facing passthrough.
# ----------------------------------------------------------------------


def test_f13_stacked_attention_threads_gpu_stage4(monkeypatch):
    """F13: stacked attention threads gpu stage4."""
    import mlframe.feature_engineering.transformer.stacked_attention as sa

    captured = []
    orig = sa.compute_row_attention

    def spy(*args, **kwargs):
        """Records call arguments for this test's assertions."""
        captured.append(kwargs.get("gpu_stage4"))
        return orig(*args, **kwargs)

    monkeypatch.setattr(sa, "compute_row_attention", spy)
    rng = np.random.default_rng(0)
    n = 60
    X_train = rng.normal(size=(n, 5)).astype(np.float32)
    y_train = rng.normal(size=n).astype(np.float32)
    splitter = KFold(n_splits=3, shuffle=True, random_state=0)
    sa.compute_stacked_row_attention(X_train, y_train, None, splitter, seed=0, n_layers=1, gpu_stage4="auto")
    assert captured and all(c == "auto" for c in captured)


def test_f13_stacked_attention_default_gpu_stage4_is_false():
    """F13: stacked attention default gpu stage4 is false."""
    import inspect

    import mlframe.feature_engineering.transformer.stacked_attention as sa

    sig = inspect.signature(sa.compute_stacked_row_attention)
    assert sig.parameters["gpu_stage4"].default is False


# ----------------------------------------------------------------------
# F14 -- row_attention.py: cupy stage4 dispatch uses identity, not a __name__ string match.
# ----------------------------------------------------------------------


def test_f14_stage4_dispatch_uses_identity_not_name_collision():
    """F14: stage4 dispatch uses identity not name collision."""
    from mlframe.feature_engineering.transformer.row_attention import attend, build_key_bank

    class _FakeCupyNamed:
        """Deliberately __name__-collides with the real cupy kernel but is NOT that function; it does not
        accept the k_proj_device/y_train_device kwargs the real cupy path receives."""

        __name__ = "row_attention_stage4_cupy"

        def __call__(self, q, k, y, ids, temp, y_mean_v, y_std_v, x_mean_v):
            y_mean_v[:] = 1.0
            y_std_v[:] = 0.0
            x_mean_v[:] = 0.0

    rng = np.random.default_rng(0)
    n_train, d, head_dim = 40, 3, 2
    X_train = rng.normal(size=(n_train, d)).astype(np.float32)
    y_train = rng.normal(size=n_train).astype(np.float32)
    bank = build_key_bank(X_train, y_train, seed=0, n_heads=1, head_dim=head_dim, standardize=True)
    # Pretend GPU-resident so the k_proj_device branch is live -- pre-fix, the __name__ string match would
    # route through this fake callable's incompatible signature (TypeError on the device kwargs); post-fix
    # the identity check correctly recognises it is NOT the real cupy kernel.
    bank.k_proj_device = ["sentinel"]
    bank.y_train_device = "sentinel"

    X_query = rng.normal(size=(3, d)).astype(np.float32)
    out = attend(bank, X_query, k=8, stage4_callable=_FakeCupyNamed())
    assert np.allclose(out["y_mean_h0"], 1.0)


# ----------------------------------------------------------------------
# F15 -- anomaly_score_features.py: global_mean_train uses a subsample, not a full Xt rescore.
# ----------------------------------------------------------------------


def test_f15_anomaly_score_global_mean_never_rescoring_full_xt(monkeypatch):
    """F15: anomaly score global mean never rescoring full xt."""
    from sklearn.ensemble import IsolationForest

    call_sizes: list = []
    orig = IsolationForest.score_samples

    def spy(self, X):
        """Records call arguments for this test's assertions."""
        call_sizes.append(len(X))
        return orig(self, X)

    monkeypatch.setattr(IsolationForest, "score_samples", spy)

    rng = np.random.default_rng(0)
    n_train, n_query = 500, 10
    X_train = rng.normal(size=(n_train, 4)).astype(np.float32)
    X_query = rng.normal(size=(n_query, 4)).astype(np.float32)
    y_train = rng.normal(size=n_train).astype(np.float32)
    df = compute_anomaly_score_features(X_train, y_train, X_query, seed=0, standardize=False)
    assert df.shape[0] == n_query
    # Pre-fix, 2 of these calls scored the full n_train=500 rows; post-fix every call is <= max_samples (256).
    assert call_sizes, "IsolationForest.score_samples was never called"
    assert all(sz <= 256 for sz in call_sizes), call_sizes


def test_f15_anomaly_score_output_still_finite_and_reasonable():
    """F15: anomaly score output still finite and reasonable."""
    rng = np.random.default_rng(0)
    n_train, n_query = 500, 20
    X_train = rng.normal(size=(n_train, 4)).astype(np.float32)
    X_query = rng.normal(size=(n_query, 4)).astype(np.float32)
    y_train = rng.normal(size=n_train).astype(np.float32)
    df = compute_anomaly_score_features(X_train, y_train, X_query, seed=0, standardize=True)
    assert np.isfinite(df.to_numpy()).all()
