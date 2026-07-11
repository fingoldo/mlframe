"""biz_value test for ``feature_engineering.transformer.compute_denoising_autoencoder_features`` /
``swap_noise_augment``.

The win: a swap-noise DENOISING autoencoder's bottleneck representation should be more ROBUST to input
perturbation than a PLAIN (non-denoising) autoencoder's bottleneck -- that robustness (not raw reconstruction
quality) is the entire point of the Porto Seguro 1st-place swap-noise DAE technique, since real production
inputs are never perfectly clean. Quantified by comparing each encoder's bottleneck-activation shift between
the clean input and a mildly swap-noise-perturbed version of the SAME input (simulating realistic imperfect
production data): the DAE's shift should be substantially smaller than the plain AE's.

Also verifies ``swap_noise_augment`` preserves each column's marginal distribution (the core reason swap
noise beats Gaussian/multiplicative corruption for tabular data: every corrupted value is a real value from
that same column, never an out-of-distribution one).
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_engineering.transformer.autoencoder import _extract_bottleneck as _extract_ae
from mlframe.feature_engineering.transformer.autoencoder import _fit_autoencoder
from mlframe.feature_engineering.transformer.denoising_autoencoder import _extract_bottleneck as _extract_dae
from mlframe.feature_engineering.transformer.denoising_autoencoder import _extract_multilayer, _fit_dae
from mlframe.feature_engineering.transformer.swap_noise import swap_noise_augment


def _make_latent_structured_dataset(n: int, d: int, seed: int):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 3))
    W = rng.normal(size=(3, d))
    X = latent @ W + rng.normal(scale=0.2, size=(n, d))
    return X


def test_biz_val_denoising_autoencoder_bottleneck_more_robust_to_perturbation_than_plain_autoencoder():
    seed = 2
    X = _make_latent_structured_dataset(n=3000, d=12, seed=seed)

    dae, dae_scaler = _fit_dae(X, hidden_size=24, bottleneck_dim=4, max_iter=500, swap_prob=0.3, seed=seed)
    ae, ae_scaler = _fit_autoencoder(X, hidden_size=24, bottleneck_dim=4, max_iter=500, seed=seed)

    # Mildly-perturbed input simulates realistic imperfect production data (a swap-noise corruption of the
    # SAME clean rows the encoders were fit on -- not a held-out set; this test is about representation
    # stability under perturbation, not generalization).
    X_perturbed = swap_noise_augment(X, swap_prob=0.2, rng=np.random.default_rng(99 + seed))

    dae_shift = float(np.mean((_extract_dae(dae, dae_scaler, X, 4) - _extract_dae(dae, dae_scaler, X_perturbed, 4)) ** 2))
    ae_shift = float(np.mean((_extract_ae(ae, ae_scaler, X, 4) - _extract_ae(ae, ae_scaler, X_perturbed, 4)) ** 2))

    robustness_gain = 1.0 - dae_shift / ae_shift
    assert robustness_gain > 0.45, (
        f"expected the swap-noise DAE's bottleneck to shift >45% less than a plain AE's under the same input "
        f"perturbation, got {robustness_gain:.4f} (dae_shift={dae_shift:.5f}, ae_shift={ae_shift:.5f})"
    )


def test_swap_noise_augment_preserves_column_marginals():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=[0, 100, -5], scale=[1, 20, 0.5], size=(5000, 3))
    X_corrupted = swap_noise_augment(X, swap_prob=0.3, rng=np.random.default_rng(1))

    # Every corrupted value came from the SAME column of X, so the corrupted column's sorted unique values
    # must be a subset of the original column's values (no synthesized out-of-distribution values).
    for j in range(X.shape[1]):
        assert np.isin(X_corrupted[:, j], X[:, j]).all()
        # Marginal mean/std of the corrupted column should stay close to the original (resampling from the
        # same column doesn't change its distribution, only which row holds which value).
        assert abs(X_corrupted[:, j].mean() - X[:, j].mean()) < 0.1 * X[:, j].std()
        assert abs(X_corrupted[:, j].std() - X[:, j].std()) < 0.1 * X[:, j].std()


def test_swap_noise_augment_swap_rate_matches_swap_prob():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20000, 4))
    X_corrupted = swap_noise_augment(X, swap_prob=0.15, rng=np.random.default_rng(1))
    changed_rate = float((X_corrupted != X).mean())
    assert abs(changed_rate - 0.15) < 0.01


def test_swap_noise_augment_zero_prob_is_identity():
    X = np.random.default_rng(0).normal(size=(50, 3))
    assert np.array_equal(swap_noise_augment(X, swap_prob=0.0), X)


def _make_mixed_abstraction_dataset(n: int, seed: int):
    """Signal split across two levels of abstraction: a low-rank latent (3 dims -> 6 correlated columns,
    exactly what a small bottleneck is built to compress) and a separate high-rank block (14 near-independent
    columns) that a narrow bottleneck cannot preserve without discarding most of the low-rank signal, but a
    wider hidden layer can carry alongside it."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 3))
    W = rng.normal(size=(3, 6))
    X_low = latent @ W + rng.normal(scale=0.15, size=(n, 6))
    X_high = rng.normal(size=(n, 14))
    X = np.concatenate([X_low, X_high], axis=1).astype(np.float32)
    y = 2.0 * np.tanh(latent[:, 0]) - latent[:, 1] + 0.9 * X_high[:, :5].sum(axis=1) + rng.normal(scale=0.3, size=n)
    return X, y


def test_biz_val_denoising_autoencoder_multilayer_extraction_beats_bottleneck_only():
    """The win: when predictive signal lives at multiple levels of abstraction (some in a low-rank latent
    the bottleneck compresses well, some in a higher-rank block the narrow bottleneck must discard),
    concatenating the wider hidden-layer activations (``extract_layers="multi"``) alongside the bottleneck
    recovers signal a bottleneck-only extraction loses, measured by held-out R2 of a linear probe."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    X, y = _make_mixed_abstraction_dataset(n=4000, seed=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    dae, scaler = _fit_dae(X_train, hidden_size=16, bottleneck_dim=4, max_iter=400, swap_prob=0.2, seed=1)

    Z_train_bottleneck = _extract_dae(dae, scaler, X_train, 4)
    Z_test_bottleneck = _extract_dae(dae, scaler, X_test, 4)
    Z_train_multi, _ = _extract_multilayer(dae, scaler, X_train, 4)
    Z_test_multi, _ = _extract_multilayer(dae, scaler, X_test, 4)

    r2_bottleneck = r2_score(y_test, Ridge(alpha=1.0).fit(Z_train_bottleneck, y_train).predict(Z_test_bottleneck))
    r2_multi = r2_score(y_test, Ridge(alpha=1.0).fit(Z_train_multi, y_train).predict(Z_test_multi))

    assert r2_multi > r2_bottleneck + 0.25, (
        f"expected multi-layer DAE features to beat bottleneck-only by >0.25 held-out R2 "
        f"(measured multi={r2_multi:.4f} - bottleneck={r2_bottleneck:.4f} = {r2_multi - r2_bottleneck:.4f})"
    )
    assert r2_multi >= 0.60, f"expected multi-layer held-out R2 >= 0.60, got {r2_multi:.4f}"
