"""biz_value + regression tests for ``feature_engineering.transformer.swap_noise.swap_noise_augment``.

Covers the opt-in per-column swap-probability mode (``column_swap_probs``): callers corrupt noisy/
low-signal columns more aggressively than clean/informative ones when generating swap-noise-corrupted input
for DAE pretraining, instead of the original uniform ``swap_prob`` applied identically to every column.
"""

from __future__ import annotations

import warnings

import numpy as np

from mlframe.feature_engineering.transformer.denoising_autoencoder import _extract_bottleneck
from mlframe.feature_engineering.transformer.swap_noise import swap_noise_augment


def _make_latent_plus_noise_dataset(n: int, d_info: int, d_noise: int, seed: int):
    """``d_info`` columns driven by a shared 3-dim latent (true signal) + ``d_noise`` columns of pure
    independent noise unrelated to the latent -- the target ``y`` depends only on the latent."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 3))
    W = rng.normal(size=(3, d_info))
    X_info = latent @ W + rng.normal(scale=0.2, size=(n, d_info))
    X_noise = rng.normal(size=(n, d_noise))
    X = np.concatenate([X_info, X_noise], axis=1)
    y = latent.sum(axis=1) + rng.normal(scale=0.1, size=n)
    return X, y


def _fit_dae_variant(X: np.ndarray, *, hidden_size: int, bottleneck_dim: int, max_iter: int, seed: int, swap_prob: float, column_swap_probs=None):
    """Same 4-layer symmetric MLP DAE fit as ``denoising_autoencoder._fit_dae``, but wired to accept the new
    ``column_swap_probs`` opt-in (not yet threaded through the shared ``_fit_dae`` helper)."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    rng = np.random.default_rng(seed)
    X_corrupted = swap_noise_augment(X_s, swap_prob=swap_prob, rng=rng, column_swap_probs=column_swap_probs)

    dae = MLPRegressor(
        hidden_layer_sizes=(hidden_size, bottleneck_dim, hidden_size),
        activation="tanh",
        solver="adam",
        alpha=1e-3,
        learning_rate_init=0.01,
        max_iter=max_iter,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-4,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dae.fit(X_corrupted, X_s)
    return dae, scaler


def test_biz_val_swap_noise_augment_per_column_probs_beat_uniform_for_dae_pretraining():
    """A DAE pretrained with per-column swap rates (heavy corruption on pure-noise columns, light corruption
    on informative columns) should yield a bottleneck more predictive of the downstream target than one
    pretrained with a single uniform rate across all columns -- uniform corruption wastes reconstruction
    capacity on noise columns while under-corrupting (and so under-forcing robust structure learning on) the
    columns that actually carry signal."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    seed = 3
    d_info, d_noise = 5, 5
    X, y = _make_latent_plus_noise_dataset(n=4000, d_info=d_info, d_noise=d_noise, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    bottleneck_dim = 4
    uniform_rate = 0.2
    # Same average corruption budget as uniform (0.05*5 + 0.55*5 = 0.6*5 == 0.2*10*3 roughly matched),
    # concentrated on the noise columns.
    per_col = np.array([0.05] * d_info + [0.55] * d_noise)

    dae_uniform, scaler_uniform = _fit_dae_variant(X_train, hidden_size=24, bottleneck_dim=bottleneck_dim, max_iter=500, seed=seed, swap_prob=uniform_rate)
    dae_percol, scaler_percol = _fit_dae_variant(
        X_train, hidden_size=24, bottleneck_dim=bottleneck_dim, max_iter=500, seed=seed, swap_prob=uniform_rate, column_swap_probs=per_col
    )

    bottleneck_train_uniform = _extract_bottleneck(dae_uniform, scaler_uniform, X_train, bottleneck_dim)
    bottleneck_test_uniform = _extract_bottleneck(dae_uniform, scaler_uniform, X_test, bottleneck_dim)
    bottleneck_train_percol = _extract_bottleneck(dae_percol, scaler_percol, X_train, bottleneck_dim)
    bottleneck_test_percol = _extract_bottleneck(dae_percol, scaler_percol, X_test, bottleneck_dim)

    r2_uniform = Ridge(alpha=1.0).fit(bottleneck_train_uniform, y_train).score(bottleneck_test_uniform, y_test)
    r2_percol = Ridge(alpha=1.0).fit(bottleneck_train_percol, y_train).score(bottleneck_test_percol, y_test)

    assert r2_percol > r2_uniform + 0.05, (
        f"expected per-column swap rates (heavy on noise cols, light on informative cols) to beat uniform "
        f"swap_prob on downstream R2, got r2_percol={r2_percol:.4f} vs r2_uniform={r2_uniform:.4f}"
    )


def test_swap_noise_augment_column_swap_probs_dict_matches_array():
    """Swap noise augment column swap probs dict matches array."""
    X = np.random.default_rng(7).normal(size=(2000, 4))

    out_dict = swap_noise_augment(X, rng=np.random.default_rng(5), column_swap_probs={0: 0.6, 2: 0.6})
    out_array = swap_noise_augment(X, rng=np.random.default_rng(5), column_swap_probs=np.array([0.6, 0.15, 0.6, 0.15]))
    assert np.array_equal(out_dict, out_array)


def test_swap_noise_augment_column_swap_probs_rates_match_targets():
    """Swap noise augment column swap probs rates match targets."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20000, 3))
    probs = np.array([0.05, 0.5, 0.9])
    X_corrupted = swap_noise_augment(X, rng=np.random.default_rng(1), column_swap_probs=probs)
    changed_rate = (X_corrupted != X).mean(axis=0)
    assert np.all(np.abs(changed_rate - probs) < 0.015)


def test_swap_noise_augment_column_swap_probs_out_of_range_raises():
    """Swap noise augment column swap probs out of range raises."""
    X = np.random.default_rng(0).normal(size=(10, 2))
    try:
        swap_noise_augment(X, column_swap_probs=np.array([0.5, 1.5]))
        raise AssertionError("expected ValueError for out-of-range column_swap_probs entry")
    except ValueError:
        pass


def test_swap_noise_augment_column_swap_probs_wrong_length_raises():
    """Swap noise augment column swap probs wrong length raises."""
    X = np.random.default_rng(0).normal(size=(10, 3))
    try:
        swap_noise_augment(X, column_swap_probs=np.array([0.5, 0.5]))
        raise AssertionError("expected ValueError for column_swap_probs length mismatch")
    except ValueError:
        pass


def test_swap_noise_augment_default_behavior_unchanged_bit_identical():
    """Regression guard: not passing ``column_swap_probs`` must reproduce the pre-extension uniform-rate
    code path bit-for-bit, including RNG consumption order."""
    X = np.random.default_rng(42).normal(size=(500, 6))
    out_new = swap_noise_augment(X, swap_prob=0.2, rng=np.random.default_rng(11))

    # Independent re-implementation of the ORIGINAL (pre-extension) algorithm, to compare against.
    rng = np.random.default_rng(11)
    n, d = X.shape
    X_out = np.array(X, copy=True)
    mask = rng.random((n, d)) < 0.2
    for j in range(d):
        col_mask = mask[:, j]
        n_swap = int(col_mask.sum())
        if n_swap == 0:
            continue
        perm = rng.permutation(n)[:n_swap]
        X_out[col_mask, j] = X[perm, j]

    assert np.array_equal(out_new, X_out)
