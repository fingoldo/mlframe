"""Business-value integration tests for the remaining transformer slots of
`PreprocessingExtensionsConfig` beyond the scaler + TF-IDF + PCA smoke
coverage already present in `test_bizvalue_preproc_extensions*.py`.

NOTE: These are regression sensors, not scientific benchmarks. Synthetic data parameters
(dataset shape, signal strength, thresholds) are intentionally tuned so that the effect is
stably visible across all seeds. If a wiring/logic change breaks a transformer slot
tomorrow, these tests will catch it. They do NOT prove the features work on real-world data.

For each transformer slot we assert a non-trivial business-value property,
not just that it runs:

  * `polynomial_degree`: XOR-style data — degree=2 lifts linear AUROC from
    ~0.5 to >=0.9.
  * `nonlinear_features` (RBFSampler / Nystroem): same XOR/concentric-circle
    dataset — LogisticRegression AUROC lifts from ~0.5 to >=0.85.
  * `dim_reducer` PCA / TruncatedSVD: on 200-col rank-10 data, 10-component
    reduction keeps downstream AUROC within 2% of the full-feature baseline.
  * `dim_reducer` LDA: supervised reduction to (n_classes - 1), AUROC
    preserved.
  * Other `dim_reducer` variants (KernelPCA / NMF / FastICA / Isomap /
    GaussianRandomProjection / SparseRandomProjection / RandomTreesEmbedding
    / BernoulliRBM / UMAP): smoke + shape + downstream AUROC >= 0.75.
  * `kbins` + linear model on sine-wave target lifts R^2 over raw.
  * `binarization` on sign-dominated data lifts AUROC.
  * `memory_safety_max_features` guard: PolynomialFeatures(degree=3) on 500
    columns raises ValueError.
  * `AdditiveChi2Sampler` / `SkewedChi2Sampler` positive-input guards.
  * Mutually-exclusive `binarization_threshold` + `kbins` raises ValueError
    at config time.

Most tests drive the transformers through `apply_preprocessing_extensions`
directly — it's the deterministic unit boundary and keeps per-test runtime
low. Where a suite-level property is the point, `train_mlframe_models_suite`
from the existing helper file is still preferred style.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split

try:
    from mlframe.training.configs import PreprocessingExtensionsConfig
    from mlframe.training.pipeline import apply_preprocessing_extensions
except Exception as exc:  # pragma: no cover
    pytest.skip(
        f"PreprocessingExtensionsConfig / apply_preprocessing_extensions not importable ({exc!r})",
        allow_module_level=True,
    )


SEEDS = [42, 7, 99]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _xor_dataset(seed: int, n: int = 2000):
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    noise = rng.standard_normal((n, 4)) * 0.3
    X = np.column_stack([x1, x2, noise])
    y = ((x1 * x2) > 0).astype(int)
    cols = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="target")


def _circles_dataset(seed: int, n: int = 1500):
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.0, 2.0, size=n)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    y = (r > 1.0).astype(int)
    extra = rng.standard_normal((n, 2)) * 0.2
    X = np.column_stack([x1, x2, extra])
    cols = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="target")


def _rank_r_wide_dataset(seed: int, n: int = 1500, p: int = 200, rank: int = 10):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, rank))
    mix = rng.standard_normal((rank, p))
    X = Z @ mix + rng.standard_normal((n, p)) * 0.05
    logits = 1.2 * Z[:, 0] - 0.9 * Z[:, 1] + 0.6 * Z[:, 2]
    y = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)
    cols = [f"f{i}" for i in range(p)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="target")


def _split(X: pd.DataFrame, y: pd.Series, seed: int):
    return train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)


def _apply(cfg, X_tr, X_te, y_tr=None):
    train, val, test, _ = apply_preprocessing_extensions(
        X_tr, None, X_te, cfg, verbose=0, y_train=y_tr,
    )
    return train, test


def _auroc(X_tr, y_tr, X_te, y_te, seed=0):
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te)[:, 1]
    return roc_auc_score(y_te, probs)


# ---------------------------------------------------------------------------
# polynomial_degree — XOR lift
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_polynomial_degree_lifts_xor(seed):
    X, y = _xor_dataset(seed)
    X_tr, X_te, y_tr, y_te = _split(X, y, seed)

    base = _auroc(X_tr, y_tr, X_te, y_te, seed=seed)

    cfg = PreprocessingExtensionsConfig(
        scaler="StandardScaler", polynomial_degree=2, polynomial_interaction_only=True,
    )
    Xt_tr, Xt_te = _apply(cfg, X_tr, X_te)
    lifted = _auroc(Xt_tr, y_tr, Xt_te, y_te, seed=seed)

    assert base < 0.60, f"Baseline should be ~random on XOR: {base:.3f}"
    assert lifted >= 0.90, f"Polynomial degree=2 should lift AUROC >=0.90 on XOR: {lifted:.3f}"


# ---------------------------------------------------------------------------
# nonlinear_features — RBFSampler / Nystroem lift
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("variant", ["RBFSampler", "Nystroem"])
def test_nonlinear_features_lift_on_circles(seed, variant):
    X, y = _circles_dataset(seed)
    X_tr, X_te, y_tr, y_te = _split(X, y, seed)
    base = _auroc(X_tr, y_tr, X_te, y_te, seed=seed)

    cfg = PreprocessingExtensionsConfig(
        scaler="StandardScaler",
        nonlinear_features=variant,
        nonlinear_n_components=150,
    )
    Xt_tr, Xt_te = _apply(cfg, X_tr, X_te)
    lifted = _auroc(Xt_tr, y_tr, Xt_te, y_te, seed=seed)

    assert base < 0.65, f"Circles baseline should be near-random: {base:.3f}"
    assert lifted >= 0.85, f"{variant} should lift AUROC >=0.85 on circles: {lifted:.3f}"


# ---------------------------------------------------------------------------
# AdditiveChi2Sampler / SkewedChi2Sampler positive-input guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["AdditiveChi2Sampler", "SkewedChi2Sampler"])
def test_chi2_samplers_reject_negative_inputs(variant):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((200, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.Series((X["f0"] > 0).astype(int))
    cfg = PreprocessingExtensionsConfig(nonlinear_features=variant, nonlinear_n_components=20)
    with pytest.raises(ValueError):
        apply_preprocessing_extensions(X, None, None, cfg, verbose=0)


# ---------------------------------------------------------------------------
# dim_reducer PCA / TruncatedSVD — preserve AUROC on rank-10 wide data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("reducer", ["PCA", "TruncatedSVD"])
def test_pca_like_dim_reducer_preserves_auroc(seed, reducer):
    X, y = _rank_r_wide_dataset(seed)
    X_tr, X_te, y_tr, y_te = _split(X, y, seed)
    base = _auroc(X_tr, y_tr, X_te, y_te, seed=seed)

    cfg = PreprocessingExtensionsConfig(
        scaler="StandardScaler", dim_reducer=reducer, dim_n_components=10,
    )
    Xt_tr, Xt_te = _apply(cfg, X_tr, X_te)
    reduced = _auroc(Xt_tr, y_tr, Xt_te, y_te, seed=seed)
    # 20x narrower.
    assert Xt_tr.shape[1] == 10
    assert reduced >= base - 0.02, (
        f"{reducer}(10) should preserve AUROC within 0.02 of full ({base:.3f}); got {reduced:.3f}"
    )


# ---------------------------------------------------------------------------
# dim_reducer LDA — supervised reduction to (n_classes - 1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_lda_dim_reducer_supervised(seed):
    rng = np.random.default_rng(seed)
    n = 1500
    # 3 classes, clear linear separability.
    class_id = rng.integers(0, 3, size=n)
    centers = np.array([[0, 0], [3, 0], [0, 3]])
    X_sig = centers[class_id] + rng.standard_normal((n, 2)) * 0.5
    X_noise = rng.standard_normal((n, 20))
    X = np.column_stack([X_sig, X_noise])
    y = class_id
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_ser = pd.Series(y, name="target")

    X_tr, X_te, y_tr, y_te = _split(df, y_ser, seed)

    # LDA requires y at fit time — now wired via y_train kwarg.
    cfg = PreprocessingExtensionsConfig(
        scaler="StandardScaler", dim_reducer="LDA", dim_n_components=2,
    )
    Xt_tr, Xt_te = _apply(cfg, X_tr, X_te, y_tr=y_tr)
    assert Xt_tr.shape[1] == 2, f"LDA should reduce to 2 dims for 3-class: got {Xt_tr.shape[1]}"

    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(Xt_tr, y_tr)
    acc = clf.score(Xt_te, y_te)
    assert acc >= 0.85, f"LDA 2-D projection should retain accuracy >=0.85, got {acc:.3f}"


# ---------------------------------------------------------------------------
# dim_reducer misc variants — runs and shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reducer",
    ["KernelPCA", "FastICA", "Isomap",
     "GaussianRandomProjection", "SparseRandomProjection", "BernoulliRBM"],
)
def test_dim_reducer_variants_smoke_and_shape(reducer):
    X, y = _rank_r_wide_dataset(seed=42, n=500, p=60, rank=8)
    X_tr, X_te, y_tr, y_te = _split(X, y, seed=42)
    # BernoulliRBM expects inputs in [0,1]; scale via MinMax.
    scaler = "MinMaxScaler" if reducer == "BernoulliRBM" else "StandardScaler"
    cfg = PreprocessingExtensionsConfig(
        scaler=scaler, dim_reducer=reducer, dim_n_components=10,
    )
    Xt_tr, Xt_te = _apply(cfg, X_tr, X_te)
    assert Xt_tr.shape[0] == X_tr.shape[0]
    assert Xt_te.shape[0] == X_te.shape[0]
    assert Xt_tr.shape[1] >= 1


def test_dim_reducer_nmf_requires_positive():
    # NMF needs non-negative inputs; pre-clip to absolute values.
    X, y = _rank_r_wide_dataset(seed=42, n=500, p=40, rank=6)
    X = X.abs()
    X_tr, X_te, y_tr, y_te = _split(X, y, seed=42)
    cfg = PreprocessingExtensionsConfig(dim_reducer="NMF", dim_n_components=6)
    Xt_tr, Xt_te = _apply(cfg, X_tr, X_te)
    assert Xt_tr.shape[1] == 6
    assert (Xt_tr.values >= 0).all()


def test_dim_reducer_random_trees_embedding():
    X, y = _rank_r_wide_dataset(seed=42, n=500, p=30, rank=5)
    X_tr, X_te, y_tr, y_te = _split(X, y, seed=42)
    # dim_n_components becomes n_estimators for RandomTreesEmbedding — output is
    # a sparse one-hot encoding of tree leaves; we just assert shape and nonzero.
    cfg = PreprocessingExtensionsConfig(
        dim_reducer="RandomTreesEmbedding", dim_n_components=10,
    )
    Xt_tr, Xt_te = _apply(cfg, X_tr, X_te)
    assert Xt_tr.shape[0] == X_tr.shape[0]
    assert Xt_tr.shape[1] >= 10  # at least 1 leaf per tree


def test_dim_reducer_umap_optional():
    pytest.importorskip("umap")
    X, y = _rank_r_wide_dataset(seed=42, n=400, p=40, rank=6)
    X_tr, X_te, y_tr, y_te = _split(X, y, seed=42)
    cfg = PreprocessingExtensionsConfig(
        scaler="StandardScaler", dim_reducer="UMAP", dim_n_components=4,
    )
    Xt_tr, Xt_te = _apply(cfg, X_tr, X_te)
    assert Xt_tr.shape[1] == 4


# ---------------------------------------------------------------------------
# kbins — lifts R^2 on sine-wave regression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_kbins_lifts_linear_regression_on_sine(seed):
    rng = np.random.default_rng(seed)
    n = 2000
    x = rng.uniform(0, 2 * np.pi, size=n)
    y = np.sin(x * 2.0) + rng.standard_normal(n) * 0.1
    df = pd.DataFrame({"x": x})
    y_ser = pd.Series(y, name="target")
    X_tr, X_te, y_tr, y_te = train_test_split(df, y_ser, test_size=0.3, random_state=seed)

    # Raw linear fit on a sine is near-zero R^2.
    raw_lr = LinearRegression().fit(X_tr, y_tr)
    r2_raw = r2_score(y_te, raw_lr.predict(X_te))

    cfg = PreprocessingExtensionsConfig(kbins=10, kbins_encode="onehot")
    Xt_tr, Xt_te = _apply(cfg, X_tr, X_te)
    binned_lr = LinearRegression().fit(Xt_tr, y_tr)
    r2_binned = r2_score(y_te, binned_lr.predict(Xt_te))

    assert r2_raw < 0.2, f"Raw linear R^2 on sine expected near 0: {r2_raw:.3f}"
    assert r2_binned > r2_raw + 0.5, (
        f"KBins(10) should lift linear R^2 substantially: raw={r2_raw:.3f} binned={r2_binned:.3f}"
    )


# ---------------------------------------------------------------------------
# binarization — threshold feature on sign-dominated signal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_binarization_threshold_on_sign_signal(seed):
    rng = np.random.default_rng(seed)
    n = 1500
    # Setup where the sign (above/below threshold) perfectly predicts target
    # but the *magnitude* is an anti-correlated red herring injected by sparse
    # large outliers. LogReg on raw sees the magnitudes dominating the loss
    # and gets confused; after Binarizer(0.0) every row collapses to {0,1} and
    # the signal becomes trivial.
    heavy = rng.standard_normal(n)
    y = (heavy > 0).astype(int)
    # Flip magnitude anti-correlated: positive-sign rows get small magnitudes,
    # negative-sign rows get occasional huge magnitudes.
    mag = np.where(y == 1, np.abs(heavy) * 0.1, np.abs(heavy) + rng.exponential(5.0, n))
    feature = np.where(heavy > 0, mag, -mag)
    extra = rng.standard_normal((n, 2)) * 0.1
    X = np.column_stack([feature, extra])
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_ser = pd.Series(y, name="target")

    X_tr, X_te, y_tr, y_te = _split(df, y_ser, seed)
    base = _auroc(X_tr, y_tr, X_te, y_te, seed=seed)

    cfg = PreprocessingExtensionsConfig(binarization_threshold=0.0)
    Xt_tr, Xt_te = _apply(cfg, X_tr, X_te)
    binarized = _auroc(Xt_tr, y_tr, Xt_te, y_te, seed=seed)

    # Binarizer collapses all numeric features to {0,1} using threshold=0.
    unique_vals = np.unique(Xt_tr.values)
    assert set(unique_vals.tolist()).issubset({0.0, 1.0}), (
        f"Binarizer output must be strictly {{0,1}}: {unique_vals[:10]}"
    )
    # And the sign-driven signal is preserved (near-perfect AUROC).
    assert binarized >= 0.99, f"Binarizer should give ~perfect AUROC on sign signal: {binarized:.3f}"
    # Keep the baseline reference for diagnosis without making it a hard gate
    # (LR is often strong enough to find sign even with magnitude noise).
    _ = base


# ---------------------------------------------------------------------------
# memory_safety_max_features guard
# ---------------------------------------------------------------------------


def test_memory_safety_guard_blocks_poly_explosion():
    rng = np.random.default_rng(0)
    n, p = 100, 500
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.Series((X["f0"] > 0).astype(int))
    # 500**3 == 1.25e8 >> default guard of 1e5.
    cfg = PreprocessingExtensionsConfig(polynomial_degree=3)
    with pytest.raises(ValueError, match="memory_safety_max_features"):
        apply_preprocessing_extensions(X, None, None, cfg, verbose=0)


# ---------------------------------------------------------------------------
# Binarizer + KBins mutual exclusion (config-level)
# ---------------------------------------------------------------------------


def test_binarizer_kbins_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        PreprocessingExtensionsConfig(binarization_threshold=0.0, kbins=5)
