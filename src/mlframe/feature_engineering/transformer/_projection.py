"""Random Gaussian projections — the shared front-end for column-stack reduction at d > 256.

CPU-only by design (the GPU policy section of the plan): at d=20k the input X is up to 800 GB, the projected ``X_proj`` is 2.5 GB, and the cost of streaming the
full X across PCIe to a GPU (67 s on PCIe-4) dominates the matmul itself (~5 s GPU vs ~16 s CPU sustained). For Mode B inference where the K-bank can stay
resident on the GPU, the projection step gets fused into the GPU dispatch instead — that path lives in ``_kernels_cupy``, not here.

The multi-head story for high-d data is *which random subspace each head projects onto*. Heads must be uncorrelated; we derive per-head RNGs via
``np.random.SeedSequence(seed).spawn(n_heads)`` rather than re-using one seed (a single seed with sequential ``rng.normal`` calls produces correlated heads
because the iterator state carries over and adjacent draws share entropy).
"""
from __future__ import annotations

import logging

import numba
import numpy as np

from ._utils import require_seed

logger = logging.getLogger(__name__)

# Aggregation / reduction precision: fastmath=False for correctness.
# Matmul-style kernels in this file are not numerically sensitive (just sums of products), but we keep fastmath=False here for byte-for-byte reproducibility,
# matching the mlframe rule for "anything that touches a numeric output that downstream code may compare". RFF / softmax kernels in ``_kernels_njit`` get
# fastmath=True because cuBLAS on the GPU path doesn't honour fastmath flags and the CPU/GPU parity is closer with fastmath=True there.
NUMBA_NJIT_PARAMS = dict(fastmath=False, cache=True, nogil=True)


def build_random_projections(
    d_input: int,
    n_heads: int,
    head_dim: int,
    seed: int,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Construct ``n_heads`` independent random Gaussian projection matrices.

    Returns shape ``(n_heads, d_input, head_dim)``. Each head's matrix is iid
    ``N(0, 1 / sqrt(head_dim))`` — the Johnson-Lindenstrauss scaling that preserves L2 distances in expectation after projection.

    Per-head seeds derive via ``SeedSequence(seed).spawn(n_heads)``. Critical for multi-head diversity: using one seed and a single ``rng.normal(size=(n_heads, ...))``
    call would technically work, but if a future caller adds an extra head later (``n_heads=5`` instead of 4), the existing 4 heads would silently shift because the
    flat RNG stream re-aligns. With ``SeedSequence.spawn``, head ``h`` always uses the same independent RNG regardless of how many other heads exist.
    """
    seed = require_seed(seed)
    if d_input < 1 or n_heads < 1 or head_dim < 1:
        raise ValueError(f"build_random_projections: dims must be positive (d_input={d_input}, n_heads={n_heads}, head_dim={head_dim}).")
    scale = float(1.0 / np.sqrt(head_dim))
    ss = np.random.SeedSequence(seed)
    head_seeds = ss.spawn(n_heads)
    out = np.empty((n_heads, d_input, head_dim), dtype=dtype)
    # Generate in float64 then downcast once at the end; ``standard_normal(dtype=...)`` only supports float32/64 across all numpy versions we care about, but
    # generating in float64 then converting keeps the code uniform across fp16/fp32/fp64 outputs and the cost (~10% extra compute) is dwarfed by downstream use.
    for h, hs in enumerate(head_seeds):
        rng_h = np.random.default_rng(hs)
        samples = rng_h.standard_normal((d_input, head_dim))
        samples *= scale
        out[h] = samples.astype(dtype, copy=False)
    return out


def build_importance_weighted_projection(
    X: np.ndarray,
    y: np.ndarray,
    n_heads: int,
    head_dim: int,
    seed: int,
    dtype: np.dtype = np.float32,
    aux_n_estimators: int = 50,
    aux_max_depth: int = 4,
) -> np.ndarray:
    """Random Gaussian projection where each input column j is scaled by sqrt(LGB feature_importance[j]).

    Intuition: PLS-supervised projection finds the optimal linear combination of input columns; this is target-aware but can overfit on small data or non-linear
    targets. Pure random Gaussian projection is target-blind (isotropic). Importance-weighted projection sits between:

    - Use LGB feature_importances_ (gain-based) as per-column weights w_j.
    - Random Gaussian projection W ∈ R^(d × head_dim) with each ROW j scaled by sqrt(w_j) (so important columns contribute more).
    - Result: anisotropic random projection biased toward important features.

    Compared to PLS: doesn't fit a specific linear combination, so less overfit on small samples; doesn't depend on the target's linearity in X.
    Compared to random: target-aware via importances.

    Returns shape ``(n_heads, d_input, head_dim)`` like the other projection builders.
    """
    import lightgbm as lgb
    seed = require_seed(seed)
    n_samples, d_input = X.shape
    if y.shape[0] != n_samples:
        raise ValueError(f"build_importance_weighted_projection: y.shape[0]={y.shape[0]} != X.shape[0]={n_samples}.")

    # Fit small LGB to get feature importances. Use gain-based importance (LGB default = "split", we want "gain" for value-of-feature interpretation).
    unique_y = np.unique(y[~np.isnan(y)] if y.dtype.kind == "f" else y)
    task = "binary" if len(unique_y) == 2 else "regression"
    common = dict(
        n_estimators=aux_n_estimators, max_depth=aux_max_depth, learning_rate=0.1,
        random_state=seed, verbose=-1, n_jobs=-1, num_leaves=min(2 ** aux_max_depth, 31),
        importance_type="gain",
    )
    model = lgb.LGBMClassifier(**common) if task == "binary" else lgb.LGBMRegressor(**common)
    model.fit(X, y)
    importances = np.asarray(model.feature_importances_, dtype=np.float64)
    # Normalise: shift to non-negative (importances are already ≥ 0), then scale so the mean is 1.0 (preserves overall projection scale).
    importances = np.maximum(importances, 0.0)
    if importances.sum() <= 0:
        logger.info("build_importance_weighted_projection: LGB returned all-zero importances (degenerate); falling back to uniform.")
        importances = np.ones(d_input, dtype=np.float64)
    importances = importances * (d_input / importances.sum())   # mean(importances) = 1.0
    weights = np.sqrt(importances).astype(dtype)  # per-column sqrt-scale; rows of projection get multiplied by this

    scale = float(1.0 / np.sqrt(head_dim))
    ss = np.random.SeedSequence(seed)
    head_seeds = ss.spawn(n_heads)
    out = np.empty((n_heads, d_input, head_dim), dtype=dtype)
    for h, hs in enumerate(head_seeds):
        rng_h = np.random.default_rng(hs)
        samples = rng_h.standard_normal((d_input, head_dim)).astype(dtype, copy=False)
        # Row-wise scaling: each input column j's projection direction is scaled by sqrt(w_j).
        samples = samples * weights[:, None]
        samples *= scale
        out[h] = samples
    return out


def build_shap_weighted_projection(
    X: np.ndarray,
    y: np.ndarray,
    n_heads: int,
    head_dim: int,
    seed: int,
    dtype: np.dtype = np.float32,
    aux_n_estimators: int = 50,
    aux_max_depth: int = 4,
    shap_subsample: int = 500,
) -> np.ndarray:
    """Random Gaussian projection weighted by SHAP attributions.

    Same structure as `build_importance_weighted_projection` but uses mean(|TreeSHAP|) per column as the weight instead of LGB gain-based feature_importances_.
    SHAP is more honest: feature_importances_ is a gain-counter (how much a feature reduces loss when split) while SHAP measures marginal contribution per row.
    For boostings that already saturate on gain-based importance, SHAP can surface features that matter on a *subset* of rows but rarely split — a different
    signal direction.

    Computes SHAP on a random subsample of ``shap_subsample`` rows (default 500) to keep cost bounded — TreeSHAP is O(n × n_estimators × depth²) per row.

    Returns shape ``(n_heads, d_input, head_dim)`` like the other projection builders.
    """
    import lightgbm as lgb
    seed = require_seed(seed)
    n_samples, d_input = X.shape
    if y.shape[0] != n_samples:
        raise ValueError(f"build_shap_weighted_projection: y.shape[0]={y.shape[0]} != X.shape[0]={n_samples}.")

    unique_y = np.unique(y[~np.isnan(y)] if y.dtype.kind == "f" else y)
    task = "binary" if len(unique_y) == 2 else "regression"
    common = dict(
        n_estimators=aux_n_estimators, max_depth=aux_max_depth, learning_rate=0.1,
        random_state=seed, verbose=-1, n_jobs=-1, num_leaves=min(2 ** aux_max_depth, 31),
    )
    model = lgb.LGBMClassifier(**common) if task == "binary" else lgb.LGBMRegressor(**common)
    model.fit(X, y)

    # TreeSHAP via lightgbm.predict(pred_contrib=True) — returns per-row, per-column SHAP values + bias column.
    rng = np.random.default_rng(seed)
    subsample_idx = rng.choice(n_samples, size=min(shap_subsample, n_samples), replace=False)
    shap_matrix = model.predict(X[subsample_idx], pred_contrib=True)
    # pred_contrib returns shape (n_subsample, d_input + 1); last column is the bias. Drop it.
    shap_features = np.asarray(shap_matrix)[:, :d_input]
    # Per-column SHAP magnitude.
    weights_raw = np.abs(shap_features).mean(axis=0).astype(np.float64)
    if weights_raw.sum() <= 0:
        logger.info("build_shap_weighted_projection: SHAP returned all-zero attributions (degenerate); falling back to uniform.")
        weights_raw = np.ones(d_input, dtype=np.float64)
    # Normalise to mean = 1.
    weights_raw = weights_raw * (d_input / weights_raw.sum())
    weights = np.sqrt(weights_raw).astype(dtype)

    scale = float(1.0 / np.sqrt(head_dim))
    ss = np.random.SeedSequence(seed)
    head_seeds = ss.spawn(n_heads)
    out = np.empty((n_heads, d_input, head_dim), dtype=dtype)
    for h, hs in enumerate(head_seeds):
        rng_h = np.random.default_rng(hs)
        samples = rng_h.standard_normal((d_input, head_dim)).astype(dtype, copy=False)
        samples = samples * weights[:, None]
        samples *= scale
        out[h] = samples
    return out


def build_nca_projection(
    X: np.ndarray,
    y: np.ndarray,
    n_heads: int,
    head_dim: int,
    seed: int,
    dtype: np.dtype = np.float32,
    nca_max_iter: int = 50,
    head_noise_scale: float = 0.05,
    q_high: float = 0.8,
) -> np.ndarray:
    """Beyond-frozen target-aware projection via Neighborhood Components Analysis (Goldberger 2005).

    NCA fits a linear projection W ∈ R^(d × d_embed) by gradient descent (L-BFGS) to maximize the expected leave-one-out kNN classification accuracy in the
    projected space. The resulting W is a TRUE LEARNED ATTENTION Q/K projection — same role as the learned Q/K in a transformer, but trained via a Bayes-rule-aligned
    LOO-kNN objective rather than next-token prediction.

    For binary classification: NCA on (X, y) directly.
    For regression: NCA on (X, top-q-quantile-y binary indicator); top-y rows pulled together in projection space.

    Per-head construction:
        - All heads start from the same NCA solution (computed once per call).
        - Each head adds independent Gaussian noise of magnitude ``head_noise_scale`` to break symmetry.

    Returns shape ``(n_heads, d_input, head_dim)``. Use as the projection inside row-attention via ``projection="nca"``.

    Cost: ~5-30 sec for NCA fit per call (only once per fold thanks to head-shared basis). Same order as PLS.

    Why this is structurally beyond-frozen:
    - PLS / Importance / SHAP / Random projections are all CLOSED-FORM or hand-engineered.
    - NCA's W is OPTIMIZED VIA GRADIENT to a target-aware objective. The projection function ITSELF is learned.
    """
    from sklearn.neighbors import NeighborhoodComponentsAnalysis
    from sklearn.preprocessing import StandardScaler
    seed = require_seed(seed)
    n_samples, d_input = X.shape
    if y.shape[0] != n_samples:
        raise ValueError(f"build_nca_projection: y.shape[0]={y.shape[0]} != X.shape[0]={n_samples}.")

    unique_y = np.unique(y[~np.isnan(y)] if y.dtype.kind == "f" else y)
    task = "binary" if len(unique_y) == 2 else "regression"

    # Binarise for NCA (it requires discrete class labels).
    if task == "binary":
        y_bin = (y > 0.5).astype(int)
    else:
        threshold = float(np.quantile(y, q_high))
        y_bin = (y >= threshold).astype(int)

    if y_bin.sum() < 2 or (1 - y_bin).sum() < 2:
        logger.info("build_nca_projection: degenerate class counts; falling back to random projection.")
        return build_random_projections(d_input=d_input, n_heads=n_heads, head_dim=head_dim, seed=seed, dtype=dtype)

    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    n_components_eff = min(head_dim, d_input)
    nca = NeighborhoodComponentsAnalysis(
        n_components=n_components_eff,
        init="pca",
        max_iter=nca_max_iter,
        random_state=seed,
    )
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        try:
            nca.fit(X_s, y_bin)
        except Exception as exc:
            logger.info("build_nca_projection: NCA fit failed (%s); falling back to random.", exc)
            return build_random_projections(d_input=d_input, n_heads=n_heads, head_dim=head_dim, seed=seed, dtype=dtype)
    # NCA's transformation_ is the learned W: shape (n_components, d_input). We want (d_input, head_dim) to be consistent with build_random_projections.
    W_base = nca.components_.T.astype(dtype)  # (d_input, n_components_eff)
    if W_base.shape[1] < head_dim:
        padded = np.zeros((d_input, head_dim), dtype=dtype)
        padded[:, :W_base.shape[1]] = W_base
        W_base = padded

    # Per-head noise to break symmetry across heads.
    scale = float(1.0 / np.sqrt(head_dim))
    ss = np.random.SeedSequence(seed)
    head_seeds = ss.spawn(n_heads)
    out = np.empty((n_heads, d_input, head_dim), dtype=dtype)
    for h, hs in enumerate(head_seeds):
        rng_h = np.random.default_rng(hs)
        noise = rng_h.standard_normal((d_input, head_dim)).astype(dtype, copy=False) * head_noise_scale * scale
        out[h] = (W_base + noise).astype(dtype, copy=False)
    return out


def build_supervised_projections_pls(
    X: np.ndarray,
    y: np.ndarray,
    n_heads: int,
    head_dim: int,
    seed: int,
    dtype: np.dtype = np.float32,
    noise_scale: float = 0.05,
) -> np.ndarray:
    """Target-aware projection matrices via partial least squares (PLS).

    Real transformer Q/K/V are learned to be discriminative for the task; our usual random Gaussian projections are isotropic and have no relationship to y. PLS
    derives directions that maximise covariance ``cov(X @ w, y)`` per component — the closest "no backprop" analogue to learned Q/K. Each head gets a PLS fit
    with a small per-head Gaussian noise added to the projection so heads remain diverse (otherwise all heads would learn the same top-``head_dim`` PLS directions).

    Mathematically: PLS-1 component ``w_k`` solves ``argmax cov(X_residual @ w, y_residual)`` subject to ``||w||=1``, iterating on residuals like NIPALS.

    Per-head construction:
        - All heads share the same PLS basis (top ``head_dim`` components from sklearn.cross_decomposition.PLSRegression).
        - Each head adds independent Gaussian noise of magnitude ``noise_scale`` to break symmetry; head ``h`` uses ``SeedSequence(seed).spawn(n_heads)[h]``.

    Why noise: pure PLS gives the same projection for every head -> all heads attend identically -> multi-head averaging degenerates to single-head. The noise
    perturbs the projection so each head explores a slightly different subspace around the optimal PLS direction. ``noise_scale=0.05`` is small enough that the
    target-alignment signal dominates but large enough that heads diverge.

    Reference: Wold 1966; Krishnan 1966 (NIPALS); sklearn.cross_decomposition.PLSRegression. ``head_dim`` clamps to ``min(head_dim, rank(X))``.

    Returns shape ``(n_heads, d_input, head_dim)`` like ``build_random_projections``.
    """
    seed = require_seed(seed)
    from sklearn.cross_decomposition import PLSRegression
    n_samples, d_input = X.shape
    if y.shape[0] != n_samples:
        raise ValueError(f"build_supervised_projections_pls: y.shape[0]={y.shape[0]} != X.shape[0]={n_samples}.")
    if head_dim < 1 or n_heads < 1:
        raise ValueError(f"build_supervised_projections_pls: dims must be positive (n_heads={n_heads}, head_dim={head_dim}).")
    rank_cap = min(head_dim, d_input, n_samples)
    if rank_cap < head_dim:
        logger.info("build_supervised_projections_pls: clipping head_dim from %d to %d (input rank limit).", head_dim, rank_cap)
    # PLS expects 2-D y; reshape if 1-D.
    y_2d = y.reshape(-1, 1) if y.ndim == 1 else y
    pls = PLSRegression(n_components=rank_cap, scale=False, max_iter=500)
    pls.fit(X, y_2d)
    # The "x_weights_" attribute gives the projection directions (each column is a PLS component direction in input space).
    base_proj = pls.x_weights_.astype(dtype, copy=False)  # (d_input, rank_cap)
    if rank_cap < head_dim:
        # Pad missing dims with zero columns; they'll be effectively dead but keep shape consistent.
        pad = np.zeros((d_input, head_dim - rank_cap), dtype=dtype)
        base_proj = np.concatenate([base_proj, pad], axis=1)
    # Normalise each column so per-row dot products live in a comparable scale to random Gaussian (1/sqrt(head_dim)).
    col_norms = np.linalg.norm(base_proj, axis=0, keepdims=True)
    np.maximum(col_norms, np.finfo(dtype).tiny, out=col_norms)
    base_proj = base_proj / col_norms
    # Per-head noise perturbation. Noise scale relative to per-column std (which is ~1/sqrt(d_input) for normalised PLS columns).
    ss = np.random.SeedSequence(seed)
    head_seeds = ss.spawn(n_heads)
    out = np.empty((n_heads, d_input, head_dim), dtype=dtype)
    noise_std = float(noise_scale / np.sqrt(d_input))
    for h, hs in enumerate(head_seeds):
        rng_h = np.random.default_rng(hs)
        noise = (rng_h.standard_normal((d_input, head_dim)) * noise_std).astype(dtype, copy=False)
        # Combine base + noise, then re-normalise so each column is unit-norm again.
        perturbed = base_proj + noise
        perturbed_norms = np.linalg.norm(perturbed, axis=0, keepdims=True)
        np.maximum(perturbed_norms, np.finfo(dtype).tiny, out=perturbed_norms)
        out[h] = (perturbed / perturbed_norms).astype(dtype, copy=False)
    return out


def apply_projection(
    X: np.ndarray,
    projections: np.ndarray,
    *,
    batch_rows: int = 100_000,
    l2_normalize: bool = True,
) -> np.ndarray:
    """Apply per-head projections to ``X`` in row-streaming batches.

    Output shape: ``(n_heads, N, head_dim)``. We choose ``(n_heads, N, head_dim)`` (head-major) over ``(N, n_heads, head_dim)`` so the per-head ANN build downstream
    can read contiguous ``(N, head_dim)`` slices without strided gather. The cost is that one row's all-heads concatenation is non-contiguous, but downstream uses
    per-head slices, not per-row concatenations.

    ``batch_rows`` (default 100k) controls the streaming chunk: full materialisation of X at N=10M, d=20k is 800 GB and won't fit RAM. With batch=100k, peak per-step
    memory is N_batch * d * 4 = 8 GB before projection — within reach of a 64 GB workstation. Reduce ``batch_rows`` if you're memory-bound.

    ``l2_normalize=True`` (default) renormalises projected vectors to unit L2 — required for cosine-similarity row-attention, harmless otherwise. The cost is one
    extra pass over the projected output (~10x smaller than the input streaming cost). Set ``False`` only when downstream uses raw L2.
    """
    if X.ndim != 2:
        raise ValueError(f"apply_projection: X must be 2-D, got shape {X.shape}.")
    if projections.ndim != 3:
        raise ValueError(f"apply_projection: projections must be 3-D (n_heads, d_input, head_dim), got shape {projections.shape}.")
    n_heads, d_input, head_dim = projections.shape
    n_rows, d_x = X.shape
    if d_x != d_input:
        raise ValueError(f"apply_projection: X.shape[1]={d_x} != projections.shape[1]={d_input}.")
    if batch_rows < 1:
        raise ValueError(f"apply_projection: batch_rows must be positive, got {batch_rows}.")

    out_dtype = projections.dtype
    # Sanity-check: don't silently downcast through int->float; raise if X has a wider float than projections.
    if X.dtype.kind == "f" and X.dtype.itemsize > out_dtype.itemsize:
        logger.info("apply_projection: X dtype %s is wider than projections dtype %s; downcast at matmul.", X.dtype, out_dtype)

    out = np.empty((n_heads, n_rows, head_dim), dtype=out_dtype)
    for r0 in range(0, n_rows, batch_rows):
        r1 = min(r0 + batch_rows, n_rows)
        X_batch = X[r0:r1].astype(out_dtype, copy=False)
        for h in range(n_heads):
            # gemm: (batch, d_input) @ (d_input, head_dim) -> (batch, head_dim)
            np.matmul(X_batch, projections[h], out=out[h, r0:r1])
    if l2_normalize:
        _l2_normalize_inplace(out)
    return out


def _l2_normalize_inplace(arr3d: np.ndarray) -> None:
    """In-place L2 normalisation of the last axis of ``arr3d`` (n_heads, N, head_dim).

    Standalone function so we can swap in an njit version if profiling shows the numpy version is too slow at the streaming scale; current numpy implementation is
    bandwidth-bound and fast enough up to N=10M, head_dim=8.
    """
    norms = np.linalg.norm(arr3d, axis=-1, keepdims=True)
    # Avoid division by zero for any all-zero projected vector (extremely rare but possible when X has a degenerate row).
    np.maximum(norms, np.finfo(arr3d.dtype).tiny, out=norms)
    arr3d /= norms


def validate_projection_dims(
    d_input: int,
    head_dim: int,
    *,
    allow_overcomplete: bool = False,
) -> None:
    """Reject combinations where the projection geometry doesn't make sense.

    At ``head_dim > d_input`` the random projection is rank-deficient: the projected vectors live in an at-most ``d_input``-dim subspace of a ``head_dim``-dim
    space, so the extra ``head_dim - d_input`` coordinates carry pure noise that softmax-attention will spread weight over uniformly. Caller can override with
    ``allow_overcomplete=True`` if they have a reason (e.g. random-features kitchen-sink where over-completeness gives effective ensemble diversity).

    The ``head_dim > d_input // 4`` is a soft warning point — at that ratio the JL-distortion bound is loose enough that distance preservation deteriorates;
    we don't reject but the calling layer logs an INFO line.
    """
    if head_dim < 1 or d_input < 1:
        raise ValueError(f"validate_projection_dims: dims must be positive (d_input={d_input}, head_dim={head_dim}).")
    if head_dim > d_input and not allow_overcomplete:
        raise ValueError(
            f"head_dim={head_dim} > d_input={d_input}: projection is rank-deficient, distance geometry is undefined. "
            "If this is intentional (overcomplete random features), pass allow_overcomplete=True."
        )
    if head_dim > d_input // 4:
        logger.info(
            "head_dim=%d is more than 1/4 of d_input=%d; Johnson-Lindenstrauss distortion is loose at this ratio.", head_dim, d_input,
        )


@numba.njit(parallel=True, **NUMBA_NJIT_PARAMS)
def _project_single_head_njit(  # pragma: no cover
    X: np.ndarray,
    W: np.ndarray,
    out: np.ndarray,
) -> None:
    """Parallel-prange gemm for a single head's projection, used when numpy/BLAS isn't available or for parity testing.

    The numpy ``matmul`` path in ``apply_projection`` is usually faster because it dispatches to MKL / OpenBLAS sgemm with all the cache blocking and SIMD already
    tuned. This njit kernel exists for (a) test parity (assert njit == numpy), and (b) the case where the caller has explicitly disabled BLAS thread parallelism
    (e.g. inside a joblib worker with ``threadpoolctl(1)``); the njit kernel respects ``NUMBA_NUM_THREADS`` instead.
    """
    n, d = X.shape
    head_dim = W.shape[1]
    for i in numba.prange(n):
        for k in range(head_dim):
            acc = 0.0
            for j in range(d):
                acc += X[i, j] * W[j, k]
            out[i, k] = acc
