"""Maximum-Sample-Reuse Banzhaf semivalue estimate over the additive proxy game (gt_03).

Wang & Jia (AISTATS 2023) prove the Banzhaf semivalue is the most noise-robust of the family --
uniform coalition weighting (unlike Shapley's 1/(P*C(P-1,|S|)) size-dependent weights) makes its
ranking the least sensitive to per-coalition value noise, exactly the property a top-K prescreen
RANKING needs when the underlying proxy loss is itself noisy (OOF-SHAP fold variance, booster seed
jitter, finite-sample loss). Maximum Sample Reuse (MSR) estimates beta for ALL features from ONE
shared pool of m sampled coalitions: beta_j = mean_{S ni j} v(S) - mean_{S not ni j} v(S) -- every
sampled coalition informs every feature's estimate, unlike a per-feature sampling scheme.
"""

from __future__ import annotations

import warnings

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_objective import METRIC_CODES, resolve_metric, score_margin_batch


def banzhaf_msr(
    phi: np.ndarray,
    base: np.ndarray,
    y: np.ndarray,
    *,
    classification: bool,
    metric: str | None,
    n_coalitions: int = 4096,
    rng: np.random.Generator,
    batch: int = 256,
) -> tuple[np.ndarray, dict]:
    """MSR-Banzhaf estimate of per-feature semivalues over the additive coalition-margin proxy game.

    Samples ``n_coalitions`` boolean masks over the ``P`` proxy columns (each feature included
    independently w.p. 0.5), scores every sampled coalition's margin ``base + mask @ phi.T`` against
    ``y`` with the resolved metric (AUC excluded -- ``score_margin_auto`` only covers the pointwise
    njit metrics; the prescreen ranking is proxy-loss based like the rest of this module), and
    negates the loss into a value ``v(S) = -loss(S)`` (higher value = better coalition, matching the
    semivalue convention). ``beta_j`` is then the mean value gap between coalitions that DID and did
    NOT include feature ``j`` -- the MSR estimator, reusing every sampled coalition for every feature
    instead of a fresh sample per feature.

    Processes masks in chunks of ``batch`` (not the whole ``(n_coalitions, P)`` matrix -> ``(m, n)``
    margins at once) to bound peak memory: at n_samples=3000 a full (4096, 3000) float64 margins
    matrix is ~98MB, cheap, but this stays O(batch * n_samples) regardless of ``n_coalitions`` so the
    estimator scales to large coalition counts without a memory cliff.

    Returns ``(beta, info)`` where ``beta`` is shape ``(P,)`` and ``info`` holds
    ``n_coalitions``, ``v_mean``, ``v_std`` (over all sampled coalitions) and per-feature
    ``beta_stderr`` (two-sample mean stderr: sqrt(var_in/n_in + var_out/n_out)). Features sampled
    into every / no coalition (guarded against, though unlikely at m>=64, P>=2) get beta=0 and a
    warning is appended to ``info['degenerate_features']``.
    """
    phi = np.ascontiguousarray(phi, dtype=np.float64)
    base = np.ascontiguousarray(base, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    metric = resolve_metric(classification, metric)
    if metric == "auc":
        # score_margin_auto has no AUC kernel (needs a per-subset sort); route AUC callers to the
        # nearest supported pointwise metric rather than silently mis-scoring the coalitions.
        metric = "brier" if classification else "rmse"
    metric_code = METRIC_CODES["rmse" if metric == "mse" else metric]
    is_rmse = metric == "rmse"

    _n_samples, n_features = phi.shape
    phi_T = np.ascontiguousarray(phi.T)  # (P, n_samples)

    m = int(n_coalitions)
    masks = rng.random((m, n_features)) < 0.5  # (m, P) bool

    v = np.empty(m, dtype=np.float64)  # v(S) = -loss(S) for every sampled coalition
    for start in range(0, m, batch):
        end = min(start + batch, m)
        chunk = masks[start:end].astype(np.float64)  # (b, P)
        margins_chunk = np.ascontiguousarray(base[None, :] + chunk @ phi_T)  # (b, n_samples)
        # score_margin_batch scores the whole chunk in ONE compiled call instead of one call per
        # coalition -- njit call-dispatch overhead otherwise dominates at n_coalitions in the
        # thousands (measured: ~0.41s of a 0.55s stage wall was per-row dispatch, not compute).
        losses = score_margin_batch(margins_chunk, y, metric_code)
        if is_rmse:
            losses = np.sqrt(losses)
        v[start:end] = -losses

    beta = np.zeros(n_features, dtype=np.float64)
    beta_stderr = np.zeros(n_features, dtype=np.float64)
    degenerate: list[int] = []
    for j in range(n_features):
        in_mask = masks[:, j]
        n_in = int(in_mask.sum())
        n_out = m - n_in
        if n_in == 0 or n_out == 0:
            degenerate.append(j)
            continue
        v_in = v[in_mask]
        v_out = v[~in_mask]
        beta[j] = float(v_in.mean() - v_out.mean())
        beta_stderr[j] = float(np.sqrt(v_in.var(ddof=1) / n_in + v_out.var(ddof=1) / n_out)) if n_in > 1 and n_out > 1 else 0.0

    if degenerate:
        warnings.warn(
            f"banzhaf_msr: {len(degenerate)} feature(s) never/always sampled at n_coalitions={m} "
            f"(indices {degenerate}); beta forced to 0 for these -- increase n_coalitions if this persists.",
            stacklevel=2,
        )

    info = dict(
        n_coalitions=m,
        v_mean=float(v.mean()),
        v_std=float(v.std(ddof=1)) if m > 1 else 0.0,
        beta_stderr=beta_stderr,
        degenerate_features=degenerate,
    )
    return beta, info
