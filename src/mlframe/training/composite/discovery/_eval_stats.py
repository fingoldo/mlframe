"""Statistical + de-duplication helpers for the per-transform gain evaluator.

Carved out of ``_eval`` / ``_fit`` so those files stay under the monolith
threshold. The concerns living here:

* :func:`near_collinear_keep_mask` -- which feature columns to KEEP when a base's
  ``x_remaining`` is de-duplicated before the MI baseline.
* :func:`bootstrap_gain_p_value` -- a one-sided bootstrap p-value for the null
  ``mi_gain <= 0`` from the already-computed bootstrap gain replicates, used as
  the per-spec input to family-wise FDR control.
* :func:`benjamini_hochberg_reject` / :func:`apply_fdr_control_to_candidates` --
  the Benjamini-Hochberg step-up procedure and its application over the whole
  candidate family at a target FDR level.
* :func:`apply_alpha_drift_gate` -- the rolling-origin alpha-drift Chow test for
  ``linear_residual`` specs, lifted from ``_fit`` to keep that file small.

The pure numpy helpers (mask / p-value / BH) are safe to call from the threading
pool that drives candidate evaluation; the two ``apply_*`` orchestrators run
serially in ``fit`` after the candidates are collected.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


def apply_fdr_control_to_candidates(
    candidates: list[dict[str, Any]], *, alpha: float,
) -> int:
    """Mark candidate entries that family-wise FDR control rejects.

    ``candidates`` is the list of per-spec entry dicts ``eval_one_transform``
    returns; each carries a ``bootstrap_p_value`` (NaN when no bootstrap ran).
    A per-spec bootstrap CI controls only that spec's OWN error rate, so testing
    a whole family of (base, transform) specs in one sweep inflates the chance
    that a noise spec spuriously "beats baseline". This runs Benjamini-Hochberg
    over the per-spec one-sided p-values (H0 ``mi_gain <= 0``) and stamps
    ``fdr_dropped=True`` + a ``reason`` on every spec BH does NOT reject at the
    target family FDR ``alpha``; the caller then skips those before the eps gate.

    No-op under the shipped defaults: with the bootstrap disabled every p-value
    is NaN, so the finite-p set is empty and nothing is dropped. Returns the
    number of specs dropped so the caller can log the family-wise effect.
    """
    scored = [
        e for e in candidates
        if e.get("spec") is not None
        and np.isfinite(e.get("bootstrap_p_value", float("nan")))
    ]
    if not scored:
        return 0
    p_values = np.array(
        [float(e["bootstrap_p_value"]) for e in scored], dtype=np.float64,
    )
    reject = benjamini_hochberg_reject(p_values, alpha)
    n_dropped = 0
    for entry, is_rejected in zip(scored, reject):
        if not is_rejected:
            entry["fdr_dropped"] = True
            entry["reason"] = (
                f"BH-FDR: bootstrap p={float(entry['bootstrap_p_value']):.4g} "
                f"not significant at family FDR alpha={alpha:.3f} "
                f"({len(scored)} specs tested)"
            )
            n_dropped += 1
    if n_dropped:
        logger.info(
            "[CompositeTargetDiscovery] BH-FDR dropped %d/%d bootstrapped specs "
            "as not significant at family alpha=%.3f.",
            n_dropped, len(scored), alpha,
        )
    return n_dropped


def near_collinear_keep_mask(
    feature_matrix: np.ndarray, *, corr_threshold: float,
) -> np.ndarray:
    """Boolean keep-mask dropping near-duplicate columns of ``feature_matrix``.

    Walks columns left to right; a column is DROPPED when its absolute Pearson
    correlation with any already-kept column exceeds ``corr_threshold``. The
    FIRST column of every collinear group is always kept, so the result is
    deterministic and order-stable (matching the column order the caller built).
    A near-duplicate sibling of a removed base left in ``x_remaining`` carries
    almost the base's information and inflates ``MI(y, x_remaining)`` without
    helping ``MI(T, x_remaining)`` -- dropping it de-biases ``mi_gain``.

    Correlation uses per-pair finite masking (rows where BOTH columns are
    finite); a pair with fewer than 3 jointly-finite rows is treated as
    uncorrelated (kept) since the estimate is meaningless. A constant column has
    zero variance and never correlates with anything, so it is always kept.

    Returns a length-``n_cols`` boolean array; an empty / 1-column / degenerate
    matrix returns an all-True mask (nothing to dedup).

    The ``O(B^2)`` pair walk dispatches to a numba kernel for large inputs (see
    ``_collinear_numba.near_collinear_keep_mask_fast``); tiny inputs stay on the
    numpy reference below. The dispatcher is bit-identical to this reference
    (borderline pairs are re-decided with the exact numpy primitives), so the
    public contract is unchanged.
    """
    from ._collinear_numba import near_collinear_keep_mask_fast

    return near_collinear_keep_mask_fast(
        feature_matrix,
        corr_threshold=corr_threshold,
        reference_fn=_near_collinear_keep_mask_numpy,
    )


def _near_collinear_keep_mask_numpy(
    feature_matrix: np.ndarray, *, corr_threshold: float,
) -> np.ndarray:
    """Pure-numpy reference walk for :func:`near_collinear_keep_mask`.

    Kept as the correctness baseline AND the small-input / no-numba fast path;
    the numba dispatcher must stay bit-identical to this implementation.
    """
    if feature_matrix.ndim != 2:
        raise ValueError("near_collinear_keep_mask expects a 2-D matrix")
    n_cols = feature_matrix.shape[1]
    keep = np.ones(n_cols, dtype=bool)
    if n_cols < 2 or feature_matrix.shape[0] < 3:
        return keep
    thr = float(corr_threshold)
    if not (thr < 1.0):  # threshold >= 1.0 disables dedup (no pair can exceed it).
        return keep
    # Precompute per-column finite masks + centred values lazily; the kept set is
    # small in practice (a handful of bases), so the O(kept * n_cols) pairwise
    # walk is cheap relative to the MI compute it protects.
    finite = np.isfinite(feature_matrix)
    kept_idx: list[int] = []
    for j in range(n_cols):
        col_j = feature_matrix[:, j]
        fin_j = finite[:, j]
        drop = False
        for k in kept_idx:
            col_k = feature_matrix[:, k]
            pair = fin_j & finite[:, k]
            n_pair = int(pair.sum())
            if n_pair < 3:
                continue
            a = col_j[pair]
            b = col_k[pair]
            a_dev = a - a.mean()
            b_dev = b - b.mean()
            va = float(np.dot(a_dev, a_dev))
            vb = float(np.dot(b_dev, b_dev))
            if va < 1e-24 or vb < 1e-24:
                continue  # constant on the joint support -> no correlation.
            corr = abs(float(np.dot(a_dev, b_dev)) / np.sqrt(va * vb))
            if corr > thr:
                drop = True
                break
        if drop:
            keep[j] = False
        else:
            kept_idx.append(j)
    return keep


def bootstrap_gain_p_value(boot_gains: np.ndarray) -> float:
    """One-sided bootstrap p-value for H0: ``mi_gain <= 0`` from gain replicates.

    ``boot_gains`` is the per-replicate ``mi_t_b - mi_y_b`` array (possibly with
    NaN for failed replicates). The p-value is the bootstrap mass at or below
    zero: ``(#{g <= 0} + 1) / (n_finite + 1)`` -- the add-one (Davison-Hinkley)
    smoothing keeps the p-value strictly positive so a spec with zero replicates
    below zero is not assigned an impossible p=0 that the BH step would over-trust.
    Returns 1.0 (most conservative) when no finite replicate exists, so a spec
    whose bootstrap entirely failed never spuriously survives FDR control.
    """
    g = np.asarray(boot_gains, dtype=np.float64)
    g = g[np.isfinite(g)]
    n = g.size
    if n == 0:
        return 1.0
    n_le0 = int(np.count_nonzero(g <= 0.0))
    return (n_le0 + 1.0) / (n + 1.0)


def benjamini_hochberg_reject(p_values, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg step-up: which of ``p_values`` are rejected at ``alpha``.

    Standard BH (1995): sort the m p-values ascending, find the largest rank k
    with ``p_(k) <= (k/m) * alpha``, and reject every hypothesis with
    ``p_(i) <= p_(k)``. Returns a boolean array aligned to the INPUT order
    (True = reject H0 = the spec's gain is judged real at FDR ``alpha``).
    NaN p-values are treated as non-rejectable (kept as False). An empty input
    returns an empty array.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    out = np.zeros(m, dtype=bool)
    if m == 0:
        return out
    a = float(alpha)
    finite = np.isfinite(p)
    n_test = int(finite.sum())
    if n_test == 0:
        return out
    idx_finite = np.flatnonzero(finite)
    p_fin = p[idx_finite]
    order = np.argsort(p_fin, kind="stable")
    ranks = np.arange(1, n_test + 1, dtype=np.float64)
    thresh = ranks * a / n_test
    sorted_p = p_fin[order]
    below = sorted_p <= thresh
    if not below.any():
        return out
    k = int(np.flatnonzero(below).max())  # largest rank index satisfying BH.
    cutoff = sorted_p[k]
    out[idx_finite] = p_fin <= cutoff
    return out


def apply_alpha_drift_gate(
    self,
    kept_specs: list,
    *,
    df,
    train_idx: np.ndarray,
    y_full: np.ndarray,
    extract_column_array: Callable,
    linear_residual_fit: Callable,
) -> list:
    """Rolling-origin alpha-drift Chow test for ``linear_residual`` specs.

    Fits the OLS slope on the first and second halves of train, computes a
    Chow-style z-score on ``|alpha_1 - alpha_2|`` using a residual-based pooled
    SE, and records it in ``self._alpha_drift_flags``. When
    ``reject_on_alpha_drift`` is set, specs whose z exceeds
    ``alpha_drift_z_threshold`` are dropped. Returns the (possibly filtered)
    ``kept_specs`` list; a no-op (returns the input unchanged) when the feature
    is disabled, no ``linear_residual`` spec survived, or train is too small.
    """
    if not (getattr(self.config, "detect_linear_residual_alpha_drift", True)
            and any(s.transform_name == "linear_residual" for s in kept_specs)):
        return kept_specs
    self._alpha_drift_flags: dict[str, dict[str, float]] = {}
    drift_threshold = float(getattr(self.config, "alpha_drift_z_threshold", 3.0))
    reject_on_drift = bool(getattr(self.config, "reject_on_alpha_drift", False))
    half = len(train_idx) // 2
    if half < 50:
        return kept_specs
    drift_dropped: list[tuple[str, float]] = []
    drift_kept: list = []
    y_train_for_drift = y_full[train_idx]
    for s in kept_specs:
        if s.transform_name != "linear_residual":
            drift_kept.append(s)
            continue
        # ``self._auto_base_pool[base]`` already holds ``base_full[train_idx]`` (set during per-base setup); ``pool[:half]/pool[half:]/pool`` are bit-identical to re-extracting the column and indexing it.
        base_pool = self._auto_base_pool.get(s.base_column)
        if base_pool is not None:
            base_t = base_pool
            base_h1 = base_pool[:half]
            base_h2 = base_pool[half:]
        else:
            base_full = extract_column_array(df, s.base_column)
            base_t = base_full[train_idx]
            base_h1 = base_full[train_idx[:half]]
            base_h2 = base_full[train_idx[half:]]
        try:
            # Batched closed-form: the two half-fits are independent single-base
            # OLS systems; solving them in one pass via
            # _linear_residual_fit_batched pays the per-call dispatch ONCE
            # instead of two lstsq/SVD launches. Bit-identical to applying the
            # scalar closed-form per half (see the batched solver's contract).
            from ..transforms.linear import _linear_residual_fit_batched
            _alphas, _betas = _linear_residual_fit_batched(
                [np.asarray(base_h1), np.asarray(base_h2)],
                [np.asarray(y_train_for_drift[:half]),
                 np.asarray(y_train_for_drift[half:])],
            )
            # A non-finite base/y half makes the closed-form sums NaN; mirror the
            # legacy lstsq-raises-on-NaN behaviour by skipping the spec.
            if not (np.isfinite(_alphas).all() and np.isfinite(_betas).all()):
                drift_kept.append(s)
                continue
            params1 = {"alpha": float(_alphas[0]), "beta": float(_betas[0])}
            params2 = {"alpha": float(_alphas[1]), "beta": float(_betas[1])}
        except Exception:
            drift_kept.append(s)
            continue
        a1 = float(params1.get("alpha", 0.0))
        a2 = float(params2.get("alpha", 0.0))
        # Residual-based OLS slope SE: SE(alpha) = sqrt(SSE/(n-2)) / (sqrt(n)*base_std); the marginal y_std form overstates SE when the regressor explains most variance.
        finite_pair = np.isfinite(base_t) & np.isfinite(y_train_for_drift)
        base_finite = base_t[finite_pair]
        base_std = float(base_finite.std()) if base_finite.size > 1 else 1.0
        if base_std < 1e-12 or half < 2:
            drift_kept.append(s)
            continue
        y_finite = y_train_for_drift[finite_pair]
        n_pair = int(finite_pair.sum())
        if n_pair > 2:
            # Pooled OLS estimate (mean of the two half-fits) to compute residuals for the pooled residual scale.
            b1 = float(params1.get("beta", 0.0))
            b2 = float(params2.get("beta", 0.0))
            alpha_pool = 0.5 * (a1 + a2)
            beta_pool = 0.5 * (b1 + b2)
            residuals = y_finite - (alpha_pool * base_finite + beta_pool)
            sse = float(np.sum(residuals * residuals))
            sigma_resid = float(np.sqrt(max(sse / (n_pair - 2), 0.0)))
        else:
            sigma_resid = float(y_finite.std()) if y_finite.size > 1 else 1.0
        # a1-a2 is a difference of two independent half-fits, so Var(a1-a2)=2*sigma^2/(half*var_base); the sqrt(2) keeps the drift z from being inflated ~1.41x.
        se_alpha = sigma_resid * np.sqrt(2.0) / (np.sqrt(half) * base_std)
        z = abs(a1 - a2) / max(se_alpha, 1e-12)
        self._alpha_drift_flags[s.name] = {
            "alpha_first_half": a1,
            "alpha_second_half": a2,
            "z_score": float(z),
        }
        if z > drift_threshold:
            if reject_on_drift:
                drift_dropped.append((s.name, float(z)))
                continue
            # DEBUG not WARNING: many drift-flagged specs are later rejected by the raw-y baseline / Wilcoxon gate, so a per-spec WARNING here is dead-noise; a summary WARNING is emitted only for survivors at the end of discovery.
            logger.debug(
                "[CompositeTargetDiscovery] alpha drift candidate spec=%s "
                "(alpha first-half=%.4f, second-half=%.4f, z=%.2f > %.2f).",
                s.name, a1, a2, z, drift_threshold,
            )
        drift_kept.append(s)
    if drift_dropped:
        logger.info(
            "[CompositeTargetDiscovery] alpha drift gate dropped %d "
            "linear_residual spec(s): %s",
            len(drift_dropped),
            ", ".join(f"{n}(z={z:.2f})" for n, z in drift_dropped[:5]),
        )
    return drift_kept
