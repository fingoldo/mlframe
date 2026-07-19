"""Two-phase residual attribution (gt_09).

SHAP attribution on a full model divides credit among all features jointly; when a few strong
features explain most of the target, genuinely-predictive weak features receive small mean|phi| not
because they carry no signal but because the strong features absorb the shared credit. This module
runs a SECOND SHAP pass on pass-1's residuals -- the strong features' signal is already explained
away there, so weak features become the dominant explanators of what remains and earn full credit
(boosting's insight applied to attribution rather than prediction). Carved out of
``_shap_proxied_fit.py`` to keep it under the 1k LOC ceiling.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-clipped logistic sigmoid."""
    x = np.clip(x, -60.0, 60.0)
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


def compute_residual_target(phi: np.ndarray, base: np.ndarray, y_phi: np.ndarray, *, classification: bool) -> np.ndarray:
    """Pass-1 residual target: what the additive coalition proxy failed to explain.

    Regression: ``y - (base + sum(phi))`` -- the plain proxy residual, free (no extra OOF-predict
    pass) because it's exactly the full-coalition proxy margin pass 1 already computed.

    Classification: the pseudo-residual of logistic loss ``y - sigmoid(margin)``. ``margin_pred =
    base + phi.sum(axis=1)`` is already log-odds (TreeSHAP on a binary xgboost classifier attributes
    margin space, and ``base`` is the margin base value), but a discrete ``{0,1}`` label has no
    finite log-odds target -- so instead of an ``inverse-sigmoid(clip(y))`` infinity-handling hack,
    this uses the standard boosting gradient-of-logloss residual: bounded, continuous, exactly what
    a boosting step would fit next.
    """
    margin_pred = np.asarray(base, dtype=np.float64) + phi.sum(axis=1)
    y_arr = np.asarray(y_phi, dtype=np.float64)
    if classification:
        return np.asarray(y_arr - _sigmoid(margin_pred), dtype=np.float64)
    return np.asarray(y_arr - margin_pred, dtype=np.float64)


def _validate_residual_params(self: Any) -> int:
    """Validate ``residual_*`` constructor params at fit time; returns the resolved pass count."""
    rp = int(self.residual_passes)
    if rp < 0 or rp > 2:
        raise ValueError(
            f"ShapProxiedFS: residual_passes must be 0, 1, or 2; got {self.residual_passes!r}. "
            "Each pass costs one full OOF-SHAP fit; more than 2 has no plausible attribution payoff."
        )
    if str(self.residual_merge).lower() not in ("rescue", "blend"):
        raise ValueError(f"ShapProxiedFS: residual_merge must be 'rescue' or 'blend'; got {self.residual_merge!r}")
    if float(self.residual_lambda) < 0:
        raise ValueError(f"ShapProxiedFS: residual_lambda must be >= 0; got {self.residual_lambda!r}")
    if self.residual_top_k is not None and int(self.residual_top_k) <= 0:
        raise ValueError(f"ShapProxiedFS: residual_top_k must be a positive int or None; got {self.residual_top_k!r}")
    if int(self.residual_exclude_top) < 0:
        raise ValueError(f"ShapProxiedFS: residual_exclude_top must be >= 0; got {self.residual_exclude_top!r}")
    return rp


def _proxy_idx_to_names(idx, unit_to_members, working_cols, X_cols) -> list:
    """Expand proxy(unit)/raw-column indices to original input-column names (input column order)."""
    if unit_to_members is not None:
        cols = sorted({int(working_cols[m]) for u in idx for m in unit_to_members[int(u)]})
    else:
        cols = sorted(int(working_cols[i]) for i in idx)
    return [str(X_cols[i]) for i in cols]


def run_residual_pass(
    self: Any,
    phi: np.ndarray,
    base: np.ndarray,
    y_phi: np.ndarray,
    X_proxy: pd.DataFrame,
    model_template: Any,
    unit_to_members,
    working_cols,
    X_cols,
    report: dict,
    _stage: Callable,
) -> tuple:
    """Second SHAP pass on pass-1's residual; returns ``(rescue_proxy_idx, blend_importance, protected_working_cols)``.

    MUST be called on the PRE-prescreen ``phi``/``X_proxy`` (rescue only helps if it can save a
    column the prescreen would otherwise cut -- computing it post-prescreen would be too late for
    the cut it is meant to rescue from). ``unit_to_members`` entries at this point are WORKING-space
    column positions -- the same space ``within_cluster_refine``'s ``member_cols``/``protected_cols``
    consume -- so ``protected_working_cols`` needs no further remapping downstream.

    ``rescue_proxy_idx`` (``residual_merge="rescue"``): top ``residual_top_k`` proxy columns by
    mean|phi2|, to be unioned into the prescreen keep-set as a fourth member (alongside top-K,
    noise-floor rescue, and su_seeded rescue).

    ``blend_importance`` (``residual_merge="blend"``): ``phi1_importance + residual_lambda *
    phi2_importance_aligned`` (full ``n_proxy``-length vector, zero contribution from excluded
    columns) for the prescreen's RANKING order only -- the search/coalition proxy always consumes
    raw phi1, never this blended vector.
    """
    n_passes = _validate_residual_params(self)
    rescue_proxy_idx: set = set()
    blend_importance: Optional[np.ndarray] = None
    protected_working_cols: set = set()
    if n_passes <= 0:
        return rescue_proxy_idx, blend_importance, protected_working_cols

    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import compute_shap_matrix, make_default_estimator

    n_proxy = phi.shape[1]
    importance1 = np.abs(phi).mean(axis=0)
    residual = compute_residual_target(phi, base, y_phi, classification=self.classification)
    residual_std_before = float(np.std(residual))

    # "Hard residual": drop the top residual_exclude_top pass-1 columns from pass-2's X entirely, so
    # pass 2 must explain the residual WITHOUT the strong features at all (never just soft-discount
    # them via low proxy loss). Index MAP (not positional assumption): phi2's column j corresponds to
    # proxy column kept_proxy_idx[j] -- required because pass-2 X can be a column subset.
    excluded: set = set()
    if self.residual_exclude_top > 0 and n_proxy > 1:
        n_excl = min(int(self.residual_exclude_top), n_proxy - 1)
        excluded = {int(i) for i in np.argsort(-importance1)[:n_excl]}
    kept_proxy_idx = np.array(sorted(set(range(n_proxy)) - excluded), dtype=np.int64)
    if kept_proxy_idx.size == 0:
        report["residual_pass"] = dict(
            n_passes=n_passes,
            merge=str(self.residual_merge),
            lambda_=float(self.residual_lambda),
            rescued=[],
            excluded_top=_proxy_idx_to_names(excluded, unit_to_members, working_cols, X_cols),
            pass2_top_importance=[],
            residual_std_before=residual_std_before,
            residual_std_after=residual_std_before,
            skipped_reason="residual_exclude_top left no proxy columns for pass 2",
        )
        return rescue_proxy_idx, blend_importance, protected_working_cols

    # Pass 2 runs on the SAME already-narrowed X_proxy (post-prefilter, possibly post-cluster), never
    # the raw frame -- memory discipline at C4-scale widths where phi2 is another significant (n, f)
    # float64 matrix.
    X_pass2 = X_proxy.iloc[:, kept_proxy_idx] if kept_proxy_idx.size < n_proxy else X_proxy

    # Pass 2 ALWAYS runs as a REGRESSION booster on continuous residuals, regardless of the outer
    # task: a classifier fit on a continuous pseudo-residual target would misinterpret it as discrete
    # labels. When the caller supplied a custom classifier, derive an xgboost regressor with the same
    # tree budget rather than attempt a generic classifier->regressor clone.
    if self.model is not None:
        n_est = int(getattr(self.model, "n_estimators", 300) or 300)
        model_template_regression = make_default_estimator(False, random_state=int(self.random_state), n_estimators=n_est)
    else:
        booster_kind = self._resolve_booster_kind()
        model_template_regression = make_default_estimator(
            False, random_state=int(self.random_state), booster_kind=booster_kind, cat_features=self.cat_features
        )

    with _stage("residual_pass"):
        phi2, base2, _y2, per_fold_phi_mean2 = compute_shap_matrix(
            model_template_regression,
            X_pass2,
            residual,
            classification=False,
            out_of_fold=True,
            n_splits=self.n_splits,
            n_models=1,
            rng=self._rng,
            tqdm_desc=("shap-residual" if self.tqdm else None),
            n_jobs=self.n_jobs,
            n_estimators_cap=self.oof_shap_n_estimators,
            inner_n_jobs_cap=self.inner_n_jobs_cap,
            return_per_fold_phi_mean=True,
            cache_dir=self.cache_dir,
        )

    importance2 = np.abs(phi2).mean(axis=0)
    residual_pred = np.asarray(base2, dtype=np.float64) + phi2.sum(axis=1)
    residual_std_after = float(np.std(residual[: residual_pred.shape[0]] - residual_pred)) if residual_pred.shape[0] == residual.shape[0] else float("nan")

    # Cross-fold rank-CONSISTENCY gate (not a magnitude floor). A magnitude-only floor
    # (``median(bottom half) * safety_factor``, the same rule ``noise_floor_rescue_keep_set`` uses
    # for phi1) was tried and REJECTED here: measured on the p=3000 mixed-strength biz_val fixture,
    # the TRUE weak feature is phi2's rank-1 column by OOF mean|phi2| yet its magnitude still sits
    # below a 4x floor (the residual's own sampling noise is too large relative to a weight=0.25
    # signal for any fixed multiplier to cleanly separate them) -- so a magnitude floor either killed
    # all recall (tight) or let noise columns leak into the protected set (loose), and an ungated
    # search-pool alternative let the search stage itself pick spurious noise independent of
    # protection (measured: 6->8 selected, 2 noise columns, with NO protection involved at all).
    #
    # The fix used here needs no extra model fits (reuses ``per_fold_phi_mean2``, already computed
    # by the SAME OOF folds pass 2 already pays for): a genuinely-informative column should rank
    # near the top consistently across INDEPENDENT folds, while a noise column's rank is a one-fold
    # sampling fluke that does not replicate. Require a column's per-fold rank to clear a generous
    # per-fold pool (``3 * top_k``) in EVERY fold (``n_splits`` independent estimates) before it is
    # eligible for rescue -- a much better-calibrated signal-vs-noise separator than the column's
    # single aggregate OOF magnitude.
    top_k = int(self.residual_top_k) if self.residual_top_k is not None else int(self.brute_force_max_features)
    top_k = min(top_k, importance2.shape[0])
    per_fold = np.asarray(per_fold_phi_mean2, dtype=np.float64)  # (n_splits, n_kept_proxy)
    fold_pool = min(per_fold.shape[1], max(top_k * 3, 1))
    consistent = np.ones(per_fold.shape[1], dtype=bool)
    for fold_row in per_fold:
        fold_top = set(np.argsort(-fold_row)[:fold_pool].tolist())
        consistent &= np.array([j in fold_top for j in range(per_fold.shape[1])], dtype=bool)
    eligible_local = np.nonzero(consistent)[0]

    order_eligible = eligible_local[np.argsort(-importance2[eligible_local])]
    top_local = order_eligible[:top_k]
    rescue_proxy_idx = {int(kept_proxy_idx[i]) for i in top_local}
    pass2_top_pairs = [
        (_proxy_idx_to_names([int(kept_proxy_idx[i])], unit_to_members, working_cols, X_cols), float(importance2[i])) for i in np.argsort(-importance2)[:10]
    ]

    if str(self.residual_merge).lower() == "blend":
        blend_importance = importance1.copy()
        gated_importance2 = np.where(consistent, importance2, 0.0)
        blend_importance[kept_proxy_idx] = blend_importance[kept_proxy_idx] + float(self.residual_lambda) * gated_importance2
    else:  # "rescue" (default)
        if unit_to_members is not None:
            for u in rescue_proxy_idx:
                protected_working_cols.update(int(c) for c in unit_to_members[int(u)])
        else:
            protected_working_cols = set(rescue_proxy_idx)

    report["residual_pass"] = dict(
        n_passes=n_passes,
        merge=str(self.residual_merge),
        lambda_=float(self.residual_lambda),
        rescued=_proxy_idx_to_names(rescue_proxy_idx, unit_to_members, working_cols, X_cols),
        excluded_top=_proxy_idx_to_names(excluded, unit_to_members, working_cols, X_cols),
        pass2_top_importance=pass2_top_pairs,
        residual_std_before=residual_std_before,
        residual_std_after=residual_std_after,
        n_fold_consistent=int(eligible_local.size),
        fold_pool=int(fold_pool),
        n_protected=len(protected_working_cols),
    )
    del phi2  # memory discipline: another (n, f) float64 matrix, freed once the rescue/blend vector is extracted
    return rescue_proxy_idx, blend_importance, protected_working_cols
