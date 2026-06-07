"""init_score baseline computation for BaselineDiagnostics.

Carved out of ``baseline_diagnostics`` via method-rebinding (W10E pattern).
"""
from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _fit_init_score_baseline(
    self,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: Sequence[str],
    cat_features: Sequence[str],
    ablation: list,
    metric_name: str,
    higher_is_better: bool,
    raw_metric: float,
    target_type: str = "regression",
):
    """Refit quick-LightGBM with the top-K dominant features combined via
    OLS (regression) or logistic regression (binary) and passed via
    ``init_score`` so the model learns the residual instead of the full
    target. Mirrors LightGBM / XGBoost's native ``base_margin`` /
    ``init_score`` interface -- the cheapest possible "what if I learn
    the residual instead of the target?" baseline.

    For binary classification the init_score lives in **logit space**:
    ``init_score = logit(prior_p)`` where ``prior_p`` is the OLS-style
    linear combination of the top-K dominant features squashed through
    sigmoid. LightGBM treats the per-row init_score as the additive
    logit baseline; the booster learns the residual logit and
    ``predict_proba`` returns ``sigmoid(init_score + tree_output)``.

    Use case: CTR prediction with ``base = baseline_click_rate``, fraud
    detection with ``base = prior_fraud_score``, any binary task with a
    probability-scale dominant feature.

    For ``init_score_top_k > 1`` the top-K features are combined via OLS
    (regression) or LogisticRegression (binary). Falls back to top-1 if
    the combiner is degenerate.
    """
    from .diagnostics import InitScoreBaseline, _delta_pct

    top_k = max(1, min(self.config.init_score_top_k, len(ablation)))
    # Use only features with positive ablation delta - dropping them actually
    # hurts. Sorted ablation is already ranked; take prefix.
    chosen = [e for e in ablation[:top_k] if e.delta_pct > 0]
    if not chosen:
        return None

    if len(chosen) == 1:
        feat = chosen[0].feature
        base_values = X[feat].to_numpy().astype(np.float64)
        feature_used = feat
    else:
        # Linear combiner: OLS for regression, LR for binary.
        try:
            # Single-select / loc materialisation hoists per-feature dispatch out of the loop. On K>=4
            # candidate features this saved ~3x over the prior list-comprehension of independent .to_numpy()
            # calls (each call dispatched through pandas/polars column-access overhead).
            _feats_for_combiner = [e.feature for e in chosen]
            if hasattr(X, "loc"):
                B = X.loc[:, _feats_for_combiner].to_numpy(dtype=np.float64, copy=False)
            else:
                # Polars DataFrame branch -- single select -> arrow -> numpy.
                B = X.select(_feats_for_combiner).to_numpy()
                if B.dtype != np.float64:
                    B = B.astype(np.float64, copy=False)
            ptp = B.ptp(axis=0)
            keep_cols = ptp > 1e-12
            if keep_cols.sum() == 0:
                return None
            B = B[:, keep_cols]
            if target_type == "binary_classification":
                # Logistic regression -> probability scale, then converted to
                # logit by the binary branch below.
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression(max_iter=200, C=1.0)
                lr.fit(B, y)
                base_values = lr.predict_proba(B)[:, 1]
            else:
                B_aug = np.column_stack([B, np.ones(len(B))])
                coef, *_ = np.linalg.lstsq(
                    B_aug, y.astype(np.float64), rcond=None,
                )
                base_values = B_aug @ coef
            kept_feats = [e.feature for e, k in zip(chosen, keep_cols) if k]
            feature_used = "+".join(kept_feats) if kept_feats else chosen[0].feature
        except (ValueError, np.linalg.LinAlgError, RuntimeError, TypeError) as exc:
            # OLS / LR combiner can fail on rank-deficient B or NaN-rich y;
            # fall back to single-feature mode. KeyboardInterrupt / MemoryError
            # still propagate.
            logger.info(
                "BaselineDiagnostics: init_score combiner failed (%s); "
                "falling back to top-1.", exc,
            )
            feat = chosen[0].feature
            base_values = X[feat].to_numpy().astype(np.float64)
            feature_used = feat

    if not np.all(np.isfinite(base_values)):
        logger.info(
            "BaselineDiagnostics: init_score base values contain "
            "non-finite entries; skipping."
        )
        return None

    # Convert to the right scale for the inner family. LightGBM's init_score
    # for binary expects a LOGIT, not a probability.
    if target_type == "binary_classification":
        # Treat base_values as a probability-scale signal. Squash through
        # sigmoid first if it's outside [0, 1] (raw feature scale), else
        # clip to (eps, 1-eps) and take logit.
        mn, mx = float(np.min(base_values)), float(np.max(base_values))
        if mn < 0.0 or mx > 1.0:
            # Not on probability scale; squash via sigmoid relative to the
            # sample mean so the prior centres on the observed class balance.
            base_values = base_values - float(np.mean(base_values))
            # Scale so std ~= 1 (keeps logit values reasonable).
            std = float(np.std(base_values)) or 1.0
            base_values = base_values / std
            # Centre the logit baseline on the empirical positive rate so
            # init_score isn't biased toward 50/50.
            p_pos = float(np.mean(y))
            p_pos = min(max(p_pos, 1e-3), 1 - 1e-3)
            init_logit = base_values + math.log(p_pos / (1.0 - p_pos))
        else:
            # Probability scale; clip + logit directly.
            eps = 1e-6
            p = np.clip(base_values, eps, 1.0 - eps)
            init_logit = np.log(p / (1.0 - p))
        init_score_arg = init_logit
    else:
        init_score_arg = base_values

    try:
        metric_val, _ = self._fit_quick_and_score(
            X, y, feature_cols, cat_features, target_type,
            metric_name, init_score=init_score_arg,
        )
    except (ValueError, RuntimeError, TypeError, IndexError) as exc:
        # init_score path on LightGBM can fail on dtype mismatch / NaN-rich
        # init_score / single-class y. KeyboardInterrupt / MemoryError still propagate.
        logger.warning(
            "BaselineDiagnostics: init_score refit failed: %s; baseline skipped.",
            exc,
        )
        return None

    return InitScoreBaseline(
        feature_used=feature_used,
        model_family=self.config.quick_model_family,
        metric=metric_val,
        delta_vs_raw_pct=_delta_pct(raw_metric, metric_val, higher_is_better),
    )
