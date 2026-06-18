"""Feature selection for the LTR ranker suite, driven by the COMMON ``FeatureSelectionConfig``.

The ranker suite is a separate dispatch path, so it does not go through the main per-target FS loop. This module
re-wires it onto the SAME selector builder the main suite uses (``core._setup_helpers_pre_pipelines._build_pre_pipelines``)
so ``use_mrmr_fs`` / ``rfecv_models`` / ``use_boruta_shap`` work for LEARNING_TO_RANK exactly as for every other
target type -- no LTR-specific FS config. The graded-relevance label is the selection target; ``target_type`` is
threaded into the builder so each selector picks regression-appropriate settings (e.g. BorutaShap regressor mode,
RFECV regressor estimator) rather than the classification defaults that reject a multi-grade relevance label.

Nothing here modifies the core MRMR / RFECV / selector procedures -- it only constructs + fits them.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _to_pandas_features(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    if hasattr(X, "to_pandas"):  # polars
        return X.to_pandas()
    return pd.DataFrame(np.asarray(X))


def _build_rfecv_models_params(rfecv_models, X_df, y, *, random_seed, verbose):
    """Build RFECV instances for the (relevance-as-regression) LTR target via the standard config path."""
    if not rfecv_models:
        return {}
    from ..trainer import configure_training_params

    _, _, cb_rfecv, lgb_rfecv, xgb_rfecv, _, _ = configure_training_params(
        train_df=X_df, train_target=np.asarray(y), target=np.asarray(y),
        use_regression=True,  # graded relevance -> regression estimator/scorer for the RFECV wrapper
        mlframe_models=list(rfecv_models),
        prefer_gpu_configs=False, verbose=bool(verbose), rfecv_model_verbose=False,
    )
    return {"cb_rfecv": cb_rfecv, "lgb_rfecv": lgb_rfecv, "xgb_rfecv": xgb_rfecv}


def select_ltr_features(
    X,
    y: np.ndarray,
    *,
    feature_selection_config,
    rfecv_models: Optional[list] = None,
    target_type: Any = None,
    fs_random_seed: int = 42,
    verbose: int = 0,
) -> Optional[list]:
    """Select feature columns for the LTR rankers using the common ``FeatureSelectionConfig``.

    Builds whichever selectors the config enables (MRMR via ``use_mrmr_fs``, RFECV via ``rfecv_models``, BorutaShap
    via ``use_boruta_shap``) with the SAME ``_build_pre_pipelines`` the main suite uses, fits each on
    ``(X, graded relevance)``, and returns the UNION of selected columns. Returns ``None`` when no FS is enabled
    (caller keeps all features). Never raises -- a selector that fails is skipped with a warning.
    """
    fsc = feature_selection_config
    if fsc is None:
        return None
    use_mrmr = bool(getattr(fsc, "use_mrmr_fs", False))
    use_bs = bool(getattr(fsc, "use_boruta_shap", False))
    rfecv_models = list(rfecv_models or [])
    if not (use_mrmr or use_bs or rfecv_models):
        return None

    X_df = _to_pandas_features(X)
    y_ser = pd.Series(np.asarray(y), name="relevance")

    rfecv_models_params = {}
    if rfecv_models:
        try:
            rfecv_models_params = _build_rfecv_models_params(rfecv_models, X_df, y_ser, random_seed=fs_random_seed, verbose=verbose)
        except Exception as exc:
            logger.warning("LTR RFECV construction failed (%s: %s); skipping RFECV for ranking FS.", type(exc).__name__, exc)
            rfecv_models = []

    from ..core._setup_helpers_pre_pipelines import _build_pre_pipelines

    pre_pipelines, _names = _build_pre_pipelines(
        use_ordinary_models=False,  # no pass-through baseline; we only want the selectors
        rfecv_models=rfecv_models,
        rfecv_models_params=rfecv_models_params,
        use_mrmr_fs=use_mrmr,
        mrmr_kwargs=dict(getattr(fsc, "mrmr_kwargs", None) or {}),
        use_boruta_shap=use_bs,
        boruta_shap_kwargs=dict(getattr(fsc, "boruta_shap_kwargs", None) or {}),
        use_sample_weights_in_fs=bool(getattr(fsc, "use_sample_weights_in_fs", False)),
        target_type=target_type,
        fs_random_seed=fs_random_seed,
    )

    selected: set = set()
    ran_any = False
    for pp in pre_pipelines:
        if pp is None:
            continue
        try:
            Xt = pp.fit_transform(X_df, y_ser)
            ran_any = True
            cols = list(Xt.columns) if hasattr(Xt, "columns") else None
            if cols is None:
                support = getattr(pp, "support_", None)
                if support is not None:
                    sup = np.asarray(support)
                    cols = ([X_df.columns[i] for i in np.where(sup)[0]] if sup.dtype == bool
                            else [X_df.columns[int(i)] for i in sup.tolist() if 0 <= int(i) < X_df.shape[1]])
            if cols:
                selected.update(c for c in cols if c in set(X_df.columns))
        except Exception as exc:
            logger.warning("LTR feature selector %s failed (%s: %s); skipping it.", type(pp).__name__, type(exc).__name__, exc)

    if not ran_any:
        return None
    out = [c for c in X_df.columns if c in selected]  # preserve original column order
    if not out:
        logger.warning("LTR feature selection produced an empty union; keeping all features.")
        return list(X_df.columns)
    return out
