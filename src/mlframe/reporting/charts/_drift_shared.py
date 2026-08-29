"""Frame plumbing and adversarial-validation constants shared by the drift builders.

The PSI heatmap and the adversarial panels both need to turn a caller's frame (pandas, polars, or a bare
ndarray) into a list of columns and a row count, and both need the sampling caps. Keeping those in
``drift.py`` meant the carved adversarial module had to import its own parent back, which makes the edge
bidirectional and the import order load-bearing.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

# Two-sided 95% normal quantile for the adversarial AUC's no-shift bar.
_ADV_Z: float = 1.96


def _frame_columns(feature_frame: Any, feature_names: Optional[Sequence[str]]) -> Tuple[List[np.ndarray], List[str]]:
    """Yield per-column ndarrays + names from ndarray / pandas / polars without copying the whole frame.

    ``feature_names`` (when given) RESTRICTS + ORDERS the pulled columns for a DataFrame
    too -- not only the ndarray-labelling path. Ignoring it on frames silently trained
    PSI / adversarial validation on every column (target / id leakage, mismatched order).
    """
    if hasattr(feature_frame, "columns") and hasattr(feature_frame, "__getitem__") and not isinstance(feature_frame, np.ndarray):
        selected = list(feature_names) if feature_names is not None else list(feature_frame.columns)
        # polars exposes ``to_numpy`` per Series; pandas ``.values``. Pull one column at a time (narrow ndarray pull).
        cols = []
        for c in selected:
            s = feature_frame[c]
            arr = s.to_numpy() if hasattr(s, "to_numpy") else np.asarray(s)
            cols.append(arr)
        return cols, [str(c) for c in selected]
    arr = np.asarray(feature_frame)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(arr.shape[1])]
    return [arr[:, i] for i in range(arr.shape[1])], names
def _frame_rows(frame: Any) -> int:
    """Row count of a 2-D ndarray / pandas / polars frame without materialising it."""
    if hasattr(frame, "shape"):
        return int(frame.shape[0])
    return len(frame)


def _adversarial_auc_bar(n_a: int, n_b: int) -> float:
    """How far above 0.5 an adversarial AUC must sit before it means anything, at these per-side row counts.

    Under the null of identical distributions the AUC (a Mann-Whitney statistic) has variance
    ``(n_a + n_b + 1) / (12 * n_a * n_b)``; scaling its standard error by z gives a bar that shrinks as the sets
    grow. The old fixed 0.6 line called ordinary small-sample noise a distribution shift, and simultaneously missed
    genuine, reproducible shifts on large sets where 0.55 is far outside anything the null can produce.
    """
    if n_a <= 0 or n_b <= 0:
        return 0.5
    se = float(np.sqrt((n_a + n_b + 1.0) / (12.0 * n_a * n_b)))
    return _ADV_Z * se
# Per-side row cap for the adversarial classifier. A LightGBM split-classifier converges on distribution-shift signal
# long before 200k rows/side; sampling caps the fit cost at large n without changing the verdict.
ADV_MAX_ROWS_PER_SIDE: int = 200_000
ADV_TOP_FEATURES: int = 20
# Trees in the adversarial LightGBM separator. The adversarial AUC is a COARSE drift signal (is it ~0.5, or elevated?),
# not a tuned predictor, so it saturates far below 200 trees: reducing 200 -> 75 shifts the OOF AUC by <=0.007 and never
# flips the drift verdict (validated across 12 drift regimes from no-drift to heavy), while cutting the fit ~2x (the
# separator is trained 3x for CV + once for importances, and this was ~17s across a report's drift panels at 300k).
ADV_N_ESTIMATORS: int = 75
# Minimum rows per side for the adversarial CV: a stratified 2-fold needs >= 2 of each class per fold, so fewer
# rows per side makes cross_val_predict raise on a 0-sample fold.
MIN_ADV_ROWS_PER_SIDE: int = 4


__all__ = ["ADV_MAX_ROWS_PER_SIDE", "ADV_N_ESTIMATORS", "ADV_TOP_FEATURES", "MIN_ADV_ROWS_PER_SIDE"]
