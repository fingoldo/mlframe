"""Feature selection within ML pipelines. Wrappers methods. Currently includes recursive feature elimination."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from pyutilz.numbalib import set_numba_random_seed  # noqa: F401
from sklearn.base import (
    clone,
)
from sklearn.dummy import DummyClassifier, DummyRegressor  # noqa: F401
from sklearn.model_selection import (
    GroupShuffleSplit,  # noqa: F401
    StratifiedShuffleSplit,  # noqa: F401
)

logger = logging.getLogger(__name__)



def _sffs_swap_pass(
    self, X, y, estimator, cv, scoring,
    best_nfeatures: int, best_score_ref: float,
    selected_features_per_nfeatures: dict, feature_importances: dict,
    original_features, evaluated_scores_mean: dict,
    evaluated_scores_std: dict, verbose: int, ndigits: int,
    X_estimator=None, col_pos=None,
) -> None:
    """Replace each of the K worst-FI kept features with each of the K best-FI dropped features and accept any swap that improves the
    CV score. Mutates selected_features_per_nfeatures / evaluated_scores_* in place. Cost: O(K) extra CV evaluations executed via
    sklearn.model_selection.cross_val_score.
    """
    from sklearn.model_selection import cross_val_score

    K = int(self.swap_top_k)
    best_set = list(selected_features_per_nfeatures.get(best_nfeatures, []))
    if len(best_set) < 1:
        return

    # Aggregate FI across all runs (mean of non-NaN values per feature).
    from collections import defaultdict
    fi_acc = defaultdict(list)
    for _key, _fi in feature_importances.items():
        for feat, val in _fi.items():
            try:
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    fi_acc[feat].append(float(val))
            except (TypeError, ValueError):
                continue
    fi_mean = {f: float(np.mean(v)) for f, v in fi_acc.items() if v}

    # Worst K kept (features with no FI history get 0).
    # Wave 57 (2026-05-20): add feature name as deterministic tiebreaker --
    # many features tie at fi_mean=0 (no FI history); without the tiebreak,
    # which feature wins the swap depends on Python set/list iteration
    # order, silently flipping selection across runs.
    kept_sorted = sorted(best_set, key=lambda f: (fi_mean.get(f, 0.0), str(f)))
    swap_out = kept_sorted[:K]
    # Best K dropped.
    not_in_set = [f for f in original_features if f not in set(best_set)]
    # For the reverse-desc side, negate score then ascend alphabetically
    # so the deterministic tiebreaker is uniform-direction.
    not_sorted = sorted(not_in_set, key=lambda f: (-fi_mean.get(f, 0.0), str(f)))
    swap_in = not_sorted[:K]

    cur_set = list(best_set)
    cur_score = float(best_score_ref)
    n_swaps_accepted = 0
    for out_f, in_f in zip(swap_out, swap_in):
        trial_set = [in_f if f == out_f else f for f in cur_set]
        try:
            if X_estimator is not None and col_pos is not None:
                # All-numeric fast path: feed the estimator numpy column-subsets by integer position so
                # cross_val_score's inner fits/predicts skip the per-call pandas reconversion. float64
                # mirror -> bit-identical to ``X[trial_set]`` for the all-numeric case.
                pos = [col_pos[f] for f in trial_set]
                trial_X = X_estimator[:, pos]
            elif isinstance(X, pd.DataFrame):
                trial_X = X[trial_set]
            else:
                idx = [list(original_features).index(f) for f in trial_set]
                trial_X = X[:, idx]
            trial_scores = cross_val_score(
                clone(estimator), trial_X, y, cv=cv,
                scoring=scoring, n_jobs=1,
            )
        except Exception as _exc:
            if verbose:
                logger.warning(
                    "SFFS swap %s -> %s evaluation failed (%s); skipping pair.",
                    out_f, in_f, _exc,
                )
            continue
        if trial_scores is None or len(trial_scores) == 0:
            continue
        trial_mean = float(np.nanmean(trial_scores))
        trial_std = float(np.nanstd(trial_scores))
        if trial_mean > cur_score:
            if verbose:
                logger.info(
                    "SFFS swap accepted: %s -> %s improved %.*f -> %.*f",
                    out_f, in_f, ndigits, cur_score, ndigits, trial_mean,
                )
            cur_set = trial_set
            cur_score = trial_mean
            n_swaps_accepted += 1
            # |trial_set| == |best_set|, so this overwrites the entry for that nfeatures count.
            _n = len(trial_set)
            selected_features_per_nfeatures[_n] = trial_set
            evaluated_scores_mean[_n] = trial_mean
            evaluated_scores_std[_n] = trial_std

    if verbose:
        logger.info(
            "SFFS swap pass: %d/%d paired swaps accepted (best score %.*f).",
            n_swaps_accepted, len(swap_out), ndigits, cur_score,
        )

# sklearn 1.6 deprecated _get_tags / _more_tags in favour of __sklearn_tags__, which returns a sklearn.utils.Tags dataclass carrying
# classifier/regressor type info, input contract, and request metadata. Without overriding it, downstream sklearn helpers
# (estimator_html_repr, check_is_fitted, set_config(transform_output=...)) see RFECV as a generic transformer with no estimator-type
# tag and may mis-route routing requests. Delegate to the wrapped estimator's tags.
