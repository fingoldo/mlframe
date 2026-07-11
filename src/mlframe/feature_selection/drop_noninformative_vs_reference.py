"""``drop_noninformative_vs_reference``: KS-test drop of features that DON'T distinguish two populations.

Source: 3rd_mechanisms-of-action-moa-prediction.md -- dropped features where a KS-test between control and
treated distributions gives p > 0.1 (the feature doesn't distinguish control from treated at all, so it's
likely pure noise/batch artifact). This is the OPPOSITE direction from mlframe's existing
``ks_stability_filter`` (which drops features whose distribution SHIFTS between train/test, p <= alpha --
a drift-remediation concern) -- here, a feature that shows NO significant difference between a reference
group and the rest is the one to drop, since it carries no discriminative signal about the very distinction
the reference cohort encodes (e.g. dosed vs control samples in an assay).

Reuses ``ks_stability_filter``'s per-column KS computation (same ``scipy.stats.ks_2samp`` machinery) rather
than reimplementing it -- this module is purely the inverted selection criterion plus the "reference cohort"
framing.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import ks_stability_filter


def _noninformative_columns_vs_one_reference(
    df: pd.DataFrame,
    reference_mask: np.ndarray,
    feature_cols: Optional[Sequence[str]],
    alpha: float,
) -> List[str]:
    mask = np.asarray(reference_mask, dtype=bool)
    reference_df = df.loc[mask]
    rest_df = df.loc[~mask]

    report = ks_stability_filter(reference_df, rest_df, feature_cols=feature_cols, p_value_threshold=alpha)
    # ks_stability_filter's "stable" flag is p_value > threshold -- exactly the "does NOT differ" condition
    # this entry wants to drop, just under the opposite name (there, "stable" means "safe to keep because it
    # didn't drift"; here the same condition means "no signal, drop it").
    return [str(c) for c in report.loc[report["stable"], "column"].tolist()]


def drop_noninformative_vs_reference(
    df: pd.DataFrame,
    reference_mask: Union[np.ndarray, Sequence[np.ndarray]],
    feature_cols: Optional[Sequence[str]] = None,
    alpha: float = 0.1,
    require_all_cohorts: bool = False,
) -> List[str]:
    """Return feature names whose distribution does NOT significantly differ between the reference subgroup
    and the rest of ``df`` (KS-test p-value > ``alpha``) -- candidates to drop as likely noise/batch artifact.

    Parameters
    ----------
    df
        Frame containing both the reference subgroup(s) and the rest.
    reference_mask
        Boolean mask selecting the reference/control subpopulation; ``~reference_mask`` is the "rest"
        (e.g. treated samples) it's compared against. When ``require_all_cohorts`` is True, this may
        instead be a sequence of several such masks (multiple independent reference/control cohorts,
        e.g. separate assay batches) -- each compared against its own complement.
    feature_cols
        Numeric columns to screen; defaults to every numeric column of ``df``.
    alpha
        A feature is flagged non-informative (drop candidate) when its KS p-value EXCEEDS this threshold
        (the source convention: p > 0.1 means no significant distributional difference).
    require_all_cohorts
        Opt-in multi-reference-cohort mode. When False (default), ``reference_mask`` must be a single mask
        and behavior is identical to the original single-cohort function. When True, ``reference_mask`` is
        a sequence of masks and a feature is only flagged as a drop candidate if it is non-informative
        (p > alpha) against EVERY cohort -- guards against a single reference batch spuriously looking
        similar to the rest by chance for a feature that is genuinely informative, which a single-cohort
        KS-test cannot distinguish from true noise.

    Returns
    -------
    list of str
        Column names to consider dropping (non-informative vs the reference cohort(s)).
    """
    if not require_all_cohorts:
        return _noninformative_columns_vs_one_reference(df, reference_mask, feature_cols, alpha)  # type: ignore[arg-type]

    masks: Sequence[np.ndarray] = reference_mask  # type: ignore[assignment]
    if len(masks) == 0:
        raise ValueError("require_all_cohorts=True requires at least one reference mask in the sequence")

    per_cohort_candidates = [set(_noninformative_columns_vs_one_reference(df, m, feature_cols, alpha)) for m in masks]
    # only drop a feature that failed to discriminate against ALL cohorts -- a feature that looks
    # noninformative against one cohort but informative against another is real signal, not noise.
    common = per_cohort_candidates[0]
    for candidates in per_cohort_candidates[1:]:
        common &= candidates

    # preserve df column order in the output for readability/determinism.
    return [str(c) for c in df.columns if c in common]


__all__ = ["drop_noninformative_vs_reference"]
