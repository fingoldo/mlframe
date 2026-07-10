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

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._ks_stability import ks_stability_filter


def drop_noninformative_vs_reference(
    df: pd.DataFrame,
    reference_mask: np.ndarray,
    feature_cols: Optional[Sequence[str]] = None,
    alpha: float = 0.1,
) -> List[str]:
    """Return feature names whose distribution does NOT significantly differ between the reference subgroup
    and the rest of ``df`` (KS-test p-value > ``alpha``) -- candidates to drop as likely noise/batch artifact.

    Parameters
    ----------
    df
        Frame containing both the reference subgroup and the rest.
    reference_mask
        Boolean mask selecting the reference/control subpopulation; ``~reference_mask`` is the "rest"
        (e.g. treated samples) it's compared against.
    feature_cols
        Numeric columns to screen; defaults to every numeric column of ``df``.
    alpha
        A feature is flagged non-informative (drop candidate) when its KS p-value EXCEEDS this threshold
        (the source convention: p > 0.1 means no significant distributional difference).

    Returns
    -------
    list of str
        Column names to consider dropping (non-informative vs the reference cohort).
    """
    mask = np.asarray(reference_mask, dtype=bool)
    reference_df = df.loc[mask]
    rest_df = df.loc[~mask]

    report = ks_stability_filter(reference_df, rest_df, feature_cols=feature_cols, p_value_threshold=alpha)
    # ks_stability_filter's "stable" flag is p_value > threshold -- exactly the "does NOT differ" condition
    # this entry wants to drop, just under the opposite name (there, "stable" means "safe to keep because it
    # didn't drift"; here the same condition means "no signal, drop it").
    return [str(c) for c in report.loc[report["stable"], "column"].tolist()]


__all__ = ["drop_noninformative_vs_reference"]
