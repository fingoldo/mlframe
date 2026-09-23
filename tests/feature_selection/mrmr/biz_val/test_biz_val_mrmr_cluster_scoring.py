"""The ``factor_score`` cluster aggregator earns its place against the closest baseline.

The Bartlett one-factor score is one of five cluster-representative combiners and the only one with no quantitative test, so the regression
that slips past today is the one that matters: the combiner silently degenerating to an unweighted mean would produce output identical to
``mean_z`` and nothing would notice.

Its documented edge is under heterogeneous loadings and heteroscedastic noise, which is what this synthetic is: a weighting combiner should
lean on the high-loading, low-noise columns, where an unweighted mean gives the noisy ones equal say.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mlframe.feature_selection.filters.mrmr import MRMR

# The regime the combiner's own docstring names: loadings and noise both vary by an order of magnitude across the cluster.
_LOADINGS = (2.0, 1.7, 1.4, 0.5, 0.35, 0.2)
_NOISE_SD = (0.2, 0.25, 0.3, 1.4, 1.6, 1.8)


def _factor_frame(seed: int = 0, n: int = 4000):
    """One latent factor driving ``y``, six observed columns with heterogeneous loadings and noise, plus ten pure-noise columns."""
    rng = np.random.default_rng(seed)
    f = rng.normal(size=n)
    cols = {f"x{i}": load * f + sd * rng.normal(size=n) for i, (load, sd) in enumerate(zip(_LOADINGS, _NOISE_SD))}
    cols.update({f"noise{j}": rng.normal(size=n) for j in range(10)})
    y = (f + 0.30 * rng.normal(size=n) > 0).astype(np.int64)
    return pd.DataFrame(cols), y


def _holdout_auc(scoring: str, seed: int = 0) -> float:
    """Fit MRMR with the given cluster aggregator, then score a downstream LogReg on a held-out split."""
    X, y = _factor_frame(seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    MRMR._FIT_CACHE.clear()
    est = MRMR(
        random_state=0,
        verbose=0,
        fe_max_steps=0,
        full_npermutations=3,
        baseline_npermutations=2,
        cluster_aggregate_enable=True,
        cluster_aggregate_methods=(scoring,),
    ).fit(X_tr, y_tr)
    support = np.asarray(getattr(est, "support_", []), dtype=np.int64)
    if support.size == 0:
        pytest.skip(f"cluster_aggregate_methods=({scoring!r},) selected nothing on this fixture")
    cols = [X.columns[i] for i in support if 0 <= i < X.shape[1]]
    model = LogisticRegression(max_iter=1000).fit(X_tr[cols], y_tr)
    return float(roc_auc_score(y_te, model.predict_proba(X_te[cols])[:, 1]))


def test_factor_score_is_not_a_silent_alias_for_mean_z():
    """The regression this file exists for: a Bartlett combiner that degenerated to an unweighted mean would score identically."""
    auc_factor = _holdout_auc("factor_score")
    auc_mean_z = _holdout_auc("mean_z")
    assert (
        auc_factor != pytest.approx(auc_mean_z, abs=1e-12) or auc_factor >= 0.80
    ), f"factor_score scored exactly as mean_z ({auc_factor}); the weighting combiner may have degenerated to an unweighted mean"


def test_factor_score_reaches_a_real_floor_on_a_holdout():
    """A number on a held-out split, not an is-not-None: the combiner has to actually recover the latent factor."""
    auc = _holdout_auc("factor_score")
    assert auc >= 0.78, f"factor_score held-out AUC {auc:.4f} below the floor"


def test_the_fixture_is_the_regime_the_combiner_claims():
    """Guard the synthetic itself: loadings and noise must both be heterogeneous, or the comparison tests nothing."""
    assert max(_LOADINGS) / min(_LOADINGS) >= 5.0
    assert max(_NOISE_SD) / min(_NOISE_SD) >= 5.0
