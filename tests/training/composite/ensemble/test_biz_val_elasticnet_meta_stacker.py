"""biz_value: ElasticNet meta-stacker is more STABLE than Lasso on a highly-correlated component group.

Synthetic: two informative components (``a``, ``b``, weighted 0.7/0.3 into ``y``) plus a GROUP of 3 near-identical
copies of ``a`` (small added noise -- a highly-correlated redundant group, not just a single duplicate) and 3
pure-noise components. Lasso's L1 penalty is known in the statistics literature to arbitrarily pick ONE member of a
correlated group to keep and zero the rest -- WHICH member survives is unstable across resamples. ElasticNet's added
L2 term should make the fit prefer to keep-or-drop the correlated group TOGETHER, more consistently, across resamples.

This is validated empirically here (not assumed from theory): fit both stackers on many independent resamples of the
same generative process and measure how often each keeps the correlated group's zero/nonzero pattern IDENTICAL across
resamples (group-consistency) and how often the whole group is either fully kept or fully dropped together (a proxy
for "the group is treated as ONE unit" rather than sliced arbitrarily).
"""

from __future__ import annotations

import warnings
from collections import Counter

import numpy as np

from mlframe.training.composite.ensemble._stackers import fit_elasticnet_meta_stacker, fit_lasso_meta_stacker

warnings.filterwarnings("ignore")

_GROUP_IDX = (2, 3, 4)  # columns of the 3 near-identical copies of `a`


def _gen_correlated_group_and_noise(n, seed, n_group=3, n_noise=3, group_noise=0.02):
    """Gen correlated group and noise."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    y = 0.7 * a + 0.3 * b + 0.05 * rng.normal(size=n)
    group = [a + group_noise * rng.normal(size=n) for _ in range(n_group)]
    noise = [rng.normal(size=n) for _ in range(n_noise)]
    X = np.column_stack([a, b, *group, *noise])
    return X, y


def _group_zero_patterns(fit_fn, n_resamples, n, seed_offset):
    """Fit ``fit_fn`` on ``n_resamples`` independent resamples; return the group's zero/nonzero pattern per resample."""
    patterns = []
    for i in range(n_resamples):
        X, y = _gen_correlated_group_and_noise(n, seed=seed_offset + i)
        model = fit_fn(X, y, X.shape[1])
        coef = np.asarray(model.coef_)
        patterns.append(tuple(int(c == 0.0) for c in coef[list(_GROUP_IDX)]))
    return patterns


def test_biz_val_elasticnet_meta_stacker_keeps_correlated_group_together_more_often_than_lasso():
    """Biz val elasticnet meta stacker keeps correlated group together more often than lasso."""
    n_resamples = 40
    lasso_patterns = _group_zero_patterns(fit_lasso_meta_stacker, n_resamples, n=2000, seed_offset=100)
    en_patterns = _group_zero_patterns(fit_elasticnet_meta_stacker, n_resamples, n=2000, seed_offset=100)

    def frac_kept_or_dropped_together(patterns):
        """Frac kept or dropped together."""
        return sum(1 for p in patterns if p == (0, 0, 0) or p == (1, 1, 1)) / len(patterns)

    lasso_together = frac_kept_or_dropped_together(lasso_patterns)
    en_together = frac_kept_or_dropped_together(en_patterns)

    # Measured on this synthetic: lasso=0.07 (2-3/40), elasticnet=0.33 (13/40) -- threshold set below the measured
    # elasticnet value and above the measured lasso value, with margin.
    assert (
        en_together >= 0.20
    ), f"expected ElasticNet to keep/drop the correlated group together in >=20% of resamples, got {en_together:.2f} (patterns={Counter(en_patterns)})"
    assert (
        lasso_together <= 0.15
    ), f"sanity check on the synthetic: expected Lasso's arbitrary single-member pick to keep the group together in <=15% of resamples, got {lasso_together:.2f} (patterns={Counter(lasso_patterns)})"
    assert (
        en_together > lasso_together
    ), f"expected ElasticNet to be strictly more stable than Lasso on the correlated group, got en={en_together:.2f} lasso={lasso_together:.2f}"


def test_biz_val_elasticnet_meta_stacker_lower_zero_indicator_variance_than_lasso():
    """Biz val elasticnet meta stacker lower zero indicator variance than lasso."""
    n_resamples = 40
    lasso_patterns = np.array(_group_zero_patterns(fit_lasso_meta_stacker, n_resamples, n=2000, seed_offset=100), dtype=float)
    en_patterns = np.array(_group_zero_patterns(fit_elasticnet_meta_stacker, n_resamples, n=2000, seed_offset=100), dtype=float)

    # Per-group-member variance of the zero/nonzero indicator across resamples -- lower means that member's
    # in/out-of-blend status is more predictable (stable) across independent resamples of the same process.
    lasso_var = float(np.var(lasso_patterns, axis=0).mean())
    en_var = float(np.var(en_patterns, axis=0).mean())

    # Measured: lasso=0.1815, elasticnet=0.1479. Threshold set with margin around the measured values.
    assert en_var <= 0.17, f"expected ElasticNet's per-member zero-indicator variance to be low, got {en_var:.4f}"
    assert en_var < lasso_var, f"expected ElasticNet's zero-indicator variance to be strictly lower than Lasso's, got en={en_var:.4f} lasso={lasso_var:.4f}"


def test_biz_val_elasticnet_meta_stacker_holdout_rmse_not_worse_than_lasso():
    """Biz val elasticnet meta stacker holdout rmse not worse than lasso."""
    X_train, y_train = _gen_correlated_group_and_noise(n=2000, seed=0)
    X_test, y_test = _gen_correlated_group_and_noise(n=2000, seed=1)
    n_components = X_train.shape[1]

    elasticnet = fit_elasticnet_meta_stacker(X_train, y_train, n_components)
    lasso = fit_lasso_meta_stacker(X_train, y_train, n_components)

    rmse_en = float(np.sqrt(np.mean((elasticnet.predict(X_test) - y_test) ** 2)))
    rmse_lasso = float(np.sqrt(np.mean((lasso.predict(X_test) - y_test) ** 2)))

    assert rmse_en <= rmse_lasso * 1.10, f"expected ElasticNet holdout RMSE to be within 10% of Lasso, got en={rmse_en:.4f} lasso={rmse_lasso:.4f}"
