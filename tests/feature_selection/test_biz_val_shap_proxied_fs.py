"""biz_val tests for ``ShapProxiedFS`` end-to-end on synthetic data.

The business value of this selector is: cheaply rank feature subsets so that the chosen subset, when
a model is HONESTLY retrained on it, beats both a random same-size subset AND a plain
SHAP-importance-top-k subset, while excluding noise + redundant-correlated columns -- at a fraction
of the cost of exhaustive honest retraining.

Synthetic design mirrors the user's poker-style construction: K informative features driving the
target, plus pure-noise columns and columns correlated with informative ones (the redundancy trap
that breaks naive importance ranking).

Measured dev run (seed=0): the selector recovers the informative set, proxy_honest_loss <=
importance_honest_loss, and proxy_honest_loss << random-baseline loss. Floors carry 5-15% headroom.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")
pytest.importorskip("xgboost")


def _make_dataset(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    inf = rng.normal(size=(n, 5))                       # 5 informative
    noise = rng.normal(size=(n, 4))                     # 4 pure noise
    corr = inf[:, :2] + 0.3 * rng.normal(size=(n, 2))   # 2 redundant-with-informative
    X = pd.DataFrame(
        np.column_stack([inf, noise, corr]),
        columns=[f"inf{i}" for i in range(5)] + [f"noise{i}" for i in range(4)] + ["corr0", "corr1"],
    )
    logit = 0.9 * inf[:, 0] + 0.8 * inf[:, 1] - 0.7 * inf[:, 2] + 0.6 * inf[:, 3] + 0.4 * inf[:, 4]
    y = (logit + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, y


@pytest.mark.slow
def test_biz_val_shap_proxied_fs_recovers_informative_and_beats_baselines():
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = _make_dataset(seed=0)
    sel = ShapProxiedFS(
        classification=True, metric="brier", optimizer="bruteforce",
        max_features=7, top_n=20, n_splits=3, n_revalidation_models=2,
        random_state=0, verbose=False,
    )
    sel.fit(X, pd.Series(y))
    selected = set(sel.selected_features_)
    rep = sel.shap_proxy_report_

    informative_kept = selected & {f"inf{i}" for i in range(5)}
    noise_kept = [c for c in selected if c.startswith("noise")]

    # Recovery: at least 4 of 5 informative features (measured 5/5; floor 4 leaves seed headroom).
    assert len(informative_kept) >= 4, f"too few informative kept: {sorted(informative_kept)}"
    # Discrimination: at most 1 pure-noise column (measured 0).
    assert len(noise_kept) <= 1, f"too many noise columns kept: {noise_kept}"

    # Proxy fidelity was measured on this data.
    assert rep["trust"]["spearman"] > 0.7, rep["trust"]

    # Unique value: proxy subset at least ties SHAP-importance-top-k (measured: proxy strictly wins).
    abl = rep["importance_ablation"]
    assert abl["proxy_at_least_ties"], abl

    # Honest win over a random same-size subset by a wide margin.
    best = rep["revalidation"]["ranked"][0]["honest_loss"]
    baseline = rep["revalidation"]["random_baseline"]["honest_loss"]
    assert best < 0.9 * baseline, f"chosen honest loss {best} not clearly below random baseline {baseline}"


@pytest.mark.slow
def test_biz_val_regression_recovers_informative():
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    rng = np.random.default_rng(1)
    n = 2500
    inf = rng.normal(size=(n, 4))
    noise = rng.normal(size=(n, 5))
    X = pd.DataFrame(np.column_stack([inf, noise]),
                     columns=[f"inf{i}" for i in range(4)] + [f"noise{i}" for i in range(5)])
    y = inf[:, 0] + 0.8 * inf[:, 1] - 0.6 * inf[:, 2] + 0.5 * inf[:, 3] + 0.1 * rng.normal(size=n)

    sel = ShapProxiedFS(classification=False, metric="rmse", optimizer="bruteforce",
                        max_features=6, top_n=15, n_splits=3, n_revalidation_models=2,
                        random_state=0, verbose=False)
    sel.fit(X, y)
    selected = set(sel.selected_features_)
    informative_kept = selected & {f"inf{i}" for i in range(4)}
    noise_kept = [c for c in selected if c.startswith("noise")]
    assert len(informative_kept) >= 3
    assert len(noise_kept) <= 2
