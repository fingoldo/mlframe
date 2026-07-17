"""iter75: SU is the unconditional default clustering backend in ShapProxiedFS.

Locks:
* Default ``cluster_backend='auto'`` picks SU at moderate width WITHOUT precomputed bins;
  X is binned on the fly via ``categorize_dataset`` so every default user gets non-linear
  redundancy detection automatically. Report surfaces ``backend='su'`` and
  ``bins_source='on_the_fly'``.
* ``cluster_backend='pearson'`` deterministically falls back to the legacy correlation path.
* ``cluster_backend='su'`` forces SU even at widths above the auto cap.
* Wide-regime auto: ``n_features > cluster_su_auto_max_features`` falls back to Pearson under
  auto so the O(f^2) pairwise SU does not pay an unbounded wall-clock cost.
* Precomputed-bins reuse: when MRMR's ``export_artifacts['bins']`` already covers every working
  column the on-the-fly binner is bypassed; ``bins_source='precomputed'``.
* biz_value conservation: at the default auto config the chosen subset's recall on synthetic
  regime data is NO WORSE than the explicit Pearson backend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_regime(n_samples=1500, n_informative=5, n_redundant=5, n_noise=70, seed=0):
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import (
        make_regime_dataset,
    )

    return make_regime_dataset(
        n_samples=n_samples,
        n_informative=n_informative,
        n_redundant=n_redundant,
        redundancy_rho=0.8,
        n_noise=n_noise,
        snr=8.0,
        task="binary",
        seed=seed,
    )


def _common_kwargs():
    return dict(
        random_state=0,
        verbose=False,
        prefilter_top=30,
        max_features=5,
        n_models=1,
        n_splits=2,
        out_of_fold=False,
        revalidate=False,
        trust_guard=False,
        run_importance_ablation=False,
        cluster_features=True,
        cluster_auto_threshold=10,
        brute_force_max_features=12,
        shap_prefilter_enabled=False,
    )


def test_default_auto_picks_su_without_precomputed_at_small_width():
    """iter75 capstone: the default ShapProxiedFS().fit() ships SU clustering with X binned
    on the fly. Without any precomputed bundle the selector still routes through SU."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y, _ = _make_regime(seed=0)
    sps = ShapProxiedFS(**_common_kwargs()).fit(X, y)
    rep = sps.shap_proxy_report_
    assert rep["clustering"]["backend"] == "su"
    assert rep["clustering"]["bins_source"] == "on_the_fly"


def test_explicit_pearson_overrides_default_su():
    """Opt-out via ``cluster_backend='pearson'`` keeps the legacy Pearson clustering."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y, _ = _make_regime(seed=0)
    sps = ShapProxiedFS(cluster_backend="pearson", **_common_kwargs()).fit(X, y)
    rep = sps.shap_proxy_report_
    assert rep["clustering"]["backend"] == "pearson"
    assert rep["clustering"]["bins_source"] == "n/a"


def test_forced_su_runs_even_above_auto_cap():
    """``cluster_backend='su'`` forces SU regardless of the auto-width gate."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y, _ = _make_regime(seed=2)
    kw = _common_kwargs()
    sps = ShapProxiedFS(
        cluster_backend="su",
        # Force the auto cap below the actual width so 'auto' would fall back, but the
        # explicit 'su' still runs SU.
        cluster_su_auto_max_features=8,
        **kw,
    ).fit(X, y)
    rep = sps.shap_proxy_report_
    assert rep["clustering"]["backend"] == "su"


def test_auto_falls_back_to_pearson_above_width_cap():
    """At ``n_features > cluster_su_auto_max_features`` the auto mode reverts to Pearson."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y, _ = _make_regime(seed=3)
    # Cap intentionally below n_features so the auto branch picks Pearson without precomputed.
    sps = ShapProxiedFS(
        cluster_backend="auto",
        cluster_su_auto_max_features=4,
        **_common_kwargs(),
    ).fit(X, y)
    rep = sps.shap_proxy_report_
    assert rep["clustering"]["backend"] == "pearson"


def test_precomputed_bins_path_skips_on_the_fly_binning():
    """When MRMR.export_artifacts['bins'] covers every working column the on-the-fly
    binner is bypassed and SU consumes the precomputed view directly."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y, _ = _make_regime(seed=4)
    artifacts = (
        MRMR(
            retain_artifacts=True,
            dcd_enable=False,
            build_friend_graph=False,
            cluster_aggregate_enable=False,
            verbose=0,
        )
        .fit(X, y)
        .export_artifacts()
    )
    sps = ShapProxiedFS(precomputed=artifacts, **_common_kwargs()).fit(X, y)
    rep = sps.shap_proxy_report_
    assert rep["clustering"]["backend"] == "su"
    assert rep["clustering"]["bins_source"] == "precomputed"


def test_invalid_backend_rejected():
    """Constructor validates the backend literal."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    with pytest.raises(ValueError, match="cluster_backend"):
        ShapProxiedFS(cluster_backend="bogus")


def _recall(selected_names, roles) -> float:
    """Fraction of informative columns ('inf{i}') captured in the selection."""
    informative_names = {n for n, role in roles.items() if role == "informative"}
    chosen = set(selected_names)
    hit = len(informative_names & chosen)
    return hit / max(len(informative_names), 1)


def test_biz_value_recall_conservation_auto_vs_pearson():
    """KEY CONSERVATION CHECK: at the default config the SU auto backend must not drop recall
    on the synthetic regime informative set vs the explicit Pearson backend. Either backend
    may surface a different cluster boundary but SU should be no worse on retained informative
    feature recall (any tie is fine; SU strictly worse triggers HONEST-NEGATIVE)."""
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    n_informative = 5
    X, y, roles = make_regime_dataset(
        n_samples=1500,
        n_informative=n_informative,
        n_redundant=5,
        redundancy_rho=0.8,
        n_noise=70,
        snr=8.0,
        task="binary",
        seed=0,
    )

    common = _common_kwargs()
    common["max_features"] = 10

    sps_su = ShapProxiedFS(cluster_backend="auto", **common).fit(X, y)
    sps_p = ShapProxiedFS(cluster_backend="pearson", **common).fit(X, y)

    sel_su = list(sps_su.selected_features_)
    sel_p = list(sps_p.selected_features_)

    recall_su = _recall(sel_su, roles)
    recall_p = _recall(sel_p, roles)

    assert recall_su >= recall_p - 1e-9, (
        f"iter75 recall regression: SU auto={recall_su:.3f} < Pearson={recall_p:.3f}\n  selected_su={sorted(sel_su)}\n  selected_p={sorted(sel_p)}"
    )
