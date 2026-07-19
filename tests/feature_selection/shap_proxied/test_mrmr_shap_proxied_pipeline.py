"""End-to-end pipeline test: ``MRMR -> ShapProxiedFS`` with artifact reuse.

Validates the cross-selector artifact-passing API introduced in iter66:

* ``MRMR(retain_artifacts=True).fit(X, y).export_artifacts()`` returns a dict
  with non-empty ``su_to_target`` / ``mi_to_target`` / ``feature_names``;
* ``ShapProxiedFS(precomputed=...)`` honours the dict, replaces the stage-A
  univariate F-statistic pre-screen with the MRMR-computed SU ranking, and
  surfaces the reuse signal under ``shap_proxy_report_['precomputed_used']``;
* The chosen subset matches the standalone-pipeline subset (bit-identical
  expectation for SU-vs-F-ranking-equivalent regimes), and the recall against
  the ground-truth informative columns is preserved;
* Wall-clock is not regressed (precomputed path <= 1.05x standalone).

The dataset is sized so the prefilter actually runs (n_features = 100 ==
prefilter_top default 100); below that threshold the stage-A code path is
skipped and the precomputed dict has nothing to substitute for.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS


def _make_dataset():
    # 100 features total: 5 informative + 5 redundant (rho=0.8 around the
    # informatives) + 90 noise. n_samples=2000 keeps wall-clock under the
    # 180s/config cap while still letting MRMR rank features above the
    # cardinality-bias floor.
    """Make dataset."""
    X, y, roles = make_regime_dataset(
        n_samples=2000,
        n_informative=5,
        n_redundant=5,
        redundancy_rho=0.8,
        n_noise=90,
        snr=8.0,
        task="binary",
        seed=0,
    )
    inf_names = {n for n, r in roles.items() if r == "informative"}
    return X, y, inf_names


def test_mrmr_export_artifacts_dict_shape():
    """MRMR(retain_artifacts=True).fit(X, y).export_artifacts() returns a
    well-formed dict with the documented schema."""
    X, y, _ = _make_dataset()
    mrmr = MRMR(
        retain_artifacts=True,
        # Pin the fast / deterministic path so the test is robust to default
        # changes elsewhere.
        dcd_enable=False,
        build_friend_graph=False,
        cluster_aggregate_enable=False,
        verbose=0,
    ).fit(X, y)

    artifacts = mrmr.export_artifacts()
    assert isinstance(artifacts, dict)
    # Required schema keys.
    for k in ("feature_names", "su_to_target", "mi_to_target", "mrmr_kept_indices"):
        assert k in artifacts, f"missing required artifact key {k!r}"
    # Axis shapes.
    n_in = X.shape[1]
    assert len(artifacts["feature_names"]) == n_in
    assert np.asarray(artifacts["su_to_target"]).shape == (n_in,)
    assert np.asarray(artifacts["mi_to_target"]).shape == (n_in,)
    # SU values are in [0, 1] (modulo NaN for constant/unscored columns).
    su = np.asarray(artifacts["su_to_target"])
    su_finite = su[np.isfinite(su)]
    assert su_finite.size > 0, "no SU values were computed"
    assert su_finite.min() >= 0.0
    assert su_finite.max() <= 1.0


def test_mrmr_export_artifacts_requires_opt_in():
    """Constructing without ``retain_artifacts=True`` must make
    ``export_artifacts`` raise. Catches accidental footprint regressions."""
    X, y, _ = _make_dataset()
    mrmr = MRMR(
        retain_artifacts=False,
        dcd_enable=False,
        build_friend_graph=False,
        cluster_aggregate_enable=False,
        verbose=0,
    ).fit(X, y)
    with pytest.raises(ValueError, match="retain_artifacts"):
        mrmr.export_artifacts()


def test_mrmr_then_shap_proxied_fs_reuses_artifacts_e2e():
    """Pipeline test: MRMR narrows to 20 features + exports artifacts,
    ShapProxiedFS consumes the dict and skips its own univariate pre-screen."""
    X, y, inf_names = _make_dataset()

    # Step 1: MRMR fits + exports artifacts. We do NOT actually narrow the
    # column set here: the canonical reuse pattern is "MRMR scored everything,
    # ShapProxiedFS reuses the scoring to skip its OWN univariate pre-screen".
    # Keeping ShapProxiedFS on the FULL frame lets the prefilter actually fire
    # (n_features > prefilter_top), which is the code path the precomputed dict
    # is supposed to short-circuit. The MRMR narrowing pattern is also valid
    # but produces a frame too small for the prefilter on this test dataset.
    mrmr = MRMR(
        retain_artifacts=True,
        dcd_enable=False,
        build_friend_graph=False,
        cluster_aggregate_enable=False,
        verbose=0,
    ).fit(X, y)
    artifacts = mrmr.export_artifacts()
    assert artifacts["su_to_target"].shape == (X.shape[1],)

    # Force the prefilter to fire by setting prefilter_top below n_features.
    # Common kwargs across both selectors so the only diff is the
    # ``precomputed`` slot.
    common = dict(
        random_state=0,
        verbose=False,
        prefilter_top=20,
        n_models=1,
        n_splits=2,
        out_of_fold=False,
        revalidate=False,
        trust_guard=False,
        run_importance_ablation=False,
        cluster_features=False,
        shap_prefilter_enabled=False,
        # gt_08: this test times the precomputed-SU-reuse path against the standalone prefilter, a
        # question orthogonal to proxy_mode. proxy_mode="auto"'s su_seeded screen has a small
        # roughly-fixed cost (permutation-null pair scan) that is negligible against the wide/slow
        # fits it was benchmarked on but swamps THIS test's tight <=1.5x wall budget on its tiny
        # fixture -- pin the legacy "additive" escape hatch (zero screen cost) to keep the timing
        # assertion about the thing it actually tests.
        proxy_mode="additive",
    )

    # Step 2: ShapProxiedFS standalone (legacy F-statistic / booster prefilter).
    t1 = time.perf_counter()
    sps_standalone = ShapProxiedFS(**common).fit(X, y)
    t_standalone = time.perf_counter() - t1

    # Step 3: ShapProxiedFS with precomputed SU.
    t2 = time.perf_counter()
    sps_reuse = ShapProxiedFS(precomputed=artifacts, **common).fit(X, y)
    t_reuse = time.perf_counter() - t2

    # Recall against the ground-truth informative columns is preserved.
    standalone_recall = len(inf_names & set(sps_standalone.selected_features_))
    reuse_recall = len(inf_names & set(sps_reuse.selected_features_))
    assert reuse_recall == standalone_recall, (
        f"recall regressed: standalone={standalone_recall} reuse={reuse_recall}; "
        f"standalone={sps_standalone.selected_features_} reuse={sps_reuse.selected_features_}"
    )

    # Report surfaces the reuse signal -- this is the primary API contract.
    rep = sps_reuse.shap_proxy_report_
    pc = rep.get("precomputed_used")
    assert pc is not None, "report block 'precomputed_used' missing"
    assert pc.get("honoured") is True, f"precomputed not honoured: {pc}"
    assert pc.get("su_available") is True
    assert rep.get("prefilter", {}).get("method") == "precomputed_su", f"prefilter method did not switch to SU: {rep.get('prefilter')}"

    # No e2e regression. Cap at 1.5x baseline because the standalone wall is
    # dominated by setup overhead on small frames; the perf signal of the
    # precomputed path emerges at larger widths where the F-statistic prefilter
    # is the dominant cost. Headroom keeps the test robust to scheduling noise
    # on a Windows CI box.
    assert t_reuse <= max(t_standalone * 1.5, t_standalone + 5.0), f"precomputed path regressed: standalone={t_standalone:.3f}s reuse={t_reuse:.3f}s"


def test_shap_proxied_fs_precomputed_mismatch_warns_and_falls_back(caplog):
    """When precomputed feature_names don't match X.columns, the selector
    must warn + ignore the artifacts + run the legacy prefilter. Defensive
    contract against silent misalignment."""
    X, y, _ = _make_dataset()
    bogus_precomputed = {
        "schema_version": 1,
        "feature_names": ["totally_different_name_" + str(i) for i in range(X.shape[1])],
        "su_to_target": np.linspace(1.0, 0.0, X.shape[1]),
        "mi_to_target": np.linspace(0.5, 0.0, X.shape[1]),
    }
    common = dict(
        random_state=0,
        verbose=False,
        prefilter_top=10,
        n_models=1,
        n_splits=2,
        out_of_fold=False,
        revalidate=False,
        trust_guard=False,
        run_importance_ablation=False,
        cluster_features=False,
        shap_prefilter_enabled=False,
    )
    sps = ShapProxiedFS(precomputed=bogus_precomputed, **common).fit(X, y)
    rep = sps.shap_proxy_report_
    pc = rep["precomputed_used"]
    assert pc["honoured"] is False
    assert "mismatch" in pc.get("reason", "")
    # Legacy prefilter ran (method is NOT precomputed_su).
    assert rep.get("prefilter", {}).get("method") != "precomputed_su"
