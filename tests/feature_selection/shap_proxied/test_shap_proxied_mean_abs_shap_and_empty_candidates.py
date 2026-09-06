"""Regression tests for two ``ShapProxiedFS`` defects surfaced by the Phase-0 FS benchmark.

Defect 1: ``shap_proxy_report_['mean_abs_shap']`` was never written by ANY module in
``shap_proxied_fs``, so ``registry._report_extract_shap_proxied_fs`` always returned
``scores=None`` and the benchmark's ShapProxiedFS arm had to declare ``score_kind='none'``
(no matched-K row, no ranking metric). The selector now persists the per-feature mean |SHAP|
its subset search ranks by, plus a ``mean_abs_shap_coverage`` block stating exactly how much
of the input frame the map spans.

Defect 2: ``min_features`` is an ORIGINAL-feature-space floor but the subset search enumerates
PROXY columns (correlated-feature clustering units, post-prescreen). On a frame whose columns all
collapse into ONE unit (the real ``hill-valley`` bed does exactly this: 100 columns, one unit),
``min_card=3`` over a 1-column proxy makes every cardinality illegal and the search returns zero
candidates, an unsatisfiable constraint rather than an honest "found no signal". The floor is now
clamped to the proxy width, and the genuinely-empty case raises the distinct
``ShapProxiedNoCandidatesError`` so a caller can tell "found nothing" from "crashed".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.shap_proxied_fs._shap_proxied_resolvers import (
    ShapProxiedNoCandidatesError, resolve_effective_min_features, unit_importance_to_feature_map)


def _collapsing_frame(n_rows: int = 400, n_cols: int = 12, seed: int = 0):
    """Build a frame whose columns are near-perfectly correlated, so clustering collapses them to one unit.

    Mirrors the hill-valley regime that produced the empty-candidate crash: a single latent driver
    plus tiny per-column jitter, so the correlated-feature clustering step folds every column into a
    single proxy unit while the target stays learnable from that unit.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n_rows)
    cols = {f"f{i}": latent + rng.normal(scale=1e-4, size=n_rows) for i in range(n_cols)}
    X = pd.DataFrame(cols)
    y = pd.Series((latent > 0).astype(int))
    return X, y


def test_resolve_effective_min_features_clamps_to_proxy_width():
    """The clamp keeps a satisfiable floor and degrades to the proxy width when the request cannot fit."""
    assert resolve_effective_min_features(3, 10) == 3, "a satisfiable floor must pass through untouched"
    assert resolve_effective_min_features(3, 1) == 1, "a 1-column proxy cannot honour min_card=3"
    assert resolve_effective_min_features(3, 2) == 2
    assert resolve_effective_min_features(0, 5) == 1, "never hand the optimizer a zero floor on a non-empty proxy"
    assert resolve_effective_min_features(3, 0) == 0, "degenerate empty proxy stays empty"


def test_no_candidates_error_is_a_runtime_error_subclass():
    """Existing ``except RuntimeError`` callers keep working while gaining a way to tell the cases apart."""
    assert issubclass(ShapProxiedNoCandidatesError, RuntimeError)


def test_unit_importance_expands_to_original_feature_names():
    """Cluster members share their unit's value; prefilter-dropped originals stay ABSENT, never zero-filled."""
    # Two units over working columns [0, 1, 2]; working col 2 maps to original col 5. Original
    # columns 1, 2 and 4 were dropped by the prefilter and must not appear in the map at all.
    out = unit_importance_to_feature_map(
        importance=np.array([0.5, 0.25]), unit_to_members=[[0, 1], [2]], working_cols=np.array([0, 3, 5]), feature_names=[f"c{i}" for i in range(6)]
    )
    assert out == {"c0": 0.5, "c3": 0.5, "c5": 0.25}
    assert "c1" not in out and "c4" not in out, "a feature with no SHAP attribution must not be invented as 0.0"


def test_registry_extractor_does_not_use_default_via_or():
    """``x.get(a) or x.get(b)`` falls through on an empty dict AND on a falsy value; the chain must test ``is None``."""
    from mlframe.feature_selection.registry import _report_extract_shap_proxied_fs

    class _Sel:
        """Minimal stand-in exposing only the attributes the extractor reads."""

        shap_proxy_report_ = {"mean_abs_shap": {}, "importances": {"a": 1.0}}
        selected_features_: list = []
        feature_names_in_ = ["a"]

    # Pre-fix the empty ``mean_abs_shap`` fell through the ``or`` to ``importances``; post-fix the
    # present-but-empty primary key wins and the fallback is never consulted.
    out = _report_extract_shap_proxied_fs(_Sel(), None)
    assert out["scores"] is None, f"empty primary key must not fall through to the fallback: {out['scores']}"


pytest.importorskip("shap")
pytest.importorskip("xgboost")


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_mean_abs_shap_is_persisted_and_readable_by_the_registry():
    """The selector writes ``mean_abs_shap`` and the registry extractor turns it into non-None ``scores``.

    Pre-fix signature: the key is absent from ``shap_proxy_report_`` entirely and
    ``_report_extract_shap_proxied_fs`` returns ``scores=None``.
    """
    from mlframe.feature_selection.registry import _report_extract_shap_proxied_fs
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(8)})
    y = pd.Series((X["f0"] + X["f1"] + 0.3 * rng.normal(size=n) > 0).astype(int))
    sel = ShapProxiedFS(
        classification=True, n_splits=3, top_n=6, min_features=2, oof_shap_n_estimators=40, revalidate=False, trust_guard=False, random_state=0, verbose=False
    )
    sel.fit(X, y)
    report = sel.shap_proxy_report_
    assert "mean_abs_shap" in report, f"mean_abs_shap never written; keys={sorted(report)}"
    scores = report["mean_abs_shap"]
    assert isinstance(scores, dict) and scores
    coverage = report["mean_abs_shap_coverage"]
    assert coverage["n_input_features"] == X.shape[1]
    assert coverage["n_covered"] == len(scores)
    assert coverage["complete"] is True, f"no prefilter ran on an 8-column frame, so coverage must be complete: {coverage}"
    assert set(scores) == set(X.columns)
    assert all(float(v) >= 0.0 for v in scores.values()), "mean |SHAP| is nonnegative by construction"
    out = _report_extract_shap_proxied_fs(sel, None)
    assert out["scores"] is not None and len(out["scores"]) == X.shape[1]


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_min_features_wider_than_proxy_does_not_raise_no_candidate_subsets():
    """A ``min_features`` floor the proxy width cannot satisfy must degrade, not empty the candidate list.

    Deterministic stand-in for the hill-valley regime: there the 100 columns collapse to ONE clustering
    unit so ``min_features=3`` becomes unsatisfiable; here the floor exceeds the frame width outright, so
    the same unsatisfiable-constraint branch is hit without depending on how aggressively clustering
    happens to collapse a correlated frame.

    Pre-fix signature: ``RuntimeError: ShapProxiedFS: search produced no candidate subsets.``
    """
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = _collapsing_frame(n_cols=6)
    sel = ShapProxiedFS(classification=True, n_splits=3, top_n=6, min_features=8, oof_shap_n_estimators=40, revalidate=False, random_state=0, verbose=False)
    sel.fit(X, y)
    assert len(sel.selected_features_) >= 1, "an informative frame must yield a non-empty selection"
    clamp = sel.shap_proxy_report_.get("min_features_clamped")
    assert clamp is not None, "the clamp actually fired here, so the report must record it"
    assert clamp["requested"] == 8 and 1 <= clamp["effective"] <= clamp["n_proxy_cols"] <= X.shape[1]
