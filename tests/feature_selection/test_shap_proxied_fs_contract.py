"""sklearn-contract tests for ``ShapProxiedFS`` (mirrors the BorutaShap contract tests).

Covers: NotFittedError before fit, support_ shape/dtype + agreement with selected_features_,
name-based transform that preserves input column order under reordering, polars input, registry
registration, and RNG isolation (construction must not mutate the global numpy RNG).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")
pytest.importorskip("xgboost")


def _fit_small():
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    rng = np.random.default_rng(0)
    n = 1500
    inf = rng.normal(size=(n, 3))
    X = pd.DataFrame(np.column_stack([inf, rng.normal(size=(n, 3))]),
                     columns=["a", "b", "c", "d", "e", "f"])
    y = (0.9 * inf[:, 0] + 0.8 * inf[:, 1] - 0.7 * inf[:, 2] + 0.3 * rng.normal(size=n) > 0).astype(int)
    sel = ShapProxiedFS(classification=True, metric="brier", optimizer="bruteforce",
                        max_features=4, top_n=10, n_splits=3, n_revalidation_models=1,
                        trust_guard=False, random_state=0, verbose=False,
                        n_jobs=1)
    sel.fit(X, y)
    return sel, X, y


def test_transform_before_fit_raises_notfitted():
    from sklearn.exceptions import NotFittedError

    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    sel = ShapProxiedFS()
    with pytest.raises(NotFittedError):
        sel.transform(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    with pytest.raises(NotFittedError):
        sel.get_support()


def test_support_and_selected_features_consistent():
    sel, X, _ = _fit_small()
    assert sel.support_.shape == (6,)
    assert sel.support_.dtype == bool
    assert sel.n_features_in_ == 6
    support_named = {c for c, m in zip(X.columns, sel.support_) if m}
    assert support_named == set(sel.selected_features_)
    assert list(sel.get_support(indices=True)) == list(np.where(sel.support_)[0])


def test_transform_preserves_input_column_order():
    sel, X, _ = _fit_small()
    # transform on a column-reordered frame must still return the selected columns (name-based).
    X_shuffled = X[list(reversed(X.columns))]
    out = sel.transform(X_shuffled)
    assert list(out.columns) == list(sel.selected_features_)
    # values must match the original (name-based selection, not positional)
    for c in sel.selected_features_:
        np.testing.assert_array_equal(out[c].to_numpy(), X[c].to_numpy())


def test_polars_input_supported():
    pl = pytest.importorskip("polars")
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    rng = np.random.default_rng(0)
    n = 1200
    inf = rng.normal(size=(n, 3))
    data = np.column_stack([inf, rng.normal(size=(n, 2))])
    Xpl = pl.DataFrame(data, schema=["a", "b", "c", "d", "e"])
    y = (0.9 * inf[:, 0] + 0.8 * inf[:, 1] - 0.6 * inf[:, 2] + 0.3 * rng.normal(size=n) > 0).astype(int)
    sel = ShapProxiedFS(classification=True, metric="brier", optimizer="bruteforce", max_features=4,
                        top_n=8, n_splits=3, n_revalidation_models=1, trust_guard=False,
                        random_state=0, verbose=False, n_jobs=1)
    sel.fit(Xpl, pl.Series(y))
    assert sel.n_features_in_ == 5
    out = sel.transform(Xpl)
    assert out.shape[1] == len(sel.selected_features_)


def test_construction_does_not_mutate_global_numpy_rng():
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    np.random.seed(12345)
    before = np.random.get_state()[1][:5].copy()
    ShapProxiedFS(random_state=7)
    after = np.random.get_state()[1][:5].copy()
    np.testing.assert_array_equal(before, after)


def test_registered_in_registry():
    from mlframe.feature_selection import registry

    assert "ShapProxiedFS" in registry.available()
    spec = registry.get("ShapProxiedFS")
    sel = spec.instantiate(classification=True, optimizer="beam")
    assert type(sel).__name__ == "ShapProxiedFS"


def test_facade_spearman_floor_kwarg_deprecated_and_aliased_to_fidelity_floor():
    """Iter18 rename sentinel at the FACADE level: ``ShapProxiedFS(spearman_floor=...)`` must still
    work but emit a DeprecationWarning at fit-time. Setting BOTH ``fidelity_floor`` and
    ``spearman_floor`` on the facade must raise ``ValueError``. The legacy kwarg flows into the
    same gate as the new name (verified via the report's ``fidelity_floor`` field)."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    rng = np.random.default_rng(0)
    n = 600
    inf = rng.normal(size=(n, 3))
    X = pd.DataFrame(np.column_stack([inf, rng.normal(size=(n, 3))]),
                     columns=["a", "b", "c", "d", "e", "f"])
    y = (0.9 * inf[:, 0] + 0.8 * inf[:, 1] - 0.7 * inf[:, 2] + 0.3 * rng.normal(size=n) > 0).astype(int)

    # Construction with legacy kwarg succeeds (no deprecation at __init__).
    sel = ShapProxiedFS(classification=True, metric="brier", optimizer="bruteforce",
                        max_features=4, top_n=10, n_splits=3, n_revalidation_models=1,
                        trust_guard=True, n_anchors=8, spearman_floor=0.55,
                        random_state=0, verbose=False, n_jobs=1)
    assert sel.spearman_floor == 0.55
    # iter18 (commit 63a296fd): ``fidelity_floor`` defaults to the ``None`` unset
    # sentinel, not the literal 0.5 -- this preserves sklearn ``clone()`` identity
    # and lets the both-floors-set conflict be detected via ``fidelity_floor is not
    # None`` (an explicit ``fidelity_floor=0.5`` is no longer mistaken for the
    # default). The EFFECTIVE default is unchanged at 0.5: ``effective_floor =
    # self.fidelity_floor if self.fidelity_floor is not None else 0.5`` at fit time.
    assert sel.fidelity_floor is None

    # Fit emits the deprecation warning AND the legacy value reaches the trust report.
    with pytest.warns(DeprecationWarning, match="spearman_floor"):
        sel.fit(X, y)
    assert sel.shap_proxy_report_["trust"]["fidelity_floor"] == 0.55

    # Setting both raises at fit-time (we can't catch it at __init__ without breaking sklearn's
    # "no validation in __init__" rule).
    sel_both = ShapProxiedFS(classification=True, metric="brier", optimizer="bruteforce",
                             max_features=4, top_n=10, n_splits=3, n_revalidation_models=1,
                             trust_guard=True, n_anchors=8,
                             fidelity_floor=0.4, spearman_floor=0.6,
                             random_state=0, verbose=False, n_jobs=1)
    with pytest.raises(ValueError, match="fidelity_floor.*spearman_floor"):
        sel_both.fit(X, y)
