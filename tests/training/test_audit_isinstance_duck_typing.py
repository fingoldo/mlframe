"""Wave-29 sensors: isinstance narrow checks missing duck-typed alternatives.

5 sites where ``isinstance(x, ConcreteClass)`` rejected legitimate
alternatives. Audit also confirmed strong overall hygiene
(~120/130 dispatch sites use proper pd/pl/np branches).

P1 #1 extractors.py:817 -- ``isinstance(exact_val, list)`` rejected
   tuples; ``classification_exact_values={"col": (1,2,3)}`` got
   wrapped as ``[(1,2,3)]`` then ``col_data == (1,2,3)`` raised.

P1 #2 mrmr.py:1281 -- polars DataFrame slipped past the np.ndarray
   coerce; downstream ``X[target_name] = y`` raised on polars
   in-place mutation. Added explicit polars -> pandas branch.

P1 #3 boruta_shap.py:872 -- ``np.array(explainer.shap_values(...))``
   wrapped BEFORE the ``isinstance(..., list)`` check, making the
   list branch DEAD CODE. Multi-class SHAP aggregation silently
   mis-counted importances (3-D ndarray branch ran instead).

P2 #4 pipeline.py:_filter_to_numeric -- polars DataFrame silently
   passed through; downstream ``_df.select_dtypes(...)`` raised
   AttributeError with no diagnostic naming the type. Coerce
   polars -> pandas explicitly.

P2 #5 core/main.py -- ``isinstance(df, (pd, pl, str))`` rejected
   ``pathlib.Path`` with a confusing "must be ... path string"
   message. Common caller idiom from yaml/Click configs. Coerce
   PathLike -> str at the boundary.
"""

from __future__ import annotations

import pytest

# ---- #1 extractors classification_exact_values accepts iterables -------


def _targets_for(exact_values):
    """Build targets over a two-row frame with ``classification_exact_values=exact_values``."""
    import pandas as pd

    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

    extractor = SimpleFeaturesAndTargetsExtractor(
        classification_targets=["label"],
        classification_exact_values=exact_values,
    )
    built = extractor.build_targets(pd.DataFrame({"label": [1, 2]}))
    names: set = set()
    for group in built.values():
        names |= set(group)
    return names


@pytest.mark.parametrize(
    "container",
    [
        pytest.param([1, 2], id="list"),
        pytest.param((1, 2), id="tuple"),
        pytest.param({1, 2}, id="set"),
        pytest.param(frozenset({1, 2}), id="frozenset"),
    ],
)
def test_exact_values_accepts_any_container_not_only_list(container):
    """Every iterable container must expand to one target per value.

    Behavioural since 2026-09-03. This asserted that the pre-fix line
    `exact_vals = exact_val if isinstance(exact_val, list) else [exact_val]` is absent and that
    `isinstance(exact_val, (list, tuple, set, frozenset))` is present. Both are claims about the
    text of one branch; neither says a tuple actually produces two targets, and both would pass a
    branch that had been made unreachable.

    Pre-fix, a tuple was wrapped whole as `[(1, 2)]`, so the comparison became
    `col_data == (1, 2)` -- which raises on pandas and on polars alike.
    """
    names = _targets_for({"label": container})

    assert names == {"label_eq_1", "label_eq_2"}, names


def test_a_scalar_exact_value_still_means_one_target():
    """The other side of the branch: a bare scalar must not be iterated into characters or digits."""
    assert _targets_for({"label": 1}) == {"label_eq_1"}


def test_a_string_exact_value_is_one_target_not_one_per_character():
    """`str` is iterable, so a container check that used a bare `Iterable` would explode "ab" into
    two targets. This is why the check enumerates container types rather than duck-typing."""
    import pandas as pd

    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

    extractor = SimpleFeaturesAndTargetsExtractor(
        classification_targets=["label"],
        classification_exact_values={"label": "ab"},
    )
    built = extractor.build_targets(pd.DataFrame({"label": ["ab", "cd"]}))
    names: set = set()
    for group in built.values():
        names |= set(group)

    assert names == {"label_eq_ab"}, names


# ---- #2 mrmr polars-coerce ---------------------------------------------


def test_mrmr_fit_handles_polars_input_without_inplace_mutation():
    """MRMR.fit accepts a polars frame and leaves it untouched: no injected target column, no changed values.

    The Wave-29 failure was ``X[target_name] = y`` on the caller's frame. This used to be pinned by searching
    ``_mrmr_class.py`` for an isinstance line, which broke when the fit was split into stage modules while the
    behaviour stayed correct; the behaviour is what is tested now.
    """
    import warnings

    import numpy as np
    import polars as pl

    from mlframe.feature_selection.filters import MRMR

    rng = np.random.default_rng(0)
    n = 600
    X = pl.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    y = (X["a"].to_numpy() + 0.1 * rng.normal(size=n) > 0).astype(int)
    before = X.clone()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs = MRMR(random_seed=0, verbose=0).fit(X=X, y=y)
    assert X.columns == before.columns and X.equals(before), "MRMR.fit mutated the caller's polars frame"
    assert "a" in list(getattr(fs, "support_names_", None) or np.asarray(before.columns)[fs.support_])


# ---- #3 boruta_shap multi-class branch revived -------------------------


def _explain_with_fake_shap(monkeypatch, shap_return, n_features):
    """Run ``BorutaShap.explain`` against a stub TreeExplainer returning ``shap_return``; return the aggregated importances."""
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from mlframe.feature_selection.boruta_shap import _fit_explain

    class _Explainer:
        """Stub explainer returning a canned SHAP payload."""

        def __init__(self, model, **kwargs):
            """Accept and ignore the model and kwargs."""

        def shap_values(self, basis):
            """Return the canned payload."""
            return shap_return

    monkeypatch.setattr(_fit_explain, "import_optional", lambda *a, **k: SimpleNamespace(TreeExplainer=_Explainer))
    n = 12
    frame = pd.DataFrame(np.zeros((n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    owner = SimpleNamespace(
        model_=object(),
        sample=False,
        X_boruta_=frame,
        X_=frame,
        y_=np.zeros(n, dtype=int),
        classification=True,
    )
    _fit_explain.explain(owner)
    return owner.shap_values_


def test_boruta_shap_inspects_raw_shap_type_before_array_wrap(monkeypatch):
    """A list of per-class SHAP arrays is aggregated as the mean over classes of the per-class mean |shap|."""
    import numpy as np

    rng = np.random.default_rng(0)
    per_class = [rng.normal(size=(12, 4)) for _ in range(3)]
    got = _explain_with_fake_shap(monkeypatch, per_class, n_features=4)
    expected = np.mean([np.abs(a).mean(0) for a in per_class], axis=0)
    assert got.shape == (4,)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


# ---- #4 pipeline _filter_to_numeric coerces polars ---------------------


def test_pipeline_filter_to_numeric_handles_polars():
    """A polars frame is coerced to pandas and filtered to its numeric columns, with the dropped names reported."""
    import pandas as pd
    import polars as pl

    from mlframe.training.pipeline._pipeline_extensions import _filter_to_numeric

    df = pl.DataFrame({"a": [1, 2, 3], "s": ["x", "y", "z"], "b": [0.5, 1.5, 2.5]})
    out, dropped = _filter_to_numeric(df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["a", "b"]
    assert dropped == ["s"]
    assert out["a"].tolist() == [1, 2, 3]
    assert out["b"].tolist() == [0.5, 1.5, 2.5]


# ---- #5 main.py PathLike coercion --------------------------------------


def test_main_accepts_pathlike_df_argument(tmp_path):
    """A ``pathlib.Path`` df is coerced to its string form at the suite boundary instead of being rejected."""
    from mlframe.training.core._main_train_suite_phases import validate_suite_inputs

    path = tmp_path / "data.parquet"
    out = validate_suite_inputs(path, "target", "model", object())
    assert out == str(path)
    assert isinstance(out, str)


def test_main_typeerror_message_mentions_pathlike():
    """The TypeError for an unsupported df type names PathLike as an accepted type and the offending type."""
    from mlframe.training.core._main_train_suite_phases import validate_suite_inputs

    with pytest.raises(TypeError, match="PathLike") as info:
        validate_suite_inputs(123, "target", "model", object())
    assert "int" in str(info.value)
