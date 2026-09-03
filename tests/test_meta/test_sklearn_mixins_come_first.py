"""Every sklearn mixin must precede `BaseEstimator` in the class bases, repo-wide.

sklearn resolves `__sklearn_tags__` through the MRO and `BaseEstimator`'s implementation is terminal: once it
precedes a mixin, the mixin's override is never reached. Under sklearn >= 1.6 that makes `is_classifier` and
`is_regressor` answer False for the estimator -- silently, with no error at import or at fit.

The 2026-09-01 audit found this on the seven estimators in `estimators/`, where the visible symptom was
`get_best_dummy_score` raising `TypeError: estimator must be a sklearn classifier or regressor`. Sweeping the
tree found 66 classes with the same ordering. The wider costs are quieter: `GridSearchCV` and `cross_val_score`
pick plain `KFold` instead of `StratifiedKFold` for a classifier they cannot recognise, and default-scoring
resolution picks the wrong metric.

This is a source-shape gate rather than an instantiate-everything gate on purpose: many of these estimators
require constructor arguments, and a shape check has no import-order or heavy-dependency cost. The behavioural
half is covered by the spot checks at the bottom.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "mlframe"

# The mixins whose `__sklearn_tags__` override decides the estimator type. Project-local mixins are deliberately
# not listed: their ordering carries no sklearn meaning and is the author's to choose.
SKLEARN_MIXINS = frozenset(
    {
        "ClassifierMixin",
        "RegressorMixin",
        "TransformerMixin",
        "MultiOutputMixin",
        "SelectorMixin",
        "ClusterMixin",
        "BiclusterMixin",
        "DensityMixin",
        "OutlierMixin",
    }
)


def _base_names(node: ast.ClassDef) -> list:
    """Base class names as written, ignoring subscripted or attribute forms we cannot order meaningfully."""
    out = []
    for b in node.bases:
        if isinstance(b, ast.Name):
            out.append(b.id)
        elif isinstance(b, ast.Attribute):
            out.append(b.attr)
    return out


def _class_bases() -> dict:
    """Repo-wide map of class name -> declared base names, for a transitive `BaseEstimator` check."""
    out: dict = {}
    for path in sorted(SRC.rglob("*.py")):
        if "_benchmarks" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                out.setdefault(node.name, set()).update(_base_names(node))
    return out


def _is_estimator_base(name: str, graph: dict, seen: "set | None" = None) -> bool:
    """True when `name` is `BaseEstimator` or reaches it through declared bases."""
    if name == "BaseEstimator":
        return True
    seen = seen if seen is not None else set()
    if name in seen:
        return False
    seen.add(name)
    return any(_is_estimator_base(b, graph, seen) for b in graph.get(name, ()))


def _offenders() -> list:
    """Every class where a sklearn mixin is declared after a base that carries `BaseEstimator`.

    The rule is specifically about `BaseEstimator`, not about ordering against project bases generally. A
    project mixin may legitimately need to come FIRST -- `SelectorMixin` declares `_get_support_mask` abstract
    and also defines `transform`, so a local mixin supplying both has to precede it or the class is left
    abstract and its own `transform` is shadowed. Both of those happened while this gate was written as
    "before any non-mixin base".
    """
    graph = _class_bases()
    found = []
    for path in sorted(SRC.rglob("*.py")):
        if "_benchmarks" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = _base_names(node)
            mixins = [i for i, b in enumerate(bases) if b in SKLEARN_MIXINS]
            estimators = [i for i, b in enumerate(bases) if b not in SKLEARN_MIXINS and _is_estimator_base(b, graph)]
            if mixins and estimators and max(mixins) > min(estimators):
                found.append(f"{path.relative_to(SRC).as_posix()}:{node.lineno}:{node.name}({', '.join(bases)})")
    return found


def test_no_estimator_declares_a_sklearn_mixin_after_its_base():
    """The gate. A new class written the wrong way round is silent at runtime until something asks its type."""
    offenders = _offenders()
    assert not offenders, (
        "sklearn mixins must precede BaseEstimator (or any base inheriting it), otherwise "
        "BaseEstimator.__sklearn_tags__ shadows the mixin's and is_classifier / is_regressor return False:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize(
    "dotted,kind",
    [
        ("mlframe.estimators.custom:PureRandomClassifier", "classifier"),
        ("mlframe.estimators.custom:IdentityClassifier", "classifier"),
        ("mlframe.estimators.custom:IdentityRegressor", "regressor"),
        ("mlframe.estimators.base:ClassifierWithEarlyStopping", "classifier"),
        ("mlframe.estimators.base:RegressorWithEarlyStopping", "regressor"),
        ("mlframe.training.composite.classification:CompositeClassificationEstimator", "classifier"),
        ("mlframe.training.composite.estimator._estimator:CompositeTargetEstimator", "regressor"),
    ],
)
def test_sklearn_recognises_the_estimator_type(dotted, kind):
    """The behaviour the shape gate protects, asserted on real instances."""
    import importlib

    from sklearn.base import is_classifier, is_regressor

    module_name, cls_name = dotted.split(":")
    obj = getattr(importlib.import_module(module_name), cls_name)()
    assert is_classifier(obj) if kind == "classifier" else is_regressor(obj), f"{cls_name} is not recognised as a {kind}"


def test_the_dummy_baseline_helper_accepts_our_own_estimators():
    """The concrete breakage the audit surfaced: this helper rejected every in-repo estimator."""
    from sklearn.base import is_classifier

    from mlframe.estimators.custom import PureRandomClassifier

    assert is_classifier(PureRandomClassifier()), "get_best_dummy_score raises TypeError for anything failing this"


def test_every_estimator_in_the_registry_is_instantiable():
    """`SelectorMixin` declares `_get_support_mask` abstract, so a local mixin supplying it must come FIRST.

    Getting that backwards leaves the class abstract, and the failure surfaces far away -- `MRMR` became
    uninstantiable through the registry factory with `TypeError: Can't instantiate abstract class MRMR without
    an implementation for abstract method '_get_support_mask'`, from a suite-level end-to-end test. A base-order
    gate that only reads source shape cannot see this; instantiating can.
    """
    from mlframe.feature_selection.filters import MRMR
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    assert not MRMR.__abstractmethods__, sorted(MRMR.__abstractmethods__)
    assert MRMR() is not None
    assert ShapProxiedFS() is not None


def test_the_local_implementation_wins_over_the_sklearn_mixin():
    """The quieter half of the same defect: `SelectorMixin` also DEFINES `transform`, so a mixin placed after it
    keeps the class concrete-looking while its own `transform` is silently shadowed."""
    from mlframe.feature_selection.filters.mrmr._mrmr_class_transform import _MRMRTransformMixin
    from mlframe.feature_selection.filters import MRMR
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS
    from mlframe.feature_selection.shap_proxied_fs._shap_proxied_methods import ShapProxiedMethodsMixin

    assert MRMR._get_support_mask is _MRMRTransformMixin._get_support_mask
    assert ShapProxiedFS.transform is ShapProxiedMethodsMixin.transform


def test_both_sklearn_init_subclass_chains_still_run():
    """Moving a mixin ahead of `BaseEstimator` changes which `__init_subclass__` is the entry point
    (`_SetOutputMixin` instead of `_MetadataRequester`). Both must still run, or metadata routing or the
    set_output transform wrapping goes missing with no error anywhere."""
    from mlframe.feature_selection.ace import ACESelector

    inst = ACESelector()
    assert inst._get_metadata_request() is not None
    assert getattr(type(inst), "_sklearn_auto_wrap_output_keys", None) == {"transform"}
