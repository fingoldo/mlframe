"""CatBoost non-symmetric trees (``grow_policy`` Depthwise / Lossguide) as a flat ``TreeEnsemble``, so mlframe's own TreeSHAP kernels explain them.

CatBoost's native ``ShapValues`` is exact for symmetric (oblivious) trees but its non-symmetric code path does not add up to the prediction (measured: up to 0.9 on Depthwise and 2.5 on Lossguide models, on individual
rows, with a zero mean error), so those models are explained here from the JSON export instead: splits, leaf values and leaf weights (node weights are the sums of their leaves). Models with categorical or
text features, multi-output leaves and symmetric trees are declined (``None``) and keep the native path.
"""

from __future__ import annotations

import logging
import tempfile
import weakref
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import orjson

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import _NO_CHILD, TreeEnsemble, _ensemble_expected_value

from mlframe.utils.log_throttle import log_throttle
logger = logging.getLogger(__name__)


def _model_json(est: Any) -> dict:
    """The CatBoost JSON export of a fitted model."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.json"
        est.save_model(str(path), format="json")
        out: dict = orjson.loads(path.read_bytes())
        return out


def is_nonsymmetric_catboost(est: Any) -> bool:
    """True for a fitted CatBoost model whose trees are not oblivious (the case the native SHAP gets wrong)."""
    try:
        return str(est.get_params().get("grow_policy", "SymmetricTree")) in ("Depthwise", "Lossguide")
    except Exception as exc:
        log_throttle(logger, "treeshap_catboost.params_unreadable", logging.WARNING, "CatBoost grow_policy unreadable (%s: %s); the model is treated as symmetric", type(exc).__name__, exc)
        return False


def extract_catboost_nonsymmetric(est: Any) -> Optional[TreeEnsemble]:
    """The flat ensemble of a fitted non-symmetric single-output CatBoost model with float features only, else ``None``."""
    if not is_nonsymmetric_catboost(est):
        return None
    d = _model_json(est)
    info = d.get("features_info", {})
    if info.get("categorical_features") or info.get("text_features") or info.get("embedding_features"):
        return None
    if "trees" not in d or d.get("oblivious_trees"):
        return None
    floats = info.get("float_features", [])
    n_features = len(floats)
    nan_right = {int(f["feature_index"]): f.get("nan_value_treatment") == "AsTrue" for f in floats}
    scale_bias = d.get("scale_and_bias", [1.0, [0.0]])
    scale = float(scale_bias[0])
    bias = scale_bias[1]
    if not isinstance(bias, list) or len(bias) != 1:
        return None
    cl: list[int] = []
    cr: list[int] = []
    cd: list[int] = []
    feat: list[int] = []
    thr: list[float] = []
    val: list[float] = []
    cover: list[float] = []
    roots: list[int] = []
    max_depth = 0
    for tree in d["trees"]:
        roots.append(len(cl))
        flat: list[dict] = []

        def _collect(node: dict, depth: int, flat: list = flat, base: int = len(cl)) -> float:
            """Pre-order ids; returns the node weight (a leaf's weight, else the sum of its children)."""
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            node["_gid"] = base + len(flat)
            flat.append(node)
            if "split" in node:
                node["_w"] = _collect(node["left"], depth + 1) + _collect(node["right"], depth + 1)
            else:
                node["_w"] = float(node["weight"])
            return float(node["_w"])

        _collect(tree, 0)
        for node in flat:
            if "split" in node:
                sp = node["split"]
                if sp.get("split_type") != "FloatFeature":
                    return None
                f = int(sp["float_feature_index"])
                border = np.float32(sp["border"])
                cl.append(node["left"]["_gid"])
                cr.append(node["right"]["_gid"])
                cd.append(node["right"]["_gid"] if nan_right.get(f, False) else node["left"]["_gid"])
                feat.append(f)
                thr.append(float(np.nextafter(border, np.float32(np.inf))))  # CatBoost sends x > border right, so x <= border is x < nextafter(border)
                val.append(0.0)
            else:
                v = node["value"]
                if isinstance(v, list):
                    return None
                cl.append(_NO_CHILD)
                cr.append(_NO_CHILD)
                cd.append(_NO_CHILD)
                feat.append(-1)
                thr.append(0.0)
                val.append(float(v) * scale)
            cover.append(node["_w"])
    children_left = np.asarray(cl, dtype=np.int32)
    children_right = np.asarray(cr, dtype=np.int32)
    values = np.asarray(val, dtype=np.float64)  # Intentional float64: leaf values and covers are double, as in the reference TreeSHAP
    weights = np.maximum(np.asarray(cover, dtype=np.float64), 1e-12)  # a leaf no training row reached has weight 0; a zero cover would make its parent's child fractions 0/0
    roots_arr = np.asarray(roots, dtype=np.int32)
    base = _ensemble_expected_value(children_left, children_right, values, weights, roots_arr) + float(bias[0])
    return TreeEnsemble(
        children_left=children_left,
        children_right=children_right,
        children_default=np.asarray(cd, dtype=np.int32),
        features=np.asarray(feat, dtype=np.int32),
        thresholds=np.asarray(thr, dtype=np.float32),
        values=values,
        node_sample_weight=weights,
        tree_roots=roots_arr,
        max_depth=int(max_depth),
        base_offset=float(base),
        n_features=int(n_features),
    )


_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()  # not an attribute of the model, so pickling it is unaffected


def cached_nonsymmetric_ensemble(est: Any) -> Optional[TreeEnsemble]:
    """``extract_catboost_nonsymmetric`` memoised per model object (the JSON export is the expensive part)."""
    try:
        if est in _CACHE:
            return cast(Optional[TreeEnsemble], _CACHE[est])
    except TypeError:
        return extract_catboost_nonsymmetric(est)
    ens = extract_catboost_nonsymmetric(est)
    _CACHE[est] = ens
    return ens
