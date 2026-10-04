"""Wave-19 follow-up sensor: .meta.json sidecar pattern reused at the
three remaining persistence boundaries flagged by the audit.

Original wave-19 P0 #1 (49d68ab) added the sidecar to save_mlframe_model
in training/io.py. This sensor pins three additional integrations:

1. P0 #3 -- training/ranker_suite.py write loop: per-flavor booster
   joblib.dump now triggers _write_save_meta_sidecar so CB/LGB/XGB
   minor-upgrade skew gets a WARN log at load instead of a cryptic
   AttributeError deep in predict().

2. P1 -- calibration/post.py per-calibrator joblib.dump now writes a
   sidecar. _PerClassIsotonicCalibrator / _PostHocMultiCalibratedModel
   carry attributes (n_classes, is_exclusive, _target_type) whose
   semantics could shift across mlframe versions.

3. P1 -- inference/predict.py read_trained_models calls
   validate_load_meta_sidecar before joblib.load so the operator sees
   library-version drift instead of chasing a downstream crash.

All three sites are source-level guards: a behavioural fixture would
require full ranker / calibrator / inference fixtures which already
exist in their own dedicated test files; what we pin HERE is the
contract that the sidecar helper is wired in at each call site.
Failure mode: future refactor removes the wire-up, this sensor names
the boundary that lost its version check.
"""

from __future__ import annotations

import logging
import os

import numpy as np


def test_ranker_suite_per_flavor_dump_writes_sidecar(tmp_path):
    """The ranker suite's save step writes a ``.meta.json`` library-version envelope beside every per-flavor booster artefact."""
    import joblib
    from sklearn.linear_model import LinearRegression

    from mlframe.training.io import _meta_sidecar_path
    from mlframe.training.ranking._ranker_suite_train_helpers import _train_mlframe_rank_save_dir

    flavors = ["catboost", "lightgbm"]
    models_dict = {f: {"model": LinearRegression().fit(np.arange(6.0).reshape(-1, 1), np.arange(6.0))} for f in flavors}
    _train_mlframe_rank_save_dir(str(tmp_path), "rk", flavors, models_dict, False, {"n": 1})
    artefacts = sorted(tmp_path.glob("rk_*.joblib"))
    assert [a.name for a in artefacts] == ["rk_catboost.joblib", "rk_lightgbm.joblib"]
    for artefact in artefacts:
        assert os.path.isfile(_meta_sidecar_path(str(artefact)))
        assert joblib.load(artefact).coef_.shape == (1,)


def test_calibrator_post_dump_writes_sidecar(tmp_path):
    """Every calibrator dump ``train_postcalibrators`` writes gets a ``.meta.json`` version envelope beside it."""
    from pathlib import Path
    from unittest.mock import patch

    from pyutilz.strings import slugify
    from sklearn.isotonic import IsotonicRegression

    from mlframe.calibration.post import named_calibrator, train_postcalibrators
    from mlframe.training import TargetTypes

    rng = np.random.default_rng(0)
    p1 = rng.random(300)
    probs = np.column_stack([1 - p1, p1])
    target = (p1 + rng.normal(0, 0.1, 300) > 0.5).astype(int)

    class _FakeModel:
        """FakeModel."""
        columns = ["y"]

    (tmp_path / slugify("t") / slugify("fs") / slugify(str(TargetTypes.BINARY_CLASSIFICATION)) / slugify("m")).mkdir(parents=True)
    fake_calibrators = [named_calibrator(IsotonicRegression(out_of_bounds="clip"), name="Iso", lib="sklearn")]
    with patch("mlframe.calibration.post.get_postcalibrators", return_value=fake_calibrators):
        train_postcalibrators(
            models={"m1": _FakeModel()}, model_name="m", models_dir=str(tmp_path), target_name="t", featureset_name="fs",
            include_patterns=["sklearn"], ensembling_method="harm", verbose=0, calib_probs_per_model=[probs], calib_target=target,
        )
    dumps = sorted(Path(tmp_path).rglob("*.dump"))
    assert dumps, "train_postcalibrators wrote no calibrator dump"
    for dump in dumps:
        assert Path(str(dump) + ".meta.json").is_file(), f"{dump.name} has no .meta.json version envelope"


def test_inference_read_trained_models_validates_sidecar(tmp_path, caplog):
    """read_trained_models warns about library-version drift recorded in a model's ``.meta.json`` and still loads the model."""
    import json

    import joblib
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    from mlframe.inference.predict import read_trained_models
    from mlframe.training.io import _meta_sidecar_path, _write_save_meta_sidecar
    from mlframe.utils.safe_pickle import write_sidecar

    frame = pd.DataFrame({"a": np.arange(8.0), "b": np.arange(8.0) ** 2})
    model = LinearRegression().fit(frame, np.arange(8.0))
    featureset_dir = tmp_path / "infer" / "fs"
    featureset_dir.mkdir(parents=True)
    features_json = featureset_dir / "features.dump.json"
    features_json.write_text(json.dumps(["a", "b"]))
    write_sidecar(str(features_json))
    model_path = featureset_dir / "model.pkl"
    joblib.dump(model, str(model_path))
    write_sidecar(str(model_path))
    _write_save_meta_sidecar(str(model_path), durable=False)
    envelope = _meta_sidecar_path(str(model_path))
    with open(envelope, encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["lib_versions"], "the envelope must record library versions"
    lib = sorted(meta["lib_versions"])[0]
    meta["lib_versions"][lib] = "0.0.0-saved-elsewhere"
    with open(envelope, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)

    with caplog.at_level(logging.WARNING, logger="mlframe.training.io"):
        models, _X = read_trained_models("fs", frame, inference_folder=str(tmp_path / "infer"))
    assert list(models) == ["model"]
    drift = [r for r in caplog.records if "library-version drift detected" in r.getMessage()]
    assert len(drift) == 1
    assert f"{lib}: saved='0.0.0-saved-elsewhere'" in drift[0].getMessage()


def test_sidecar_helpers_remain_importable_from_io():
    """The reusable sidecar helpers MUST stay public in training/io for
    callers (ranker_suite / calibrator / inference predict) to use them."""
    from mlframe.training.io import (
        _write_save_meta_sidecar,
        validate_load_meta_sidecar,
        load_save_meta_sidecar,
        _meta_sidecar_path,
        _collect_lib_versions,
    )

    # These are private-but-shared (single-underscore prefix) helpers; we
    # rely on the contract that the public + adjacent callers use them.
    assert callable(_write_save_meta_sidecar)
    assert callable(validate_load_meta_sidecar)
    assert callable(load_save_meta_sidecar)
    assert callable(_meta_sidecar_path)
    assert callable(_collect_lib_versions)
