"""What a trained composite suite leaves on disk, and what it serves from there.

One module-scoped suite run (TVT fixture, linear strategy, ``linear_residual``) backs every contract here:

- the per-model ``.dump`` of a composite target holds the y-scale wrapper, not the T-scale inner it was saved as before
  post-processing wrapped it, and the disk entry point reproduces the in-memory predictions exactly;
- the cross-target ensemble entry and the metadata keys post-processing writes reach disk at all;
- for the additive ``linear_residual`` spec the y-scale error equals the inner's T-scale error, which only holds when the inner
  receives its own pipeline stage and the base is read raw;
- a fresh interpreter can load the suite and serve every composite target, including an auto-chain transform that exists only in
  the training process's registry.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess  # nosec B404 - the fresh-process contract needs a real second interpreter
import sys
import textwrap

import numpy as np
import pytest

from tests.training.composite.test_composite_integration import (
    _LEAN_OUTPUT_CONFIG_KWARGS,
    _LEAN_REPORTING_CONFIG_KWARGS,
    _build_minimal_fte,
    _tvt_dataset,
)

pytest.importorskip("lightgbm")

TARGET = "target"
BASE_COLUMN = "TVT_prev"


@pytest.fixture(scope="module")
def composite_suite(tmp_path_factory):
    """Train one composite suite with a data_dir and return ``(models, metadata, models_path, df)``."""
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    from mlframe.training.core import train_mlframe_models_suite

    data_dir = str(tmp_path_factory.mktemp("composite_suite"))
    df = _tvt_dataset(n=400)
    cfg = CompositeTargetDiscoveryConfig(
        # These tests are about what a TRAINED composite target does (wrapping, persistence, serving), so they need
        # one to be trained: on a 400-row fixture a spec beats raw but cannot clear the default 2-SE significance
        # floor on its paired gain, which is the right production call and would leave every assertion vacuous.
        min_honest_gain_z=0.0,
        enabled=True,
        base_candidates=[BASE_COLUMN],
        transforms=["linear_residual"],
        mi_sample_n=200,
        top_k_after_mi=1,
        eps_mi_gain=-1.0,
    )
    models, metadata = train_mlframe_models_suite(
        df=df,
        target_name=TARGET,
        model_name="persist",
        features_and_targets_extractor=_build_minimal_fte(),
        mlframe_models=["linear"],
        output_config={"data_dir": data_dir, "models_dir": "models", **_LEAN_OUTPUT_CONFIG_KWARGS},
        reporting_config=_LEAN_REPORTING_CONFIG_KWARGS,
        verbose=0,
        composite_target_discovery_config=cfg,
    )
    metadata_files = glob.glob(os.path.join(data_dir, "**", "metadata.pkl.zst"), recursive=True)
    assert metadata_files, f"the suite saved no metadata under {data_dir}"
    return models, metadata, os.path.dirname(metadata_files[0]), df


def _spec_list(metadata) -> list[dict]:
    """The composite specs discovered for the fixture's target."""
    specs = metadata.get("composite_target_specs") or {}
    by_target = specs.get("regression") or {}
    return list(by_target.get(TARGET) or [])


def _composite_names(metadata) -> list[str]:
    """Names of the composite targets the suite trained."""
    return [s["name"] for s in _spec_list(metadata)]


def test_suite_trained_composite_targets(composite_suite):
    """Fixture premise: discovery produced composite targets and models for them."""
    models, metadata, _models_path, _df = composite_suite
    names = _composite_names(metadata)
    assert names, "discovery produced no composite spec; every contract below would be vacuous"
    by_name = models["regression"]
    for name in names:
        assert by_name.get(name), f"composite target {name!r} has no trained model entry"


def test_composite_dumps_hold_the_y_scale_wrapper(composite_suite):
    """Each composite target's on-disk dump deserialises to the wrapper the suite serves in memory, not the T-scale inner."""
    from mlframe.training.composite import CompositeTargetEstimator
    from mlframe.training.io import load_mlframe_model

    _models, metadata, models_path, _df = composite_suite
    for name in _composite_names(metadata):
        dumps = glob.glob(os.path.join(models_path, "regression", name, "*.dump"))
        assert dumps, f"no dump written for composite target {name!r}"
        for path in dumps:
            loaded = load_mlframe_model(path)
            assert isinstance(loaded.model, CompositeTargetEstimator), f"{path} holds {type(loaded.model).__name__}, so a suite loaded from disk would serve T-scale values"


def test_cross_target_ensemble_entry_and_metadata_reach_disk(composite_suite):
    """The CT ensemble entry and every metadata key composite post-processing writes are persisted, not only returned."""
    import pickle  # nosec B403 - reading back the suite's own metadata file

    import zstandard

    _models, metadata, models_path, _df = composite_suite
    ct_dumps = glob.glob(os.path.join(models_path, "regression", "_CT_ENSEMBLE__*", "*.dump"))
    assert ct_dumps, "the cross-target ensemble entry was never saved, so predict-from-disk loses the composite combiner"
    with open(os.path.join(models_path, "metadata.pkl.zst"), "rb") as f:
        saved = pickle.loads(zstandard.ZstdDecompressor().decompress(f.read()))  # nosec B301 - the suite's own file, written by this test's fixture
    for key in ("composite_target_specs", "composite_target_y_scale_metrics", "ensembles_chosen", "cross_target_ensemble_metrics"):
        if key in metadata:
            assert key in saved, f"metadata key {key!r} is in the returned metadata but missing on disk"


def test_disk_predictions_match_the_in_memory_models(composite_suite):
    """Serving the saved suite reproduces the in-memory predictions for every composite target."""
    from mlframe.training.core.predict import predict_from_models, predict_mlframe_models_suite

    models, metadata, models_path, df = composite_suite
    X = df.drop(columns=[TARGET])
    in_memory = predict_from_models(X, models, metadata, return_probabilities=False, verbose=0)
    from_disk = predict_mlframe_models_suite(X, models_path, return_probabilities=False, verbose=0)
    for name in _composite_names(metadata):
        mem_keys = [k for k in in_memory["predictions"] if name in k]
        disk_keys = [k for k in from_disk["predictions"] if name in k]
        assert mem_keys, f"predict_from_models returned nothing for composite target {name!r}"
        assert disk_keys, f"predict_mlframe_models_suite returned nothing for composite target {name!r}"
        np.testing.assert_allclose(
            np.asarray(from_disk["predictions"][disk_keys[0]], dtype=np.float64),
            np.asarray(in_memory["predictions"][mem_keys[0]], dtype=np.float64),
            rtol=1e-9, atol=1e-9,
        )


def test_additive_composite_y_error_equals_inner_t_error(composite_suite):
    """For an additive residual the y-scale error must equal the inner's T-scale error, on the suite's own served frame."""
    from mlframe.training.composite.transforms import get_transform
    from mlframe.training.core.predict import predict_mlframe_models_suite

    _models, metadata, models_path, df = composite_suite
    X = df.drop(columns=[TARGET])
    y = df[TARGET].to_numpy(dtype=np.float64)
    result = predict_mlframe_models_suite(X, models_path, return_probabilities=False, verbose=0)
    stage = result["input_df"]
    from mlframe.training.io import load_mlframe_model

    for spec in _spec_list(metadata):
        if spec["transform_name"] != "linear_residual":
            continue
        name = spec["name"]
        path = glob.glob(os.path.join(models_path, "regression", name, "*.dump"))[0]
        wrapper = load_mlframe_model(path).model
        y_pred = np.asarray(result["predictions"][next(k for k in result["predictions"] if name in k)], dtype=np.float64)
        transform = get_transform(spec["transform_name"])
        base = np.asarray(stage[spec["base_column"]], dtype=np.float64)
        t_true = np.asarray(transform.forward(y, base, spec["fitted_params"]), dtype=np.float64)
        from mlframe.training.composite.estimator._routing import inner_input

        t_pred = np.asarray(wrapper.estimator_.predict(inner_input(wrapper, stage, transform)), dtype=np.float64)
        rmse_y = float(np.sqrt(np.mean((y_pred - y) ** 2)))
        rmse_t = float(np.sqrt(np.mean((t_pred - t_true) ** 2)))
        assert abs(rmse_y - rmse_t) < 1e-6, (
            f"composite {name!r}: y-scale RMSE {rmse_y:.6f} != inner T-scale RMSE {rmse_t:.6f}. For an additive residual the two "
            "are the same number unless the inner or the base is read at the wrong pipeline stage."
        )


def test_fresh_process_serves_every_composite_target(composite_suite, tmp_path):
    """A second interpreter loads the saved suite and serves y-scale values, including auto-chain transforms it never registered."""
    _models, metadata, models_path, df = composite_suite
    csv_path = tmp_path / "predict_input.csv"
    df.drop(columns=[TARGET]).to_csv(csv_path, index=False)
    out_path = tmp_path / "fresh_predictions.json"
    import mlframe

    package_root = os.path.dirname(os.path.dirname(os.path.abspath(mlframe.__file__)))
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, r"{package_root}")
        import json
        import numpy as np
        import pandas as pd
        from mlframe.training.core.predict import predict_mlframe_models_suite

        X = pd.read_csv(r"{csv_path}")
        res = predict_mlframe_models_suite(X, r"{models_path}", return_probabilities=False, verbose=0)
        out = {{k: [float(np.mean(v)), float(np.std(v))] for k, v in res["predictions"].items()}}
        with open(r"{out_path}", "w", encoding="utf-8") as f:
            json.dump(out, f)
        """
    )
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMBA_NUM_THREADS="2", LOKY_MAX_CPU_COUNT="1")
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=900)  # nosec B603 - fixed interpreter and generated script
    assert proc.returncode == 0, f"fresh-process predict failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    fresh = json.loads(out_path.read_text(encoding="utf-8"))
    y = df[TARGET].to_numpy(dtype=np.float64)
    for name in _composite_names(metadata):
        keys = [k for k in fresh if name in k]
        assert keys, f"fresh process served no prediction for composite target {name!r}"
        mean, std = fresh[keys[0]]
        assert abs(mean - float(y.mean())) < 0.5 * float(y.std()), f"composite {name!r} served mean {mean:.3f} against y mean {y.mean():.3f}: not the y scale"
        assert std > 0.3 * float(y.std()), f"composite {name!r} served std {std:.3f} against y std {y.std():.3f}: predictions collapsed"
