"""One composite suite run, checked against identities that hold only when persistence, routing and reporting agree.

The suite saved metadata before composite post-processing created the CT ensemble, re-saved nothing after wrapping the
composite models (a reloaded suite served bare inner models in T-scale), kept capped specs in the metadata, widened the
``transforms`` whitelist with chains, and reported CT-ensemble metrics for a model that no longer shipped. Each of those
passed the tests that ran one suite per assertion, because none of them compared the saved suite with the returned one.
Everything here is an identity between two views of the same run, not a tuned threshold.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pytest

from .test_composite_integration import _LEAN_OUTPUT_CONFIG_KWARGS, _LEAN_REPORTING_CONFIG_KWARGS, _build_minimal_fte, _tvt_dataset

pytest.importorskip("lightgbm")

_WHITELIST = ["linear_residual", "additive_residual"]


class _Records(logging.Handler):
    """Collects every record emitted while the fixture's suite runs."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    """One suite run with two whitelisted transforms and a cap of one composite: ``(models, metadata, df, models_path, records)``."""
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    from mlframe.training.core import train_mlframe_models_suite

    tmp = str(tmp_path_factory.mktemp("composite_suite"))
    df = _tvt_dataset(n=600)
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, base_candidates=["TVT_prev"], transforms=list(_WHITELIST), mi_sample_n=300, eps_mi_gain=-1.0,
        max_total_composite_targets=1, min_honest_gain_to_train=None,
    )
    handler, root = _Records(), logging.getLogger("mlframe")
    root.addHandler(handler)
    try:
        models, metadata = train_mlframe_models_suite(
            df=df, target_name="target", model_name="ct", features_and_targets_extractor=_build_minimal_fte(),
            mlframe_models=["linear", "lgb"], output_config={"data_dir": tmp, "models_dir": "models", **_LEAN_OUTPUT_CONFIG_KWARGS},
            reporting_config=_LEAN_REPORTING_CONFIG_KWARGS, verbose=0, composite_target_discovery_config=cfg,
        )
    finally:
        root.removeHandler(handler)
    return models, metadata, df, os.path.join(tmp, "models", "target", "ct"), handler.records


def _trained(models) -> dict:
    """``{(target_type, key): [entries with a model]}`` for every trained slot."""
    out = {}
    for tt, by_key in models.items():
        for key, entries in by_key.items():
            live = [e for e in entries if getattr(e, "model", None) is not None]
            if live:
                out[(str(getattr(tt, "value", tt)), key)] = live
    return out


def _composite_keys(models, metadata) -> set:
    """Composite target keys that were trained (in the model dict and not a raw target or the CT slot)."""
    names = {s["name"] for by_t in metadata["composite_target_specs"].values() for specs in by_t.values() for s in specs}
    return {k for (_tt, k) in _trained(models) if k in names}


def test_the_saved_suite_serves_what_the_run_returned(suite):
    """Every trained slot reloads with the same model types and the reloaded suite predicts identically (INT-01, INT-02)."""
    from mlframe.training.core._predict_main_from_models import predict_from_models
    from mlframe.training.core.predict import load_mlframe_suite

    models, metadata, df, path, _ = suite
    loaded_models, loaded_md = load_mlframe_suite(path)
    mem, disk = _trained(models), _trained(loaded_models)
    assert set(mem) == set(disk), f"slots missing on disk: {sorted(set(mem) - set(disk))}; extra: {sorted(set(disk) - set(mem))}"
    for slot, entries in mem.items():
        assert sorted(type(e.model).__name__ for e in entries) == sorted(type(e.model).__name__ for e in disk[slot]), slot
    X = df.drop(columns=["target"]).iloc[:60]
    p_mem = predict_from_models(X, models, metadata, verbose=0)
    p_disk = predict_from_models(X, loaded_models, loaded_md, verbose=0)
    assert set(p_mem["predictions"]) == set(p_disk["predictions"])
    for key, pred in p_mem["predictions"].items():
        np.testing.assert_allclose(np.asarray(p_disk["predictions"][key], dtype=float), np.asarray(pred, dtype=float), rtol=1e-9, err_msg=key)
    np.testing.assert_allclose(p_disk["ensemble_predictions"], p_mem["ensemble_predictions"], rtol=1e-9)


def test_the_saved_metadata_covers_the_returned_metadata(suite):
    """Every metadata key the run returned is on disk (INT-02: the CT-ensemble keys were stamped after the save)."""
    from mlframe.training.core.predict import load_mlframe_suite

    _models, metadata, _df, path, _ = suite
    _, loaded_md = load_mlframe_suite(path)
    missing = sorted(set(metadata) - set(loaded_md))
    assert not missing, f"metadata keys stamped after the last save: {missing}"
    assert loaded_md.get("cross_target_ensemble_metrics") == metadata.get("cross_target_ensemble_metrics")


def test_the_spec_record_is_exactly_what_was_trained(suite):
    """Exported spec names equal the trained composite keys; the capped spec is a failure with the cap reason (INT-11)."""
    models, metadata, _df, _path, _ = suite
    listed = {s["name"] for by_t in metadata["composite_target_specs"].values() for specs in by_t.values() for s in specs}
    assert listed and listed == _composite_keys(models, metadata)
    capped = [f for by_t in metadata.get("composite_target_failures", {}).values() for fs in by_t.values() for f in fs
              if "global cap max_total_composite_targets=1" in f.get("reason", "")]
    assert capped, "the second whitelisted transform's spec must be recorded as capped"


def test_every_spec_is_in_the_transforms_whitelist(suite):
    """No spec outside ``transforms`` (INT-07: auto-chain used to add chain specs to any whitelist)."""
    _models, metadata, _df, _path, _ = suite
    names = {s["transform_name"] for by_t in metadata["composite_target_specs"].values() for specs in by_t.values() for s in specs}
    assert names <= set(_WHITELIST), sorted(names - set(_WHITELIST))


def test_composite_predictions_are_finite_on_the_y_scale(suite):
    """``predict_mlframe_models_suite`` returns finite composite predictions within the target's range (INT-03, EST-01)."""
    from mlframe.training.core._predict_main_suite import predict_mlframe_models_suite

    models, metadata, df, path, _ = suite
    out = predict_mlframe_models_suite(df.drop(columns=["target"]).iloc[:60], path, verbose=0)
    comp = [k for k in out["predictions"] if any(ck in k for ck in _composite_keys(models, metadata))]
    assert comp, f"no composite prediction in {list(out['predictions'])}"
    lo, hi = float(df["target"].min()), float(df["target"].max())
    for k in comp:
        p = np.asarray(out["predictions"][k], dtype=float)
        assert np.all(np.isfinite(p)), k
        assert lo - (hi - lo) < p.min() and p.max() < hi + (hi - lo), f"{k} is off the y scale: [{p.min():.3f}, {p.max():.3f}] vs y [{lo:.3f}, {hi:.3f}]"


def test_the_targets_table_reports_composites_from_their_y_scale_metrics(suite):
    """The composite rows of the targets table carry the recorded y-scale test RMSE (INT-10)."""
    from mlframe.training.targets_performance import targets_performance_frame

    models, metadata, _df, _path, _ = suite
    frame = targets_performance_frame(models, metadata).set_index("target_name")
    yscale = metadata["composite_target_y_scale_metrics"]
    for key in _composite_keys(models, metadata):
        assert frame.loc[key, "scale"] == "y", key
        recorded = [e["metrics"]["test"]["RMSE"] for by_t in yscale.values() for e in by_t.get(key, []) if "test" in e.get("metrics", {})]
        assert recorded and any(np.isclose(frame.loc[key, "RMSE"], r) for r in recorded), (key, frame.loc[key, "RMSE"], recorded)


def test_no_composite_phase_reports_a_failure(suite):
    """No composite phase logged a failure at WARNING or above (a swallowed failure is how INT-01 shipped unnoticed)."""
    *_, records = suite
    bad = [r.getMessage()[:200] for r in records if r.levelno >= logging.WARNING and "composite" in r.name.lower() and "fail" in r.getMessage().lower()]
    assert not bad, bad
