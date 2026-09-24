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
# The per-model y-scale hook's calls during the ``suite`` run: each carries the wrapped entry and the val/test frames it scored.
_HOOK_CALLS: list[dict] = []


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
    import mlframe.training.core._phase_composite_wrapping as wrapping

    real_hook = wrapping.emit_per_model_composite_y_scale_test

    def _hook_spy(**kw):
        out = real_hook(**kw)
        _HOOK_CALLS.append(kw)
        return out

    _HOOK_CALLS.clear()
    handler, root = _Records(), logging.getLogger("mlframe")
    root.addHandler(handler)
    mp = pytest.MonkeyPatch()
    mp.setattr(wrapping, "emit_per_model_composite_y_scale_test", _hook_spy)
    try:
        models, metadata = train_mlframe_models_suite(
            df=df, target_name="target", model_name="ct", features_and_targets_extractor=_build_minimal_fte(),
            mlframe_models=["linear", "lgb"], output_config={"data_dir": tmp, "models_dir": "models", **_LEAN_OUTPUT_CONFIG_KWARGS},
            reporting_config=_LEAN_REPORTING_CONFIG_KWARGS, verbose=0, composite_target_discovery_config=cfg,
        )
    finally:
        mp.undo()
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
    assert mem.items(), "the run trained no models"
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


@pytest.fixture(scope="module")
def rerun(suite, tmp_path_factory):
    """The same suite re-run with the first run's specs handed in as a precomputed bundle: ``(models, metadata)``."""
    from mlframe.training._precompute import TrainMlframeSuitePrecomputed
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    from mlframe.training.core import train_mlframe_models_suite

    _models, metadata, df, _path, _ = suite
    tmp = str(tmp_path_factory.mktemp("composite_rerun"))
    cfg = CompositeTargetDiscoveryConfig(enabled=True, base_candidates=["TVT_prev"], transforms=list(_WHITELIST), mi_sample_n=300,
                                         eps_mi_gain=-1.0, max_total_composite_targets=1, min_honest_gain_to_train=None)
    return train_mlframe_models_suite(
        df=df, target_name="target", model_name="ct", features_and_targets_extractor=_build_minimal_fte(),
        mlframe_models=["linear", "lgb"], output_config={"data_dir": tmp, "models_dir": "models", **_LEAN_OUTPUT_CONFIG_KWARGS},
        reporting_config=_LEAN_REPORTING_CONFIG_KWARGS, verbose=0, composite_target_discovery_config=cfg,
        precomputed=TrainMlframeSuitePrecomputed(composite_target_specs=metadata["composite_target_specs"]),
    )


def test_a_precomputed_rerun_trains_the_same_composite_targets(suite, rerun):
    """Handing the specs back in trains exactly the composite keys the first run trained (INT-06: it trained none)."""
    models, metadata, *_ = suite
    models2, metadata2 = rerun
    assert _composite_keys(models, metadata), "the first run trained no composite target"
    assert _composite_keys(models2, metadata2) == _composite_keys(models, metadata)


def test_an_additive_composite_has_the_same_error_on_y_and_on_t(suite):
    """For an additive transform ``y_hat - y == T_hat - T`` row by row: the wrapper's y-scale error on val and test equals the
    error its inner was scored at on T, on every row the out-of-range base soft-shrink leaves alone (INT-03: a wrong route to
    the inner gave y RMSE 0.607 against T RMSE 0.334 on every row)."""
    from mlframe.training.composite.transforms import get_transform

    assert _HOOK_CALLS, "the per-model y-scale hook never ran"
    checked = 0
    for kw in _HOOK_CALLS:
        entry, spec = kw["entry"], kw["composite_spec"]
        if spec["transform_name"] not in _WHITELIST:
            continue
        t = get_transform(spec["transform_name"])
        y_full = np.asarray(kw["y_full"], dtype=float)
        for split, frame, idx in (("val", kw.get("val_df"), kw.get("val_idx")), ("test", kw.get("test_df_pd"), kw.get("test_idx"))):
            if frame is None or idx is None:
                continue
            y = y_full[np.asarray(idx)]
            y_hat = np.asarray(entry.model.predict(frame), dtype=float)
            shrunk = entry.model.soft_shrink_info_.get("shrunk_mask")
            keep = np.ones(y.size, dtype=bool) if shrunk is None else ~np.asarray(shrunk, dtype=bool)
            T = np.asarray(t.forward(y, frame[spec["base_column"]].to_numpy(dtype=float), spec["fitted_params"]), dtype=float)
            np.testing.assert_allclose(T, np.asarray(getattr(entry, f"{split}_target"), dtype=float), atol=1e-9, err_msg=f"{split} T target")
            t_err = np.asarray(getattr(entry, f"{split}_preds"), dtype=float) - T
            assert keep.mean() > 0.8, "the fixture must leave most rows unshrunk for the identity to be tested"
            np.testing.assert_allclose((y_hat - y)[keep], t_err[keep], atol=1e-9, err_msg=f"{spec['name']} {split}")
            checked += 1
    assert checked, "no additive composite split was checked"


@pytest.fixture(scope="module")
def grouped_suite(tmp_path_factory):
    """A grouped suite with the MoE gate on: ``(models, metadata, records, splits)``; ``splits`` are the ``(name, frame, idx)``
    the MoE phase scored the shipped model on, captured with a spy."""
    import mlframe.training.core._phase_composite_post_moe as moe_phase
    import mlframe.training.core._phase_dummy_baselines as dummy_phase
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    from tests.training.shared import SimpleFeaturesAndTargetsExtractor
    from mlframe.training.core import train_mlframe_models_suite

    tmp = str(tmp_path_factory.mktemp("composite_grouped"))
    df = _tvt_dataset(n=900, seed=1)
    df = df.rename(columns={"TVT_prev": "target_prev"})  # the name the dummy baselines read as the lag expert the MoE gate needs
    df["g"] = np.arange(len(df)) % 40
    df["grp_id"] = df["g"]  # the extractor's group_field is bookkeeping and leaves the features; the gate routes on "g"
    df["target"] = df["target"] + 0.05 * df["g"]
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, base_candidates=["target_prev"], transforms=list(_WHITELIST), mi_sample_n=400, eps_mi_gain=-1.0,
        max_total_composite_targets=2, min_honest_gain_to_train=None, group_column="g", moe_gate_enabled=True,
    )
    captured: list = []
    real = moe_phase._restamp_shipped_metrics

    def _spy(metadata, target_type, target_name, shipped, y_full, splits):
        captured.append((target_name, np.asarray(y_full, dtype=float), [(n, f, None if i is None else np.asarray(i)) for n, f, i in splits]))
        return real(metadata, target_type, target_name, shipped, y_full, splits)

    tags: dict = {}
    real_is_composite = dummy_phase.is_composite_target

    def _tag_spy(name, names, *a, **k):
        out = real_is_composite(name, names, *a, **k)
        tags.setdefault(str(name), set()).add("MTRESID" if out else "MTTR")
        return out

    handler, root = _Records(), logging.getLogger("mlframe")
    root.addHandler(handler)
    mp = pytest.MonkeyPatch()
    mp.setattr(moe_phase, "_restamp_shipped_metrics", _spy)
    mp.setattr(dummy_phase, "is_composite_target", _tag_spy)
    try:
        models, metadata = train_mlframe_models_suite(
            df=df, target_name="target", model_name="ctg", features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True, group_field="grp_id"),
            mlframe_models=["linear", "lgb"], output_config={"data_dir": tmp, "models_dir": "models", **_LEAN_OUTPUT_CONFIG_KWARGS},
            reporting_config=_LEAN_REPORTING_CONFIG_KWARGS, verbose=0, composite_target_discovery_config=cfg,
        )
    finally:
        mp.undo()
        root.removeHandler(handler)
    return models, metadata, tags, captured


def test_the_shipped_moe_model_is_what_the_ensemble_metrics_describe(grouped_suite):
    """``cross_target_ensemble_metrics`` test RMSE equals the RMSE of the model the returned suite ships (EST-13)."""
    models, metadata, _tags, captured = grouped_suite
    assert captured, "the MoE phase never scored a shipped model; the grouped variant lost its subject"
    ct = metadata["cross_target_ensemble_metrics"]
    for target_name, y_full, splits in captured:
        shipped = [e.model for by_key in models.values() for k, es in by_key.items() if k.startswith("_CT_ENSEMBLE") for e in es
                   if target_name in k and getattr(e, "model", None) is not None]
        assert shipped, f"no CT ensemble entry shipped for {target_name}"
        for split, frame, idx in splits:
            if frame is None or idx is None or len(idx) == 0:
                continue
            rec = [slot[target_name][f"{split}_RMSE"] for slot in ct.values() if f"{split}_RMSE" in slot.get(target_name, {})]
            pred = np.asarray(shipped[0].predict(frame), dtype=float).reshape(-1)
            assert rec and np.isclose(rec[0], float(np.sqrt(np.mean((pred - y_full[idx]) ** 2))), rtol=1e-9), (target_name, split, rec)


def test_composite_targets_are_tagged_from_the_spec_set(grouped_suite):
    """The dummy-baseline tag decision marks every composite target MTRESID and the raw one MTTR (INT-08: a name heuristic
    decided it and missed chain names)."""
    _models, metadata, tags, _captured = grouped_suite
    names = {s["name"] for by_t in metadata["composite_target_specs"].values() for ss in by_t.values() for s in ss}
    assert names, "the grouped variant trained no composite target"
    for k in names:
        assert tags.get(k) == {"MTRESID"}, (k, tags.get(k))
    assert tags.get("target") == {"MTTR"}, tags.get("target")
