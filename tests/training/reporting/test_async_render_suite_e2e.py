"""End to end: ``train_mlframe_models_suite`` with ``ReportingConfig.async_render`` on vs off saves the same artifacts and returns the same numbers."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from mlframe.training import OutputConfig, PreprocessingConfig, ReportingConfig
from mlframe.training.configs import BaselineDiagnosticsConfig
from mlframe.training.core import train_mlframe_models_suite

from ..shared import SimpleFeaturesAndTargetsExtractor

pytestmark = [pytest.mark.slow, pytest.mark.requires_cb]

_UUID = re.compile(rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _frame(n: int = 3000) -> pd.DataFrame:
    """Binary classification frame with a signal in two numerics and one categorical."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame({f"num_{i}": rng.standard_normal(n).astype("float32") for i in range(5)})
    df["cat_0"] = pd.Categorical(rng.choice(["A", "B", "C"], n))
    df["target"] = (df["num_0"] - 0.5 * df["num_1"] + rng.standard_normal(n) * 0.6 > 0).astype("int32")
    return df


# The interaction-strength diagnostic decides to skip itself from a measured predict time, which is not reproducible between runs.
_STABLE = dict(interaction_strength_charts=False)


def _suite(tmp_path, async_render: Any, models=("cb",), **reporting_kw):
    """Train once with the given async_render setting; returns ``(models, metadata)``."""
    hp: Dict[str, Any] = {"iterations": 8, "cb_kwargs": {"task_type": "CPU", "verbose": 0, "thread_count": 2}}
    return train_mlframe_models_suite(
        df=_frame(), target_name="tgt", model_name="mdl",
        features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False),
        mlframe_models=list(models), hyperparams_config=hp, preprocessing_config=PreprocessingConfig(drop_columns=[]),
        use_ordinary_models=True, use_mlframe_ensembles=False,
        output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models", save_charts=True),
        reporting_config=ReportingConfig(show_perf_chart=False, async_render=async_render, adversarial_validation=False, **reporting_kw),
        # the feature-ablation baseline fits dozens of throwaway LightGBM models and has nothing to do with rendering
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        verbose=0,
    )


def _artifacts(root) -> Dict[str, str]:
    """Relative path -> content hash of every saved chart/report file (plotly's random element ids normalised)."""
    out: Dict[str, str] = {}
    for base, _dirs, files in os.walk(root):
        rel_base = os.path.relpath(base, root)
        if rel_base.split(os.sep)[0] == "models":
            continue
        for f in files:
            with open(os.path.join(base, f), "rb") as fh:
                raw = fh.read()
            if f.endswith(".html"):
                raw = _UUID.sub(b"UUID", raw)
            out[os.path.join(rel_base, f)] = hashlib.sha1(raw).hexdigest()
    return out


def _numbers(obj: Any, path: str = "") -> Dict[str, str]:
    """Flatten the returned models into path -> repr/hash leaves, skipping the chart accounting and live objects."""
    out: Dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("charts", "figure_specs"):
                continue
            out.update(_numbers(v, f"{path}/{k}"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(_numbers(v, f"{path}[{i}]"))
    elif isinstance(obj, SimpleNamespace):
        for k, v in vars(obj).items():
            # plot_file / model_file_path carry the run's own directory, and the fingerprint is derived from them.
            if k in ("model", "pre_pipeline", "columns", "plot_file", "model_file_path", "training_fingerprint_"):
                continue
            out.update(_numbers(v, f"{path}.{k}"))
    elif isinstance(obj, np.ndarray):
        out[path] = hashlib.sha1(np.ascontiguousarray(obj).tobytes()).hexdigest()
    elif isinstance(obj, (int, float, str, bool, type(None), np.floating, np.integer)):
        out[path] = repr(obj)
    return out


def _charts_accounting(obj: Any, path: str = "") -> Dict[str, Any]:
    """Collect every ``charts`` accounting dict in the returned models, keyed by where it lives."""
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "charts" and isinstance(v, dict):
                out[path] = {kk: (sorted(vv) if isinstance(vv, list) else vv) for kk, vv in v.items() if kk != "skipped"}
            else:
                out.update(_charts_accounting(v, f"{path}/{k}"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(_charts_accounting(v, f"{path}[{i}]"))
    elif isinstance(obj, SimpleNamespace):
        for k, v in vars(obj).items():
            out.update(_charts_accounting(v, f"{path}.{k}"))
    return out


def _shm() -> set:
    """This process's shared-memory segments currently present (other test processes may own their own)."""
    return {n for n in os.listdir("/dev/shm") if n.startswith(f"mlframe_rq_{os.getpid()}_")} if os.path.isdir("/dev/shm") else set()


def test_async_suite_saves_the_same_artifacts_and_numbers_as_the_sync_suite(tmp_path):
    """Same file set, byte-identical PNGs, structure-identical HTML, bit-identical metrics and the same chart accounting; no leaked shm."""
    before = _shm()
    sync_dir, async_dir = tmp_path / "sync", tmp_path / "async"
    # A cold first suite pays JIT / cache warm-up that the time-gated diagnostics (interaction strength probes one predict against a
    # wall-clock budget) react to, so whichever run came first would lose artifacts for reasons unrelated to rendering.
    _suite(tmp_path / "warmup", False, **_STABLE)
    sync_models, sync_meta = _suite(sync_dir, False, **_STABLE)
    async_models, async_meta = _suite(async_dir, True, **_STABLE)

    sync_files, async_files = _artifacts(sync_dir), _artifacts(async_dir)
    assert sync_files, "the sync run saved no charts -- the comparison would be vacuous"
    assert sorted(sync_files) == sorted(async_files), "async rendering changed which artifacts exist"
    differing = [k for k in sync_files if sync_files[k] != async_files[k] and not k.endswith(".csv")]
    assert not differing, f"artifacts differ between sync and async rendering: {differing[:5]}"

    assert _numbers(sync_models) == _numbers(async_models), "returned metrics / predictions changed under async rendering"
    sync_acc, async_acc = _charts_accounting(sync_models), _charts_accounting(async_models)
    assert sync_acc and sync_acc.keys() == async_acc.keys()

    def _rebase(acc: Dict[str, Any], root) -> Dict[str, Any]:
        """Replace the run's own directory in every recorded path so two runs' accounting is comparable."""
        return {k: {kk: [p.replace(str(root), "X") if isinstance(p, str) else p for p in (vv if isinstance(vv, list) else [vv])] for kk, vv in v.items()} for k, v in acc.items()}

    assert _rebase(sync_acc, sync_dir) == _rebase(async_acc, async_dir)

    from mlframe.reporting._async_render_hooks import active_render_queue

    assert active_render_queue() is None, "the suite must leave no active render queue behind on its thread"
    stamp = async_meta["async_render"]
    assert stamp["failed"] == [] and stamp["artifacts"] > 0
    assert "async_render" not in sync_meta
    assert _shm() == before


def test_a_failing_artifact_is_reported_by_name_and_does_not_stop_training(tmp_path, caplog, monkeypatch):
    """A renderer that raises for one figure: training and every other artifact complete, the WARNING names the file, metadata lists it."""
    from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer

    real_save = MatplotlibRenderer.save

    def _save(self, fig, path, fmt):
        """Fail for the decision-curve PNG only."""
        if "decision_curve" in path:
            raise OSError("simulated disk failure")
        return real_save(self, fig, path, fmt)

    monkeypatch.setattr(MatplotlibRenderer, "save", _save)
    with caplog.at_level(logging.WARNING):
        models, meta = _suite(tmp_path, True)
    assert models, "training must complete"
    failed = meta["async_render"]["failed"]
    assert any("decision_curve" in name for name in failed), failed
    assert any("decision_curve" in r.getMessage() and r.levelno >= logging.WARNING for r in caplog.records)
    files = _artifacts(tmp_path)
    assert not any("decision_curve" in f and f.endswith(".png") for f in files)
    assert any(f.endswith("_perfplot.png") for f in files), "other artifacts must still be written"


def test_async_render_off_never_creates_a_queue(tmp_path):
    """``async_render=False`` renders inline: no stamp in the metadata and the same suite still saves its charts."""
    _models, meta = _suite(tmp_path, False)
    assert "async_render" not in meta
    assert _artifacts(tmp_path)
