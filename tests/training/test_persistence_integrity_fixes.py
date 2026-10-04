"""Regression tests for cache-key canonicalisation, atomic publication of artefacts and reporting integrity."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlframe.training._canonical_json import canonical_json_bytes
from mlframe.training.suite_artefact_cache import SuiteKeyBuilder
from mlframe.training.utils import compute_model_input_fingerprint


def _build(cfg):
    """SuiteKeyBuilder key for ``cfg``."""
    return SuiteKeyBuilder.build(df_fp="abc", config_canonical=cfg)


def _fp(**kw):
    """Model fingerprint hash of a tiny frame with extra context ``kw``."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    return compute_model_input_fingerprint(df, **kw)[0]


DISTINCT_CONFIGS = [
    {"a": float("nan")},
    {"a": None},
    {"a": float("inf")},
    {"a": float("-inf")},
    {"a": "nan"},
    {1: "x"},
    {"1": "x"},
    {True: "x"},
    {"a": 2**70},
    {"a": str(2**70)},
    {"a": 2**70 + 1},
    {"a": 1},
    {"a": 1.5},
    {"a": [1, 2]},
    {"a": (1, 2, 3)},
    {"a": {1, 2}},
]


def test_canonical_json_distinct_configs_give_distinct_bytes():
    """Configs orjson alone would collapse or reject (NaN/None, int keys, huge ints) all keep distinct bytes."""
    blobs = [canonical_json_bytes(c) for c in DISTINCT_CONFIGS]
    assert blobs
    assert len(set(blobs)) == len(blobs)


def test_canonical_json_is_stable_across_calls_and_key_order():
    """Same config digests identically on every call and regardless of dict insertion order, including int keys."""
    a = {"z": 1, 3: {"y": float("nan"), 2: [1, 2]}, "a": 2**80}
    b = {"a": 2**80, 3: {2: [1, 2], "y": float("nan")}, "z": 1}
    assert canonical_json_bytes(a) == canonical_json_bytes(a) == canonical_json_bytes(b)


def test_canonical_json_plain_configs_keep_orjson_bytes():
    """Ordinary string-keyed finite configs keep the historical bytes so existing cache keys stay valid."""
    import orjson

    cfg = {"b": [1, 2.5, "x"], "a": {"k": None, "t": True}}
    assert canonical_json_bytes(cfg) == orjson.dumps(cfg, default=str, option=orjson.OPT_SORT_KEYS)


def test_canonical_json_never_raises_on_cyclic_or_exotic_input():
    """A self-referencing structure falls back to a repr digest instead of raising."""
    cyc: list = []
    cyc.append(cyc)
    assert isinstance(canonical_json_bytes(cyc), bytes)
    assert isinstance(canonical_json_bytes(object()), bytes)


def test_suite_key_builder_distinguishes_nan_none_and_does_not_raise():
    """build() neither raises on int keys / 70-bit ints nor merges NaN with None."""
    keys = [_build(c) for c in DISTINCT_CONFIGS]
    assert len(set(keys)) == len(keys)
    assert _build({"a": float("nan"), 1: 2}) == _build({1: 2, "a": float("nan")})


def test_model_fingerprint_config_digest_distinguishes_int_keyed_and_big_int_configs():
    """Int-keyed or >64-bit configs no longer digest to one constant placeholder."""
    hashes = [_fp(preprocessing_config=c) for c in ({1: 2}, {1: 3}, {"a": 2**70}, {"a": 2**70 + 1}, {"a": float("nan")}, {"a": None})]
    assert len(set(hashes)) == len(hashes)
    assert _fp(preprocessing_config={1: 2, "k": 5}) == _fp(preprocessing_config={"k": 5, 1: 2})


def test_model_fingerprint_split_indices_with_equal_size_and_samples_do_not_collide():
    """Two splits sharing size, first, middle and last element but differing elsewhere get different fingerprints."""
    a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    b = np.array([0, 9, 2, 3, 4, 5, 6, 10, 8])
    assert a[0] == b[0] and a[4] == b[4] and a[-1] == b[-1]
    assert _fp(train_idx=a) != _fp(train_idx=b)
    assert _fp(train_idx=a) == _fp(train_idx=a.copy())


class _Boom(Exception):
    """Injected failure."""


def _fail_replace(*_a, **_k):
    """Stand-in for os.replace that fails mid publication."""
    raise _Boom("replace failed")


def _litter(directory: Path, keep: str) -> list:
    """Files in ``directory`` other than ``keep``."""
    return [p.name for p in directory.iterdir() if p.name != keep]


def test_split_sidecar_write_is_atomic(tmp_path, monkeypatch):
    """A failing publish leaves the previous sidecar intact and no temp file behind."""
    from mlframe.training._fixed_splits import _read_json, _write_json

    target = tmp_path / "split_ids.json"
    _write_json(str(target), {"v": 1})
    monkeypatch.setattr(os, "replace", _fail_replace)
    with pytest.raises(_Boom):
        _write_json(str(target), {"v": 2})
    monkeypatch.undo()
    assert _read_json(str(target)) == {"v": 1}
    assert _litter(tmp_path, "split_ids.json") == []


def test_split_membership_parquet_write_is_atomic(tmp_path, monkeypatch):
    """A crash while publishing split_ids.parquet keeps the old membership readable and leaves no temp litter."""
    from mlframe.training._fixed_splits import SPLIT_IDS_FILENAME, record_split_membership

    ids = np.arange(10)

    def _record(split_dir):
        """Record membership for a fixed 10-row split into ``split_dir``."""
        return record_split_membership(
            row_ids=ids, id_column="row_id", train_idx=np.arange(0, 6), val_idx=np.arange(6, 8), test_idx=np.arange(8, 10),
            calib_idx=None, timestamps=None, split_dir=str(split_dir),
        )

    entry = _record(tmp_path)
    path = Path(entry["path"])
    before = path.read_bytes()
    monkeypatch.setattr(os, "replace", _fail_replace)
    with pytest.raises(_Boom):
        _record(tmp_path)
    monkeypatch.undo()
    assert path.name == SPLIT_IDS_FILENAME
    assert path.read_bytes() == before
    assert pd.read_parquet(path).shape[0] == 10
    assert sorted(_litter(tmp_path, "")) == sorted([path.name, path.with_suffix(".json").name])


@pytest.mark.parametrize("frame_kind", ["pandas", "polars"])
def test_save_series_or_df_is_atomic_and_writes_no_index_column(tmp_path, monkeypatch, frame_kind):
    """save_series_or_df publishes through a temp file; a crash keeps the previous file and no litter."""
    from mlframe.training.utils import save_series_or_df

    if frame_kind == "polars":
        pl = pytest.importorskip("polars")
        make = lambda v: pl.DataFrame({"x": [v, v + 1]})
    else:
        make = lambda v: pd.DataFrame({"x": [v, v + 1]}, index=[10, 11])
    f = tmp_path / "o.parquet"
    save_series_or_df(make(1), str(f))
    before = f.read_bytes()
    monkeypatch.setattr(os, "replace", _fail_replace)
    with pytest.raises(_Boom):
        save_series_or_df(make(5), str(f))
    monkeypatch.undo()
    assert f.read_bytes() == before
    assert _litter(tmp_path, "o.parquet") == []
    assert list(pd.read_parquet(f).columns) == ["x"]


def test_pipeline_disk_cache_persist_uses_unique_temp_names(tmp_path, monkeypatch):
    """Persisting never writes a fixed shared '<path>.tmp' file and a crash keeps the old cache file."""
    from mlframe.training.core import _setup_helpers_pipeline_cache as m

    path = tmp_path / "pc.json"
    path.write_bytes(b'{"old": true}')
    monkeypatch.setattr(m, "_pipeline_disk_cache_path", lambda: str(path))
    seen = []
    real_open = os.open

    def _spy(p, *a, **k):
        """Record every low level open of a temp file."""
        seen.append(os.fspath(p))
        return real_open(p, *a, **k)

    monkeypatch.setattr(os, "open", _spy)
    monkeypatch.setattr(os, "replace", _fail_replace)
    m._persist_pipeline_disk_cache()
    monkeypatch.undo()
    assert path.read_bytes() == b'{"old": true}'
    assert _litter(tmp_path, "pc.json") == []
    assert str(path) + ".tmp" not in seen
    assert any(s.startswith(str(path) + ".tmp.") for s in seen)


def test_key_bank_save_never_removes_a_published_directory(tmp_path):
    """A second save for an existing fingerprint keeps the published directory's files untouched."""
    from mlframe.feature_engineering.transformer._key_bank import KeyBank, save_key_bank

    bank = KeyBank(projections=np.zeros((1, 2, 3)), k_proj=np.zeros((1, 4, 3)), y_train=np.zeros(4), seed=1)
    final = tmp_path / "fp1"
    final.mkdir()
    marker = final / "reader_is_using_this.bin"
    marker.write_bytes(b"live")
    save_key_bank(bank, tmp_path, "fp1")
    assert marker.read_bytes() == b"live"
    assert [p.name for p in tmp_path.iterdir()] == ["fp1"]


def test_composite_id_is_identical_for_numpy_and_python_scalar_params():
    """np.int64 / np.float32-exact params give the same composite_id as the equivalent python values."""
    from mlframe.training.composite import CompositeProvenance, CompositeSpec

    def _id(params):
        """composite_id for a spec carrying ``params``."""
        spec = CompositeSpec(
            name="t__ratio__b", target_col="t", transform_name="ratio", base_column="b", fitted_params=params,
            mi_gain=0.1, mi_y=0.1, mi_t=0.2, valid_domain_frac=0.99, n_train_rows=900,
        )
        return CompositeProvenance.from_spec(spec, random_state=1).composite_id

    assert _id({"k": np.int64(3), "w": np.float64(0.5), "f": np.bool_(True)}) == _id({"k": 3, "w": 0.5, "f": True})
    assert _id({"k": np.int64(3), "w": np.float32(0.5)}) == _id({"k": 3, "w": 0.5})
    assert _id({"k": 3}) != _id({"k": 4})


def test_stability_plot_marks_selected_n_on_the_mean_cv_curve(tmp_path, monkeypatch):
    """The red selected-N marker sits at the plotted mean CV value, not at the penalised base_perf."""
    from mlframe.feature_selection.wrappers.rfecv._stability_select import _plot_cv_performance

    captured = {}
    real_plot = matplotlib.axes.Axes.plot

    def _spy(self, *args, **kw):
        """Capture the marker call."""
        if len(args) >= 3 and args[2] == "ro":
            captured["xy"] = (float(args[0]), float(args[1]))
        return real_plot(self, *args, **kw)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", _spy)
    plt.close("all")
    nf = np.array([1, 2, 3])
    mean = np.array([0.5, 0.8, 0.7])
    base = mean - 0.1
    _plot_cv_performance(False, str(tmp_path / "p.png"), 10, (4, 3), nf, mean, mean * 0.1, mean, 1, base)
    assert captured["xy"] == (2.0, 0.8)


def test_stability_plot_does_not_mutate_global_rcparams(tmp_path):
    """Plotting with a custom font size leaves matplotlib's global font.size unchanged and closes the figure."""
    from mlframe.feature_selection.wrappers.rfecv._stability_select import _plot_cv_performance

    plt.close("all")
    before = plt.rcParams["font.size"]
    perf = np.array([0.5, 0.6, 0.7])
    _plot_cv_performance(False, str(tmp_path / "p.png"), before + 7, (4, 3), np.array([1, 2, 3]), perf, perf, perf, 1, perf)
    assert plt.rcParams["font.size"] == before
    assert plt.get_fignums() == []


def test_search_state_plot_does_not_mutate_global_rcparams():
    """plot_search_state applies its font size inside an rc_context."""
    from mlframe.models._optimization_shared import plot_search_state

    plt.close("all")
    before = plt.rcParams["font.size"]
    x = np.arange(5)
    plot_search_state(
        search_space=x, next_cand=2, new_y=0.5, best_candidate=1, best_evaluation=0.4, nsteps=1, expected_fitness=None, y_pred=None, y_std=None,
        ground_truth=None, known_candidates=np.array([0, 1]), known_evaluations=np.array([0.1, 0.4]), skip_candidates=[], acquisition_method="EI",
        mode="m", additional_info="", font_size=before + 5,
    )
    assert plt.rcParams["font.size"] == before
    assert plt.get_fignums() == []


def test_rfecv_estimators_save_failure_logs_warning_at_default_verbosity(tmp_path, caplog):
    """A failing estimators_save_path save is a WARNING with the exception even when verbose is 0, and the call returns False."""
    from types import SimpleNamespace

    from mlframe.feature_selection.wrappers.rfecv import _finalize

    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    self_ = SimpleNamespace(estimators_save_path=str(blocker), _selected_cols_cache=["a"], n_features_=1, cv_results_=None, keep_estimators=False)
    with caplog.at_level(logging.WARNING, logger=_finalize.logger.name):
        _finalize._persist_fitted_estimators(self_, estimator=object(), fitted_estimators={}, verbose=0)
    msgs = [r for r in caplog.records if r.levelno == logging.WARNING and "persistence failed" in r.getMessage()]
    assert msgs


def test_rfecv_estimators_save_publishes_atomically_with_sidecar(tmp_path, monkeypatch):
    """required_features.dump is written with a .sha256 sidecar; a crash keeps the previous dump and leaves no litter."""
    from types import SimpleNamespace

    from mlframe.feature_selection.wrappers.rfecv import _finalize

    def _self(cols):
        """Fake RFECV carrying a selected-columns cache."""
        return SimpleNamespace(estimators_save_path=str(tmp_path), _selected_cols_cache=cols, n_features_=len(cols), cv_results_=None, keep_estimators=False)

    _finalize._persist_fitted_estimators(_self(["a"]), estimator=object(), fitted_estimators={}, verbose=0)
    dump = tmp_path / "required_features.dump"
    assert (tmp_path / "required_features.dump.sha256").exists()
    before = dump.read_bytes()
    monkeypatch.setattr(os, "replace", _fail_replace)
    _finalize._persist_fitted_estimators(_self(["a", "b"]), estimator=object(), fitted_estimators={}, verbose=0)
    monkeypatch.undo()
    assert dump.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["required_features.dump", "required_features.dump.sha256"]


def test_post_calibrator_dump_is_atomic_and_has_sidecar(tmp_path, monkeypatch):
    """train_postcalibrators publishes each calibrator through a temp file and writes its .sha256 sidecar."""
    from unittest.mock import patch

    from pyutilz.strings import slugify
    from sklearn.isotonic import IsotonicRegression

    from mlframe.calibration.post import named_calibrator, train_postcalibrators
    from mlframe.training import TargetTypes

    rng = np.random.default_rng(0)
    p1 = rng.uniform(size=300)
    probs = np.column_stack([1 - p1, p1])
    target = (rng.uniform(size=300) < p1).astype(int)
    out_dir = tmp_path / slugify("t") / slugify("fs") / slugify(str(TargetTypes.BINARY_CLASSIFICATION)) / slugify("m")
    out_dir.mkdir(parents=True)

    class _Fake:
        """Model stub."""
        columns = ["y"]

    def _run():
        """Train one isotonic post-calibrator and dump it."""
        cals = [named_calibrator(IsotonicRegression(out_of_bounds="clip"), name="Iso", lib="sklearn")]
        with patch("mlframe.calibration.post.get_postcalibrators", return_value=cals):
            return train_postcalibrators(
                models={"m1": _Fake()}, model_name="m", models_dir=str(tmp_path), target_name="t", featureset_name="fs", include_patterns=["sklearn"],
                ensembling_method="harm", verbose=0, calib_probs_per_model=[probs], calib_target=target,
            )

    _run()
    dumps = sorted(out_dir.glob("*.dump"))
    assert dumps
    for d in dumps:
        assert Path(str(d) + ".sha256").exists()
    before = {d.name: d.read_bytes() for d in dumps}
    monkeypatch.setattr(os, "replace", _fail_replace)
    with pytest.raises(_Boom):
        _run()
    monkeypatch.undo()
    assert {d.name: d.read_bytes() for d in out_dir.glob("*.dump")} == before
    assert not [p for p in out_dir.iterdir() if ".tmp." in p.name]
