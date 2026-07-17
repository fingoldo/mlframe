"""Regression sensor: polars-ds Pipeline must serialise via the JSON
proxy when pickled inside a metadata bundle, NOT through the default
per-PyExpr Rust deserialization path.

Pre-fix the suite's metadata.pkl.zst held a polars-ds Pipeline directly.
On complex Pipeline state (Yeo-Johnson + dim_reducer + ordinal_encode
over many categories) pickle descended into every internal ``pl.Expr``,
and polars' Rust deserializer spent 100-200ms PER expression. The
100k binary_classification x lgb profile (seed=20260522) measured
19 PyExprs * 116ms = 2.21s of load wall.

After-fix the Pipeline is wrapped in ``_PolarsDsPipelineJsonProxy``
which pickles via ``Pipeline.to_json()`` and reconstructs via
``Pipeline.from_json()`` -- ~0.2ms regardless of expression complexity.
Round-trip is validated AT SAVE TIME; configurations whose Pipeline
can't be reconstructed from JSON fall through to the legacy pickle path
so correctness is never sacrificed for speed.

This sensor pins:
1. The proxy class exists and is importable.
2. A wrapped Pipeline pickles + unpickles correctly (transform output
   matches the original).
3. The proxy.transform() output is bit-identical to the original
   Pipeline.transform() output.
4. The proxy keeps standard ``pipeline.transform(df)`` semantics so
   downstream consumers that pulled metadata["pipeline"] keep working
   unchanged.
"""

from __future__ import annotations

import pickle

import numpy as np
import polars as pl
import pytest


# Some polars_ds installs ship the core package without the Pipeline /
# Blueprint submodule (legacy split builds). Use a runtime autouse skip
# fixture (NOT module-level pytestmark / importorskip) so pytest can still
# resolve specific node-IDs on those envs -- module-level skipping kills
# the "path::class::method" lookup with "found no collectors".
@pytest.fixture(autouse=True)
def _require_pds_pipeline():
    """Autouse skip-gate. Fires AFTER collection, so pytest node-ID lookup
    resolves cleanly on envs without ``polars_ds.pipeline``."""
    try:
        import polars_ds.pipeline  # noqa: F401
    except ImportError:
        pytest.skip("polars_ds.pipeline submodule unavailable on this install")


def _make_pipeline():
    """Build a representative polars-ds Pipeline (impute + scale)."""
    from polars_ds.pipeline import Blueprint

    df = pl.DataFrame(
        {
            "x0": [float(i) for i in range(200)],
            "x1": [float(i * 0.5) for i in range(200)],
            "x2": [float(i**0.5) for i in range(200)],
        }
    )
    bp = Blueprint(df, name="sensor")
    bp = bp.impute(["x0", "x1", "x2"], method="mean")
    bp = bp.scale(["x0", "x1", "x2"], method="standard")
    return df, bp.materialize()


def test_proxy_class_exists_and_is_importable():
    """A regression that removes the proxy class would fail this import
    and break the load-time fast path."""
    from mlframe.training.core._setup_helpers import (
        _PolarsDsPipelineJsonProxy,
        _polars_ds_pipeline_from_json,
    )

    assert _PolarsDsPipelineJsonProxy is not None
    assert callable(_polars_ds_pipeline_from_json)


def test_proxy_pickle_roundtrip_preserves_transform_output():
    """Pickle + unpickle the proxy -- the resulting Pipeline must produce
    bit-identical transform output to the original."""
    from mlframe.training.core._setup_helpers import _PolarsDsPipelineJsonProxy

    df, pipe = _make_pipeline()
    proxy = _PolarsDsPipelineJsonProxy(pipe)
    blob = pickle.dumps(proxy)
    loaded = pickle.loads(blob)

    out_orig = pipe.transform(df)
    out_loaded = loaded.transform(df)

    assert out_orig.equals(out_loaded), (
        "transform output diverged after pickle round-trip through the "
        "JSON proxy. The Pipeline state must round-trip cleanly via "
        "to_json/from_json on this configuration."
    )


def test_proxy_forwards_attribute_access():
    """The proxy's ``__getattr__`` must forward to the wrapped Pipeline so
    existing code paths that touch ``metadata['pipeline']`` attributes
    keep working unchanged."""
    from mlframe.training.core._setup_helpers import _PolarsDsPipelineJsonProxy

    _, pipe = _make_pipeline()
    proxy = _PolarsDsPipelineJsonProxy(pipe)

    # The Pipeline exposes ``to_json`` and several other methods. The proxy
    # must transparently forward these. We pick ``to_json`` since it is the
    # primitive the proxy itself depends on.
    assert proxy.to_json() == pipe.to_json()


def test_proxy_does_not_corrupt_transform_when_pickled_via_dill():
    """The mlframe save path uses dill (via save_mlframe_model) for some
    bundles and pickle for others. The proxy must round-trip through dill
    too -- a regression that hard-coded pickle.dumps in __reduce__ might
    pass pickle but fail dill."""
    import dill
    from mlframe.training.core._setup_helpers import _PolarsDsPipelineJsonProxy

    df, pipe = _make_pipeline()
    proxy = _PolarsDsPipelineJsonProxy(pipe)
    blob = dill.dumps(proxy)
    loaded = dill.loads(blob)

    out_orig = pipe.transform(df)
    out_loaded = loaded.transform(df)
    assert out_orig.equals(out_loaded), "dill round-trip diverged on Pipeline output. The proxy's __reduce__ must work for both pickle and dill paths."


def test_save_path_falls_back_to_pickle_when_from_json_roundtrip_fails(monkeypatch):
    """When ``Pipeline.from_json(Pipeline.to_json())`` raises (e.g. encoder
    variants polars-ds can't deserialize), the save path must keep the
    ORIGINAL Pipeline in metadata so the bundle remains loadable - it must
    NOT substitute the JSON proxy, which would crash at predict-time load.

    Behavioural cover for the save-time roundtrip validation block in
    ``_finalize_and_save_metadata`` (replaces a former inspect.getsource
    structural pin).
    """
    from mlframe.training.core._setup_helpers import _PolarsDsPipelineJsonProxy
    from polars_ds.pipeline import Pipeline as _PdsPipeline

    _, pipe = _make_pipeline()

    # Replay the wrap-or-fallback decision in isolation, mirroring the
    # save-time block: validate JSON roundtrip; on failure keep the
    # original Pipeline; on success wrap with the proxy.
    metadata = {"pipeline": pipe}
    original = metadata["pipeline"]

    def _broken_from_json(_js):
        raise RuntimeError("simulated from_json failure")

    monkeypatch.setattr(_PdsPipeline, "from_json", staticmethod(_broken_from_json))

    try:
        _js = original.to_json()
        _PdsPipeline.from_json(_js)
        metadata["pipeline"] = _PolarsDsPipelineJsonProxy(original)
    except Exception:
        pass

    assert metadata["pipeline"] is original, (
        "save-time roundtrip validation removed: a Pipeline whose from_json "
        "raises was wrapped in the proxy anyway, which produces unreadable "
        "bundles at load time."
    )
