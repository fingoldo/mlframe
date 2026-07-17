"""B-P0-1 sensor: PySR replay parity.

Fit a preprocessing-extensions pipeline with PySR enabled, serialize the
resulting ``PreprocessingExtensionsBundle`` via joblib, reload, call
``_apply_extensions_pipeline`` on a fresh predict frame, and verify the
PySR-produced columns reproduce byte-identical values vs train-time output.

Skipped when PySR / Julia is not importable (CI without juliacall installed).
"""

from __future__ import annotations

import io
import os
import pickle
import numpy as np
import pandas as pd
import pytest


from tests._pysr_gate import pysr_works


pytestmark = [
    pytest.mark.skipif(
        not pysr_works(),
        reason="PySR / Julia runtime not usable (probe failed)",
    ),
    # Cold Julia fit costs ~30s + has access-violated the xdist worker on S:
    # 2026-05-20 inside PythonCall.jl. slow_only keeps it out of --fast; the
    # subprocess probe in pysr_works() blocks import-time crashes from
    # reaching the worker; --max-worker-restart at the pytest level covers
    # fit-time crashes that the probe cannot pre-detect.
    pytest.mark.slow_only,
]


def test_pysr_replay_parity(tmp_path):
    from mlframe.training.pipeline import (
        apply_preprocessing_extensions,
        PreprocessingExtensionsBundle,
        PySRTransformer,
    )
    from mlframe.training.core.predict import _apply_extensions_pipeline
    from mlframe.training.configs import PreprocessingExtensionsConfig

    rng = np.random.default_rng(0)
    n = 64
    df = pd.DataFrame(
        {
            "x0": rng.normal(size=n).astype(np.float32),
            "x1": rng.normal(size=n).astype(np.float32),
            "x2": rng.normal(size=n).astype(np.float32),
        }
    )
    y = (df["x0"] * df["x1"] + 0.1 * rng.normal(size=n)).to_numpy().astype(np.float32)

    cfg = PreprocessingExtensionsConfig(
        pysr_enabled=True,
        pysr_niterations=3,
        pysr_sample_size=n,
        pysr_top_k=2,
        scaler=None,
        polynomial_degree=None,
        dim_reducer=None,
    )

    train, val, test, ext = apply_preprocessing_extensions(
        df.copy(),
        None,
        df.copy(),
        config=cfg,
        verbose=0,
        y_train=y,
    )

    assert isinstance(ext, PreprocessingExtensionsBundle), "PySR-enabled extensions must persist as a PreprocessingExtensionsBundle"
    assert ext.pysr is not None and isinstance(ext.pysr, PySRTransformer)

    pysr_cols = [c for c in train.columns if c.startswith("pysr__")]
    assert pysr_cols, "PySR did not add any columns; check niterations / data"

    # Round-trip the bundle through pickle (joblib uses pickle under the hood; this
    # is the load-side stress test for B-P0-1).
    buf = io.BytesIO()
    pickle.dump(ext, buf)
    buf.seek(0)
    reloaded = pickle.load(buf)

    # Replay against a fresh copy of the predict frame (no PySR cols).
    replayed = _apply_extensions_pipeline(df.copy(), reloaded, verbose=0)

    for col in pysr_cols:
        a = train[col].to_numpy(dtype=np.float32)
        b = replayed[col].to_numpy(dtype=np.float32)
        # Byte-identical for deterministic PySR evaluation (same fitted model, same input).
        assert np.array_equal(a, b), f"PySR column {col} drift after reload: {a[:4]} vs {b[:4]}"
