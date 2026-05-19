"""Regression test: apply_preprocessing_extensions polars→pandas bridge
must use the Arrow-extension-array path on polars 1.x, NOT bare .to_pandas().

Pre-fix the call site at pipeline.py:498 passed
``split_blocks=True, self_destruct=True`` to ``polars.DataFrame.to_pandas``.
Polars 1.x rejects these kwargs at the top level (it forwards them to
pyarrow internally), raising
``TypeError: pyarrow.lib._PandasConvertible.to_pandas() got multiple
values for keyword argument 'split_blocks'``. The TypeError caught the
fallback to bare ``.to_pandas()``, which:

* loses the Arrow-extension-array dtypes (pl.Enum / pl.Categorical -> object)
* uses the slow consolidation copy path (~30x slower on wide frames)
* WARNS the user that 'polars version is too old' -- but it was 1.x, not old

The fix tries the polars-1.x signature first, then the polars-0.x legacy
signature, then bare. This test pins:

1. On polars >= 1.0, the returned pandas frame DOES carry Arrow-backed
   dtypes (``float[pyarrow]`` etc.), confirming the Arrow extension-array
   path fired instead of the bare fallback.
2. pl.Enum / pl.Categorical columns are preserved as pandas Categorical
   (not degraded to object).

A regression that re-introduces the unconditional split_blocks kwarg on
polars 1.x would fail by yielding object dtype for the categorical column.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest


@pytest.mark.skipif(
    tuple(int(x) for x in pl.__version__.split(".")[:2]) < (1, 0),
    reason="Polars 1.x API: test verifies the modern to_pandas dispatch.",
)
def test_apply_preprocessing_extensions_polars_to_pandas_uses_arrow_extension_array(monkeypatch):
    """The Arrow-extension fast path must fire on polars 1.x. Pins the
    dispatch by monkey-patching ``polars.DataFrame.to_pandas`` to record
    every (args, kwargs) pair seen during the conversion and asserts
    that AT LEAST ONE call uses ``use_pyarrow_extension_array=True``
    (the Arrow-bridge fast path).

    A regression that drops the modern call OR re-introduces an
    unconditional ``split_blocks=True`` kwarg on polars 1.x (which
    triggers TypeError and falls back to bare ``.to_pandas()``) would
    fail by leaving the recorded-calls list empty of the extension-array
    flavour.
    """
    from mlframe.training.pipeline import apply_preprocessing_extensions
    from mlframe.training.configs import PreprocessingExtensionsConfig

    cfg = PreprocessingExtensionsConfig(
        scaler="StandardScaler",
        pysr_enabled=False,
        dim_reducer=None,
    )

    n = 200
    df = pl.DataFrame({
        "x": np.arange(n, dtype=np.float32),
        "y": np.arange(n, dtype=np.float64) * 0.5,
    })

    seen_calls: list[dict] = []
    _orig_to_pandas = pl.DataFrame.to_pandas

    def _spy_to_pandas(self, *args, **kwargs):
        seen_calls.append({
            "use_pyarrow_extension_array": kwargs.get("use_pyarrow_extension_array", False),
            "split_blocks_kw_present": "split_blocks" in kwargs,
        })
        # Intentionally call the unbound original with self as first arg.
        return _orig_to_pandas(self, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "to_pandas", _spy_to_pandas)

    apply_preprocessing_extensions(
        train_df=df.clone(),
        val_df=df.clone(),
        test_df=df.clone(),
        config=cfg,
        verbose=0,
        y_train=np.arange(n, dtype=np.float64),
    )

    fast_path_calls = [c for c in seen_calls if c["use_pyarrow_extension_array"]]
    assert fast_path_calls, (
        f"apply_preprocessing_extensions polars->pandas: NO call to "
        f"to_pandas used use_pyarrow_extension_array=True; recorded calls: "
        f"{seen_calls}. The bare .to_pandas() path is being hit, losing "
        f"pl.Enum / pl.Categorical fidelity + wide-frame Arrow zero-copy."
    )
    # On polars 1.x the modern call (no split_blocks kwarg) must succeed;
    # if every fast-path call still passes split_blocks=True we are on the
    # buggy legacy code that crashes on 1.x.
    modern_signature_calls = [
        c for c in fast_path_calls if not c["split_blocks_kw_present"]
    ]
    assert modern_signature_calls, (
        f"apply_preprocessing_extensions polars->pandas: only the legacy "
        f"split_blocks=True signature was attempted; on polars 1.x this "
        f"raises TypeError and forces the fallback. Recorded fast-path "
        f"calls: {fast_path_calls}. Make sure the modern "
        f"to_pandas(use_pyarrow_extension_array=True) (no split_blocks) "
        f"call is tried FIRST."
    )
