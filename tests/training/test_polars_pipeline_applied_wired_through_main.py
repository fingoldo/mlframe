"""Wave-7: ``polars_pipeline_applied`` must be wired from ``_phase_fit_pipeline``'s
return tuple through main.py to ``_phase_pandas_conversion_and_cat_prep``.

Pre-fix: main.py never passed ``polars_pipeline_applied=`` to
``_phase_pandas_conversion_and_cat_prep`` so the kwarg silently relied on its
``default=True`` and the Wave-7 gating logic always saw True. The polars-pipeline-skipped
path was unreachable from production callers.

Post-fix: the kwarg is forwarded with the truthful value returned by ``_phase_fit_pipeline``.
The assertion below proves the recipient saw ``False`` (the value injected by the patched
``_phase_fit_pipeline``), not ``True`` (the default).
"""

from __future__ import annotations

import pytest


class _Sentinel(Exception):
    """Raised by the monkeypatched _phase_pandas_conversion_and_cat_prep to stop the suite
    immediately once the kwarg has been captured. Avoids paying for the downstream phases
    (outlier detection, per-target loop, finalize) which are not relevant to this wiring test."""


def test_polars_pipeline_applied_received_value_from_phase_fit_pipeline(monkeypatch):
    from mlframe.training.core import main as main_mod

    # 2026-05-22 split: ``train_mlframe_models_suite`` body lives in
    # ``_main_train_suite.py``; the live call to ``_phase_fit_pipeline`` /
    # ``_phase_pandas_conversion_and_cat_prep`` resolves from THAT
    # module's globals. Patch both namespaces.
    from mlframe.training.core import _main_train_suite as _suite_mod

    captured: dict = {}

    real_fit = main_mod._phase_fit_pipeline

    def fake_fit_pipeline(*args, **kwargs):
        # Return shape mirrors the production tuple; ``polars_pipeline_applied`` (10th element)
        # is forced to False so the wiring test can distinguish it from the default=True.
        out = real_fit(*args, **kwargs)
        return (
            out[0],
            out[1],
            out[2],  # train_df, val_df, test_df
            out[3],
            out[4],  # pipeline, extensions_pipeline
            out[5],
            out[6],  # cat_features, cat_features_polars
            out[7],
            out[8],  # was_polars_input, all_models_polars_native
            False,  # polars_pipeline_applied -- forced False so default=True can't masquerade
            out[10],
            out[11],
            out[12],  # train_df_polars_pre, val_df_polars_pre, test_df_polars_pre
            out[13],
            out[14],  # pipeline_config, preprocessing_extensions
            out[15],  # train_df_pandas_pre_meta
        )

    def fake_pandas_conv(*args, **kwargs):
        captured["polars_pipeline_applied"] = kwargs.get("polars_pipeline_applied", "MISSING")
        captured["all_kwargs"] = set(kwargs.keys())
        raise _Sentinel()

    for _mod in (main_mod, _suite_mod):
        if hasattr(_mod, "_phase_fit_pipeline"):
            monkeypatch.setattr(_mod, "_phase_fit_pipeline", fake_fit_pipeline)
        if hasattr(_mod, "_phase_pandas_conversion_and_cat_prep"):
            monkeypatch.setattr(_mod, "_phase_pandas_conversion_and_cat_prep", fake_pandas_conv)

    # Minimal valid dataset + extractor: just enough to reach the cat-prep call.
    import numpy as np
    import pandas as pd
    from tests.training.shared import SimpleFeaturesAndTargetsExtractor

    rng = np.random.default_rng(0)
    n = 64
    df = pd.DataFrame(
        {
            "x0": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "target": rng.standard_normal(n),
        }
    )
    extractor = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

    with pytest.raises(_Sentinel):
        main_mod.train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="wire_test",
            features_and_targets_extractor=extractor,
            mlframe_models=["linear"],
            use_mlframe_ensembles=False,
            verbose=0,
        )

    assert "polars_pipeline_applied" in captured["all_kwargs"], (
        "main.py must explicitly pass polars_pipeline_applied= to _phase_pandas_conversion_and_cat_prep; "
        "pre-fix the kwarg was absent and the recipient silently relied on default=True."
    )
    assert captured["polars_pipeline_applied"] is False, (
        f"main.py must forward the value returned by _phase_fit_pipeline (forced False here), "
        f"not the default True. Got: {captured['polars_pipeline_applied']!r}"
    )
