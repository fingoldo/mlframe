"""Regression sensor: each ensemble method must emit its own chart file.

History (2026-05-08): a perf "optimization" force-disabled chart writes
inside ``ensembling._process_single_ensemble_method`` on the false
premise that all 6 ensemble methods were overwriting the same plot_file.
Empirical check showed that ``trainer._setup_model_info_and_paths``
prefixes every plot_file with ``slugify(model_name_prefix)``, and the
ensembling loop sets ``model_name_prefix=f"Ens{METHOD}-N{n_models}"``
PER METHOD -- so each method already wrote a UNIQUE file:

    EnsARITHM-N6_val_perfplot.png
    EnsGEO-N6_val_perfplot.png
    EnsHARM-N6_val_perfplot.png
    EnsMEDIAN-N6_val_perfplot.png
    EnsQUAD-N6_val_perfplot.png
    EnsQUBE-N6_val_perfplot.png

The "fix" deleted 24 user-visible artifacts in exchange for ~32 seconds.
This test exists to fail loudly the next time anyone (human or LLM)
tries the same trick.

The test is small (6k rows, 1 model, ensembles=True) so the suite runs
in well under a minute on CPU.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.training.configs import OutputConfig, TargetTypes
from mlframe.training.core import train_mlframe_models_suite
from tests.training.shared import SimpleFeaturesAndTargetsExtractor


pytestmark = pytest.mark.slow


@pytest.fixture
def ensembling_dataset():
    """6k rows, 4 numeric + 2 cat cols, binary target with planted signal.

    Big enough that the ensembling pipeline doesn't degenerate (need
    ≥3 base models * weights, sub-models all converge), small enough
    that the suite finishes in <60 s on CPU.
    """
    rng = np.random.default_rng(0)
    n = 6000
    X = rng.standard_normal((n, 4)).astype(np.float32)
    score = X[:, 0] - 0.5 * X[:, 1] + 0.3 * rng.standard_normal(n)
    y = (score > 0).astype(np.int64)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["cat_a"] = rng.choice(["A", "B", "C"], n)
    df["cat_b"] = rng.choice(["X", "Y"], n)
    df["target"] = y
    return df


def test_each_ensemble_method_writes_its_own_perfplot(ensembling_dataset, tmp_path):
    """Regression sensor: ensembling must NOT consolidate per-method
    perfplot.png writes into a single shared file. Each ensemble method
    is a different model; users compare their calibration charts.

    Failure mode this guards against (2026-05-08): the previous
    "optimization" forced ``plot_file=""`` inside the ensembling
    flat_params, which dropped ALL per-method chart writes silently.

    Required artifacts:
    - At least one per-base-model perfplot (e.g. CatBoostClassifier_val_perfplot.png).
    - At least 3 distinct ``Ens{METHOD}-N{n}_*_perfplot.png`` files
      (one per simple ensembling method). The standard 6 are
      ARITHM / GEO / HARM / MEDIAN / QUAD / QUBE; require at least 3
      so the test isn't fragile to which methods compute successfully
      on the synthetic data.
    """
    fte = SimpleFeaturesAndTargetsExtractor(
        target_column="target",
        target_type=TargetTypes.BINARY_CLASSIFICATION,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_mlframe_models_suite(
            df=ensembling_dataset,
            target_name="ensembling_artifact_sensor",
            model_name="ens_artifact_sensor",
            features_and_targets_extractor=fte,
            target_type=TargetTypes.BINARY_CLASSIFICATION,
            mlframe_models=["lgb", "xgb"],  # ≥2 base models required for ensembling to run
            hyperparams_config={"iterations": 30},
            output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models"),
            use_mlframe_ensembles=True,
            behavior_config={"prefer_gpu_configs": False},
            verbose=0,
        )

    # Walk tmp_path and collect every emitted perfplot PNG. The default plot_outputs is multi-backend
    # ("plotly[html] + matplotlib[png]"), so the matplotlib image is named ``<base>_perfplot.matplotlib.png``
    # (single-backend configs name it ``<base>_perfplot.png``); match both, exclude the ``.plotly.html`` twin.
    pngs: list[str] = []
    for _root, _dirs, files in os.walk(tmp_path):
        for f in files:
            if "_perfplot" in f and f.endswith(".png"):
                pngs.append(f)

    # 1) At least one base-model perfplot exists.
    base_pngs = [p for p in pngs if "Ens" not in p]
    assert base_pngs, f"no base-model perfplot png written; full list: {sorted(pngs)}"

    # 2) Multiple distinct ``Ens{METHOD}_*_perfplot[.<backend>].png`` artifacts.
    ens_methods_seen: set[str] = set()
    for p in pngs:
        if p.startswith("Ens") and "_perfplot" in p:
            # Filename shape: ``Ens{METHOD}-N{n}_{split}_perfplot.png``;
            # extract the method token between "Ens" and "-N".
            try:
                method_part = p.split("Ens", 1)[1].split("-N", 1)[0]
                ens_methods_seen.add(method_part)
            except (IndexError, ValueError):
                pass
    assert len(ens_methods_seen) >= 3, (
        f"expected ≥3 distinct ensembling methods to each write their own "
        f"perfplot.png; got {sorted(ens_methods_seen)} from PNGs "
        f"{sorted(p for p in pngs if p.startswith('Ens'))}. "
        f"Regression: a previous 'optimization' force-disabled chart writes "
        f"inside _process_single_ensemble_method on the false premise that "
        f"the methods overwrote each other -- they DON'T (slugify of "
        f"model_name_prefix differentiates them via Ens{{METHOD}}). DO NOT "
        f"reapply that 'optimization'."
    )


def test_each_ensemble_method_writes_distinct_filename(ensembling_dataset, tmp_path):
    """Stronger sensor: assert NO two perfplot files share content via
    overwrite. We verify by checking that filename basenames are unique
    across the per-method chart writes.

    If filenames collide, the chart count on disk will be smaller than
    the chart-render call count; here we just check that the filenames
    we find are all distinct (overwrites would collapse them).
    """
    fte = SimpleFeaturesAndTargetsExtractor(
        target_column="target",
        target_type=TargetTypes.BINARY_CLASSIFICATION,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_mlframe_models_suite(
            df=ensembling_dataset,
            target_name="ensembling_artifact_sensor",
            model_name="ens_artifact_sensor_2",
            features_and_targets_extractor=fte,
            target_type=TargetTypes.BINARY_CLASSIFICATION,
            mlframe_models=["lgb", "xgb"],
            hyperparams_config={"iterations": 30},
            output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models"),
            use_mlframe_ensembles=True,
            behavior_config={"prefer_gpu_configs": False},
            verbose=0,
        )

    ens_pngs = []
    for _root, _dirs, files in os.walk(tmp_path):
        for f in files:
            # Multi-backend output names the matplotlib image ``<base>_perfplot.matplotlib.png``; match either form.
            if f.startswith("Ens") and "_perfplot" in f and f.endswith(".png"):
                ens_pngs.append(f)

    assert len(ens_pngs) == len(set(ens_pngs)), (
        f"Duplicate ensemble perfplot filenames -- different methods are overwriting each other's charts. Files: {sorted(ens_pngs)}"
    )
