"""Regression sensor for INV-44: composite-target discovery must render the winning-spec
diagnostic plots (``plot_target_distribution`` + ``plot_mi_gain_with_jitter``) into the chart
dir when charts are saved.

Pre-fix the two helpers in ``composite/diagnostics.py`` had ZERO production callers --
``run_composite_target_discovery`` never accepted a chart dir and never rendered them, so a
real run produced no target-distribution / MI-gain image on disk. This sensor calls the
production discovery entry-point with ``save_charts=True`` + a ``data_dir`` on a synthetic
where a ``linear_residual`` composite clears the MI-gain gate, and asserts the PNGs appear.
It FAILS on the pre-fix function (no ``data_dir`` / ``save_charts`` params, no render).
"""

from __future__ import annotations

import glob
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from mlframe.training.configs import CompositeTargetDiscoveryConfig, TargetTypes
from mlframe.training.core._phase_composite_discovery import run_composite_target_discovery


def _tvt_strong(n: int = 1500, seed: int = 0):
    """y = 0.95 * TVT_prev + 0.5 * x1 - 0.3 * x2 + small noise -- strong AR-lag base so a
    ``linear_residual`` composite clears the default MI-gain gate on a small sample."""
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=10.0, scale=3.0, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = 0.95 * base + 0.5 * x1 - 0.3 * x2 + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame({"TVT_prev": base, "x1": x1, "x2": x2, "x3": x3})
    return df, y.astype(np.float64)


def test_inv44_discovery_renders_winning_spec_diagnostics_on_disk(tmp_path):
    feats_df, y = _tvt_strong()
    n = len(feats_df)
    full_idx = np.arange(n)

    cfg = CompositeTargetDiscoveryConfig(enabled=True)
    target_by_type = {TargetTypes.REGRESSION: {"TVT": y}}
    metadata: dict = {}

    data_dir = str(tmp_path / "charts")

    _, metadata = run_composite_target_discovery(
        composite_target_discovery_config=cfg,
        target_by_type=target_by_type,
        mlframe_models=None,
        metadata=metadata,
        filtered_train_df=feats_df,
        filtered_train_idx=full_idx,
        train_df_pd=feats_df,
        val_df_pd=feats_df,
        test_df_pd=feats_df,
        train_idx=full_idx,
        val_idx=full_idx,
        test_idx=full_idx,
        baseline_diagnostics_config=None,
        cat_features=None,
        verbose=False,
        data_dir=data_dir,
        save_charts=True,
    )

    specs = metadata.get("composite_target_specs", {}).get(str(TargetTypes.REGRESSION), {}).get("TVT", [])
    assert specs, "discovery accepted no composite spec on the strong-AR synthetic; cannot prove chart wiring"

    saved = metadata.get("composite_target_diagnostic_charts", {}).get(str(TargetTypes.REGRESSION), {}).get("TVT", [])
    assert saved, "INV-44: no diagnostic chart paths stamped into metadata when save_charts=True"

    mi_gain_pngs = glob.glob(os.path.join(data_dir, "composite_*_mi_gain.png"))
    tdist_pngs = glob.glob(os.path.join(data_dir, "composite_*_tdist_*.png"))
    assert mi_gain_pngs, f"INV-44: MI-gain diagnostic PNG not written under {data_dir}"
    assert tdist_pngs, f"INV-44: target-distribution diagnostic PNG not written under {data_dir}"
    for _p in mi_gain_pngs + tdist_pngs:
        assert os.path.getsize(_p) > 0, f"INV-44: diagnostic PNG {_p} is empty"


def test_inv44_no_charts_when_save_charts_false(tmp_path):
    """The render is gated on ``save_charts``; with it off, no diagnostic PNG is produced even
    though discovery still accepts the spec."""
    feats_df, y = _tvt_strong()
    full_idx = np.arange(len(feats_df))
    cfg = CompositeTargetDiscoveryConfig(enabled=True)
    metadata: dict = {}
    data_dir = str(tmp_path / "charts")

    run_composite_target_discovery(
        composite_target_discovery_config=cfg,
        target_by_type={TargetTypes.REGRESSION: {"TVT": y}},
        mlframe_models=None,
        metadata=metadata,
        filtered_train_df=feats_df,
        filtered_train_idx=full_idx,
        train_df_pd=feats_df,
        val_df_pd=feats_df,
        test_df_pd=feats_df,
        train_idx=full_idx,
        val_idx=full_idx,
        test_idx=full_idx,
        baseline_diagnostics_config=None,
        cat_features=None,
        verbose=False,
        data_dir=data_dir,
        save_charts=False,
    )
    assert not glob.glob(os.path.join(data_dir, "composite_*.png")), "INV-44: diagnostic PNGs written despite save_charts=False"
