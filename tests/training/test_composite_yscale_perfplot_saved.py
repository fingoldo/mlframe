"""Regression: a composite-target model must get its perfplot / residuals / res_dist_and_acf saved.

For a composite target (e.g. ``target_hourly_rate-logY``) the in-training regression chart is skipped on purpose (its
metrics are on the transformed T-scale), and the y-scale chart emitted after the fit is supposed to replace it. That
replacement was never written: it built its path from ``output_config.plot_file``, a config field that defaults to ""
and that the suite never fills, so every composite target folder lacked the three regression charts while the y-scale
metric log line still printed. The chart path now comes from the model entry's own recorded chart prefix, so the files
land beside the model's other charts with the same names a raw-target model gets.
"""

from __future__ import annotations

import glob
import os
from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.training.core._phase_composite_wrapping import _emit_yscale_composite_chart


class CatBoostRegressor:  # stand-in carrying the inner class name used in the chart title
    pass


def _emit(entry, split, plot_file=""):
    rng = np.random.default_rng(0)
    y = rng.normal(10.0, 2.0, 400)
    p = y + rng.normal(0.0, 0.5, 400)
    _emit_yscale_composite_chart(
        y_target=y,
        y_pred=p,
        inner_entry=entry,
        composite_name="target_hourly_rate-logY",
        orig_tname="target_hourly_rate",
        target_name="target_hourly_rate-logY",
        plot_file=plot_file,
        reporting_config=SimpleNamespace(plot_outputs="matplotlib[png]", plot_dpi=60),
        rmse_y=0.5,
        mae_y=0.4,
        r2_y=0.9,
        split_name=split,
    )


def _saved(root, pattern):
    return glob.glob(os.path.join(str(root), "**", pattern), recursive=True)


@pytest.mark.parametrize("split", ["val", "test"])
def test_composite_yscale_charts_use_the_model_chart_prefix(tmp_path, split):
    """With the entry's recorded chart prefix, the three regression charts are saved under the raw-target names."""
    prefix = os.path.join(str(tmp_path), "recency__CatBoostRegressor")
    entry = SimpleNamespace(model=CatBoostRegressor(), plot_file=prefix)
    _emit(entry, split)
    for kind in ("perfplot", "residuals", "res_dist_and_acf"):
        hits = _saved(tmp_path, f"recency__CatBoostRegressor_{split}_{kind}*.png")
        assert hits, f"{split}_{kind} not saved; files: {sorted(os.listdir(tmp_path))}"
    # No composite/yscale disambiguation suffix: the file name matches the raw-target model's own perfplot exactly.
    assert not _saved(tmp_path, "*yscale*")


def test_composite_yscale_chart_directory_prefix_joins_split(tmp_path):
    """A prefix that is a directory (trailing os.sep) joins the split like the regular eval path does."""
    prefix = str(tmp_path) + os.sep
    entry = SimpleNamespace(model=CatBoostRegressor(), plot_file=prefix)
    _emit(entry, "test")
    assert _saved(tmp_path, "test_perfplot*.png")


def test_composite_yscale_chart_without_any_path_is_skipped_quietly(tmp_path):
    """No recorded prefix and no supplied base: nothing is written and nothing raises."""
    entry = SimpleNamespace(model=CatBoostRegressor())
    _emit(entry, "test", plot_file="")
    assert not _saved(tmp_path, "*.png")
