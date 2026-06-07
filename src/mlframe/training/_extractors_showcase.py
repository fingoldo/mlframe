"""``showcase_features_and_targets`` carved out of ``mlframe.training.extractors``.

Re-imported at the parent module's bottom so historical
``from mlframe.training.extractors import showcase_features_and_targets`` import
sites keep working.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from pyutilz.pythonlib import is_jupyter_notebook

from .configs import TargetTypes
from .utils import get_pandas_view_of_polars_df
from ._extractors_dtype_helpers import get_dataframe_info

logger = logging.getLogger("mlframe.training.extractors")


def showcase_features_and_targets(
    df: Union[pd.DataFrame, pl.DataFrame],
    target_by_type: Dict[str, Dict[str, Any]],
    max_hist_samples: int = 100_000,
    random_seed: int = 42,
) -> None:
    """Show distribution of features and targets.

    Args:
        df: DataFrame with features.
        target_by_type: Dictionary of targets by type (e.g., {TargetTypes.REGRESSION: {"target1": array}}).
        max_hist_samples: Maximum samples to use for histogram (performance threshold).
    """
    print(get_dataframe_info(df))

    head = df.head(5)
    if isinstance(df, pl.DataFrame):
        # Route through Arrow split-blocks bridge so pl.Enum / pl.Categorical / pl.Date columns keep their pandas-native dtypes (CategoricalDtype / DatetimeTZDtype) instead of collapsing to object; preserves the display contract for Jupyter rich rendering.
        head = get_pandas_view_of_polars_df(head)

    non_floats = head.select_dtypes(exclude=np.float32)

    caption = "Non-float32 dtypes"

    logger.info(f"{caption}: {non_floats.dtypes.to_dict()}")

    in_jupyter = is_jupyter_notebook()

    if in_jupyter:
        from IPython.display import display
        from .reporting import _style_with_caption

        display(_style_with_caption(non_floats, "Non-float32 dtypes"))

    for target_type, targets in target_by_type.items():
        for target_name, target in targets.items():
            line = f"{target_type} {target_name}"
            if in_jupyter:
                from IPython.display import display

                display(line)
            else:
                print(line)
            if target_type == TargetTypes.REGRESSION:
                # Subsample if target is large to speed up histogram. Use a
                # local seeded Generator instead of the global numpy RNG so
                # histograms shown to the user are reproducible across runs
                # and don't depend on whatever else mutated np.random state.
                _hist_rng = np.random.default_rng(random_seed)
                if len(target) > max_hist_samples:
                    if isinstance(target, (pl.Series, pd.Series)):
                        sample_idx = _hist_rng.choice(
                            len(target), max_hist_samples, replace=False
                        )
                        # polars Series ``target[sample_idx]`` with a numpy integer array works
                        # (polars treats it as ``.gather(sample_idx)``); pandas Series uses ``.iloc`` for
                        # positional indexing to avoid the FutureWarning on label-vs-position dispatch.
                        sample = (
                            target.iloc[sample_idx].values
                            if isinstance(target, pd.Series)
                            else target[sample_idx].to_numpy()
                        )
                    else:
                        sample_idx = _hist_rng.choice(
                            len(target), max_hist_samples, replace=False
                        )
                        sample = target[sample_idx]
                    # Add min and max to preserve full range (if not already in sample)
                    if isinstance(target, pl.Series):
                        min_val, max_val = target.min(), target.max()
                    elif isinstance(target, pd.Series):
                        min_val, max_val = target.min(), target.max()
                    else:  # np.ndarray
                        min_val, max_val = np.min(target), np.max(target)
                    extras = []
                    if min_val not in sample:
                        extras.append(min_val)
                    if max_val not in sample:
                        extras.append(max_val)
                    plot_data = np.concatenate([sample, extras]) if extras else sample
                else:
                    # Convert to numpy array if needed
                    if isinstance(target, pl.Series):
                        plot_data = target.to_numpy()
                    elif isinstance(target, pd.Series):
                        plot_data = target.values
                    else:
                        plot_data = target
                plt.hist(plot_data, bins=30, color="skyblue", edgecolor="black")

                # Add titles and labels
                plt.title(f"{target_name} Histogram")
                plt.xlabel("Value")
                plt.ylabel("Frequency")

                # Show the plot
                plt.show()

                # Wave 55 (2026-05-20): unknown target type (LazyFrame / torch tensor / list)
                # left desc_data undefined, causing NameError on the display below. Skip
                # gracefully instead -- this is a display-only diagnostic path.
                desc_data = None
                if isinstance(target, (pl.Series, pd.Series)):
                    desc_data = target.describe()
                elif isinstance(target, np.ndarray):
                    desc_data = pl.Series(target).describe()

                if desc_data is not None:
                    if in_jupyter:
                        from IPython.display import display

                        display(desc_data)
                    else:
                        print(desc_data)

            elif target_type == TargetTypes.BINARY_CLASSIFICATION:
                desc_data = None
                if isinstance(target,  pd.Series):
                    desc_data = target.value_counts(normalize=True)
                elif isinstance(target, pl.Series):
                    desc_data = target.value_counts(normalize=True, sort=True)
                elif isinstance(target, np.ndarray):
                    desc_data = pl.Series(target).value_counts(normalize=True, sort=True)

                if desc_data is not None:
                    if in_jupyter:
                        from IPython.display import display

                        display(desc_data)
                    else:
                        print(desc_data)

    if in_jupyter:
        from IPython.display import display

        display(head)

        tail = df.tail(5)
        if isinstance(df, pl.DataFrame):
            tail = get_pandas_view_of_polars_df(tail)

        display(tail)
