"""cProfile harness for ``honest_oof_reconstruction_rmse`` (the honest group-OOF spec-rank selector).

Run: python -m mlframe.training.composite.discovery._benchmarks.bench_honest_oof_selection

Hotspot (expected + measured): the per-spec ``model.fit`` (LightGBM, GIL-released) dominates; the selector already
threads across specs via the joblib threading pool (mirrors the tiny-rerank), so the per-spec fits run concurrently.
The selector is a single train-on-screen / predict-on-holdout fit per spec (ONE held-out set, no K-fold), so it is
cheaper than the K-fold internal CV it replaces as the rank key. No actionable Python-side hotspot beyond the booster:
the gathers (``_extract_column_array``) and the forward/inverse are vectorised numpy. The booster threads are the floor.
"""
from __future__ import annotations

import cProfile
import pstats

import numpy as np
import pandas as pd

from mlframe.training.composite import CompositeSpec, CompositeTargetDiscovery
from mlframe.training.composite.discovery._honest_oof_select import honest_oof_reconstruction_rmse
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _make(n_groups: int = 40, per: int = 5000, n_specs: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    well_level = rng.uniform(0.0, 50.0, n_groups)
    groups = np.repeat(np.arange(n_groups), per)
    n = groups.size
    cols = {f"x{i}": rng.normal(size=n) for i in range(10)}
    base = 0.5 * well_level[groups] + rng.normal(0.0, 0.3, n)
    y = well_level[groups] + 5.0 * cols["x0"] + rng.normal(0.0, 1.0, n)
    cols["base"] = base
    cols["y"] = y.astype(np.float64)
    df = pd.DataFrame(cols)
    feats = [c for c in df.columns if c != "y"]
    holdout = set(np.argsort(well_level)[-8:].tolist())
    hmask = np.array([g in holdout for g in groups])
    holdout_idx, screen_idx = np.nonzero(hmask)[0], np.nonzero(~hmask)[0]
    cfg = CompositeTargetDiscoveryConfig(enabled=True, random_state=0, tiny_model_n_estimators=60)
    disc = CompositeTargetDiscovery(cfg)
    disc._group_ids_for_rerank = groups
    disc.honest_holdout_idx_ = holdout_idx
    specs = [
        CompositeSpec(name=f"y-linres-base-{i}", target_col="y", transform_name="linear_residual",
                      base_column="base", fitted_params={"alpha": 1.0 + 0.01 * i, "beta": 0.0},
                      mi_gain=1.0, mi_y=0.0, mi_t=1.0, valid_domain_frac=1.0, n_train_rows=100)
        for i in range(n_specs)
    ]
    return disc, df, specs, feats, screen_idx, holdout_idx, y


def main() -> None:
    disc, df, specs, feats, screen_idx, holdout_idx, y = _make()
    honest_oof_reconstruction_rmse(disc, df, "y", specs[:2], feats, screen_idx, holdout_idx, y)  # warm
    pr = cProfile.Profile()
    pr.enable()
    honest_oof_reconstruction_rmse(disc, df, "y", specs, feats, screen_idx, holdout_idx, y)
    pr.disable()
    pstats.Stats(pr).sort_stats("cumulative").print_stats(25)


if __name__ == "__main__":
    main()
