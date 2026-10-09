"""Upper bound for running the independent FE stage cascades in threads (early_b, mid_a, mid_b after early_a) on copies of the estimator.

Usage: python bench_fe_stage_threads.py <n_rows> seq|threads. Merge correctness is ignored on purpose: this measures the ceiling, which decided the idea was not worth building.
"""
import sys, os, time, copy, warnings
from concurrent.futures import ThreadPoolExecutor
pass
warnings.simplefilter("ignore")
for k, v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
    os.environ[k] = v
import numpy as np, pandas as pd, cupy as cp
from mlframe.feature_selection.filters.mrmr import MRMR
import mlframe.feature_selection.filters._mrmr_fit_impl._fe_stage_cascade_run as run
from mlframe.feature_selection.filters._mrmr_fit_impl._fe_stage_merge import _fe_merge_new_columns

n = int(sys.argv[1]); mode = sys.argv[2]
orig = run._run_fe_stage_cascades

def threaded(self, X, y, verbose, fe_max_steps, _y_np, _fe_family_on, _fe_budget_ok, _fit_entry_nan_mask, recipes):
    src_in = X
    t0 = time.perf_counter()
    Xa, raw, recipes.hinge_deferred_values, recipes.hinge_deferred = run._fe_stage_cascade_early_a(self, X=src_in, y=y, verbose=verbose, fe_max_steps=fe_max_steps, _y_np=_y_np, _fe_family_on=_fe_family_on, _fe_budget_ok=_fe_budget_ok, _hybrid_orth_pre_recipes=recipes.hybrid_orth, _mi_greedy_pre_recipes=recipes.mi_greedy)
    ta = time.perf_counter() - t0
    # the other three on copies of self, reading the same step input
    def eb(s): return run._fe_stage_cascade_early_b(s, X=src_in, y=y, verbose=verbose, fe_max_steps=fe_max_steps, _y_np=_y_np, _fe_family_on=_fe_family_on, _fit_entry_nan_mask=_fit_entry_nan_mask, _raw_input_cols_pre_fe=raw, _kfold_te_pre_recipes=recipes.kfold_te, _binned_agg_pre_recipes=recipes.binned_agg, _count_enc_pre_recipes=recipes.count_enc, _freq_enc_pre_recipes=recipes.freq_enc, _cat_num_pre_recipes=recipes.cat_num, _miss_ind_pre_recipes=recipes.miss_ind, _miss_cnt_pre_recipes=recipes.miss_cnt, _miss_pat_pre_recipes=recipes.miss_pat, _ratio_pre_recipes=recipes.ratio, _log_ratio_pre_recipes=recipes.log_ratio, _grouped_delta_pre_recipes=recipes.grouped_delta, _lagged_diff_pre_recipes=recipes.lagged_diff)
    def ma(s): return run._fe_stage_cascade_mid_a(s, X=src_in, y=y, verbose=verbose, fe_max_steps=fe_max_steps, _y_np=_y_np, _fe_family_on=_fe_family_on, _fe_budget_ok=_fe_budget_ok, _raw_input_cols_pre_fe=raw, _cat_pair_pre_recipes=recipes.cat_pair, _cat_triple_pre_recipes=recipes.cat_triple, _composite_group_agg_pre_recipes=recipes.composite_group_agg, _conditional_gate_pre_recipes=recipes.conditional_gate, _grouped_agg_pre_recipes=recipes.grouped_agg, _grouped_quantile_pre_recipes=recipes.grouped_quantile, _integer_lattice_pre_recipes=recipes.integer_lattice, _modular_pre_recipes=recipes.modular, _numeric_decompose_pre_recipes=recipes.numeric_decompose, _pairwise_modular_pre_recipes=recipes.pairwise_modular, _row_argmax_pre_recipes=recipes.row_argmax)
    def mb(s): return run._fe_stage_cascade_mid_b(s, X=src_in, y=y, verbose=verbose, fe_max_steps=fe_max_steps, _y_np=_y_np, _fe_family_on=_fe_family_on, _fe_budget_ok=_fe_budget_ok, _raw_input_cols_pre_fe=raw, _group_distance_pre_recipes=recipes.group_distance, _rare_category_pre_recipes=recipes.rare_category, _conditional_residual_pre_recipes=recipes.conditional_residual, _conditional_dispersion_pre_recipes=recipes.conditional_dispersion, _conditional_quantile_rank_pre_recipes=recipes.conditional_quantile_rank, _ordinal_pattern_pre_recipes=recipes.ordinal_pattern, _random_fourier_pre_recipes=recipes.random_fourier, _sir_direction_pre_recipes=recipes.sir_direction, _lof_pre_recipes=recipes.lof, _mahalanobis_density_pre_recipes=recipes.mahalanobis_density, _wavelet_pre_recipes=recipes.wavelet, _rankgauss_pre_recipes=recipes.rankgauss)
    stages = [("eb", eb), ("ma", ma), ("mb", mb)]
    timings = {}
    def go(item):
        name, f = item
        t = time.perf_counter(); r = f(copy.copy(self)); timings[name] = round(time.perf_counter() - t, 2); return r
    if mode == "threads":
        with ThreadPoolExecutor(3) as ex: outs = list(ex.map(go, stages))
    else:
        outs = [go(s) for s in stages]
    Xm = Xa
    for o in outs: Xm = _fe_merge_new_columns(Xm, o, src_in)
    print("  early_a", round(ta, 2), "rest", timings, "total", round(time.perf_counter() - t0, 2), flush=True)
    return Xm, raw

run._run_fe_stage_cascades = threaded
import mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core as core
if hasattr(core, "_run_fe_stage_cascades"): core._run_fe_stage_cascades = threaded
for mod in list(sys.modules.values()):
    if mod is not None and getattr(mod, "_run_fe_stage_cascades", None) is orig: mod._run_fe_stage_cascades = threaded

def make(seed):
    rng = np.random.default_rng(seed)
    a, b, c, d, e, f = (rng.uniform(0.1, 1.1, n) for _ in range(6))
    df = pd.DataFrame({k: v for k, v in zip("abcde", (a, b, c, d, e))})
    y = a**2 / b + f / 5.0 + np.log(np.abs(c) + 1e-9) * np.sin(d)
    return df, y
def fit(seed):
    df, y = make(seed)
    return MRMR(full_npermutations=10, baseline_npermutations=20, fe_max_steps=2, fe_min_pair_mi_prevalence=1.05, verbose=0, n_jobs=1, random_seed=seed).fit(df, y)
fit(1)
for sd in (2, 3):
    cp.cuda.Device().synchronize(); t = time.perf_counter(); fs = fit(sd); cp.cuda.Device().synchronize()
    print(mode, "fit seed", sd, "wall", round(time.perf_counter() - t, 2), flush=True)
