from __future__ import annotations

import scipy.stats as stats
import os
import pandas as pd
from pyutilz.system import tqdmu as tqdm


def ranking2top(ranking):
    return ranking[ranking == ranking.max()].index.tolist()


def kendall_tau(df):
    res_d = {}
    for method, subset in df.items():
        res_d[method] = subset.apply(lambda x: x.split(":")[1].strip()).tolist()

    return {method: round(stats.kendalltau(res_d["AM"], method_top_k)[0], 3) for method, method_top_k in res_d.items() if method != "AM"}


def agreement_rate(df, k, top_k=True):
    # Wave 60 (2026-05-20): clamp k to actual subset size so `iloc[-k:]` doesn't
    # silently return the WHOLE subset (and divide by k anyway, inflating the
    # agreement rate vs AM). The prior code on a 3-row leaderboard with k=10
    # would count all 3 rows but still divide by 10 -- silently misreporting
    # agreement.
    res_d = {}
    _k_eff = k
    for method, subset in df.items():
        _k_eff = min(_k_eff, len(subset))
        _subset = subset.copy().iloc[:k] if top_k else subset.copy().iloc[-k:]
        res_d[method] = _subset.apply(lambda x: x.split(":")[1].strip()).tolist()

    _denom = max(1, _k_eff)
    return {method: round(len(set(method_top_k).intersection(set(res_d["AM"]))) / _denom, 2) for method, method_top_k in res_d.items() if method != "AM"}


def tracker_filename(model, task, dirpath):
    return f"{dirpath}/{model}_{task}_0/"


def get_tracker_table(data, dirpath):

    from experiment_impact_tracker.data_interface import DataInterface
    from experiment_impact_tracker.data_utils import load_initial_info
    from experiment_impact_tracker.utils import gather_additional_info

    di_attrs = ["exp_len_hours", "kg_carbon", "total_power"]
    info_attrs = ["gpu_hours"]

    tracker_cols = []
    for task in data.columns:
        tracker_cols += [task.split(".")[0] + "_" + attr for attr in di_attrs + info_attrs]

    tracker_cols = list(set(tracker_cols))
    tracker_res = pd.DataFrame(columns=tracker_cols, index=data.index)

    for f in tqdm(os.listdir(dirpath)):
        model, task, _ = f.split("_")
        if model not in data.index:
            continue

        fname = tracker_filename(model, task, dirpath)
        datain = DataInterface([fname + "impacttracker"])
        info = load_initial_info(fname)
        add_info = gather_additional_info(info, fname)

        for attr in di_attrs:
            tracker_res.loc[model, f"{task}_{attr}"] = getattr(datain, attr)
        for attr in info_attrs:
            tracker_res.loc[model, f"{task}_{attr}"] = add_info[attr]

    return tracker_res
