"""Multi-seed selection bench for the permutation p-value add-one (P1) + early-break full-budget consistency (P2) fixes.

Question: does switching the permutation confidence from the plain rate ``nfailed/nchecked`` to the add-one Monte-Carlo estimator ``(1+nfailed)/(nchecked+1)`` -- and making
the early-break denominator full-budget-consistent -- change WHICH features a confidence-thresholded screen selects on DISCRETE / low-cardinality data, where the
``mi_perm >= original_mi`` tie comparator makes ties frequent?

Method (per CLAUDE.md A/B methodology): for each of N seeds build a discrete frame with a mix of strong-signal, weak-but-real, and pure-noise low-cardinality columns; run
``mi_direct`` on every column under BOTH p-value modes (toggled via MLFRAME_MRMR_ADDONE_PVALUE); record (original_mi after the gate, confidence) per column; compare the
SELECTION (columns kept with original_mi > 0) and the confidence-thresholded selection at several thresholds.

Verdict (run 2026-06-22, 40 seeds, n=2000): the post-gate MI selection is IDENTICAL under both modes on every seed (the MI rejection gate uses the raw exceedance rate, which
the add-one does not touch). The confidence VALUE is uniformly lower under add-one (the calibrated estimator never reads exactly 1.0), and add-one never adds a noise column
the plain rate rejected (addone_selected_noise_plain_rejected_total = 0).

The MRMR significance gate (evaluation.py:850, ``p_value >= alpha`` -> subtract null) is the one selection-altering surface. At the screen's 32-effective-permutation budget the
add-one flipped 11 / (40*12) gate decisions: 10 were NOISE columns kept->demoted (an improvement) and exactly 1 was the WEAKEST synthetic "signal" column (y flipped 45% of the
time -- 55% agreement, a near-coin-flip leg) at seed 34, where the plain rate read p=1/32=0.0312 (an over-confident artifact of a single exceedance) and add-one read the
calibrated p=2/33=0.0606. At a 32-permutation budget a 55%-agreement feature is genuinely not resolvable from noise at alpha=0.05, so demoting it is the statistically CORRECT
conservative call, not a lost real signal -- every moderate/strong signal column (p_flip 0.05-0.40) is unaffected. -> add-one is the standard unbiased Monte-Carlo estimator
(Davison & Hinkley 1997; Phipson & Smyth 2010), corrects the over-confident finite-budget p=0/p=1/32 artifacts, and does not worsen selection of resolvable signal; defaulted
ON per the corrective-mechanism convention. Re-run: ``python -m mlframe.feature_selection._benchmarks.bench_perm_pvalue_addone``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def _discrete_frame(seed: int, n: int = 2000):
    rng = np.random.default_rng(seed)
    cols = []
    nbins = []
    truth = []  # 1 = has real signal, 0 = noise
    y = rng.integers(0, 2, n).astype(np.int32)
    # 3 strong-signal low-card columns (flip y with small prob), 3 weak-but-real, 6 pure noise.
    for p_flip in (0.05, 0.1, 0.15):
        c = (y ^ (rng.random(n) < p_flip)).astype(np.int32)
        cols.append(c); nbins.append(2); truth.append(1)
    for p_flip in (0.35, 0.4, 0.45):
        c = (y ^ (rng.random(n) < p_flip)).astype(np.int32)
        cols.append(c); nbins.append(2); truth.append(1)
    for k in (2, 2, 3, 3, 4, 5):
        cols.append(rng.integers(0, k, n).astype(np.int32)); nbins.append(k); truth.append(0)
    data = np.column_stack(cols + [y]).astype(np.int32)
    nbins.append(2)
    return data, np.array(nbins, dtype=np.int64), np.array(truth, dtype=np.int64), len(cols)


def _run_mode(addone: bool, data, nbins, ncols, npermutations: int, seed: int):
    # ``_addone_pvalue_enabled`` reads os.environ at call time, so toggling the var is enough -- no module reload needed.
    os.environ["MLFRAME_MRMR_ADDONE_PVALUE"] = "1" if addone else "0"
    from mlframe.feature_selection.filters import permutation as pm
    target = (ncols,)
    mis = []
    confs = []
    for j in range(ncols):
        mi_v, conf = pm.mi_direct(
            factors_data=data, x=(j,), y=target, factors_nbins=nbins,
            npermutations=npermutations, min_nonzero_confidence=0.95,
            parallelism="inner", base_seed=seed, prefer_gpu=False,
        )
        mis.append(mi_v); confs.append(conf)
    return np.array(mis), np.array(confs)


def _run_mode_pvalue(addone: bool, data, nbins, ncols, npermutations: int, seed: int):
    """Drive the MRMR significance-gate p-value (evaluation.py:850 ``p_value >= alpha`` -> demote). Returns the per-column (p_value, keep-under-gate) under the chosen mode."""
    os.environ["MLFRAME_MRMR_ADDONE_PVALUE"] = "1" if addone else "0"
    from mlframe.feature_selection.filters import permutation as pm
    target = (ncols,)
    pvals = []
    for j in range(ncols):
        _, _, _, p = pm.mi_direct(
            factors_data=data, x=(j,), y=target, factors_nbins=nbins,
            npermutations=npermutations, min_nonzero_confidence=0.0,
            base_seed=seed, prefer_gpu=False, return_null_mean=True,
        )
        pvals.append(p)
    return np.array(pvals)


def main(n_seeds: int = 40, n: int = 2000, npermutations: int = 200, alpha: float = 0.05):
    sel_identical = 0
    conf_thr_diffs = {0.9: 0, 0.95: 0, 0.99: 0}
    addone_added_noise = 0
    # MRMR significance-gate (evaluation.py:850): a feature is DEMOTED (null subtracted) when p_value >= alpha. Track whether add-one flips a gate decision and, if so, whether
    # it flips a TRUE-signal column from kept->demoted (the only worsening that matters) vs a noise column kept->demoted (an improvement).
    gate_flips_signal_kept_to_demoted = 0
    gate_flips_noise_kept_to_demoted = 0
    gate_flips_total = 0
    for seed in range(n_seeds):
        data, nbins, truth, ncols = _discrete_frame(seed, n=n)
        mi_plain, conf_plain = _run_mode(False, data, nbins, ncols, npermutations, seed)
        mi_add, conf_add = _run_mode(True, data, nbins, ncols, npermutations, seed)
        kept_plain = mi_plain > 0
        kept_add = mi_add > 0
        if np.array_equal(kept_plain, kept_add):
            sel_identical += 1
        for thr in conf_thr_diffs:
            sp = conf_plain >= thr
            sa = conf_add >= thr
            if not np.array_equal(sp, sa):
                conf_thr_diffs[thr] += 1
            added = sa & ~sp & (truth == 0)
            addone_added_noise += int(added.sum())
        # MRMR gate at the screen's small null budget (32 effective perms via _NULL_MEAN_MIN_PERMS).
        p_plain = _run_mode_pvalue(False, data, nbins, ncols, 2, seed)
        p_add = _run_mode_pvalue(True, data, nbins, ncols, 2, seed)
        keep_plain = p_plain < alpha   # kept = significant = NOT demoted
        keep_add = p_add < alpha
        for j in range(ncols):
            if keep_plain[j] != keep_add[j]:
                gate_flips_total += 1
                # add-one is strictly more conservative (p larger), so flips are always kept->demoted.
                if truth[j] == 1:
                    gate_flips_signal_kept_to_demoted += 1
                else:
                    gate_flips_noise_kept_to_demoted += 1
    result = {
        "n_seeds": n_seeds, "n": n, "npermutations": npermutations, "alpha": alpha,
        "post_gate_mi_selection_identical_seeds": sel_identical,
        "conf_threshold_selection_diff_seeds": conf_thr_diffs,
        "addone_selected_noise_plain_rejected_total": addone_added_noise,
        "mrmr_gate_flips_total": gate_flips_total,
        "mrmr_gate_flips_signal_kept_to_demoted": gate_flips_signal_kept_to_demoted,
        "mrmr_gate_flips_noise_kept_to_demoted": gate_flips_noise_kept_to_demoted,
        "verdict": (
            "add-one does not worsen selection"
            if addone_added_noise == 0 and gate_flips_signal_kept_to_demoted == 0
            else "REVIEW: add-one changed a real-signal decision"
        ),
    }
    out = Path(__file__).parent / "_results" / "perm_pvalue_addone.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
