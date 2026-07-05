"""Isolated bench: per-class post-hoc calibration METHOD for the multiclass / multilabel path.

The production multi-output calibrator (``_PerClassIsotonicCalibrator`` in
``training/_calibration_models.py``, reached from ``training/evaluation.post_calibrate_model``
for any ``(N, K!=2)`` probability matrix) hardcodes plain isotonic for every class column.
Isotonic is a free-form monotone step map: on SMALL calibration folds it interpolates the
training calibration and overfits, generalising worse on held-out data than the 2-parameter
Platt/sigmoid map. The binary path already CHOOSES between sigmoid and isotonic
(``pick_best_calibrator``); the multi-output path does not. This bench measures honest
held-out mean-per-class ECE for ``method='isotonic'`` vs ``method='sigmoid'`` so the default
can be picked on evidence.

Metric: mean over classes of held-out ECE (10 equal-width bins) of the calibrated one-vs-rest
probability column. Calibrator fit on a CALIB slice, scored on a DISJOINT held-out slice.
Lower is better. 5 synthetic scenarios x 3 seeds.

Run: python -m mlframe.calibration._benchmarks.bench_per_class_method_isotonic_vs_sigmoid
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MLFRAME_NO_CUDA_AUTOCONFIG", "1")
os.environ.setdefault("MLFRAME_KEEP_BROKEN_CUPY", "1")

from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

N_BINS = 10


def _heldout_ece_binary(y, p, n_bins=N_BINS):
    """Equal-width-bin ECE of a one-vs-rest probability column on held-out data."""
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    n = len(p)
    for b in range(n_bins):
        m = idx == b
        c = m.sum()
        if c == 0:
            continue
        ece += (c / n) * abs(p[m].mean() - y[m].mean())
    return ece


def _fit_isotonic(p, y):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y)
    return lambda q: np.clip(iso.predict(q), 0.0, 1.0)


def _fit_sigmoid(p, y):
    """Platt: 1-D logistic on the logit of the score; falls back to identity if degenerate."""
    eps = 1e-6
    pc = np.clip(p, eps, 1 - eps)
    z = np.log(pc / (1 - pc)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return lambda q: np.clip(q, 0.0, 1.0)
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(z, y)

    def _apply(q):
        qc = np.clip(q, eps, 1 - eps)
        zq = np.log(qc / (1 - qc)).reshape(-1, 1)
        return lr.predict_proba(zq)[:, 1]

    return _apply


def _scenario(name, rng, n, K):
    """Return (scores_NK, labels_N) where scores are MISCALIBRATED multiclass probs."""
    # True latent logits; produce a label, then a DISTORTED score matrix simulating an
    # over/under-confident base model.
    W = rng.normal(size=(K, 4))
    Xf = rng.normal(size=(n, 4))
    logits = Xf @ W.T
    pt = np.exp(logits - logits.max(1, keepdims=True))
    pt /= pt.sum(1, keepdims=True)
    y = np.array([rng.choice(K, p=pt[i]) for i in range(n)])

    if name == "overconfident":
        s = np.exp(2.2 * logits)
    elif name == "underconfident":
        s = np.exp(0.45 * logits)
    elif name == "shifted":
        s = np.exp(logits + rng.normal(0, 0.8, size=(1, K)))
    elif name == "skewed_temp":
        temps = rng.uniform(0.4, 2.5, size=(1, K))
        s = np.exp(logits / temps)
    else:  # noisy
        s = np.exp(logits + rng.normal(0, 1.0, size=logits.shape))
    s = s / s.sum(1, keepdims=True)
    return s, y


def _eval(method_fit, s_cal, y_cal, s_ho, y_ho, K):
    eces = []
    for k in range(K):
        yk_cal = (y_cal == k).astype(np.float64)
        yk_ho = (y_ho == k).astype(np.float64)
        npos = int(yk_cal.sum())
        if npos < 2 or npos >= len(yk_cal) - 1:
            cal_p = s_ho[:, k]  # identity (matches prod guard)
        else:
            f = method_fit(s_cal[:, k], yk_cal)
            cal_p = f(s_ho[:, k])
        eces.append(_heldout_ece_binary(yk_ho, cal_p))
    return float(np.mean(eces))


def main():
    scenarios = ["overconfident", "underconfident", "shifted", "skewed_temp", "noisy"]
    seeds = [0, 1, 2]
    K = 5
    n_ho = 4000
    # Sweep calib-slice size: small (overfit regime) vs large (isotonic flexibility regime).
    all_results = {}
    for n_cal in (200, 1000, 8000):
        rows = []
        iso_wins = sig_wins = ties = 0
        for scn in scenarios:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                s_all, y_all = _scenario(scn, rng, n_cal + n_ho, K)
                s_cal, y_cal = s_all[:n_cal], y_all[:n_cal]
                s_ho, y_ho = s_all[n_cal:], y_all[n_cal:]
                raw = _eval(lambda p, y: (lambda q: np.clip(q, 0, 1)), s_cal, y_cal, s_ho, y_ho, K)
                iso = _eval(_fit_isotonic, s_cal, y_cal, s_ho, y_ho, K)
                sig = _eval(_fit_sigmoid, s_cal, y_cal, s_ho, y_ho, K)
                if abs(iso - sig) < 1e-4:
                    ties += 1
                    w = "tie"
                elif sig < iso:
                    sig_wins += 1
                    w = "sigmoid"
                else:
                    iso_wins += 1
                    w = "isotonic"
                rows.append(dict(scn=scn, seed=seed, raw=raw, iso=iso, sig=sig, winner=w))
                print(f"n_cal={n_cal:5d} {scn:14s} seed={seed}  raw={raw:.4f}  iso={iso:.4f}  sig={sig:.4f}  -> {w}")
        mean_iso = float(np.mean([r["iso"] for r in rows]))
        mean_sig = float(np.mean([r["sig"] for r in rows]))
        print(f"  -> n_cal={n_cal}: sigmoid_wins={sig_wins} isotonic_wins={iso_wins} ties={ties}" f"  mean_iso={mean_iso:.4f} mean_sig={mean_sig:.4f}\n")
        all_results[str(n_cal)] = dict(rows=rows, sigmoid_wins=sig_wins, isotonic_wins=iso_wins, ties=ties, mean_iso=mean_iso, mean_sig=mean_sig)

    out = Path(__file__).parent / "_results"
    out.mkdir(exist_ok=True)
    (out / "per_class_method_isotonic_vs_sigmoid.json").write_text(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
