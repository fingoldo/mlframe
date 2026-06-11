# LOOP12 - Debiased ECE annotation on the reliability diagram

Axis: ACCURACY. Status: **RESOLVED**.

## Method + formula + citation

Standard fixed-bin ECE is biased UPWARD: within each bin the empirical positive rate `acc_k` is a finite binomial
estimate of the true rate, so even under PERFECT calibration `E[(acc_k - conf_k)^2] = Var(acc_k) = conf_k(1-conf_k)/n_k > 0`,
and the plug-in ECE reports a spurious positive value that GROWS with the bin count.

The debiased estimator (Kumar, Liang, Ma 2019, "Verified Uncertainty Calibration", NeurIPS - debiased estimator)
works on the squared (L2) scale where the bias is an additive variance term subtractable in closed form:

    ece2_plugin   = sum_k (n_k/N) * (conf_k - acc_k)^2
    bias_k        = conf_k*(1-conf_k) / n_k              (expected per-bin variance under perfect calibration)
    ece2_debiased = sum_k (n_k/N) * [ (conf_k - acc_k)^2 - bias_k ]
    debiased ECE  = sqrt( max(ece2_debiased, 0) )

Computed from the per-bin `(freqs_predicted, freqs_true, hits)` the reliability binning already produces - O(bins),
no extra full-n pass. Singleton bins (n_k == 1) carry no usable variance estimate and are dropped from the correction;
degenerate inputs (no populated finite bin) -> NaN -> the annotation omits the debiased term, chart still renders.

ADDITIVE: the metrics-layer ECE (`mlframe.metrics.calibration._calibration_plot`) is untouched. The new value is a
SECOND annotation embedded in the reliability scatter title ("ECE=..  ECE_debiased=..") so it reaches both the
matplotlib and plotly backends without changing the renderer layer. Default-on via `show_ece_annotation=True`.

Note on norms: standard ECE is L1 (mean |gap|); the debiased estimator is RMS-scale (sqrt of mean squared gap, minus
variance). On REAL miscalibration RMS >= L1 by Jensen, so debiased can read slightly ABOVE standard - it does not zero
out real error. The win is specifically on the perfectly-calibrated regime, where the L1 standard retains a positive
noise floor (mean of |noise| > 0) while the debiased L2 estimate collapses to ~0.

## biz_value numbers (synthetic: score ~ U(0.02,0.98), y ~ Bernoulli(true_p))

Perfectly-calibrated (true_p = score), n=20000:

| nbins | standard ECE | debiased ECE |
|------:|-------------:|-------------:|
| 5     | 0.0033       | 0.0000       |
| 15    | 0.0080       | 0.0000       |
| 40    | 0.0131 (seed 1) | 0.0000    |
| 50    | 0.0151       | 0.0000       |

- Bias reduction (nbins=40, seed 1): standard 0.0131 (spurious, >= 0.01) -> debiased 0.0000 (ratio 0.00, well under 0.5x). RESOLVED.
- Bin-count stability (seed 2): standard inflates 0.0033 -> 0.0151 (+0.0119) as bins go 5 -> 50; debiased changes 0.0000 -> 0.0000 (~bin-count-stable). RESOLVED.

Miscalibrated (overconfident: true_p = 0.5 + 0.4*(score-0.5)), n=20000, nbins=15, seed 3:

| metric | value |
|---|---|
| standard ECE | 0.147 |
| debiased ECE | 0.166 |

Both clearly flag the real miscalibration (>= 0.05); debiased does NOT mask it (stays within ~1.5x of standard). RESOLVED.

Gallery synthetic (binary_separable, n=6000, nbins=15) annotation rendered: **ECE=0.193  ECE_debiased=0.207** (miscalibrated -> debiased tracks standard, as expected).

## cProfile (O(bins), not O(n))

`debiased_ece` profiled at fixed nbins=15 for n=2000 AND n=400000, 200 iters each (400 calls total): total tt < 1s,
identical code path regardless of n - cost is per-bin only. Pinned in `test_debiased_ece_is_o_bins_not_o_n`.

## Tests (tests/reporting/test_calibration_debiased_ece.py - 14, all green)

Unit: annotation present in spec title; annotation disable knob; standard-ECE formula; debiased variance-subtraction
formula; clamp-to-zero when gap below noise; degenerate (no bins / non-finite / singleton) -> NaN; degenerate omits
debiased term but keeps chart; single-class input keeps standard ECE.
biz_value: bias-reduction on perfectly-calibrated; bin-count stability; does-not-mask real miscalibration.
cProfile: O(bins) bound.

Regression check on the existing reporting/calibration suites: 57 passed, no regressions from the title change.

## Commit

calibration.py + test + render_gallery.py + docs/gallery/binary/calibration_reliability.png committed on the current
branch with `git commit -o`. Hash: see final summary.
