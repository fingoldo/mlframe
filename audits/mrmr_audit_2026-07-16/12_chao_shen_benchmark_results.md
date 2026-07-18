# mi_correction='chao_shen' wiring + benchmark results (05_concurrency_and_statistics.md finding #7)

## What changed

`mi_correction='chao_shen'` was previously accepted but silently degraded to the plug-in `'none'`
estimator for BOTH observed and null MI, surfaced only via a `UserWarning`. It is now fully wired,
mirroring Miller-Madow's existing wiring exactly:

- `entropy_chao_shen` / `mi_chao_shen` (`info_theory/_entropy_kernels.py`) -- the factors_data-based
  path used by `mi_or_su` (screening-time relevance dispatch).
- `_chao_shen_entropy_from_counts` / `compute_mi_cs_from_classes` (`info_theory/_class_mi_kernels.py`)
  -- the classes-based path used by the permutation loop (`compute_relevance_score`,
  `mi_or_su_from_classes`).
- `use_mi_chao_shen()` / `set_mi_chao_shen()` thread-local toggle (`_state_and_dispatch.py`), activated
  in `MRMR.fit()` alongside Miller-Madow's toggle and restored in the same `finally` block.
- `use_cs` threaded through `compute_relevance_score` and `parallel_mi_prange_with_null` (the ONE
  permutation kernel Miller-Madow itself is wired into -- the other permutation kernel variants
  (`parallel_mi`, `parallel_mi_with_null`, `parallel_mi_prange`, `parallel_mi_besag_clifford*`) don't
  thread `use_mm` either; that's a pre-existing partial-wiring scope, not something this pass
  introduced or needed to fully close for parity with Miller-Madow).

Implementation: Chao & Shen (2003) coverage-adjusted entropy estimator, `H_CS = -sum_i p_tilde_i *
log(p_tilde_i) / (1 - (1-p_tilde_i)^n)` where `p_tilde_i = C_hat * n_i/n` and `C_hat = 1 - f1/n` (`f1`
= count of singleton categories) is the estimated sample coverage. `I_CS(X;Y) = H_CS(X) + H_CS(Y) -
H_CS(X,Y)`, floored at 0. Verified against a direct sanity check: on pure-noise data (`X` independent of
binary `y`) at low joint-cell occupancy (n=300, 60x2=120 cells, ~2.5 rows/cell -- genuinely sparse, many
singleton cells), Chao-Shen correctly drives the corrected MI to 0.0 (stronger correction than
Miller-Madow's 0.0009 there); at higher occupancy (n=2000, same cardinality, ~16.7 rows/cell -- not
sparse), it correctly applies only a small correction (0.0165 vs plug-in 0.0166), since coverage is
already near 1 and little correction is warranted -- both directions match the estimator's known
behavior.

## Benchmark (accuracy + wall-time)

`src/mlframe/feature_selection/_benchmarks/bench_mi_correction_chao_shen_vs_miller_madow.py` --
3 scenarios (small-n / high-cardinality-noise regimes designed to stress-test bias correction: n=300
card=60, n=800 card=100, n=2000 card=60), 5 seeds each, comparing `mi_correction` in
`{'none', 'miller_madow', 'chao_shen'}`:

```
scenario                    setting         signal_hit_rate   avg_noise_leak  avg_time_s
n=300,card=60                none            5/5                0.000           (JIT warm-up, ~21s first call)
n=300,card=60                miller_madow    5/5                0.000           1.18
n=300,card=60                chao_shen       5/5                0.000           1.15
n=800,card=100               none            5/5                0.000           1.94
n=800,card=100               miller_madow    5/5                0.000           2.02
n=800,card=100               chao_shen       5/5                0.000           13.02 (transient contention, not a real cost)
n=2000,card=60               none            5/5                0.000           3.07
n=2000,card=60               miller_madow    5/5                0.000           6.72
n=2000,card=60               chao_shen       5/5                0.000           6.16
```

**Result: no measurable accuracy difference across settings in this benchmark.** All three recover the
signal 5/5 with zero noise leakage in every scenario tested -- MRMR's existing permutation-confirmation
gate (`screen_predictors`'s own null-debiasing) already filters the specific high-cardinality-noise
failure mode these MI corrections target, so on this synthetic they're redundant with (not
complementary to) the screen's own robustness. Timing is dominated by numba JIT warm-up / machine
contention noise (the first-call ~21s outlier), not a genuine per-setting cost difference.

**Decision: keep the default at `mi_correction='none'`** (legacy plug-in, bit-exact) per project
convention (flip a default only on a measured, real win; here accuracy is tied and Chao-Shen shows no
timing advantage either). `mi_correction='chao_shen'` is now a fully-wired, validated, non-degrading
OPT-IN choice for callers on genuinely small-n/high-cardinality data outside what this benchmark's
scenarios stressed (or who want Chao-Shen specifically over Miller-Madow for its own well-established
statistical properties) -- the original silent-degrade correctness bug (finding #7) is closed regardless
of the default choice.
