# Composite Targets audit — 90 FUTURE findings, final disposition rollup

Every one of the 90 FUTURE-bucketed findings from the 2026-06-10 adversarial
audit has an explicit disposition below (per the "every finding gets a row"
rule). Buckets: **DONE** (implemented + tested this campaign), **DEFER-PERF**
(measurement-gated; bench unreliable on this contended host — kept as a tracked
option, see REJECTED≠DELETED), **DOC** (doc/API-surface tooling), **DESIGN**
(deliberate design choice / conservative direction, anti-recommendation).

Commits this campaign: `649a5a17` (25 localized) · `67e7a00e` (batch A) ·
`eb17a103` (arch wave 2, 22) · `c37c37fe` (M8/I6/N6) · `5be5e514` (E13/E23/E6/DX3)
· `68354b36` (M9) · `6fac55bd` (M7) · `b3d62bc2` (N21) · `c74570f0` (wave 3:
M2/M5/A5/A12/D13/N19/P15) · `e972f0ba` (E7/T13/E14/E17/E22/S9/I23/N9) · `44b21477`
(M4/D21/P18 + test-gaps A34/DX18/E21/N28/T25 + E7 post_shim).

**Final tally: 90/90 dispositioned — 82 implemented + tested · 7 DEFER-PERF/DOC/DESIGN · T21 resolved via D13.** A subsequent comments-only pass stripped 172 audit-finding-ID / phase / date markers from the touched source per the CLAUDE.md comment-hygiene rule (no behaviour change).

## DONE (implemented + tested)

### Headline extensions
- **M6** — time/group-aware discovery CV (`time_column` → time-sorted screen → TimeSeriesSplit everywhere).
- **M7** — classification via base-margin residuals (`CompositeClassificationEstimator`).
- **M8** — split-conformal prediction intervals (`conformal.py`, `predict_interval`).
- **M9** — time-series transform auto-wiring (`time_series_transforms_enabled`).

### Discovery rigor / leakage
M1 (stratified heavy-tail sampling) · M2 (per-fold transform refit contract) ·
M3 (decorrelated stability seeds + row subsample) · M5 (Nadeau-Bengio variance) ·
M13 (paired majority-of-folds selection) · D8 (report flag reconciliation) ·
D11/D12 (per-pair MNAR masking, knn + auto-base) · D13≡T21 (unary base-free
context + naming) · A4/A16/A29 (raw-baseline time predicate / descending time /
zero-base sentinel) · A5 (scored-row population parity) · A6/A7/A18 (stacked
pass-2 warnings + cap) · A10/A11 (GroupKFold clamp + escape hatch) · A12/P11
(early-stop conduit) · A13 (fixed-length per-seed) · A17 (stacked time-aware) ·
A19 (group-aware stepwise) · A24 (localized).

### Estimator / predict / update
E6/DX3 (grouped predict_quantile) · E7 (sample_weight signature inspection) ·
E8/E10/E11/E19 (localized) · E13 (drift-refit envelope refresh) · E14 (quantile
domain-fallback + T-clip parity) · E17 (T-clip observability in runtime_stats) ·
E22/S9 (ring-buffer update) · E23 (robust allowlist) · T13 (time-recurrent
domain mask) · T7/T8/T10/T11/T15/T20 (localized) · DX15 (localized).

### Ensemble / OOF / integration
N6 (per-fold OOF refit) · N9 (external-holdout base param resolved) · N17/DX12
(localized) · N19 (feature-stacking groups + per-fold weight) · N20 (kfold OOF
is train-row-indexed) · N21 (time-respecting kfold OOF) · I2/I11/I15 (localized)
· I6 (per-spec keying) · I7/I13/I20 (MTR-OOF observability + parity) · I23
(failed-component zero weight).

### Perf (bit-identical) / cache / provenance
P9 (per-seed hoist) · P13 (probe gather reuse) · P14 (auto-base pool reuse) ·
P15 (int16 prebin) · S3/S18/S25 (cache) · S7/S8 (streaming) · S12 (provenance).

### Statistical
M4 (Benjamini-Hochberg FDR option) · D21 (near-collinear dedup in mi_y_compare) ·
P18 (valid_screen memo).

### Test-gaps closed
A34 (stacked suite-training) · DX18 (docs-symbol smoke) · E21 (compliance matrix
unary/multi-base/grouped/polars/quantile) · N28 (group+weight OOF, downgrade pin,
conformal) · T25 (adversarial round-trip parity sweep).

## DEFER-PERF (measurement-gated — option kept, needs an uncontended bench)
- **A20** — forward_stepwise per-trial `column_stack` churn → preallocated trial buffer. Needs a before/after bench; host is contended.
- **E12** — `_drop_columns` pandas copy per grouped predict on 100+GB frames → CoW/lazy-subset. The byte-size gate is the safe sketch; the copy-elimination needs a large-frame bench.
- **P10** — per-base raw per-bin refit is bin_var-independent, but `per_bin_n_bins=0` (OFF by default) so the win is unrealised until that path is enabled; needs a `_tiny_cv_rmse_raw_y(bin_vars=...)` contract change.
- **D10** — the mi_y per-base recompute is ALREADY fixed (decomposed via `_aggregate_mi_per_feature`); the residual is `np.delete` full-matrix copies held per base, which need an `exclude_col` contract + a RAM bench — RAM-perf only, no correctness gap.

## DOC (doc / public-API tooling)
- **DX19** — 94-field config has no generated knob reference; deliverable is a `scripts/` doc generator + a doc-drift test. Doc tooling, not a code-behaviour change.
- **DX6** — package has no `__all__`; adding one + re-exporting the registry proxy reshapes the public star-import surface. Deferred deliberately: reshaping `import *` risks breaking external importers; needs a coordinated deprecation, not a drive-by.

## DESIGN (deliberate choice / anti-recommendation)
- **T24** — EWMA/frac-diff inverse cold-starts each predict batch from the train-mean anchor. Carrying recurrence state across predict calls would make `predict` stateful and break sklearn purity + `clone`; the cold-start (and its first-k-rows caveat) is the documented, intentional design. Anti-recommendation to add hidden predict state.
