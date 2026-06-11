# LOOP29 — Risk-coverage / accuracy-rejection curve (selective prediction)

## Design

New chart `src/mlframe/reporting/charts/risk_coverage.py`:

- `compute_risk_coverage(y_true, y_score, *, task, confidence=None)` — confidence = `|p-0.5|*2` (binary), top-class prob (multiclass), or a caller-supplied array (regression). ONE descending stable argsort + one cumulative pass: `np.cumsum(sorted_loss)/running_count` gives risk (= 1-accuracy classification, or running MAE regression) at every coverage. Returns `(coverage, accuracy, risk, aurc, full_risk, has_ranking_signal)`. AURC = `np.trapezoid(risk, coverage)`; the flat **random-rejection** reference = constant full risk (`full_risk`), whose AURC equals `full_risk`.
- `build_risk_coverage_spec(...)` → `RiskCoverageResult` with the `FigureSpec`. Classification panel: accuracy-vs-coverage (left axis) + flat random-rejection accuracy line + risk twin (right axis); acc@80% operating-point star. Regression panel: retained-MAE-vs-coverage + flat random-rejection MAE line; err@80% star. Decimated to <=2000 plotted points for huge n (AURC still on the full curve).

Distinction from existing charts (NOT redundant — RESOLVED, not rejected): the **threshold sweep / decile gain** vary the decision THRESHOLD at full coverage (every row scored). Risk-coverage varies **COVERAGE** by abstaining on the least-confident rows and never moves the threshold. Orthogonal deployment question ("defer the uncertain to a human"), so it is kept.

## Edge handling

Single-class / all-equal confidence → `has_ranking_signal=False`, flat curve, title annotated `[constant confidence: no ranking signal -> flat curve]`. Tiny n and NaN rows: non-finite scores/labels dropped jointly; empty-after-drop → NaN risk, no crash. Regression without `confidence` raises `ValueError`.

## biz_value numbers (measured)

Well-ranked binary synthetic (confidence ranks correctness, `correct ~ 0.45 + 0.54*conf`, n=8000):
- **accuracy@80% = 0.806 vs accuracy@100% = 0.761 → selective gain +0.045..+0.058** across seeds (min 0.053 at n=8000).
- **AURC = 0.144 vs random-rejection AURC (full risk) = 0.282** → confidence ranking roughly halves area-under-risk.

Random-confidence binary (confidence independent of correctness, same 0.75 accuracy):
- accuracy@80% ~= accuracy@100% (|gain| < 0.02), AURC ~= random AURC → **flat, no selective gain** (asserted).

Gallery PNG (n=8000 well-ranked): green accuracy curve rises 0.71→1.0 as coverage→0, flat gray random line at 0.711, red risk twin, acc@80%=0.765 star.

## cProfile (n=1e6, `_benchmarks/profile_risk_coverage.py`)

~310 ms/call. argsort = 0.989s / 1.373s over 5 calls = **73%** (the floor), cumsum ~5%, trapezoid/confidence the rest. Argsort-bound by design; no actionable speedup beyond the sort.

## Tests

- `tests/reporting/test_risk_coverage.py` — 12 unit tests: curve monotonicity (coverage strictly increasing, risk=1-acc), AURC in [0,1] + below random, accuracy rises when ranked, random flat, multiclass top-prob, regression error drops, constant-confidence flat + annotation, NaN drop, empty, tiny-n, random-reference line present + constant, decimation cap, regression-requires-confidence raise.
- `tests/reporting/test_risk_coverage_biz_value.py` — 3 biz_value: well-ranked selective gain (>=0.05 + AURC<random), random flat (|gain|<0.02), ranked AURC strictly < random AURC.
- All 15 pass; decision_curve suite still green (22 passed together).

## Wiring (RESOLVED — wired, default-on)

- `ReportingConfig.risk_coverage_charts: bool = True` (`src/mlframe/training/_reporting_configs.py`).
- `src/mlframe/reporting/_risk_coverage_diagnostic.py::render_risk_coverage_diagnostic` (separate module because `diagnostics_dispatch.py` is at the 1k LOC limit; reuses its `_save_spec/_record/_record_path`). Records `risk_coverage_aurc` + `risk_coverage_selective_gain` into metrics.
- Called per-(model, split) in `src/mlframe/training/reporting/_reporting.py`: binary (positive score), multiclass (proba matrix), regression (confidence proxy = `-|pred-mean(pred)|`).

CHANGELOG.md NOT touched (foreign-dirty from parallel composite session). README.md NOT touched per house rules.

## Gallery

`docs/gallery/risk_coverage/risk_coverage.png` re-rendered; entry added to `scripts/render_gallery.py`.

## Disposition: RESOLVED (shipped + wired).
