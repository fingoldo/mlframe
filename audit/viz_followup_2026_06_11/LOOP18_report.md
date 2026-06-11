# LOOP18 - Opt-in data-aware binary panel emphasis

Axis: UI/UX. Verdict: RESOLVED.

## Design

Added an opt-in `panel_emphasis` mode so the binary report surfaces the panels
that matter for the data's class skew, without changing default behavior.

- New config: `ReportingConfig.panel_emphasis: Literal["all","data_aware"] = "all"`
  plus `emphasis_imbalance_lo = 0.2` / `emphasis_imbalance_hi = 0.8`.
  `"all"` (default) leaves `binary_panels` unchanged - byte-identical to current
  behavior (back-compat proven by tests).
- New pure helper `select_binary_emphasis_panels(y_true, requested_panels, *,
  emphasis, imbalance_lo, imbalance_hi)` in `reporting/auto_dispatch.py`. One
  O(n) `count_nonzero` of `y_true` derives the positive base rate (no new full
  passes; threshold configurable). It only reorders / selects within the tokens
  already present in `requested_panels` - it never invents a token.
- `render_multi_target_panels` gained `panel_emphasis`, `binary_panels_is_default`,
  `emphasis_imbalance_lo/hi`. Emphasis is applied to the binary branch ONLY when
  `panel_emphasis == "data_aware"` AND `binary_panels_is_default` is True (the
  operator left `binary_panels` at its default). A custom `binary_panels` is
  never reordered.

### Selection rule
- Imbalanced (base rate < lo or > hi): lead with `PR THRESHOLD SCORE_DIST KS GAIN`;
  ROC is dropped outright (optimistic under imbalance). Any other requested
  tokens (e.g. PIT) are appended so no panel is silently lost.
- Balanced: lead with `ROC PR SCORE_DIST KS THRESHOLD`, then append the rest.
- Note: there is NO `DECISION_CURVE` token in the binary composer (decision-curve
  is a separate wired path), so it is correctly never emitted by the emphasis
  order - the intersection-with-available-tokens design handles this by
  construction.

### Edge-safety
- Single-class (`n_pos == 0` or `== n`) -> fall back to `requested_panels`.
- Tiny n (< 50 usable rows) -> fall back (base rate too noisy).
- NaN float labels are excluded from the base-rate count.
- Empty template returned as-is.

## Back-compat proof
`select_binary_emphasis_panels(y, default, emphasis="all") == default` for both
the 0.03-imbalanced and 0.5-balanced synthetics (unit + biz_value tests). The
dispatcher applies emphasis only behind `panel_emphasis="data_aware" AND
binary_panels_is_default`, both default-off, so existing callers are untouched.

## Imbalanced-vs-balanced selected sets (default binary_panels = "ROC PR SCORE_DIST KS THRESHOLD GAIN PIT")
- Imbalanced (base rate 0.03): `PR THRESHOLD SCORE_DIST KS GAIN PIT` (PR/THRESHOLD-led, ROC dropped).
- Balanced (base rate 0.50): `ROC PR SCORE_DIST KS THRESHOLD GAIN PIT` (ROC-led).

## Tests
`tests/reporting/test_panel_emphasis.py` (17 tests, all pass; full file + existing
`test_auto_dispatch.py` = 35 passed):
- unit: all-mode identity; imbalanced PR-led + ROC dropped; high-base-rate also
  imbalanced; balanced ROC-led; single-class fallback (both 0/1); tiny-n fallback;
  empty template; NaN-label handling; configurable thresholds.
- dispatcher: default mode unchanged renders binary; data_aware applies only when
  `binary_panels_is_default` (custom template passed through untouched).
- biz_value: imbalanced 0.03 leads PR/THRESHOLD + excludes ROC; balanced includes
  and leads ROC; back-compat default identical to current for both skews.
- cProfile: 20 calls over 1M rows < 0.5s (trivial O(n); no measurable dispatch
  overhead).

## Gallery PNGs (data_aware mode)
- `docs/gallery/binary/panel_emphasis_imbalanced.png` (PR/THRESHOLD-led, ROC dropped)
- `docs/gallery/binary/panel_emphasis_balanced.png` (ROC-led)

## FUTURE (not in owned scope; not clobbered)
Wiring `ReportingConfig.panel_emphasis` through the trainer chain
(`_trainer_train_and_evaluate.py` -> `_eval_helpers.py` ->
`training/reporting/_reporting.py` -> dispatcher) and computing
`binary_panels_is_default` (compare `reporting.binary_panels` to the field
default) is a separate plumbing step in non-owned files. The dispatcher + config
field + selection logic are complete and directly testable; the trainer
end-to-end wiring is left FUTURE to avoid touching a parallel session's files.

## Commit
See commit hash in final summary.
