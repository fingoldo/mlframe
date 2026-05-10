# mlframe usage examples

Copy-pasteable recipes per feature. Distinct from the deeper guides in
`docs/*.md` -- those describe design + tradeoffs; these describe "here is
the snippet to put in your script".

| File | Topic |
|---|---|
| [`composite_targets.md`](composite_targets.md) | Auto-discovery composite targets: opt-in tiers, reading metadata, Markdown report, kill-switch env var, decision tree |

For deeper conceptual material see the docs root:

- [`docs/baseline_diagnostics_guide.md`](../baseline_diagnostics_guide.md) -- BaselineDiagnostics ablation + init_score baseline
- [`docs/composite_targets_tutorial.ipynb`](../composite_targets_tutorial.ipynb) -- 16-cell TVT walkthrough notebook
- [`docs/MULTI_OUTPUT.md`](../MULTI_OUTPUT.md) -- multilabel design notes
- [`docs/SELECTION_BIAS.md`](../SELECTION_BIAS.md) -- PU-learning + drift-correction wrappers
- [`docs/NUMERICAL_STABILITY_REPORT.md`](../NUMERICAL_STABILITY_REPORT.md) -- numerical edge cases audit
- [`docs/feature_handling_examples.md`](../feature_handling_examples.md) -- FeatureHandlingConfig recipes
- [`docs/DEBUGGING_UPSTREAM_ERRORS.md`](../DEBUGGING_UPSTREAM_ERRORS.md) -- when upstream sklearn / LightGBM raises something cryptic
