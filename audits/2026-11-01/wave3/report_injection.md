# Wave 3 audit 2026-11-01: injection surfaces

Scope: `src/mlframe` (read-only). Class: data that reaches a spreadsheet, an HTML page, a plot template, a shell or a query as code.

The auditor returned only its top findings; the five remaining Info rows were not included in the text that came back, so they are not listed here.

## Summary

8 findings: 0 High, 0 Medium, 3 Low, 5 Info (3 Low listed below, 5 Info not returned).

| ID | Severity | Location | Disposition |
|---|---|---|---|
| INJ4-1 | Low | `feature_selection/boruta_shap/_io_plot.py:46` (CSV of feature names) | RESOLVED |
| INJ4-2 | Low | `training/core/_main_train_suite_phases.py:486`, `training/core/_phase_targets_performance.py:47` (CSV of targets and models) | RESOLVED |
| INJ4-4 | Low | plotly `hovertemplate` built from names | REJECTED |

## Findings

### INJ4-1 / INJ4-2 Names written to CSV start a spreadsheet formula (Low, RESOLVED)
Feature, target and model names were written as CSV cells and headers unchanged. A name starting with `=`, `+`, `-` or `@` is evaluated by Excel and
LibreOffice. Fixed with `mlframe.utils.csv_safe.csv_safe_frame`, which prefixes such strings with an apostrophe and leaves numbers alone, wired into
the three export sites. Test: `tests/utils/test_csv_formula_injection.py`.

### INJ4-4 Plotly hovertemplate built from names (Low, REJECTED)
Plotly renders `hovertemplate` as a restricted template (`%{...}` placeholders and a small HTML subset), not as script, so a hostile name cannot run
code; the worst case is a garbled tooltip.

## Checked and already safe
- The HTML reports and the composite report escape their inputs.
