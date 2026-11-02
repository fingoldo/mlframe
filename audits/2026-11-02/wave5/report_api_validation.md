# Wave 5 audit 2026-11-02: public API input validation and type confusion

Scope: public entry points, read-only, Opus 5.5 low effort (targeted). Condensed from the agent's returned text.

| ID | Severity | Location | Disposition |
|---|---|---|---|
| API5-1 | Medium | `training/_data_helpers.py:422-436`: model name used raw in the `.dump` path | RESOLVED |
| API5-2 | Medium | `training/train_eval.py:457-460`: pre-pipeline name and model name used raw in the cached model path | RESOLVED |
| API5-3 | Low | `training/honest_diagnostics.py:453`: raw target name in the calibration plot file name | RESOLVED |
| API5-4 | Medium | `training/honest_diagnostics.py:443-447`: OOF probabilities and target cut to the shorter length and paired by position | RESOLVED |

API5-1 and API5-2: a separator, a drive colon or a reserved device name sent the file into a missing subfolder, outside the models directory (an absolute name makes `os.path.join` drop it) or onto an NTFS stream. `utils/safe_filename.safe_file_stem` replaces those characters and leaves ordinary names, spaces and underscores unchanged, so existing dumps keep matching. Test: `tests/utils/test_safe_filename.py`.

API5-3 and API5-4: the plot name goes through `safe_file_stem`; a row-count mismatch now returns `status: skipped` with both counts and a warning. Test: `tests/training/test_honest_calibration_alignment.py`.

Checked and safe: `DataFrame.eval` in `_sapp.py` takes a fixed string; config dicts go into strict Pydantic configs (unknown keys raise); YAML is read only in vendored code with `safe_load`; calibrator, RFECV, ranker and slice-pareto output names are slugified; `inference/predict.py` model loading lists a caller directory with an extension allowlist and a sha256 check; feature-column names are sanitised; the decision threshold is range-checked.

Not covered: strings iterated per character where a list is expected, zero or negative counts, `rfecv/_validate.py`, selector `fit` methods and `reporting/` outputs line by line.
