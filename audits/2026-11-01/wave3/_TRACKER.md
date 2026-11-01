# Wave 3 tracker 2026-11-01

Five reports: archives, resources, env switches, fail-open, hang, injection. Dispositions: RESOLVED (code + regression test), FUTURE, DOC, REJECTED.

| ID | Sev | Disposition | Notes |
|---|---|---|---|
| HNG4-1 | M | RESOLVED | `wavedec`/`wavedec_batched` reject an empty signal; `test_wavedec_empty_signal_terminates.py` |
| HNG4-2 | M | RESOLVED | `warm_start_step < 1` rejected; `test_early_stopping_warm_start_step_positive.py` |
| HNG4-3 | L | DOC | save lock has no timeout by design (serialised saves) |
| HNG4-4 | L | DOC | helper-thread join is synchronous by contract |
| HNG4-5 | L | DOC | `close(wait=False)` is the escape hatch for the async render join |
| HNG4-6 | I | REJECTED | recipe-name simplifier advances on every pass |
| HNG4-7 | I | REJECTED | the three loops cannot stall on NaN |
| FOP4-1 | M | RESOLVED | load cache key includes `strict_version`; `test_load_model_cache_strict_version.py` |
| FOP4-2 | L | RESOLVED | the three calibration leakage guards warn when skipped; `test_post_train_guard_skips_are_logged.py` |
| FOP4-4 | L | DOC | VRAM gates are optimisations; correctness rests on per-path OOM fallbacks |
| ENV4-1 | L | RESOLVED | `io.py` opt-out read through `env_flag`; test in `test_load_model_cache_strict_version.py` |
| ENV4-6 | L | RESOLVED | eight switches moved to `env_flag`; `test_env_switch_spellings.py` |
| ENV4-10 | I | DOC | the probe subprocess inherits the parent environment on purpose |
| ENV4-11 | I | REJECTED | pyutilz already logs a WARNING on every bypass call |
| RES4-1 | L | RESOLVED | auto-created CV scratch directory removed on return; `test_pipeline_gridsearch_default_dump_dir_private.py` |
| ARC4-1/5/6 | - | RESOLVED | covered by earlier commits of this wave |
| ARC4-2/4 | - | DOC | |
| ARC4-3 | - | REJECTED | |
| INJ4-1/2 | L | RESOLVED | `csv_safe_frame`; `test_csv_formula_injection.py` |
| INJ4-4 | L | REJECTED | plotly hovertemplate is not script |

Open: none that need code. The five Info rows of the injection report were not returned by the auditor.
