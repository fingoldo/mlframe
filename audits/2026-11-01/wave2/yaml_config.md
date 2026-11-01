# Wave 2 audit: YAML, JSON and config parsing

Date: 2026-11-01. Scope: `src/mlframe` (non-benchmark code first, benchmark trees checked where they reach config). Read-only; no source, test or config file changed; no git state changed.

Checked against the trackers `audits/2026-11-01/_TRACKER.md` and `audits/2026-10-31/_TRACKER.md`. Nothing below re-reports an existing row. INP2-4 (`inference/predict.py:111-112`, caller-sized features JSON) and DYN2-3 (`calibration/post.py:79`, `__import__` on a class path) are already recorded and are not repeated.

## Method

1. `yaml` usage: `grep -rn "yaml"` over `src/mlframe`. One live loader found (vendored InfoNet config). No `yaml.load`, `unsafe_load`, `FullLoader` or `load_all` anywhere in `src`.
2. JSON with hooks: `object_hook` and `object_pairs_hook` grep, zero hits in `src`. All JSON reads are plain `json.load(s)` or `orjson.loads`, returning plain dicts or lists.
3. Environment expansion: `expandvars`, `expanduser` and `os.environ` reads. `expanduser` is used only on fixed cache-path defaults (`training/suite_artefact_cache.py:74`, `utils/_param_oracle.py:132`). No `expandvars` in `src`. Environment variables read are fixed names (`MLFRAME_*`, `CUDA_*`); none is expanded into a config value.
4. Config-named classes, modules or callables: `importlib.import_module`, `__import__`, `getattr` on strings, `import_string`, `pydoc.locate`, `from_config`. Each hit was traced to whether its name came from a fixed table in code or from a file or config value.
5. Command or path interpolation of config values: grep for `shell=True`, `os.system` and f-strings into paths. The subprocess audit (`audits/2026-11-01/subprocess.md`) found none in non-benchmark code, so this was not repeated.

## Findings

No High or Medium findings. The class is in good shape: there is no unsafe YAML loader, no object hook, and no config-driven import.

| ID | Severity | Location | Disposition |
|---|---|---|---|
| YAML3-1 | Low | `training/...` not applicable; see YAML3-2 | (merged into YAML3-2) |
| YAML3-2 | Low | `training/neural/keras_compat.py:146` | FUTURE |
| YAML3-3 | Info | `feature_selection/filters/_vendored/infonet/infer.py:17-19` and `_neural_mi.py:312-318` | RESOLVED |
| YAML3-4 | Info | `feature_selection/filters/_orthogonal_meta_scorer_fe.py:470-485` | RESOLVED |
| YAML3-5 | Info | `data/datasets/__init__.py:62-80` | RESOLVED |
| YAML3-6 | Info | `feature_selection/filters/engineered_recipes/_recipe_dispatch.py:28-40` | RESOLVED |
| YAML3-7 | Info | whole `src/mlframe` tree (no `expandvars`, no config-value env expansion) | RESOLVED |
| YAML3-8 | Info | `feature_selection/filters/_vendored/infonet/configs/config.yaml` (sizes uncapped) | REJECTED |

Renumbered to keep the list dense: YAML3-1 is withdrawn (it duplicated YAML3-2 during drafting), so the live IDs are YAML3-2 to YAML3-8.

### YAML3-2 (Low): Keras `Sequential.from_config` on unpickled state, no `safe_mode`

- Where: `src/mlframe/training/neural/keras_compat.py:139-146`. `__setstate__` reads `_keras_config_` from the pickled state and calls `Sequential.from_config(config)` (tensorflow.keras or keras, whichever imports).
- Who controls it: whoever can write a pickle that the estimator loads. That is already a deserialization trust boundary (see `audits/2026-11-01/deserialization.md`). Once the pickle is trusted or passes a sidecar check, the nested Keras config is a second, JSON-like input.
- Failure scenario: Keras 2 (`tf.keras`) deserializes `Lambda` layers from their config, which carries marshalled bytecode, so a crafted `_keras_config_` runs code inside `from_config`. Keras 3 defaults `safe_mode=True` and refuses `Lambda`, so the risk depends on the installed Keras major version. The repo never sets `safe_mode` explicitly, so the protection is version-dependent.
- Severity Low: it needs a pickle that already bypasses the deserialization gates, and the pickle loader is the primary control.
- Proposed disposition: FUTURE. Pin `keras.saving.enable_unsafe_deserialization` off and set `safe_mode=True` explicitly in `__setstate__` (or refuse configs containing `Lambda`) so the guard does not depend on the Keras version. Needs a test on a crafted config.

### YAML3-3 (Info): InfoNet vendored YAML loaded with `yaml.safe_load`

- Where: `src/mlframe/feature_selection/filters/_vendored/infonet/infer.py:17-19` (`yaml.safe_load(file)`); caller `_neural_mi.py:310-318` resolves a fixed package path.
- Result: safe loader, no tags constructed. The path is built from `Path(__file__)` and is not user-supplied. The `infer.py` `__main__` default `configs/config.yaml` is a CLI default for a developer script, not a library path.
- Proposed disposition: RESOLVED (citation above).

### YAML3-4 (Info): meta-scorer dispatch is a fixed table

- Where: `_orthogonal_meta_scorer_fe.py:470-485`. The `scorer` string is lowercased and looked up in `_SCORER_DISPATCH`. `importlib.import_module` receives only the table's `module_name`, prefixed with the package name. An unknown name raises `ValueError`.
- A caller cannot supply a module path. Proposed disposition: RESOLVED.

### YAML3-5 (Info): lazy export table resolves only a fixed map

- Where: `data/datasets/__init__.py:62-80`. `__import__` is called with `f"{__name__}.{home}"`, where `home` comes from the `_LAZY_HOME` dict literal and `name` must be a key of it, otherwise `AttributeError`. Proposed disposition: RESOLVED.

### YAML3-6 (Info): recipe dispatch imports module and function names from code

- Where: `_recipe_dispatch.py:28-40`. `_routed(module, func, ...)` takes its arguments from literal calls in the file; `recipe.kind` selects a handler from `_KIND_HANDLERS` and does not form a module name. A recipe read from a saved file can choose a kind but cannot choose an import target. Proposed disposition: RESOLVED.

### YAML3-7 (Info): no expansion of config values

- Result: no `os.path.expandvars`, no `string.Template` substitution and no `${VAR}` handling in `src/mlframe`. Environment reads are `os.environ.get("MLFRAME_...")` with fixed names, and the value is used as a literal (a path, a flag or a size), never re-expanded. Proposed disposition: RESOLVED.

### YAML3-8 (Info): size and depth of the InfoNet config are uncapped

- Where: `infonet/configs/config.yaml` (`latent_num: 256`, `seq_len: 5000`) feeds `create_model`, which allocates by these values.
- Why rejected: the file ships inside the package and is not user-supplied. A caller who edits it already controls the process. `load_config` takes an explicit path, and the checkpoint is separately sha256-gated (NET2-03, now FUTURE/implemented). Proposed disposition: REJECTED (operator-trusted local file, not untrusted input).

## Not findings (checked, clean)

- JSON parsers: all `json.load(s)` and `orjson.loads` call sites return plain containers. The scenario lock (`data/datasets/scenarios/__init__.py:533`) and the fs_hybrid manifests are read with fixed paths.
- `orjson` nesting: orjson rejects deeply nested input above its recursion limit, so the depth of JSON sidecars is bounded by the library. Sizes are not capped (see INP2-4, already recorded).
- Benchmark trees (`_benchmarks/`): `json.load` on their own output files. They are developer tools, consistent with the earlier REJECTED ruling for benchmark code.

## Limits

- Keras behaviour (YAML3-2) was reasoned from the Keras version split, not run. A crafted-config test is needed before the FUTURE item is closed.
- Config files named by users at runtime (the training suite's config dataclasses) were checked for class-name resolution only by grep. The suite takes Python objects, not YAML, so no YAML-to-class path exists in `src`.
