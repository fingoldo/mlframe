# 01 SECURITY audit (src/mlframe; MRMR, *_benchmarks and profiling excluded)

Method: ripgrep for pickle/joblib/dill/cloudpickle/torch.load/np.load(allow_pickle)/yaml.load/eval/exec/subprocess/os.system/shell=True/
extractall/urlopen/requests/verify=False/trust_remote_code over src/mlframe, then read each production hit. No source or tests were modified.
No unsafe yaml.load, shell=True, os.system, tarfile.extractall, verify=False or tempfile.mktemp hit was found outside _benchmarks trees.

| ID | severity | file:line | finding | evidence | disposition |
|----|----------|-----------|---------|----------|-------------|
| S01 | high | inference/predict.py:198-210 | Path traversal via `featureset` into joblib.load of model dir | `_validate_trusted_path(fpath, trusted_root or inference_folder)` always runs; shared helper core/helpers.py:22-33 raises if no root | RESOLVED |
| S02 | high | training/io.py:248-274 | joblib.load of pickles could run RCE gadgets | `safe_joblib_load` uses `_SafeJoblibUnpickler` with a denylist, bytes verified by sha256 sidecar | RESOLVED |
| S03 | med | training/io.py:254-256 | Denylist (not allowlist) means a bundle that passes its sidecar is code-exec-equivalent for whoever can write bundle+sidecar | Docstring "Trust model (DESER-03/04, DOC)" states it; custom classes must stay loadable | DOC (intended trust model; allowlist would break custom model classes) |
| S04 | med | training/io.py:674-682 | `safe=False` runs vanilla dill.load | Opt-in only, `warnings.warn("Loading without allowlist -- trust source")`, input bounded by `BoundedReader` | DOC (explicit caller opt-in, warned) |
| S05 | med | utils/disk_cache.py:375,437 | Cache pickles under a shared dir could be planted | Load via `safe_load` that fails closed without a .sha256 sidecar; mismatch deletes entry (l.376-394) | RESOLVED |
| S06 | low | utils/disk_cache.py:373-374 | Env `MLFRAME_ALLOW_UNVERIFIED_PICKLE` re-enables unverified pickle | Comment: only opt-in to legacy path, default closed | DOC (operator-controlled env; same trust level as process owner) |
| S07 | med | training/feature_handling/cache.py:557-594 | np.load allow_pickle fallback on cache files | Default `allow_pickle=False`; True path reads bytes verified against sha256 sidecar first (l.566-577) | RESOLVED |
| S08 | med | training/_unpickle_policy.py:142-145 | torch storage unpickle runs nested unrestricted torch.load | Replaced by `torch.load(..., weights_only=True)` | RESOLVED |
| S09 | low | training/neural/_recurrent_wrapper_base.py:632; _flat_torch_module/__init__.py:600 | torch.load of checkpoints | Both pass `weights_only=True` | RESOLVED |
| S10 | med | training/feature_handling/hf_provider.py:227-280 | HF `trust_remote_code` would execute repo code; value may come from `hf://` URL query | Hardcoded False default, `parse_trust_remote_code` strict bool parse (l.132-144), explicit opt-in warns | RESOLVED |
| S11 | low | training/composite/discovery/_tiny_rerank_process.py:92-93 | `cloudpickle.loads` of task blobs | Blobs produced in-process at l.53 and passed to own worker pool; no file/network source | REJECTED (not attacker-reachable: same-parent-process IPC only; attacker able to inject there already has code exec) |
| S12 | low | data/datasets/_sapp.py:68 | `df.eval(...)` | Expression is a string literal; only `@` locals are numpy arrays/functions, no user text | REJECTED (no external input reaches eval) |
| S13 | low | system/kernel_tuning_cache/_ensure.py:206; utils/native_import_probe.py:55; training/_polars_native_support.py:90; training/_gpu_state_probe.py:98,109; training/pipeline/_pipeline_extensions_pysr.py | subprocess use | All list-argv, no shell=True, fixed interpreter/module/registered kernel name, timeouts set, `# nosec` justified | REJECTED (no injection path) |
| S14 | low | estimators/pipelines.py:159 | Temp dump read back with joblib.load in shared temp root | Comment: name is not pre-creatable by local user (unique temp name) | RESOLVED |
| S15 | low | training/_trainer_train_and_evaluate_parts.py:126 | joblib.load of caller-influenced model path | Defaults trusted_root to model parent dir and validates via shared helper | DOC (default root is the file's own dir, so containment is weak by design; caller should pass an explicit root for untrusted paths) |

Counts: RESOLVED 7, DOC 4, REJECTED 3, FUTURE 0, TODO 0 (S03/S04/S06/S15 are DOC; no regression tests proposed since no TODO rows).
