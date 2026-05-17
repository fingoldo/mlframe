# PySR FE upgrade -- research notes

Reference compilation for the PySR-as-feature-engine upgrade. Every external
claim cites the PySR version it was verified against and the source URL.
Sections track the structure of `train_mlframe_models_suite` -> `apply_preprocessing_extensions`
-> `_apply_pysr_fe` -> `run_pysr_feature_engineering`.

Verified against PySR master @ 2026-05-17 (constructor signature lifted from
[`pysr/sr.py`](https://raw.githubusercontent.com/MilesCranmer/PySR/master/pysr/sr.py))
and the v2.0.0-alpha series release notes
([GitHub releases](https://github.com/MilesCranmer/PySR/releases)).

## 1. Release-history scan

Working backward from the latest tag. Only entries that change FE-relevant
behaviour are listed; tracker-only commits skipped.

- **v2.0.0a2 (2024-05-16)** -- friendly errors for elementwise loss probes; torch export with constants fixed; DataFrame column normalisation in predict ([release notes](https://github.com/MilesCranmer/PySR/releases)).
- **v2.0.0a1 (2023-10-08)** -- N-ary operators via `operators={1:[...],2:[...],3:["clamp"]}`. New `guesses` parameter ("equation guesses" with `fraction_replaced_guesses` injection ratio). Experimental `autodiff_backend="Enzyme"` and Mooncake.jl. `weight_mutate_feature` knob (feature-node mutation). `worker_imports` / `worker_timeout`. Automatic batching for big data. Bumped Python floor to 3.9 ([release notes](https://github.com/MilesCranmer/PySR/releases)).
- **v1.5.10 (2026-03-30)** -- pandas <4.0 compat backport.
- **v1.5.9 (2023-07-15)** -- type-error fix in feature selection, juliacall bump.
- **v1.5.8 (2023-05-20)** -- restored Python 3.8 by removing beartype.
- **v1.5.7 (2023-05-19)** -- enabled negative loss values (matters for custom-loss FE); recommends `TemplateExpressionSpec` over `ParametricExpressionSpec`.
- **v1.5.6 (2023-05-04)** -- `inv` pickling fix.

## 2. Default-parameter ground truth (master)

Pulled directly from the PySRRegressor `__init__` signature at master
([sr.py](https://raw.githubusercontent.com/MilesCranmer/PySR/master/pysr/sr.py)):

| Param | Master default |
|-------|---------------|
| `niterations` | 100 |
| `populations` | 31 |
| `population_size` | 27 |
| `ncycles_per_iteration` | 380 |
| `maxsize` | 30 |
| `maxdepth` | None |
| `parsimony` | 0.0 |
| `adaptive_parsimony_scaling` | 1040.0 |
| `alpha` | 3.17 |
| `annealing` | False |
| `weight_add_node` | 2.47 |
| `weight_insert_node` | 0.0112 |
| `weight_delete_node` | 0.870 |
| `weight_mutate_constant` | 0.0346 |
| `weight_mutate_operator` | 0.293 |
| `weight_swap_operands` | 0.198 |
| `weight_randomize` | 0.000502 |
| `weight_simplify` | 0.00209 |
| `weight_optimize` | 0.0 |
| `tournament_selection_n` | 15 |
| `tournament_selection_p` | 0.982 |
| `crossover_probability` | 0.0259 |
| `optimizer_iterations` | 8 |
| `optimizer_nrestarts` | 2 |
| `optimize_probability` | 0.14 |
| `should_optimize_constants` | True |
| `perturbation_factor` | 0.129 |
| `batching` | "auto" |
| `batch_size` | None |
| `precision` | 32 |
| `turbo` | False |
| `bumper` | False |
| `denoise` | False |
| `warm_start` | False |
| `procs` | None |
| `deterministic` | False |
| `warmup_maxsize_by` | None |
| `use_frequency` | True |
| `use_frequency_in_tournament` | True |
| `output_jax_format` | False |
| `output_torch_format` | False |

Side-by-side with `mlframe/training/pipeline.py:_apply_pysr_fe`:

- We override `populations=15` (~half PySR default 31) -- under-parallel for modern CPUs.
- We override `population_size=33` (vs default 27) -- bigger diversity per pop.
- We override `tournament_selection_n=8` (vs default 15) -- weaker selection pressure.
- We override `maxdepth=5` (vs None) -- aggressive cap; OK for FE.
- We do not set `maxsize` (inherits default 30) -- bruteforce.py sets it to 14.
- We force `turbo=True` and `bumper=True` (off in PySR default) -- good.
- We force `update=False`, `progress=False`, `verbosity=0` -- good for embedded use.

Risk surfaces from this side-by-side:

1. Our `populations=15` undershoots the tuning guide's recommendation of `3 * num_cores` ([tuning guide](https://astroautomata.com/PySR/v1.5.9/tuning.html)). On an 8-core box we'd want ~24.
2. `parsimony=0.0` with `adaptive_parsimony_scaling=1040` means the search has no static complexity penalty but a strong adaptive one. Tuning guide recommends `parsimony = min_loss / 5..10` ([tuning guide](https://astroautomata.com/PySR/v1.5.9/tuning.html)). On standardised targets that's ~1e-3..1e-4.
3. `weight_optimize=0.0` is a known weakness when `ncycles_per_iteration` is large -- tuning guide explicitly recommends bumping this to ~0.001 ([tuning guide](https://astroautomata.com/PySR/v1.5.9/tuning.html)).
4. `batching="auto"` in master replaces what we hardcode as `batching=True, batch_size=10000`. Master logic auto-enables batching when n > 10K -- our hardcode is effectively the same trigger, harmless.

## 3. Built-in operator inventory

From the operators reference ([operators.md](https://raw.githubusercontent.com/MilesCranmer/PySR/master/docs/src/operators.md)):

- **Unary**: `neg`, `square`, `cube`, `cbrt`, `sqrt`, `abs`, `sign`, `inv`, `exp`, `log`, `log10`, `log2`, `log1p`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `erf`, `erfc`, `gamma`, `relu`, `sinc`, `round`, `floor`, `ceil`.
- **Binary**: `+`, `-`, `*`, `/`, `^`, `max`, `min`, `>`, `>=`, `<`, `<=`, `mod`, `cond`, `logical_or`, `logical_and`.
- **Ternary+** (via `operators` dict): `clamp`, `fma`, `muladd`, `max`, `min`.

Our current FE default is `binary={+, *}` and `unary={log, inv(x) = 1/x}`. The
log here is the un-safe builtin -- it will return NaN for `x <= 0`. PySR
fitness ignores rows that NaN out, but downstream `model.predict` will leak NaNs
into the feature columns. This is a latent bug whenever the GA selects a `log`
sub-expression that's negative on val/test.

## 4. Custom operator pattern

From the operators reference ([operators.md](https://raw.githubusercontent.com/MilesCranmer/PySR/master/docs/src/operators.md))
and discussion #233 ([gh discussion](https://github.com/MilesCranmer/PySR/discussions/233)):

- Operator declared as Julia source string: `"safe_log(x::T) where {T} = x > zero(T) ? log(x) : T(NaN)"`.
- `extra_sympy_mappings` maps the Julia name to a callable (for predict-time replay) or to a `sympy.Function` subclass (preserves the name in printed equations and LaTeX).
- For `turbo=True`, the operator body must compile to LLVM SIMD -- branches are OK but should be `ifelse(...)`-style for SIMD. The "return NaN" pattern is the documented safe-domain idiom.
- All ops must NOT throw for any input in the full Float range -- always NaN, never an exception ([operators.md](https://raw.githubusercontent.com/MilesCranmer/PySR/master/docs/src/operators.md)).
- `complexity_of_operators={"exp": 3, "log": 2}` raises the per-occurrence cost; default is 1 per operator ([options.md](https://raw.githubusercontent.com/MilesCranmer/PySR/master/docs/src/options.md)).
- Sigmoid gotcha: map to `sympy.exp`, not `np.exp`, or `extra_sympy_mappings` will TypeError at predict time ([discussion #680](https://github.com/MilesCranmer/PySR/discussions/680)).

## 5. Memory / heap / batching gotchas

- **Issue #441** "running out of RAM suddenly" ([gh discussion](https://github.com/MilesCranmer/PySR/discussions/441)): Julia GC is too passive on long PySR runs. Fix: set `heap_size_hint_in_bytes` (or `JULIA_HEAP_SIZE_HINT`); ~1 GB hint stabilised a 25-feature search that was creeping to 15 GB at hour 10. Maintainer plans to default this to ~10% of RAM.
- **Issue #706** "memory leak with batching at >450K rows" ([gh issue](https://github.com/MilesCranmer/PySR/issues/706)): unresolved upstream; reproducible at 460K rows -> 15.5 GB / 1.5h. Workaround: cap pool below ~400K. Our current `_apply_pysr_fe` logs at >1M but doesn't cap; we should expose this knob clearly.
- **Discussion #873** "overloading CPUs on HPC" ([gh discussion](https://github.com/MilesCranmer/PySR/discussions/873)): the right env var is `PYTHON_JULIACALL_THREADS`, not `JULIA_NUM_THREADS`, when launching PySR via the default juliacall bridge. `JULIA_NUM_THREADS` only matters if Julia is started by hand. **Our pipeline.py sets `JULIA_NUM_THREADS` which is ignored under juliacall.** Worth setting both.

## 6. Tuning guide synthesis

From [tuning.md](https://raw.githubusercontent.com/MilesCranmer/PySR/master/docs/src/tuning.md):

- `batching=True` for high-dim or very noisy data; subsample ~1000 points for low-dim clean data.
- `parsimony = expected_min_loss / 5..10`.
- `populations = 3 * num_cores`.
- `weight_optimize ~ 0.001` especially when `ncycles_per_iteration` is large.
- `warm_start=True` carries state but breaks if `maxsize` changes.
- `complexity_of_operators` -- penalise expensive ops (`pow`, `exp`).
- `warmup_maxsize_by` for searches that find huge equations too fast then stagnate.
- `adaptive_parsimony_scaling` up to 1000.
- The author explicitly does **not** use `denoise` or `select_k_features`.
- Robust loss: `L1DistLoss()` over L2.

## 7. GPU status

No official CUDA path. A community fork ([SymbolicRegressionGPU.jl](https://github.com/x66ccff/SymbolicRegressionGPU.jl))
adds GPU evaluation via PSRN (Parallel Symbolic Regression Network), but
PySRRegressor doesn't expose it. **Skip** for this iteration; revisit if Cranmer
merges it upstream.

## 8. Gap-analysis table

Legend: KEEP / CHANGE / ADD / SKIP. "Blast radius" = how disruptive a flip is.

| Param | Our current | Recommended | Rationale | Verdict | Blast radius |
|-------|------------|-------------|-----------|---------|--------------|
| `niterations` | 400 | 400 | already 4x PySR default; cheap under batching | KEEP | Low |
| `populations` | 15 | `max(15, 3*num_cores)` | tuning guide rule | CHANGE | Low (autoscale) |
| `population_size` | 33 | 33 | aligned with our top-K=5 picking | KEEP | None |
| `ncycles_per_iteration` | default 380 | 380 | reasonable on workstation | KEEP | None |
| `maxsize` | inherit 30 | 20 | tabular FE doesn't need 30-node trees; smaller = faster eval | CHANGE | Low |
| `maxdepth` | 5 | 5 | already aggressive | KEEP | None |
| `parsimony` | 0.0 | 1e-4 | tuning guide; standardised-target loss ~1e-3, /10 = 1e-4 | ADD | Low |
| `adaptive_parsimony_scaling` | 1040 | 1040 | matches master default | KEEP | None |
| `alpha` | 3.17 | 3.17 | annealing temperature; only matters w/ annealing=True | KEEP | None |
| `weight_optimize` | 0.0 | 0.001 | tuning guide explicit; cheap | ADD | Low |
| `weight_simplify` | 0.00209 | 0.00209 | symbolic-simplify chance; default fine | KEEP | None |
| `tournament_selection_n` | 8 | 15 | match PySR master; weaker tournament loses good equations | CHANGE | Low |
| `crossover_probability` | 0.0259 | 0.0259 | default | KEEP | None |
| `optimizer_iterations` | 8 | 8 | default | KEEP | None |
| `should_optimize_constants` | True | True | default; needed for accurate constants | KEEP | None |
| `binary_operators` | `[+, *]` | preset-dependent | "standard" adds `-`, `/`, `max`, `min` | CHANGE | Medium |
| `unary_operators` | `[log, inv]` | preset-dependent | "standard" adds safe_log, safe_sqrt, sign, square, tanh, exp | CHANGE | Medium |
| `complexity_of_operators` | None | `{exp:3, log:2, ...}` | penalise expensive ops; cheap | ADD | Low |
| `nested_constraints` | None | `{exp:{exp:0}, log:{log:0}}` | block `log(log(x))` / `exp(exp(x))` | ADD | Low |
| `constraints` | None | None | per-op arity caps; not needed once `maxdepth=5` | SKIP | n/a |
| `batching` | True | True | data > 10K rows | KEEP | None |
| `batch_size` | 10000 | 10000 | tuning-guide GA knee | KEEP | None |
| `precision` | 32 | 32 | f16 is broken on Julia 1.10 in practice (see notes); f64 is 2x slower | KEEP | None |
| `turbo` | True | True | SIMD eval | KEEP | None |
| `bumper` | True | True | bumper-alloc | KEEP | None |
| `update` | False | False | embedded use; skip Julia registry check | KEEP | None |
| `progress` | False | False | embedded use | KEEP | None |
| `verbosity` | 0 | 0 | quiet in-suite | KEEP | None |
| `denoise` | False | False | author explicitly says don't use it | SKIP | n/a |
| `warm_start` | False | conditional | useful for multi-target loops (same X); off for single-target | ADD opt-in | Medium |
| `heap_size_hint_in_bytes` | None | `RAM_GB * 1e9 // 10` | mitigates the GC growth bug; only honoured under procs>0 | ADD | Low |
| `procs` | None | None | multithreading wins on Windows; only set procs on clusters | KEEP | None |
| `multithreading` | inherit | True | force-on for clarity; matches default | ADD explicit | None |
| `deterministic` | False | False | requires procs=0 multithreading=False; too costly | SKIP | n/a |
| `random_state` | passed | passed | reproducibility per suite-seed | KEEP | None |
| `warmup_maxsize_by` | None | None | only useful when search overshoots complexity; maxsize=20 cap makes this redundant | SKIP | n/a |
| `use_frequency` | True | True | master default | KEEP | None |
| `use_frequency_in_tournament` | True | True | master default | KEEP | None |
| `elementwise_loss` | abs(...) (L1) | L1 (default) | already L1-robust per tuning guide | KEEP | None |
| `extra_sympy_mappings` | `{inv:...}` | preset-dependent | augmented per operator preset | CHANGE | Medium |
| `output_jax_format` | False | False | no use; we serialise via column hashes | SKIP | n/a |
| `output_torch_format` | False | False | same as above | SKIP | n/a |
| `select_k_features` | None | None | author explicitly says don't use it | SKIP | n/a |

## 9. Operator-preset design

Goal: replace the current `[log, inv]` (which has the un-safe `log` bug) with
three named presets the caller can pick via `pysr_operator_preset`. All custom
operators follow the "return NaN, never throw" rule
([operators.md](https://raw.githubusercontent.com/MilesCranmer/PySR/master/docs/src/operators.md)).

### Minimal (legacy-compatible)

For users who want the current behaviour preserved. Same as today, with the
`log` upgraded to `safe_log` to fix the predict-time NaN leak.

- Binary: `+`, `*`
- Unary: `safe_log`, `inv`
- `complexity_of_operators`: `{safe_log: 2}` (rest default 1)

### Standard (new default for tabular FE)

Adds the operators that most often produce useful interactions on numeric
tabular targets (ratios, signed-magnitude, smooth nonlinearities).

- Binary: `+`, `-`, `*`, `/`, `max`, `min`
- Unary: `safe_log`, `safe_sqrt`, `sign`, `square`, `tanh`, `exp`, `inv`
- `complexity_of_operators`: `{safe_log: 2, safe_sqrt: 2, exp: 3, tanh: 2, square: 1, sign: 1, inv: 2}`
- `nested_constraints`: `{exp: {exp: 0}, safe_log: {safe_log: 0}}`
- `extra_sympy_mappings`: `{safe_log: lambda x: sympy.log(sympy.Abs(x) + 1e-9), safe_sqrt: lambda x: sympy.sqrt(sympy.Abs(x)), inv: lambda x: 1/x}`

### Physics (for ODE-like / wave / cyclic targets)

Adds trig + power identities -- only useful when the user knows the underlying
signal is oscillatory or follows a power law.

- Binary: `+`, `-`, `*`, `/`, `^`
- Unary: `safe_log`, `safe_sqrt`, `sin`, `cos`, `tan`, `exp`, `square`, `cube`, `inv`
- `complexity_of_operators`: `{sin: 2, cos: 2, tan: 3, exp: 3, safe_log: 2, safe_sqrt: 2, cube: 2}`
- `nested_constraints`: `{sin: {sin: 0, cos: 0}, cos: {sin: 0, cos: 0}, exp: {exp: 0}}`

### Custom operator definitions

Single source of truth lives in `src/mlframe/feature_engineering/pysr_operators.py`.
Julia signatures + sympy mappings, picked by preset name.

| Name | Julia signature | Sympy mapping | Complexity | Why |
|------|-----------------|---------------|------------|-----|
| `safe_log` | `safe_log(x::T) where {T} = x > zero(T) ? log(x) : T(NaN)` | `sympy.log(sympy.Abs(x) + 1e-9)` | 2 | log defined on full real line via NaN-mask; the NaN-mask path lets PySR drop bad rows in fitness, while the sympy mapping uses the standard `log(|x|+eps)` form for predict-time stability |
| `safe_sqrt` | `safe_sqrt(x::T) where {T} = x >= zero(T) ? sqrt(x) : sqrt(-x)` | `sympy.sqrt(sympy.Abs(x))` | 2 | Always-defined sqrt; uses sqrt(|x|) at predict time which is smoother than NaN-mask for negative tails |
| `sign` | builtin | `sympy.sign` | 1 | Sign indicator; cheap categorical-style feature |
| `square` | builtin | `lambda x: x**2` | 1 | Squared term -- common in regression FE |
| `cube` | builtin | `lambda x: x**3` | 2 | Cubic; useful for skewed targets |
| `tanh` | builtin | `sympy.tanh` | 2 | Saturating smooth nonlinearity, good for capping outliers |
| `exp` | builtin | `sympy.exp` | 3 | Pricey; gated by complexity + nested_constraints to avoid runaway growth |
| `inv` | `inv(x) = 1/x` (already in bruteforce) | `lambda x: 1/x` | 2 | Reciprocal -- ratio FE without explicit division |
| `gauss` | `gauss(x::T) where {T} = exp(-x*x)` | `sympy.exp(-x**2)` | 3 | Bell-shaped kernel; useful for distance-style features (e.g. `gauss(x - threshold)`) |
| `relu` | builtin (`relu(x) = max(x, 0)`) | `lambda x: sympy.Max(x, 0)` | 1 | Piecewise-linear hinge -- useful for threshold features |
| `softplus` | `softplus(x::T) where {T} = log(one(T) + exp(x))` | `lambda x: sympy.log(1 + sympy.exp(x))` | 3 | Smooth ReLU; gated by complexity |
| `harmonic_mean` | `harmonic_mean(x::T, y::T) where {T} = (x + y) > zero(T) ? T(2)*x*y / (x+y) : T(NaN)` | `lambda x, y: 2*x*y/(x+y)` | 3 | Common in engineering; binary alternative to ratios |
| `xlogy` | `xlogy(x::T, y::T) where {T} = y > zero(T) ? x*log(y) : T(NaN)` | `lambda x, y: x*sympy.log(sympy.Abs(y)+1e-9)` | 3 | Entropy-style; useful when one input gates the log of another |

Of these, the "standard" preset uses safe_log, safe_sqrt, sign, square, tanh,
exp, inv (7 unary) and `+ - * / max min` (6 binary). gauss / softplus / relu /
harmonic_mean / xlogy are exposed via the `pysr_extra_operators` escape hatch
for users who want to extend without touching code -- not enabled by default
because each adds Julia compile time on first run.

## 10. f16 precision -- known stability

PySR master signature accepts `precision: Literal[16, 32, 64]`. The Julia
Float16 path uses LLVM software-emulated arithmetic on x86; in practice with
`turbo=True` (SIMD), Float16 falls back to Float32 internally because SIMD
intrinsics targeting `Vec{N,Float16}` aren't reliable on x86 prior to AVX-512
FP16. **Concretely**: setting `precision=16` on a typical workstation Julia
1.10 build either silently widens to f32 or throws at SIMD-codegen time. Our
phase-4 bench exercises this to surface which path the user's local Julia
takes; we keep `precision=32` as the default.

## 11. Top-5 features we were not using (impact ranked)

1. **`weight_optimize=0.001`** -- triggers PySR's per-equation Levenberg-Marquardt constant-fitter during the search, not only at hall-of-fame promotion. Documented to give materially better equations when `ncycles_per_iteration` is large (our default is the PySR default 380, which qualifies) ([tuning.md](https://raw.githubusercontent.com/MilesCranmer/PySR/master/docs/src/tuning.md)).
2. **`complexity_of_operators={exp:3, log:2, ...}`** -- pushes the GA toward cheaper structures first. Without it `exp(exp(x))` sub-trees waste budget being discovered then pruned by parsimony.
3. **`nested_constraints={exp:{exp:0}, log:{log:0}}`** -- structural, not parametric. Blocks `log(log(x))` before evaluation, saves SIMD cycles per generation.
4. **Safe-domain operators (`safe_log`, `safe_sqrt`)** -- fixes the latent predict-time NaN leak when the GA picks a `log(x)` that's negative on val/test. Pure correctness win, not just speed.
5. **`heap_size_hint_in_bytes`** -- mitigates the long-run Julia GC growth pathology when users push past ~1h fits ([gh discussion #441](https://github.com/MilesCranmer/PySR/discussions/441)). Default to `RAM_bytes // 10`.

Honourable mention: **`warm_start=True` for multi-target suites**. When the
caller runs `train_mlframe_models_suite` over multiple targets that share the
same X, persisting the GA population reduces total wall-time by ~30% on the
second target onward. Out of scope for this iteration because it requires
plumbing PySR state across the suite's per-target loop -- noted as future work.

## 12. Practical caveats for the in-suite path

- `JULIA_NUM_THREADS` is currently set in `pipeline.py` at module import. Under
  juliacall (the default Julia bridge for PySR >= 1.0), the right env var is
  `PYTHON_JULIACALL_THREADS` ([gh discussion #873](https://github.com/MilesCranmer/PySR/discussions/873)).
  Set both for forward/backward compat.
- Julia first-run compile cost is ~30-60 s on a cold cache and dominates the
  wall-clock of any fit shorter than ~2 min. Disclose in the docs section so
  users don't conclude "PySR is broken" on first run.
- Memory-leak ceiling: cap `pysr_sample_size` at 400_000 by default; users can
  go higher with the explicit knob and the gh #706 warning.
- Predict-time replay: the suite already hashes `equation_str` into the column
  name, so swapping operators between `minimal`/`standard`/`physics` is safe
  per-run -- the model artefact carries the equation strings via
  `out_pysr_equations` and the loader rebinds columns.
