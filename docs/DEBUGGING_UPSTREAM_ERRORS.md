# Debugging opaque upstream errors: the bench-first doctrine

A field guide written after the 2026-04-19 CatBoost/Polars fastpath
investigation, where **three consecutive plausible-sounding
diagnoses were wrong** before the fourth — built from a 30-line
isolated bench — landed the actual root cause.

This document is prescriptive. When an upstream library (CatBoost,
XGBoost, LightGBM, sklearn, PyTorch, transformers, polars, pyarrow,
numpy — anything whose internals you don't own) throws an opaque
error inside a deeply-stacked call chain, **do not reason about the
fix from the traceback alone**. Build a minimal repro first.

## The pattern that broke round 11 into four iterations

The production error was:

```text
File _catboost.pyx:3290, in _catboost._set_features_order_data_polars_categorical_column.process()
File _catboost.pyx:2998, in _catboost.__pyx_fused_cpdef()
TypeError: No matching signature found
```

Four hypotheses, in order:

| # | Hypothesis | Plausibility | Reality |
|---|---|---|---|
| 11.0 (rounds 7, 10) | `pl.Enum` / stale `cat_features_polars` | prod schema dumps showed plausible culprits | Both disproved — the fastpath still failed after those fixes shipped |
| 11a | Polars 1.x emits `pa.large_string()`, CB dispatch table was compiled against `pa.string()` | Version numbers matched the timeline, the error shape matched a fused-cpdef miss | **Disproved** in 25 lines: a null-free Categorical built from an all-`large_string` Arrow table **fit CB cleanly** |
| 11b | Null in Categorical → bypass to pandas | Correct diagnosis, first finally-working fix | Shipped, but the bypass gave up the 15-minute Polars-native fastpath entirely — correct but over-priced |
| 11c | Null in Categorical → fill with sentinel, keep fastpath | The same diagnosis, but now we fill in place with `"__MISSING__"` | **Correct, cheapest, shipped as `a5f4fae`** |

The second and third diagnoses were only possible because 11a was
disproved by a standalone bench. Without the bench, we would have
shipped 11a as "the fix" and moved on — leaving the real null
trigger undetected and a latent failure mode in production.

## The doctrine

### 1. If you can't reproduce in ≤40 lines of standalone code, you don't understand the bug

The traceback tells you **where** it died. It rarely tells you **why**.
"Why" comes from controlling every variable between your code and
the crash site.

A good isolation bench:

- Imports only the libraries on the stack (no mlframe, no training
  scaffolding, no configs, no fixtures).
- Constructs inputs directly (`pl.DataFrame({...})`, `np.array(...)`,
  `pd.DataFrame({...})`). No file reads. No "load the prod sample."
- Runs the upstream call that crashed, verbatim.
- Prints an `OK` / `FAIL <error repr>` line.

Examples in the repo root:

- [`bench_polars_largestring_cb_xgb.py`](../bench_polars_largestring_cb_xgb.py) — tests the large_string hypothesis against CB and XGB.
- [`bench_polars_cb_repro.py`](../bench_polars_cb_repro.py) — feature-by-feature sweep (Int16, Boolean, multi-cat, text_features, null-cat) to find which axis triggers.
- [`bench_polars_cb_nullfrac.py`](../bench_polars_cb_nullfrac.py) — null-fraction binary search proving any null triggers.

### 2. Test one variable at a time

When there are many suspects, change one, keep the others constant,
observe outcome. When a hypothesis falls, the variables you
**didn't** change are still free to vary — test them next.

The null-fraction bench is 35 lines and sweeps `null_frac ∈ {0.0,
0.1, 0.5, 0.99, 1.0}`. It answered "is it near-all-null specifically,
or any null?" in one shot. The result — "any null, including 0.1" —
redefined the fix.

### 3. Trust reproducible failure more than reproducible success

11a's bench showed "CB fits fine on `Dictionary<uint32, large_string>`"
— but only when the column was **null-free**. If we had only tested
the success case, the misdiagnosis would have persisted. The
critical test was flipping one variable (adding nulls) and observing
the failure shape match the prod error exactly:
`TypeError: No matching signature found`.

### 4. Keep the benches in the repo

The three benches live in the repo root. They're not tests (don't
assert, don't fit into pytest's setup chain), but they're authoritative
evidence. Anyone debugging the same family of upstream issues should
be able to run them and see the same outcomes six months later.

### 5. Use the bench to write the sensor

Once the bench pins the trigger, the sensor writes itself:

```python
def test_null_in_categorical_triggers_the_thing():
    df = pl.DataFrame({"c": pl.Series("c", ["a", None]).cast(pl.Categorical)})
    # Pre-fix: CB.fit(df, ...) → TypeError "No matching signature found"
    # Post-fix: our detector flags this and the upstream fill_null keeps
    # the fastpath alive.
    assert _polars_nullable_categorical_cols(df, cat_features=["c"]) == ["c"]
```

The sensor documentation links back to the bench:

```python
"""Root cause (verified 2026-04-19 via direct repro in
bench_polars_cb_nullfrac.py)..."""
```

### 6. Cost budget: when to bench vs when to ship

- **~15 minutes to write a bench** that could take an hour off the
  next debug cycle → always worth it.
- **A diagnosis that's 85% likely and the fix is 2 lines** → sometimes
  ship without the bench if the fix is clearly defensive (it handles
  the suspected case AND accidentally handles other related cases).
  **But always leave a TODO to bench later**, because you will hit
  the same family of bugs again.
- **A diagnosis that motivates a 100-line refactor** → absolutely bench
  first. The refactor locks in your wrong mental model.

11a would have been a ~30-line refactor (monkey-patching `xgboost.data._arrow_dtype`
and building a proactive detection helper). It would have shipped if not
for the user's push to verify. The cost of verification: 80 lines of
bench code, 5 minutes of runtime.

### 7. The misdiagnosis table is permanent

When a hypothesis falls, document it in the CHANGELOG and in any
code comments that referenced it. `bench_polars_largestring_cb_xgb.py`
lives in the repo specifically to prove "large_string is NOT the
trigger" to anyone who revisits this area. The comment in
`_polars_nullable_categorical_cols` says:

> Earlier hypotheses (`large_string` Arrow export, `pl.Enum`) were
> disproved by isolated benches. The null-in-Categorical trigger
> explains every prior symptom.

This prevents re-running the misdiagnosis cycle when someone else
(or future-you) looks at the same code. Without it, the next
investigator would likely land on hypothesis 11a again — it's the
most plausible-sounding one.

## Checklist

When you see an opaque upstream error:

- [ ] Can I construct the exact input shape in ≤40 lines of standalone code?
- [ ] Does the minimal repro produce the **same error** as prod? (If not, keep minimizing.)
- [ ] What single variable changes "FAIL" → "OK"?
- [ ] Have I bench-proved my hypothesis, or just reasoned about it?
- [ ] Is the fix cheapest-possible given the diagnosis? (Bypass vs fix vs patch — pick the one that keeps the most capability.)
- [ ] Did I commit the bench to the repo?
- [ ] Did I write a sensor that would have caught this?
- [ ] Did I update the CHANGELOG with the misdiagnosis log?

Skip none. Especially not the first.
