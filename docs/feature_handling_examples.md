# `FeatureHandlingConfig` cookbook

Practical recipes for the feature-handling layer.

All examples below construct a `FeatureHandlingConfig` and pass it to `train_mlframe_models_suite` via the `feature_handling_config=` keyword; the suite consumes it per target / per model when building each model's feature-handling plan.

## 1. Zero-config defaults

```python
from mlframe.training.feature_handling import FeatureHandlingConfig

fhc = FeatureHandlingConfig()
print(fhc.describe(short=True))
```

What you get:
- text columns get `tfidf` (max_features=5000, ngram (1,2))
- cat columns get `native` (CB) or `ordinal` (sklearn-style models) per the model-axis support matrix
- `intfloat/multilingual-e5-small` is the default provider when an embedding-method is requested (`auto_locale_detection="fallback_only"`)
- in-memory cache enabled, disk persistence off
- memory budget auto-derived: 70% of `min(psutil.virtual_memory().total, cgroup_limit)`

## 2. TF-IDF for everyone, larger vocabulary

```python
from mlframe.training.feature_handling import tfidf_only

fhc = tfidf_only(max_features=10000, ngram_range=(1, 3))
```

Equivalent long form:

```python
from mlframe.training.feature_handling import (
    FeatureHandlingConfig, TextHandlerSpec, TfidfParams,
    CatHandlerSpec, NoParams,
)

fhc = FeatureHandlingConfig(
    default_text=[TextHandlerSpec(method="tfidf", params=TfidfParams(max_features=10000, ngram_range=(1, 3)))],
    default_cat=[CatHandlerSpec(method="native", params=NoParams(kind="native"))],
)
```

## 3. CatBoost-native everywhere, drop text for everyone else

```python
from mlframe.training.feature_handling import cb_native_only

fhc = cb_native_only()  # default cat=native; CB gets text=native, others text=drop
```

## 4. Embedding-only (frozen) for all models

```python
from mlframe.training.feature_handling import embedding_only, EmbeddingProvider

fhc = embedding_only(
    provider=EmbeddingProvider(
        kind="huggingface",
        model="BAAI/bge-small-en-v1.5",  # English-only, faster than e5-multilingual
    ),
    pool="cls",  # default "mean"
)
```

## 5. Per-model mix: TF-IDF for tree models, learnable HF block for MLP

```python
from mlframe.training.feature_handling import (
    FeatureHandlingConfig, ModelHandlingOverride,
    TextHandlerSpec, TfidfParams, LearnableEmbeddingParams,
)

fhc = FeatureHandlingConfig(
    default_text=[TextHandlerSpec(method="tfidf", params=TfidfParams(max_features=5000))],
    per_model={
        "mlp": ModelHandlingOverride(text=[
            TextHandlerSpec(
                method="learnable_text_embedding",
                params=LearnableEmbeddingParams(finetune_lr_mult=0.05),
            ),
        ]),
    },
)
```

The validator catches inconsistent combos at construction: `learnable_text_embedding` on a non-neural model would raise `ValueError` with a clear "requires a neural model; use 'frozen_text_embedding' for non-neural" message.

## 6. Multi-handler chain: TF-IDF + frozen embedding concatenated

```python
from mlframe.training.feature_handling import (
    FeatureHandlingConfig, TextHandlerSpec, TfidfParams,
    FrozenEmbeddingParams,
)

fhc = FeatureHandlingConfig(
    default_text=[
        TextHandlerSpec(method="tfidf", params=TfidfParams(max_features=2000)),
        TextHandlerSpec(method="frozen_text_embedding", params=FrozenEmbeddingParams()),
    ],
)
```

For sparse-aware models (XGB / CB / LGB / linear) the assembler emits a two-track output: TF-IDF stays sparse, the embedding goes into the dense block. Disambiguated column names: `txt__tfidf__token_word`, `txt__frozen_emb__384`.

For dense-only models (HGB / MLP / TabNet / etc) the TF-IDF block auto-applies `TruncatedSVD(n_components=256)` with a WARN log line; user can override `svd_dim` per handler.

## 7. Append a learnable HF block on top of defaults (rather than replace)

```python
fhc = FeatureHandlingConfig(
    default_text=[TextHandlerSpec(method="tfidf", params=TfidfParams())],
    per_model={
        "mlp": ModelHandlingOverride(text_append=[
            TextHandlerSpec(
                method="learnable_text_embedding",
                params=LearnableEmbeddingParams(finetune_lr_mult=0.05),
            ),
        ]),
    },
)
# mlp gets [tfidf, learnable_text_embedding]; others just [tfidf]
```

## 8. High-cardinality categorical with leakage-safe target encoding

```python
from mlframe.training.feature_handling import (
    FeatureHandlingConfig, CatHandlerSpec, TargetEncodeParams,
)

fhc = FeatureHandlingConfig(
    default_cat=[
        CatHandlerSpec(
            method="target_mean",
            params=TargetEncodeParams(smoothing=20.0, cv=5, prior="mean"),
        ),
    ],
)
```

`LeakageSafeEncoder` runs K-fold OOF inside `fit_transform()` so train rows get encodings computed without seeing themselves; held-out rows at `transform()` time use the full-train statistic. The leakage probe in the test pack pins the contract: naive train AUC ≈ 1.0, OOF train AUC < 0.7 on a synthetic perfect-leak target.

## 9. Cache configuration: opt-in disk persistence for long suite reruns

```python
from mlframe.training.feature_handling import FeatureHandlingConfig, CacheConfig

fhc = FeatureHandlingConfig(
    cache=CacheConfig(
        persistence="auto",  # write to disk on miss
        dir="/data/mlframe_cache",  # mode 0o700 enforced on POSIX
        ram_max_gb=24.0,  # explicit override (else auto-derived)
        keep_n_providers=3,  # keep 3 HF model weights warm in VRAM
        eviction_strategy="size_weighted",  # round-3 default
    ),
)
```

Inspection:

```python
print(fhc.describe(short=False))  # full sub-config dump including resolved budgets
```

## 10. Custom user-supplied transformer (sklearn pipeline) for one column

```python
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from mlframe.training.feature_handling import (
    FeatureHandlingConfig, TextHandlerSpec, CustomParams,
)

# Power-user case: log-transform a numeric column then scale.
my_pipe = Pipeline([
    ("log", FunctionTransformer(np.log1p, validate=False)),
    ("scale", RobustScaler()),
])

fhc = FeatureHandlingConfig(
    per_model={
        "linear": ModelHandlingOverride(text=[
            TextHandlerSpec(
                method="custom",
                params=CustomParams(
                    transformer=my_pipe,
                    output_kind="dense",
                ),
                apply_to_columns=["budget_amount"],
            ),
        ]),
    },
)
```

`validate_custom_transformer()` runs at handler construction so a `lambda x: x` (no `.fit()`) is rejected immediately rather than failing inside `fit_transform()` with a confusing `AttributeError`.

## 11. Reproducibility mode for CI parity tests

```python
from mlframe.training.feature_handling import FeatureHandlingConfig, ReproConfig

fhc = FeatureHandlingConfig(
    repro=ReproConfig(
        deterministic_torch=True,    # 10-25% perf cost; off by default
        pinned_revision=True,        # default True; zero perf cost
        langdetect_seed=0,           # default 0
        pinned_svd_solver_params=True,  # default True
    ),
)
```

The `repro` sub-config is opt-in. `deterministic_torch` is the expensive flag (it sets `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` so GPU matmul drops the 1-3 fp16 mantissa-bit drift between runs); the rest are zero-cost knobs.

## 12. Multi-criteria text auto-detection with explicit overrides

```python
from mlframe.training.feature_handling import FeatureHandlingConfig, TextDetectionConfig

fhc = FeatureHandlingConfig(
    text_detection=TextDetectionConfig(
        # Anti-UUID guard tightened
        min_alphabet_entropy=5.0,         # default 4.5
        min_mean_tokens_for_text=3.0,     # default 2.0
        # User-curated overrides
        explicit_text_columns=["job_description", "review_body"],
        explicit_categorical_columns=["country_code", "tier"],
        skip_columns=["raw_html_dump"],
    ),
)
```

The detector returns per-column decisions (`fhc.describe()` surfaces them) with `rule_name` showing which trigger fired. Useful for auditing why a column was classified text -- previously a single-trigger heuristic flipped surprisingly on edge cases.

---

## Discoverability

```python
fhc.describe(short=True)   # one-line per (model, column)
fhc.describe(short=False)  # full per-section dump including resolved memory budgets
```

```python
# Check provider runtime status (round-3 U-R2-24)
from mlframe.training.feature_handling import provider_status
print(provider_status())
```

## Validation

`fhc.validate_against_models(["cb", "xgb", "lgb", "mlp"])` walks every default + per_model spec against the active model list, accumulates ALL mismatches into one combined `ValueError` so users fix everything in one pass (rather than fix-one, run-again, fix-next).

## Future provider sources (phase J -- not yet shipped)

The Protocol is ready; the impl ships in phase J:

- `EmbeddingProvider(kind="sentence-transformers", model="all-MiniLM-L6-v2")`
- `EmbeddingProvider(kind="openai", model="text-embedding-3-small", params={"api_key": "env:OPENAI_API_KEY", "dimensions": 512})` -- API-based, paid; gate via `pricing.cap_usd`.
- `EmbeddingProvider(kind="onnx", model="path/to/local.onnx")` -- self-hosted ONNX runtime.
- `EmbeddingProvider(kind="fasttext", model="cc.en.300.bin")` -- multilingual baseline.
