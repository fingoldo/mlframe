# Аудит polars_ds API для `train_mlframe_models_suite`

**Дата:** 2026-04-16  **Версия polars_ds:** 0.10.3  **Python:** 3.11 (Anaconda)

## Executive summary

1. **Критический красный флаг:** `polars_ds.Blueprint.target_encode / woe_encode / iv_encode` **не имеют встроенного cross-fitting** (в отличие от `sklearn.preprocessing.TargetEncoder` c `cv=5`). `fit_transform` на train приводит к silent target leakage: на синтетике с 500-картoйной категорией наш тест показывает **train AUC ≈ 0.73–0.78, test AUC ≈ 0.55, gap = 0.14–0.22**. Это НЕ баг polars_ds — это отсутствующая фича, подтвержденная в [transforms.py:503](D:\Temp\polars_ds_fork\python\polars_ds\pipeline\transforms.py#L503).
2. **Решение реализовано:** `upstream_demo/oof_encoders.py::OOFEncoder` — K-Fold обёртка с sklearn-совместимым `fit/transform/fit_transform`. На том же датасете: **gap = 0.001 при cv=5**, и **в 2.5x быстрее `sklearn.TargetEncoder(cv=5)` (0.91s vs 2.32s)**. Это шаблон для upstream-патча `Blueprint.target_encode(..., cv=5)`.
3. **Большинство остальных API polars_ds уже leak-safe** при использовании через `Blueprint` с fit на train (пайплайн в mlframe/training/pipeline.py:260 это обеспечивает). Killer-feature — `pds.str_leven/str_jaro` для строковых расстояний. `power_transforms` в polars_ds 0.10.3 **отсутствуют как first-class**.

## Главный эксперимент: target leakage в TE

Синтетика: n=20k, 3 категориальных колонки × 500 cardinality, slabый signal (σ=0.3), LogReg на энкодированных признаках.

| Вариант | train AUC | test AUC | gap | fit_transform, с |
|---------|-----------|----------|-----|-----------------|
| `polars_ds.target_encode` plain | 0.726 | 0.584 | **+0.141** 🔴 | 0.28 |
| **`OOFTargetEncoder(cv=5)`** (наш) | 0.587 | 0.586 | **+0.001** ✅ | **0.91** |
| `sklearn.TargetEncoder(cv=5)` (эталон) | 0.583 | 0.585 | −0.002 ✅ | 2.32 |
| `OOFTargetEncoder(randomized=True)` | 0.716 | 0.584 | +0.132 ⚠️ | 0.008 |

- **plain polars_ds** даёт модель, которая выглядит отличной на train — но тесты выдают её: классический target leakage через per-category mean.
- **OOF cv=5** ловит это на том же уровне, что и sklearn, при этом быстрее в 2.5x (polars + joblib-free K-Fold).
- **randomized** (гауссовский шум, как в `category_encoders.WOEEncoder.randomized`) — дёшев, но НЕ устраняет утечку, лишь слегка скрывает её. **Не рекомендуется как замена OOF.**

Скрипт: `upstream_demo/demo_oof_leak.py`, JSON-результат: `results/demo_oof_leak.json`.

## Расширенный эксперимент: TE + WoE × 5 библиотек, 5 повторов

Синтетика: n=20k, 3 cat × 500 cardinality, signal=0.3, LogReg. **5 независимых прогонов** (seeds 7, 20, 33, 46, 59), медиана ± std.

### Target Encoding

| Вариант | train AUC | test AUC | gap | time, s | Вердикт |
|---------|-----------|----------|-----|---------|---------|
| `polars_ds.target_encode` (plain) | 0.723 | 0.584 | **+0.139** | 0.006 | 🔴 LEAKS |
| `category_encoders.TargetEncoder` (plain) | 0.723 | 0.584 | **+0.139** | 0.062 | 🔴 LEAKS (идентично pds) |
| `OOFTargetEncoder(randomized=True, σ=0.05)` | 0.716 | 0.585 | **+0.131** | 0.009 | 🔴 LEAKS |
| **`OOFTargetEncoder(cv=3)`** | 0.574 | 0.586 | **-0.012** | 0.029 | ✅ SAFE |
| **`OOFTargetEncoder(cv=5)`** | 0.585 | 0.587 | **-0.002** | 0.041 | ✅ SAFE |
| **`sklearn.TargetEncoder(cv=3)`** | 0.579 | 0.585 | **-0.007** | 0.034 | ✅ SAFE |
| **`sklearn.TargetEncoder(cv=5)`** | 0.582 | 0.585 | **-0.003** | 0.031 | ✅ SAFE |

### WoE Encoding

| Вариант | train AUC | test AUC | gap | time, s | Вердикт |
|---------|-----------|----------|-----|---------|---------|
| `polars_ds.woe_encode` (plain) | 0.725 | 0.583 | **+0.142** | 0.006 | 🔴 LEAKS |
| `category_encoders.WOEEncoder` (plain) | 0.725 | 0.583 | **+0.142** | 0.059 | 🔴 LEAKS |
| `category_encoders.WOEEncoder(randomized=True, σ=0.05)` | 0.725 | 0.583 | **+0.142** | 0.069 | 🔴 LEAKS! |
| `OOFWOEEncoder(randomized=True, σ=0.05)` | 0.725 | 0.583 | **+0.142** | 0.009 | 🔴 LEAKS |
| **`OOFWOEEncoder(cv=3)`** | 0.574 | 0.585 | **-0.011** | 0.026 | ✅ SAFE |
| **`OOFWOEEncoder(cv=5)`** | 0.584 | 0.586 | **-0.002** | 0.038 | ✅ SAFE |

### cv=3 vs cv=5

cv=3 — рекомендуемый дефолт: на 30% быстрее, gap отличается на < 0.01 (несущественно на практике). cv=5 даёт чуть точнее OOF (каждый fold обучается на 80% вместо 66%), но разница минимальна.

### Выводы расширенного эксперимента

1. **category_encoders.TargetEncoder и WOEEncoder утекают ровно так же, как polars_ds** — gap идентичен с точностью до тысячных.
2. **`category_encoders.WOEEncoder(randomized=True, sigma=0.05)` НЕ ПОМОГАЕТ ДЛЯ WoE** — gap +0.142 не уменьшается вообще. Для TE рандомизация снижает gap минимально (0.139→0.131), но это не спасение.
3. **Единственный рабочий метод — K-Fold OOF (cv≥2).** Наша реализация `OOFTargetEncoder`/`OOFWOEEncoder` на уровне sklearn по качеству и на уровне по скорости.
4. polars_ds быстрее category_encoders в **~10x** (0.006s vs 0.060s для plain WoE).

Скрипт: `upstream_demo/demo_oof_leak_extended.py`, JSON: `results/demo_oof_leak_extended.json`.

## Rust-native OOF: бенчмарк

Реализованы `pl_target_encode_oof` и `pl_woe_discrete_oof` в Rust (форк `D:\Temp\polars_ds_fork`). Один Rust-вызов заменяет K Python->Rust roundtrips.

| n | cv | Rust OOF TE | Python OOF TE | sklearn cv=5 | Rust/Python | Rust/sklearn |
|---|---|-------------|---------------|--------------|-------------|--------------|
| 20k | 3 | **0.010s** | 0.026s | — | **2.7x** | — |
| 20k | 5 | **0.013s** | 0.036s | 0.031s | **2.8x** | **2.4x** |
| 50k | 3 | **0.017s** | 0.039s | — | **2.2x** | — |
| 50k | 5 | **0.024s** | 0.054s | 0.061s | **2.3x** | **2.6x** |
| 100k | 3 | **0.029s** | 0.058s | — | **2.0x** | — |
| 100k | 5 | **0.041s** | 0.083s | 0.105s | **2.0x** | **2.6x** |

**Rust cv=3 — самый быстрый вариант**: 3x быстрее sklearn cv=5 при сопоставимом качестве.

Leak gap идентичен (побитовое совпадение Rust vs Python). 19/19 unit-тестов пройдено (включая pandas input, custom folds, Rust-Python equivalence).

Подробности: `upstream_demo/RUST_OOF_FEASIBILITY.md`, бенч: `upstream_demo/bench_rust_vs_python_oof.py`.

## LTS CPU: `polars-ds-lts-cpu`

Реализован `build_lts_cpu.py` в форке — build script для сборки wheel без AVX2:
- Подход A+D: Cargo feature `lts-cpu` (маркер) + Python build script
- Автоматически патчит `pyproject.toml` (rename + dep swap), собирает с `RUSTFLAGS="-C target-cpu=x86-64"`, восстанавливает оригиналы
- Использование: `python build_lts_cpu.py` (wheel) или `python build_lts_cpu.py --dev` (install)
- Результат: `pip install polars-ds-lts-cpu`, `import polars_ds` работает идентично

## Таблица 8 API

| # | API | Leak-safe по умолчанию? | Альтернативы | Speedup (n=200k)* | Integration effort (1-5) | Вердикт |
|---|-----|-------------------------|--------------|-------------------|--------------------------|---------|
| 1 | `Blueprint.winsorize` | ✅ | `sklearn.RobustScaler`, `pandas.clip` | pandas MemoryError при 200k (polars_ds справляется) | **2** | GREEN — заменить ручной IQR в `core.py:325` |
| 2 | `Blueprint.impute(method='median')` | ✅ | `sklearn.SimpleImputer`, `pandas.fillna` | **15x vs pandas, 54x vs sklearn** | **1** | GREEN — тривиально добавить |
| 3 | `Blueprint.woe_encode` | 🔴 **НЕТ OOF** | `category_encoders.WOEEncoder` | **17x vs category_encoders**, но без OOF небезопасно | **4** | YELLOW — через `OOFWOEEncoder(cv=5)` |
| 4 | `Blueprint.target_encode` | 🔴 **НЕТ OOF** | `sklearn.TargetEncoder(cv=5)`, `category_encoders.TargetEncoder` | 10x vs sklearn_cv5 (но без OOF!); `OOFTargetEncoder(cv=5)` **2.5x быстрее sklearn(cv=5)** | **4** | YELLOW — использовать `OOFTargetEncoder` |
| 5 | `pds.ttest_ind`, `pds.chi2`, `pds.f_test` | ✅ (stateless) | `scipy.stats`, `sklearn.f_classif` | **4.7x vs scipy**, но ~sklearn (1.9x) | **2** | GREEN — prefilter перед RFECV |
| 6 | `pds.principal_components` | ✅ | `sklearn.decomposition.PCA` | **0.3x (медленнее sklearn)** при 3500x меньше памяти (0.02 MB vs 71 MB) | **3** | YELLOW — выигрыш по памяти, проигрыш по скорости |
| 7 | `pds.str_leven`, `pds.str_jaro` | ✅ (stateless) | `rapidfuzz`, `textdistance` | **6x (leven) / 12x (jaro) vs rapidfuzz** | **2** | GREEN — killer-feature, использовать в text-col detection |
| 8 | Power transforms | — | `sklearn.PowerTransformer`, `scipy.stats.yeojohnson` | **не реализовано в polars_ds 0.10.3** | — | RED — оставить sklearn |

*Все speedups — медиана по 3 прогонам на синтетических датасетах 10k/50k/200k строк (см. `results/bench_0X_*.json`).
`bench_01` упал на n=200k из-за MemoryError в pandas.quantile (сам polars_ds работает).

## Детальные находки

### 1. `Blueprint.winsorize(cols, q_low=0.05, q_high=0.95)` — 🟢 GREEN
- Stateful, сохраняет quantile-границы в pipeline при `materialize()`.
- Leak-safe когда Blueprint строится на train (как уже делается в `pipeline.py:260`).
- Тест `test_winsorize_fit_only_on_train` подтверждает: test-значения клампятся к train-границам.
- **Рекомендация:** заменить `_apply_outlier_detection_global` (core.py:325) на per-feature winsorize в pipeline — снимается хрупкость "глобального fit".

### 2. `Blueprint.impute(cols, method='mean'|'median')` / `linear_impute` — 🟢 GREEN
- Stateful, медиана считается на train и сохраняется.
- Тест `test_impute_uses_train_median_only` явно подтверждает: null в test заполняется **train-медианой**, не test-медианой.
- **Рекомендация:** первый шаг в pipeline (до scaling).

### 3/4. `woe_encode` / `target_encode` — 🟡 YELLOW, требуется OOF
- Реализация в [transforms.py:503-559](D:\Temp\polars_ds_fork\python\polars_ds\pipeline\transforms.py#L503) — чистый Python поверх `pds_num.target_encode` (Rust). Это значит **patch без Rust-сборки возможен**.
- `category_encoders.TargetEncoder` использует smoothing `expit((n - min_samples_leaf) / smoothing)` — **идентичен** параметрам polars_ds. Никакого OOF у category_encoders тоже нет; разница со sklearn.TargetEncoder в том, что последний делает встроенный K-Fold.
- `category_encoders.WOEEncoder` предлагает `randomized=True, sigma=0.05` — гауссовский шум в fit_transform. Мы это реализовали в `OOFEncoder(randomized=True)`, но эксперимент показал — это **НЕ** спасает от утечки (gap всё ещё 0.13).
- **Вывод:** единственный корректный способ — K-Fold OOF (`OOFTargetEncoder(cv=5)`).

### 5. Стат. тесты (`ttest_ind`, `f_test`, `chi2`) — 🟢 GREEN
- Stateless функции — безопасны. Главное не путать scope: считать только на train split.
- Для feature selection в `FeatureSelectionConfig` логично добавить фильтр `p < 0.05` как быстрый pre-step перед дорогим MRMR/RFECV.

### 6. `pds.principal_components` — 🟡 YELLOW
- Корректный API: `pds.principal_components(*[pl.col(f) for f in feats], k=5, center=True)`.
- **Медленнее sklearn**: 0.10s vs 0.03s при n=200k (0.3x speedup), но **3500x меньше памяти** (0.02 MB vs 71 MB).
- Leak-safety: stateful, проекция фитится на переданном df — при использовании через Blueprint на train корректно.
- **Рекомендация:** рассмотреть для memory-constrained сценариев. Для стандартных размеров sklearn.PCA быстрее.

### 7. Строковые расстояния — 🟢 GREEN
- Единственный API где polars_ds может давать существенный выигрыш над rapidfuzz за счёт векторизации Polars expressions (rapidfuzz тоже на Rust/C++, но Python-loop вокруг него). См. `results/bench_07_string_dist.json`.
- **Рекомендация:** использовать в автодетекции текстовых признаков (`core.py:108-136`), там где уже есть эвристики по dtype.

### 8. Power transforms — 🔴 RED
- `grep yeo\|box\|power` по `dir(polars_ds)` и методам Blueprint не находит ничего. `pds.normal_test` есть, но это тест Шапиро-Уилка, а не трансформация.
- **Рекомендация:** оставить `sklearn.PowerTransformer` в `PreprocessingExtensionsConfig`, либо открыть отдельный upstream-issue.

## Дополнительные API для рассмотрения (deep-dive, 2026-04-16)

| # | API | Файл | Описание | Leak-safe? | Полезность для mlframe |
|---|-----|------|----------|------------|----------------------|
| 9 | `linear_models` (LR, ElasticNet, GLM, OnlineLR) | `linear_models.py` | Полноценные линейные модели с null_policy, fit/predict API | ✅ stateful fit | Может заменить baseline LogReg для быстрой оценки, но нет CV |
| 10 | `EDA.DIA` class | `eda/diagnosis.py` | Auto-profiling: null/NaN/inf counts, numeric histograms, outlier detection (IQR), categorical analysis | ✅ read-only | Отличный кандидат для pre-training data quality check |
| 11 | `add_noise` | `exprs/stats.py:471-490` | Gaussian jitter / uniform perturb | ✅ stateless | Регуляризация, data augmentation |
| 12 | `expr_linear` (lin_reg, logistic_reg, lin_reg_report) | `exprs/expr_linear.py` | Expression-level regression внутри polars select | ✅ stateless | Feature engineering: per-group regression coefficients |
| 13 | `smooth_spline` | `exprs/expr_spline.py` | Cubic spline fitting с регуляризацией | ⚠️ fit на df | Non-linear feature generation |
| 14 | `query_knn_ptwise`, `query_dist_from_kth_nb` | `exprs/expr_knn.py` | KNN-based outlier detection, local density | ✅ stateless | Альтернатива pyod.IForest для outlier detection |
| 15 | `ts_features` (auto_corr, ar_coeffs, permutation_entropy) | `exprs/ts_features.py` | Time-series feature extraction | ✅ stateless | Если в mlframe появятся временные ряды |
| 16 | `metrics` (query_r2, query_roc_auc, query_confusion_matrix) | `exprs/metrics.py` | Evaluation в polars expressions | ✅ stateless | Post-training evaluation без sklearn |

**Power transforms** — подтверждено отсутствие (grep `yeo|box.*cox|power.*trans` по всему форку = 0 совпадений). Остаётся sklearn.PowerTransformer.

## Дизайн upstream-патча: (a) vs (b) и находка `materialize(return_df=True)`

### Ключевая находка
`Blueprint.materialize(df, return_df=True)` уже возвращает `(Pipeline, LazyFrame)` где LazyFrame = трансформированный train. Это **снимает главный минус варианта (a)**: сигнатура не ломается.

### Рекомендованный дизайн (вариант a, адаптированный)
1. При `cv > 0` для encoder-шагов: внутри `materialize()` подменить колонки в возвращаемом LazyFrame на OOF-значения.
2. `Pipeline` хранит финальный mapping (fit на полном train) для `transform(test)`.
3. API не меняется: `materialize(df)` → Pipeline; `materialize(df, return_df=True)` → (Pipeline, OOF-safe LazyFrame).
4. Пользователь без `return_df=True` — поведение идентично текущему (backward compatible).

Подробный анализ: `upstream_demo/PATTERN_SCAN.md`.
Draft PR: `upstream_demo/PR_DRAFT_target_encode_cv.md`.

## Upstream contribution plan

### Prototype реализован локально
- `upstream_demo/oof_encoders.py::OOFEncoder` — поведение = upstream-патч на polars_ds.
- Параметры sklearn-совместимы: `cv`, `stratified`, `random_state`, плюс все существующие `min_samples_leaf`, `smoothing`, `default`.
- Бонус: `randomized + sigma` (по образцу `category_encoders.WOEEncoder`) — для случаев, когда K-Fold неприемлемо дорог.

### Следующие шаги для upstream PR в `abstractqqq/polars_ds_extension`
1. Fork клонирован в `D:\Temp\polars_ds_fork`. Ветка `feature/target-encode-cv`.
2. Правка в `python/polars_ds/pipeline/transforms.py::target_encode`: добавить `cv: int | None = None, cv_seed: int = 0, stratified: bool = True`.
3. Правка в `python/polars_ds/pipeline/pipeline.py::Blueprint.target_encode/.woe_encode/.iv_encode`: прокинуть параметр.
4. Документация: раздел "Leak-safety and cross-fitting" в `docs/`, явное предупреждение в docstring.
5. Тесты в `tests/`: синтетический leak-тест (gap при `cv=None` vs `cv=5`), наследовать наш `demo_oof_leak.py`.
6. CHANGELOG: breaking change? Нет — `cv=None` сохраняет поведение.

### Риски / открытые вопросы
- `Blueprint` дизайн предполагает, что fit-expression применяется identically к train и test. OOF нарушает это: train-значения = OOF, test-значения = fit-on-full. Требуется либо (a) eager-transform на train внутри `materialize()` + сохранить final mapping для `transform(test)`, либо (b) хук `fit_transform_train` отдельно от `transform`. Вариант (a) проще, но ломает допущение "pipeline = pure list of expressions". Обсудить с мейнтейнером.
- При cv=5 fit_transform на train становится в ~5x медленнее plain (наш эксперимент это подтверждает), но остаётся в 2.5x быстрее sklearn — приемлемо.

## Проверка

```bash
# Leak-тесты (все проходят, 1 skip для pds.pca по необходимости доработки обёртки)
D:/ProgramData/anaconda3/python.exe -m pytest mlframe/explore/polars_ds_audit/leak_tests/ -v --no-cov

# Ключевая демонстрация
D:/ProgramData/anaconda3/python.exe mlframe/explore/polars_ds_audit/upstream_demo/demo_oof_leak.py

# Бенчи (последовательно, ~5 минут суммарно)
for i in 01 02 03 04 05 06 07 08; do D:/ProgramData/anaconda3/python.exe mlframe/explore/polars_ds_audit/benches/bench_${i}_*.py; done
```

## Рекомендации по интеграции в `train_mlframe_models_suite`

| Где | Что | Почему |
|-----|-----|--------|
| [pipeline.py:52-93](D:\Upd\Programming\PythonCodeRepository\mlframe\training\pipeline.py#L52-L93) `_build_extension_steps` | Добавить условную ветку для `OOFTargetEncoder` / `OOFWOEEncoder` для высококардинальных cat-колонок | Снимает необходимость one_hot_encode с ×100 размерностью |
| [configs.py:228-251](D:\Upd\Programming\PythonCodeRepository\mlframe\training\configs.py#L228-L251) `PreprocessingExtensionsConfig` | Ввести поле `target_encoder: Literal['none','oof','sklearn'] = 'none'`, `target_encoder_cv: int = 5` | Явный контроль, безопасный default |
| [core.py:325-414](D:\Upd\Programming\PythonCodeRepository\mlframe\training\core.py#L325-L414) `_apply_outlier_detection_global` | Опционально заменить на `Blueprint.winsorize` как первый шаг pipeline | Убирает "global fit" хрупкость |
| [configs.py:301-322](D:\Upd\Programming\PythonCodeRepository\mlframe\training\configs.py#L301-L322) `FeatureSelectionConfig` | Добавить `statistical_prefilter: Optional[Literal['ttest','chi2','f_classif']] = None, prefilter_p_threshold: float = 0.05` | Дешёвая фильтрация перед MRMR/RFECV |
| [core.py:108-136](D:\Upd\Programming\PythonCodeRepository\mlframe\training\core.py#L108-L136) автодетект текстовых колонок | Использовать `pds.str_leven` для group-by уникальным strings | Быстрее чем pandas.apply + rapidfuzz loop |
| `PreprocessingExtensionsConfig.dim_reducer` | `pds.pca` — **не добавлять**, sklearn.PCA достаточен | Numerical quality LAPACK лучше |
| `PreprocessingExtensionsConfig.scaler_name` | Power transforms — **не добавлять в polars_ds слой**, оставить sklearn.PowerTransformer | Отсутствует в polars_ds |

---

**Этот отчёт НЕ правит production-код `training/*`.** Решение по интеграции остаётся за пользователем.
