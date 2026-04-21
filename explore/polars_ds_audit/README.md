# polars_ds audit

Исследование 8 API из `polars_ds` для интеграции в `train_mlframe_models_suite`:
1. Outliers (`Blueprint.winsorize`)
2. Imputation (`Blueprint.impute`, `Blueprint.linear_impute`)
3. WoE (`Blueprint.woe_encode`)
4. Target Encoding (`Blueprint.target_encode`)
5. Статистика (`pds.ttest_ind`, `pds.chi2`, `pds.f_test`)
6. PCA (`pds.pca`)
7. String distances (`pds.str_leven`, `pds.str_jaro`, `pds.filter_by_levenshtein`)
8. Power transforms (Yeo-Johnson / Box-Cox) — **НЕ найдено как first-class**

## Структура

- `_common.py` — синт.данные, timing, scorer.
- `leak_tests/` — pytest-тесты, проверяют отсутствие утечки test→train.
- `benches/` — бенчи vs sklearn/pandas/category_encoders/pyod/rapidfuzz.
- `upstream_demo/` — демонстрация upstream-доработок polars_ds (OOF для TE/WoE).
- `results/` — JSON/CSV из прогонов.
- `REPORT.md` — финальный отчёт.

## Запуск

```bash
D:/ProgramData/anaconda3/python.exe -m pytest mlframe/explore/polars_ds_audit/leak_tests/ -v
for f in mlframe/explore/polars_ds_audit/benches/*.py; do D:/ProgramData/anaconda3/python.exe "$f"; done
```

## Версия polars_ds: 0.10.3
