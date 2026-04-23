"""Repro ValueError: DataFrame.dtypes for data must be int, float, bool or category."""
import sys, os
sys.path.insert(0, r'D:/Upd/Programming/PythonCodeRepository/mlframe')
sys.path.insert(0, r'D:/Upd/Programming/PythonCodeRepository/mlframe/tests/training')

os.environ['FUZZ_SEED'] = '20260430'

from _fuzz_combo import enumerate_combos, build_frame_for_combo
from shared import SimpleFeaturesAndTargetsExtractor
from mlframe.training.core import train_mlframe_models_suite
from mlframe.training.configs import PolarsPipelineConfig, FeatureTypesConfig, TrainingBehaviorConfig

from _fuzz_combo import FuzzCombo
# Construct combo matching the failing pattern directly
c = FuzzCombo(
    models=('hgb', 'lgb', 'xgb'),
    input_type='polars_utf8',
    n_rows=600,
    cat_feature_count=1,
    null_fraction_cats=0.1,
    use_mrmr_fs=False,
    weight_schemas=('uniform',),
    target_type='binary_classification',
    auto_detect_cats=False,
    align_polars_categorical_dicts=True,
    seed=21,
    use_polarsds_pipeline=True,
    use_text_features=True,
    honor_user_dtype=False,
    text_col_count=0,
    embedding_col_count=0,
)

print(f'FOUND: models={c.models} input={c.input_type} polarsds={c.use_polarsds_pipeline} autocat={c.auto_detect_cats} align={c.align_polars_categorical_dicts} null={c.null_fraction_cats} ncats={c.cat_feature_count}', flush=True)
df, target_col, _ = build_frame_for_combo(c)

fte = SimpleFeaturesAndTargetsExtractor(target_column=target_col, regression=c.target_type=='regression')
init_params = {'drop_columns': [], 'verbose': 0}
if 'linear' in c.models and c.cat_feature_count > 0:
    import category_encoders as ce
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    init_params['category_encoder'] = ce.CatBoostEncoder()
    init_params['scaler'] = StandardScaler()
    init_params['imputer'] = SimpleImputer(strategy='mean')

import tempfile
tmp = tempfile.mkdtemp()

try:
    trained, _ = train_mlframe_models_suite(
        df=df, target_name=c.short_id(), model_name=c.short_id(),
        features_and_targets_extractor=fte,
        mlframe_models=list(c.models),
        hyperparams_config={'iterations': 3, 'cb_kwargs': {'task_type':'CPU','verbose':0},
            'xgb_kwargs': {'device':'cpu','verbosity':0}, 'lgb_kwargs':{'device_type':'cpu','verbose':-1}},
        init_common_params=init_params,
        use_mrmr_fs=c.use_mrmr_fs,
        use_ordinary_models=True, use_mlframe_ensembles=False,
        data_dir=tmp, models_dir='models', verbose=1,
        pipeline_config=PolarsPipelineConfig(use_polarsds_pipeline=c.use_polarsds_pipeline),
        feature_types_config=FeatureTypesConfig(
            auto_detect_feature_types=c.auto_detect_cats,
            use_text_features=c.use_text_features, honor_user_dtype=c.honor_user_dtype,
        ),
        behavior_config=TrainingBehaviorConfig(align_polars_categorical_dicts=c.align_polars_categorical_dicts),
    )
    print(f'PASS: {list(trained.keys())}')
except Exception as e:
    import traceback; traceback.print_exc()
    print(f'FAIL: {type(e).__name__}: {str(e)[:400]}')
