"""c0085 diag — text_feature=float error."""
import sys
sys.path.insert(0, r'D:/Upd/Programming/PythonCodeRepository/mlframe')
sys.path.insert(0, r'D:/Upd/Programming/PythonCodeRepository/mlframe/tests/training')

from _fuzz_combo import enumerate_combos, build_frame_for_combo
from shared import SimpleFeaturesAndTargetsExtractor
from mlframe.training.core import train_mlframe_models_suite
from mlframe.training.configs import PolarsPipelineConfig, FeatureTypesConfig, TrainingBehaviorConfig
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import catboost as cb

_orig_pool = cb.Pool.__init__
def _diag_pool(self, *args, **kwargs):
    data = kwargs.get('data', args[0] if args else None)
    tf = kwargs.get('text_features')
    cf = kwargs.get('cat_features')
    print(f'\n[POOL] type={type(data).__name__} shape={getattr(data,"shape",None)}', flush=True)
    print(f'[POOL] cat_features={cf} text_features={tf}', flush=True)
    if hasattr(data, 'schema'):
        print(f'[POOL] schema: {data.schema}', flush=True)
        for c in (cf or []) + (tf or []):
            if c in data.columns:
                col = data[c]
                print(f'[POOL]   {c}: dtype={col.dtype} nulls={col.null_count()} head={list(col.head(3))}', flush=True)
    elif hasattr(data, 'dtypes'):
        print(f'[POOL] pandas dtypes: {dict(data.dtypes)}', flush=True)
    # When the failing build happens, dump full frame for forensics
    try:
        return _orig_pool(self, *args, **kwargs)
    except Exception as e:
        print(f'[POOL] BUILD FAIL: {type(e).__name__}: {e}', flush=True)
        print(f'[POOL] FULL FRAME dump:', flush=True)
        if hasattr(data, 'schema'):
            print(f'[POOL]   schema: {data.schema}')
            print(f'[POOL]   cols: {list(data.columns)}')
            print(f'[POOL]   head: {data.head(3)}')
        elif hasattr(data, 'dtypes'):
            print(f'[POOL]   dtypes: {dict(data.dtypes)}')
            print(f'[POOL]   cols: {list(data.columns)}')
            print(f'[POOL]   head:\\n{data.head(3)}')
        raise
cb.Pool.__init__ = _diag_pool

combos = enumerate_combos(target=150, master_seed=20260422)
c = next(x for x in combos if x.short_id() == 'c0085_39d4cb7b')
print(f'c0085: models={c.models} input={c.input_type} ncats={c.cat_feature_count} mrmr={c.use_mrmr_fs} polarsds={c.use_polarsds_pipeline} autocat={c.auto_detect_cats} text={c.text_col_count} emb={c.embedding_col_count} null={c.null_fraction_cats}', flush=True)
df, target_col, _ = build_frame_for_combo(c)
fte = SimpleFeaturesAndTargetsExtractor(target_column=target_col, regression=c.target_type=='regression')
init_params = {'drop_columns': [], 'verbose': 0,
    'category_encoder': ce.CatBoostEncoder(),
    'scaler': StandardScaler(), 'imputer': SimpleImputer(strategy='mean')}

import tempfile
tmp = tempfile.mkdtemp()

text_feats = ['text_0'] if c.text_col_count > 0 and 'cb' in c.models else None
emb_feats = ['emb_0'] if c.embedding_col_count > 0 and 'cb' in c.models and c.input_type != 'pandas' else None

try:
    trained, _ = train_mlframe_models_suite(
        df=df, target_name=c.short_id(), model_name=c.short_id(),
        features_and_targets_extractor=fte,
        mlframe_models=list(c.models),
        hyperparams_config={'iterations': 3, 'cb_kwargs': {'task_type':'CPU','verbose':0},
            'xgb_kwargs': {'device':'cpu','verbosity':0}},
        init_common_params=init_params,
        use_mrmr_fs=c.use_mrmr_fs,
        mrmr_kwargs={'verbose': 0, 'max_runtime_mins': 1, 'n_workers': 1,
            'quantization_nbins': 5, 'use_simple_mode': True,
            'min_nonzero_confidence': 0.9, 'max_consec_unconfirmed': 3,
            'full_npermutations': 3} if c.use_mrmr_fs else None,
        use_ordinary_models=True, use_mlframe_ensembles=False,
        data_dir=tmp, models_dir='models', verbose=0,
        pipeline_config=PolarsPipelineConfig(use_polarsds_pipeline=c.use_polarsds_pipeline),
        feature_types_config=FeatureTypesConfig(
            auto_detect_feature_types=c.auto_detect_cats,
            use_text_features=c.use_text_features, honor_user_dtype=c.honor_user_dtype,
            text_features=text_feats, embedding_features=emb_feats,
        ),
        behavior_config=TrainingBehaviorConfig(align_polars_categorical_dicts=c.align_polars_categorical_dicts),
    )
    print(f'PASS: {list(trained.keys())}', flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    print(f'FAIL: {type(e).__name__}: {str(e)[:400]}', flush=True)
