"""c0121 diag — NaN in cat_feature."""
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
import polars as pl

_orig_pool = cb.Pool.__init__
def _diag_pool(self, *args, **kwargs):
    data = kwargs.get('data', args[0] if args else None)
    tf = kwargs.get('text_features')
    cf = kwargs.get('cat_features')
    print(f'\n[POOL] type={type(data).__name__} shape={getattr(data,"shape",None)} cat_features={cf} text_features={tf}', flush=True)
    if hasattr(data, 'schema'):
        for c in data.columns:
            col = data[c]
            print(f'[POOL]   pl {c}: dtype={col.dtype} nulls={col.null_count()} head={list(col.head(3))}', flush=True)
    elif hasattr(data, 'dtypes'):
        for c in data.columns:
            col = data[c]
            nn = col.isnull().sum()
            print(f'[POOL]   pd {c}: dtype={col.dtype} nulls={nn} head={list(col.head(3))}', flush=True)
    try:
        return _orig_pool(self, *args, **kwargs)
    except Exception as e:
        print(f'[POOL] BUILD FAIL: {type(e).__name__}: {e}', flush=True)
        raise
cb.Pool.__init__ = _diag_pool

# Intercept get_pandas_view_of_polars_df (ASCII only — Windows cp1251 safe)
from mlframe.training import utils as _mlu
_orig_gpv = _mlu.get_pandas_view_of_polars_df
def _diag_gpv(df, *a, **kw):
    res = _orig_gpv(df, *a, **kw)
    import inspect
    caller = inspect.stack()[1]
    print(f'\n[GPV] caller={caller.filename.split(chr(92))[-1]}:{caller.lineno} -> pandas shape={res.shape}', flush=True)
    for c in res.columns:
        if c.startswith('cat_') or c.startswith('text_') or c.startswith('emb_'):
            col = res[c]
            nn = col.isnull().sum()
            print(f'[GPV]   {c}: dtype={col.dtype} nulls={nn} head={list(col.head(3))}', flush=True)
    return res
_mlu.get_pandas_view_of_polars_df = _diag_gpv
import mlframe.training.core as _core
import mlframe.training.trainer as _trainer
for mod in (_core, _trainer):
    if hasattr(mod, 'get_pandas_view_of_polars_df'):
        mod.get_pandas_view_of_polars_df = _diag_gpv

combos = enumerate_combos(target=150, master_seed=20260422)
c = next(x for x in combos if x.short_id() == 'c0088_2fa08bef')
print(f'c0121: models={c.models} input={c.input_type} polarsds={c.use_polarsds_pipeline} autocat={c.auto_detect_cats}', flush=True)
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
        hyperparams_config={'iterations': 3, 'cb_kwargs': {'task_type':'CPU','verbose':0}},
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
