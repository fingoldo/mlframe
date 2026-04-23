"""c0086 — CB + MRMR + polars_enum + nulls (ncats=8, n=1200)."""
import sys
sys.path.insert(0, r"D:/Upd/Programming/PythonCodeRepository/mlframe")
sys.path.insert(0, r"D:/Upd/Programming/PythonCodeRepository/mlframe/tests/training")

from _fuzz_combo import FuzzCombo, build_frame_for_combo, enumerate_combos
from shared import SimpleFeaturesAndTargetsExtractor
from mlframe.training.core import train_mlframe_models_suite
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# The c0086 combo from seed=2026_04_24 enumeration.
combos = enumerate_combos(target=150, master_seed=2026_04_24)
combo = combos[86]
print(f"combo: {combo}")
print(f"models={combo.models} input={combo.input_type} ncats={combo.cat_feature_count} nulls={combo.null_fraction_cats}")

df, target_col, _ = build_frame_for_combo(combo)
fte = SimpleFeaturesAndTargetsExtractor(target_column=target_col, regression=False)

try:
    trained, _ = train_mlframe_models_suite(
        df=df, target_name='r', model_name='r',
        features_and_targets_extractor=fte,
        mlframe_models=list(combo.models),
        hyperparams_config={'iterations': 5,
            'cb_kwargs': {'task_type': 'CPU', 'verbose': 0},
            'lgb_kwargs': {'device_type': 'cpu', 'verbose': -1},
            'xgb_kwargs': {'device': 'cpu', 'verbosity': 0}},
        init_common_params={'drop_columns': [], 'verbose': 0,
            'category_encoder': ce.CatBoostEncoder(),
            'scaler': StandardScaler(), 'imputer': SimpleImputer(strategy='mean')},
        use_ordinary_models=True, use_mlframe_ensembles=False,
        data_dir=r'D:/Temp/repro_c0086_v2', models_dir='models', verbose=0,
        use_mrmr_fs=combo.use_mrmr_fs, mrmr_kwargs={'verbose': 0, 'max_runtime_mins': 1, 'n_workers': 1,
            'quantization_nbins': 5, 'use_simple_mode': True,
            'min_nonzero_confidence': 0.9, 'max_consec_unconfirmed': 3, 'full_npermutations': 3},
    )
    print('PASS')
except Exception as e:
    import traceback; traceback.print_exc()
    print(f'FAIL: {type(e).__name__}: {str(e)[:250]}')
