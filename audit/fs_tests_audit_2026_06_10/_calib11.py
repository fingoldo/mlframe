"""Pin the EMPTY-support bug: make_signal_plus_noise + MRMR(min_features_fallback=1)
returns 0-feature support despite the documented non-empty guarantee. Check several
configs to isolate whether FE or the redundancy gate is responsible, and whether
min_features_fallback engages."""
import sys, warnings, numpy as np
sys.path.insert(0, "tests")
from feature_selection._biz_val_synth import make_signal_plus_noise, as_df


def fit(X, y, seed, **kw):
    from mlframe.feature_selection.filters.mrmr import MRMR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MRMR(random_seed=seed, verbose=0, cv=3, run_additional_rfecv_minutes=False, **kw).fit(X, y)


configs = {
    "default_fb1": dict(min_relevance_gain=0.0, full_npermutations=3, min_features_fallback=1),
    "fe_off_fb1": dict(min_relevance_gain=0.0, full_npermutations=3, min_features_fallback=1,
                       fe_max_steps=0),
    "simple_fb1": dict(min_relevance_gain=0.0, full_npermutations=3, min_features_fallback=1,
                       use_simple_mode=True),
}
for tag, kw in configs.items():
    empties = 0
    sizes = []
    for seed in range(6):
        Xn, yn, sig = make_signal_plus_noise(n=1500, p_signal=3, p_noise=16, seed=seed)
        X, y = as_df(Xn, yn)
        try:
            sel = fit(X, y, seed, **kw)
            sup = np.asarray(sel.support_, dtype=int)
            nout = sel.transform(X).shape[1]
        except Exception as e:
            sup = np.array([]); nout = -1
            print(f"  {tag} seed={seed} EXC {type(e).__name__}: {e}")
        sizes.append(len(sup))
        if len(sup) == 0:
            empties += 1
    print(f" [{tag}] support sizes={sizes} EMPTY={empties}/6")
