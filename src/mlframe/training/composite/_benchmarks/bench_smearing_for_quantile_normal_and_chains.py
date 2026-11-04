import logging

logger = logging.getLogger(__name__)
"""Does smearing help the quantile-normal and chain transforms on the y scale? RMSE with and without smearing on right-skewed targets (3 seeds).

Result 2026-11-04: quantile_normal_y 6.86/6.86, 6.37/6.26, 5.67/5.64; chain_linres_cbrt_qn 6.89/6.91, 6.40/6.42, 5.80/5.81; chain_linres_cbrt
6.89/7.01, 6.45/6.65, 5.75/5.93 (without / with smearing). The gain is under 2% and the chains get worse, so they stay out of SMEARED_TRANSFORMS.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from mlframe.training.composite.estimator import _smearing as sm
from mlframe.training.composite.estimator._estimator import CompositeTargetEstimator
def run(name, smear, seed):
    rng=np.random.default_rng(seed); n=6000
    x=rng.normal(size=(n,3)); mu=1.0+0.8*x[:,0]+0.5*x[:,1]
    y=np.exp(mu+rng.normal(0,0.7,n))          # right-skewed, y-scale mean differs from median
    df=pd.DataFrame(x,columns=list("abc"))
    tr=slice(0,4000); te=slice(4000,None)
    if smear: sm.SMEARED_TRANSFORMS=frozenset(set(sm.SMEARED_TRANSFORMS)|{name})
    else: sm.SMEARED_TRANSFORMS=frozenset(set(sm.SMEARED_TRANSFORMS)-{name})
    est=CompositeTargetEstimator(base_estimator=HistGradientBoostingRegressor(max_iter=150),transform_name=name,base_column="a")
    est.fit(df.iloc[tr],y[tr])
    p=est.predict(df.iloc[te])
    return float(np.sqrt(np.mean((p-y[te])**2)))
for name in ("quantile_normal_y","chain_linres_cbrt_qn","chain_linres_cbrt"):
    try:
        r = [(run(name, False, s), run(name, True, s)) for s in range(3)]
        print(name, [(round(a, 2), round(b, 2)) for a, b in r])
    except Exception as e:
        logger.warning("%s failed: %r", name, e)
