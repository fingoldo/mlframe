"""Does the MI under equal-width bins pick the target transform that lowers the error on the original scale? (2026-11-23)

For three synthetic regression targets (lognormal: log linearises; cube: cbrt linearises; linear: nothing helps) and three seeds, prints for every transform f of the target the sum over features of
MI(x_j; f(y)) with equal-width bins, and the RMSE on the ORIGINAL scale of a ridge and a gradient-boosting model fitted on f(y) and inverted. No shift is applied to the target (a shift of y - min + 1 broke the
linearisation of the cube case in a first run); ``log`` is skipped where the target is not positive.

Run: ``python target_transform_mi.py``. Results and the plan they feed: ``audits/2026-11-23/PLAN_target_transform_in_mrmr.md``.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
def make(kind, n=20000, p=8, seed=0):
    rng=np.random.default_rng(seed); X=rng.standard_normal((n,p))
    if kind=="lognormal": y=np.exp(0.9*X[:,0]+0.6*X[:,1]+0.4*rng.standard_normal(n))
    elif kind=="cube": y=(X[:,0]+0.7*X[:,1]+0.3*rng.standard_normal(n))**3
    else: y=X[:,0]+0.7*X[:,1]+0.3*rng.standard_normal(n)
    return X,y
def bins(v,k,mode):
    e=np.quantile(v,np.linspace(0,1,k+1)[1:-1]) if mode=="quantile" else np.linspace(v.min(),v.max(),k+1)[1:-1]
    return np.searchsorted(e,v,side="right")
def mi(a,b):
    ka,kb=a.max()+1,b.max()+1; j=np.zeros((ka,kb)); np.add.at(j,(a,b),1); j/=j.sum()
    pa=j.sum(1,keepdims=True); pb=j.sum(0,keepdims=True); m=j>0
    return float((j[m]*np.log(j[m]/(pa@pb)[m])).sum())
def ssqrt(t):
    """Signed square root: defined for negative targets, monotone."""
    return np.sign(t) * np.sqrt(np.abs(t))


def ssq(z):
    """Inverse of the signed square root."""
    return np.sign(z) * z**2

T={"identity":(lambda t:t,lambda z:z,True),"log":(np.log,np.exp,False),"cbrt":(np.cbrt,lambda z:z**3,True),"signed_sqrt":(ssqrt,ssq,True)}
for kind in ("lognormal","cube","linear"):
    for seed in (0,1,2):
        X,y=make(kind,seed=seed); n=len(y); tr=np.arange(n)<n*0.7; te=~tr
        print(f"== {kind} seed {seed}  y min {y.min():.3f}")
        for name,(f,inv,any_sign) in T.items():
            if not any_sign and y.min()<=0: continue   # log needs a positive target; no shift is applied
            z=f(y)
            mi_u=sum(mi(bins(X[:,j],10,"uniform"),bins(z,10,"uniform")) for j in range(X.shape[1]))
            r=[]
            for mdl in (Ridge(1.0), HistGradientBoostingRegressor(max_iter=100)):
                mdl.fit(X[tr],z[tr]); r.append(float(np.sqrt(np.mean((inv(mdl.predict(X[te]))-y[te])**2))))
            print(f"   f={name:11s} MI_uniform {mi_u:.4f}  RMSE_orig ridge {r[0]:.4f}  gbm {r[1]:.4f}")
