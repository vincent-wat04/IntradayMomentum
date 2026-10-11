"""Small-sample RAVEN-inspired adaptive-context diagnostic, NOT a paper replica.

NumPy / scikit-learn baseline for two short ETF series. It implements
per-instance time normalization, learned-on-validation patch importance,
three nested CIT prefixes, separately fitted ridge experts, cosine-redundancy
weighting and a global branch. It does NOT implement the paper's trainable
Transformer encoders, attention, entropy/diversity losses or joint backprop.

Run: python raven_lite_timing.py --spy SPY_5min.csv --qqq QQQ_5min.csv --output results.json
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from extended_timing_research import daily_intraday_features
from index_timing_framework import StudyConfig, event_prices, load_market, orthogonalize_past_k

FEATURES=('SKEW_DIFF','LOGVOL_SKEW','OPEN_VOL_SHARE','AM_PM_OPEN_RATIO',
          'UP_RVOL','VP_MED_SKEW','MOM_RV','RV_RATIO','VOL_20','ILLIQ_20')
LENGTH=20
PATCH=5
THRESHOLDS=(.3,.6,.9)

def build_dataset(path,panel,cfg):
    daily=daily_intraday_features(path,panel,cfg)
    residual=[]
    for name in FEATURES:
        f=daily[name].rolling(5,min_periods=5).mean()
        residual.append(orthogonalize_past_k(f,panel,5,cfg).rename(name))
    residual=pd.concat(residual,axis=1).reindex(panel.dates).ffill(limit=5)
    matrix=residual.to_numpy(float)
    windows=[];positions=[]
    for t in range(LENGTH-1,len(matrix)-2):
        if panel.dates[t] not in daily.index: continue
        x=matrix[t-LENGTH+1:t+1]
        if np.isfinite(x).all():
            windows.append(x)
            positions.append(t)
    return np.asarray(windows),np.asarray(positions,int)

def adaptive_features(windows,theta):
    """Causal variable prefixes in latest-first order; no label enters routing."""
    experts=[];global_features=[];weights=[];lengths=[]
    for x in windows:
        mu=x.mean(axis=0);sd=x.std(axis=0);z=(x-mu)/(sd+1e-6)
        p=z.reshape(LENGTH//PATCH,PATCH,-1)[::-1]
        means=p.mean(axis=1)
        vol=p.std(axis=1).mean(axis=1)
        surprise=np.mean(np.abs(p),axis=(1,2))
        score=np.exp(np.clip(theta*surprise+(1-theta)*vol,-10,10))
        prob=score/score.sum();cum=np.cumsum(prob)
        n=[int(np.searchsorted(cum,tau)+1) for tau in THRESHOLDS]
        zks=[]
        for count in n:
            selected=p[:count].reshape(-1,p.shape[-1])
            zks.append(np.r_[selected.mean(axis=0),selected.std(axis=0)])
        zks=np.asarray(zks)
        normalized=zks/(np.linalg.norm(zks,axis=1,keepdims=True)+1e-8)
        similarity=normalized@normalized.T
        redundancy=np.maximum(similarity,0).sum(axis=1)-1
        confidence=np.exp(-np.maximum(redundancy,0))
        confidence/=confidence.sum()
        experts.append(zks)
        global_features.append(np.r_[z.mean(axis=0),z.std(axis=0),z[-1]])
        weights.append(confidence);lengths.append(n)
    return np.asarray(experts),np.asarray(global_features),np.asarray(weights),np.asarray(lengths)

def fit_and_predict(experts,global_x,weights,y,train_idx,pred_idx,alpha):
    preds=[]
    for k in range(len(THRESHOLDS)):
        mdl=Ridge(alpha=alpha).fit(experts[train_idx,k],y[train_idx])
        preds.append(mdl.predict(experts[pred_idx,k]))
    local=np.sum(np.asarray(preds).T*weights[pred_idx],axis=1)
    global_model=Ridge(alpha=alpha).fit(global_x[train_idx],y[train_idx])
    return .5*local+.5*global_model.predict(global_x[pred_idx])

def measure(y,pred):
    ok=np.isfinite(y)&np.isfinite(pred)
    y=y[ok];pred=pred[ok]
    return {'n':len(y),'pearson':float(np.corrcoef(y,pred)[0,1]) if len(y)>2 and np.std(pred)>0 else None,
            'mse':float(np.mean((y-pred)**2)) if len(y) else None,
            'direction_hit':float(np.mean(np.sign(pred)==np.sign(y))) if len(y) else None}

def backtest(panel,signal_positions,pred,execution,fee_bp=1):
    # Fixed 2025 test period, long/cash. At t+1 open the old position holds
    # overnight; at t+1 close the old position holds that whole session.
    forecast={int(t):float(v) for t,v in zip(signal_positions,pred)}
    previous=0.;strategy=[];market=[];dates=[]
    for j in range(min(forecast)+1,max(forecast)+2):
        t=j-1
        target=float(forecast[t]>0) if t in forecast else previous
        overnight=panel.daily_open[j]/panel.close[j-1]-1
        intra=panel.close[j]/panel.daily_open[j]-1
        m=panel.close[j]/panel.close[j-1]-1
        if execution=='next_open':
            gross=(1+previous*overnight)*(1+target*intra)-1
        else:
            gross=previous*m
        strategy.append(gross-fee_bp*1e-4*abs(target-previous))
        market.append(m);dates.append(panel.dates[j]);previous=target
    r=np.array(strategy);m=np.array(market)
    # Market benchmark on the exact same return dates.
    def perf(x):
        curve=np.cumprod(1+x);peak=np.maximum.accumulate(np.r_[1.,curve])[1:]
        return {'total_return':float(curve[-1]-1),
                'sharpe':float(x.mean()/x.std(ddof=1)*np.sqrt(252)) if len(x)>1 and x.std(ddof=1)>0 else None,
                'max_drawdown':float((curve/peak-1).min())}
    return {'n':len(r),'start':dates[0],'end':dates[-1],
            'strategy':perf(r),'buy_hold_price':perf(m)}

def experiment(path,symbol,cfg,execution='next_open'):
    panel=load_market(path,cfg)
    x,pos=build_dataset(path,panel,cfg)
    # Training labels mature at the exit close. For next_open this is s+1;
    # for next_close it is s+2. Both fit and hyperparameter selection obey
    # the corresponding information boundary.
    eligible,entry,exit_=event_prices(panel,pos,1,execution)
    assert np.array_equal(eligible,pos)
    y=exit_/entry-1
    # Fixed calendar split, declared prior to inspecting outcomes.
    val_start=int(np.searchsorted(panel.dates,'2025-03-03'))
    test_start=int(np.searchsorted(panel.dates,'2025-06-02'))
    exit_lag=1 if execution=='next_open' else 2
    train=np.where(pos+exit_lag<val_start)[0]
    val=np.where((pos>=val_start)&(pos+exit_lag<test_start))[0]
    fit=np.where(pos+exit_lag<test_start)[0]
    test=np.where(pos>=test_start)[0]
    if len(train)<35 or len(val)<20 or len(test)<30: raise ValueError('Insufficient chronological folds')
    candidates=[]
    for theta in (0.,.5,1.):
        e,g,w,l=adaptive_features(x,theta)
        for alpha in (1.,10.,100.):
            pred=fit_and_predict(e,g,w,y,train,val,alpha)
            candidates.append((float(np.mean((y[val]-pred)**2)),theta,alpha))
    _,theta,alpha=min(candidates)
    single_candidates=[]
    for j,name in enumerate(FEATURES):
        fit_one=make_pipeline(StandardScaler(),Ridge(alpha=10.))
        fit_one.fit(x[train,-1,j,None],y[train])
        p=fit_one.predict(x[val,-1,j,None])
        single_candidates.append((float(np.mean((y[val]-p)**2)),j,name))
    _,j,best_name=min(single_candidates)
    best_single=make_pipeline(StandardScaler(),Ridge(alpha=10.))
    best_single.fit(x[fit,-1,j,None],y[fit])
    single_pred=best_single.predict(x[test,-1,j,None])
    e,g,w,lengths=adaptive_features(x,theta)
    out=fit_and_predict(e,g,w,y,fit,test,alpha)
    baseline=Ridge(alpha=alpha).fit(g[fit],y[fit]).predict(g[test])
    score={'symbol':symbol,'execution':execution,'n_train':len(train),'n_validation':len(val),
           'n_fit_before_test':len(fit),'n_test':len(test),
           'first_test_signal':panel.dates[pos[test[0]]],
           'last_test_signal':panel.dates[pos[test[-1]]],
           'theta':theta,'ridge_alpha':alpha,
           'adaptive_mean_prefix_patches':lengths[test].mean(axis=0).tolist(),
           'raven_lite':measure(y[test],out),
           'fixed_context_ridge':measure(y[test],baseline),
           'validation_selected_single_factor':best_name,
           'selected_single_factor':measure(y[test],single_pred),
           'strategy':backtest(panel,pos[test],out,execution),
           'fixed_context_strategy':backtest(panel,pos[test],baseline,execution),
           'selected_single_strategy':backtest(panel,pos[test],single_pred,execution),
           'predictions':[{'signal_day':panel.dates[t],'prediction':float(p),'realized_entry_exit':float(v)}
                          for t,p,v in zip(pos[test],out,y[test])]}
    return score

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spy',required=True);p.add_argument('--qqq',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();cfg=StudyConfig()
    res=[experiment(path,sym,cfg,execution) for sym,path in [('SPY',a.spy),('QQQ',a.qqq)]
         for execution in ('next_open','next_close')]
    with open(a.output,'w') as o:json.dump(res,o,allow_nan=False,indent=2)
    print(a.output,[(r['symbol'],r['execution'],r['n_test'],r['raven_lite']['pearson']) for r in res])
