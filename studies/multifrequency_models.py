"""Causal horizon-specific selection and SPY/QQQ timing model comparison.

The exploratory factor matrix is NEVER used to select models. Each horizon's
features and K are selected using labels that mature before the frozen test
start. Daily walk-forward fits use only labels exited by the signal close.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

from extended_timing_research import FACTOR_CATALOG, WINDOWS, aggregate, daily_intraday_features
from index_timing_framework import (StudyConfig, event_prices, load_market,
    nonoverlap_count, orthogonalize_past_k)
from raven_lite_timing import adaptive_features, backtest, fit_and_predict, measure

HORIZONS=(1,5,10,20,30,60)
EXECUTIONS=('next_open','next_close')
TEST_START='2025-06-02'
MAX_FEATURES=4


def factor_matrix(path,panel,cfg):
    daily=daily_intraday_features(path,panel,cfg)
    columns={}
    for name in FACTOR_CATALOG:
        for k in WINDOWS:
            if k==1 and FACTOR_CATALOG[name][1]=='variance': continue
            raw=aggregate(daily,name,k)
            columns[(name,k)]=orthogonalize_past_k(raw,panel,k,cfg).reindex(panel.dates)
    return pd.DataFrame(columns,index=panel.dates),daily.index


def target(panel,h,execution):
    p=np.arange(len(panel.dates),dtype=int)
    valid,entry,exit_=event_prices(panel,p,h,execution)
    y=np.full(len(p),np.nan)
    y[valid]=np.log(exit_/entry)
    # Exit date (at close) is exactly when the label first becomes known.
    maturity=p+h+(execution=='next_close')
    return y,maturity


def _corr(x,y,rank=False):
    if len(x)<12 or np.std(x)<1e-12 or np.std(y)<1e-12: return 0.
    c=spearmanr(x,y).statistic if rank else np.corrcoef(x,y)[0,1]
    return float(c) if np.isfinite(c) else 0.


def screen_factors(matrix,y,maturity,split,max_features=MAX_FEATURES):
    """Fit screen strictly within pretest, split by time for sign stability.

    Quantile spreads use in-fold ranks for diagnostics only. Both folds have
    fully matured labels. Screening is exploratory, with multiple comparisons.
    """
    candidates=[]
    all_pos=np.arange(len(y))
    available=(maturity<split)&np.isfinite(y)
    for column in matrix:
        x=matrix[column].to_numpy(dtype=float)
        ix=all_pos[available&np.isfinite(x)]
        if len(ix)<45: continue
        halfway=len(ix)//2
        halves=(ix[:halfway],ix[halfway:])
        if min(map(len,halves))<20: continue
        ic=[_corr(x[z],y[z]) for z in halves]
        ric=[_corr(x[z],y[z],rank=True) for z in halves]
        if ic[0]*ic[1]<=0 or ric[0]*ric[1]<=0: continue
        sign=np.sign(ic[0]);ranks=pd.Series(x[ix]).rank(method='first').to_numpy()
        bucket=np.minimum(2,(3*(ranks-1)/len(ranks)).astype(int))
        means=np.array([np.mean(y[ix][bucket==j]) for j in range(3)])
        if not np.isfinite(means).all(): continue
        spread=float(sign*(means[-1]-means[0]))
        monotone=int(np.sum(sign*np.diff(means)>0))
        # Stability takes precedence over absolute IC; small samples still
        # make this a very noisy selection and require independent validation.
        score=min(abs(ic[0]),abs(ic[1]))+0.5*min(abs(ric[0]),abs(ric[1]))
        score+=0.02*monotone+0.02*(spread>0)
        candidates.append(dict(name=column[0],k=int(column[1]),n=int(len(ix)),
            ic_halves=ic,rank_ic_halves=ric,means=means.tolist(),
            signed_spread=spread,monotone=monotone,score=float(score),direction=int(sign)))
    best={}
    for c in candidates:
        if c['name'] not in best or c['score']>best[c['name']]['score']:
            best[c['name']]=c
    selected=[]
    for c in sorted(best.values(),key=lambda z:-z['score']):
        col=(c['name'],c['k'])
        similar=False
        for old in selected:
            pair=matrix[[col,(old['name'],old['k'])]].iloc[:split].dropna()
            if len(pair)>25 and abs(pair.iloc[:,0].corr(pair.iloc[:,1]))>.85:
                similar=True;break
        if not similar: selected.append(c)
        if len(selected)==max_features: break
    return selected,dict(candidate_count=len(candidates),unique_factors=len(best))


def windows_for(matrix,features,eligible_days,length=20):
    vals=matrix[[(f['name'],f['k']) for f in features]].ffill(limit=5).to_numpy(float)
    windows=[];valid=[]
    for t in eligible_days:
        if t<length-1:continue
        arr=vals[t-length+1:t+1]
        if np.isfinite(arr).all(): windows.append(arr);valid.append(int(t))
    return np.asarray(windows),np.asarray(valid,int)


def matured_train_indices(t,maturity,y,valid,window=None,positions=None):
    """Labels exited no later than t close; discard signal t itself."""
    if positions is None:positions=np.arange(len(y))
    ix=np.where((positions<t)&(maturity<=t)&np.isfinite(y)&valid)[0]
    return ix[-window:] if window else ix


def _regularized_fit(x,y,xp,kind,alpha):
    scaler=StandardScaler().fit(x)
    xx=scaler.transform(x);pp=scaler.transform(xp)
    if kind=='elastic_net':
        model=ElasticNet(alpha=alpha,l1_ratio=.5,max_iter=10000,random_state=7)
    else: model=Ridge(alpha=alpha)
    model.fit(xx,y)
    return float(model.predict(pp)[0]),int(np.count_nonzero(np.abs(model.coef_)>1e-10))


def model_diagnostics(y,pred,positions,h,execution):
    result=measure(y,pred)
    result['rank_ic']=_corr(pred,y,rank=True)
    # Descriptive test-period sort only, never a live bucket threshold.
    order=np.argsort(pred,kind='stable');bins=np.empty(len(pred),dtype=int)
    bins[order]=np.minimum(4,(5*np.arange(len(pred))/len(pred)).astype(int))
    means=[float(np.mean(y[bins==i])) if np.any(bins==i) else None for i in range(5)]
    result['test_quintile_means']=means
    result['monotone_up_steps']=sum(means[i+1]>means[i] for i in range(4))
    result['greedy_nonoverlap_count']=nonoverlap_count(positions,h,execution)
    return result


def run_one(path,symbol,cfg,h,execution,model,window,matrix=None,valid_days=None,
            selected=None,screen_diag=None,panel=None):
    if panel is None:panel=load_market(path,cfg)
    if matrix is None:matrix,valid_days=factor_matrix(path,panel,cfg)
    y,maturity=target(panel,h,execution)
    split=int(np.searchsorted(panel.dates,TEST_START))
    if selected is None:selected,screen_diag=screen_factors(matrix,y,maturity,split)
    if not selected:return dict(symbol=symbol,horizon=h,execution=execution,model=model,
                                status='no_stable_features',selected=[],screen=screen_diag)
    # Screen is frozen before testing; windows only use information available
    # through t. Never fill missing factor values from the future.
    eligible=np.array([i for i in range(split,len(y)) if np.isfinite(y[i]) and
                       panel.dates[i] in valid_days],int)
    x,pos=windows_for(matrix,selected,eligible)
    if len(pos)<20:return dict(symbol=symbol,horizon=h,execution=execution,model=model,
                              status='too_few_test_windows',selected=selected)
    if model=='raven_lite':
        all_pos=np.array([i for i in range(len(y)) if panel.dates[i] in valid_days],int)
        all_x,all_pos=windows_for(matrix,selected,all_pos)
        ex,g,w,lengths=adaptive_features(all_x,.5)
        loc={p:i for i,p in enumerate(all_pos)}
    else:
        vals=matrix[[(f['name'],f['k']) for f in selected]].to_numpy(float)
        valid_train=np.isfinite(vals).all(axis=1)
    pred=[];train_counts=[];nonzero=[]
    for t in pos:
        if model=='raven_lite':
            train=matured_train_indices(t,maturity[all_pos],y[all_pos],
                      np.ones(len(all_pos),dtype=bool),window,positions=all_pos)
            if len(train)<30:pred.append(np.nan);train_counts.append(len(train));continue
            # Refit at most once per 10 test days with the label cutoff at
            # the actual fit day. Later predictions may use a stale model.
            ix=loc[t]
            val=fit_and_predict(ex,g,w,y[all_pos],train,np.array([ix]),10.)[0]
            nonzero.append(None)
        else:
            train=matured_train_indices(t,maturity,y,valid_train,window)
            if len(train)<30:pred.append(np.nan);train_counts.append(len(train));continue
            # Alpha fixed ex ante in scaled target units; no test-time tuning.
            scale=max(float(np.std(y[train])),1e-6)
            val,nz=_regularized_fit(vals[train],y[train]/scale,vals[t:t+1],model,
                                    .05 if model=='elastic_net' else 10.)
            val*=scale;nonzero.append(nz)
        pred.append(val);train_counts.append(len(train))
    pred=np.asarray(pred,float);ok=np.isfinite(pred)
    tp=pos[ok];pr=pred[ok]
    out=dict(symbol=symbol,horizon=h,execution=execution,model=model,
        training_window='expanding' if window is None else window,
        status='ok' if len(tp)>=20 else 'too_few_predictions',
        selected=selected,screen=screen_diag,n_test=int(len(tp)),
        n_train_min=int(np.min(np.asarray(train_counts)[ok])) if ok.any() else 0,
        n_train_max=int(np.max(np.asarray(train_counts)[ok])) if ok.any() else 0,
        metric=model_diagnostics(y[tp],pr,tp,h,execution) if len(tp) else None,
        active_coef_mean=float(np.mean(nonzero)) if nonzero and nonzero[0] is not None else None,
        first_signal=panel.dates[tp[0]] if len(tp) else None,
        last_signal=panel.dates[tp[-1]] if len(tp) else None,
        predictions=[dict(signal_day=panel.dates[t],signal_index=int(t),
                          label_maturity=panel.dates[maturity[t]],
                          forecast=float(v),realized=float(y[t]),train_n=int(n))
                     for t,v,n in zip(tp,pr,np.asarray(train_counts)[ok])])
    if len(tp)>=20:out['daily_long_cash']=backtest(panel,tp,pr,execution)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--spy',required=True);ap.add_argument('--qqq',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--models',nargs='+',choices=['ridge','elastic_net','raven_lite'],
                    default=['ridge','elastic_net','raven_lite'])
    ap.add_argument('--execution',nargs='+',choices=EXECUTIONS,default=list(EXECUTIONS))
    a=ap.parse_args();cfg=StudyConfig();results=[]
    for sym,path in [('SPY',a.spy),('QQQ',a.qqq)]:
        panel=load_market(path,cfg);matrix,valid_days=factor_matrix(path,panel,cfg)
        for h in HORIZONS:
            for ex in a.execution:
                y,maturity=target(panel,h,ex)
                selected,screen=screen_factors(matrix,y,maturity,
                            int(np.searchsorted(panel.dates,TEST_START)))
                print(sym,h,ex,'selected',[(z['name'],z['k']) for z in selected],flush=True)
                for model in a.models:
                    ws=(60,120,None) if model in ('ridge','elastic_net') else (None,)
                    for win in ws:
                        r=run_one(path,sym,cfg,h,ex,model,win,matrix,valid_days,selected,screen,panel)
                        results.append(r)
                        print(' ',model,win,r['status'],r.get('n_test'),
                              r.get('metric',{}).get('pearson') if r.get('metric') else None,flush=True)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(results,indent=2,allow_nan=False))

if __name__=='__main__':main()
