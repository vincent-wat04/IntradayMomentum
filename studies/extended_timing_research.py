"""Point-in-time intraday-volume and price proxies for SPY/QQQ timing.

Run: python extended_timing_research.py --spy SPY_5min.csv --qqq QQQ_5min.csv --output-dir research_output
No source data is bundled. See README_EXTENDED.md for the factor catalog.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from index_timing_framework import (EXECUTIONS, MarketPanel, StudyConfig,
    event_prices, load_market, nonoverlap_count, nw_slope_t,
    orthogonalize_past_k, trailing_bucket_ids)

WINDOWS=(1,5,10,20,30,60)
HORIZONS=(1,5,10,20,30,60,90,120)

# These are time-series *adaptations*, not reproductions of the stock-sort alphas.
# "variance" follows the report's stability formula, with population divisor K.
FACTOR_CATALOG={
 'LOGVOL_STD':('report','mean','Intraday SD of log(1+5min volume)'),
 'LOGVOL_SKEW':('report','mean','Intraday standardized third moment of log(1+5min volume)'),
 'OPEN_VOL_SHARE':('report','mean','09:30-10:00 volume / daily volume'),
 'OPEN_VOL_SHARE_VAR':('report','variance','Rolling variance of daily open share'),
 'OPEN_VOL_SD':('report','mean','SD of first six bar volumes / daily volume'),
 'OPEN_VOL_SD_VAR':('report','variance','Rolling variance of normalized open-volume SD'),
 'AM_PM_OPEN_RATIO':('report','mean','09:30-10:00 volume / 13:00-13:30 volume'),
 'AM_PM_OPEN_RATIO_VAR':('report','variance','Rolling variance of morning/afternoon opening ratio'),
 'PM_MID_CLOSE_RATIO':('report','mean','15:30-16:00 volume / 11:30-12:00 volume'),
 'PM_MID_CLOSE_RATIO_VAR':('report','variance','Rolling variance of close/midday ratio'),
 'UP_RVOL':('report','mean','SD of significantly positive 5min returns times selected mean-volume ratio'),
 'UP_RVOL_VAR':('report','variance','Rolling variance of daily UP_RVOL'),
 'UP_VOLSD':('report','mean','SD of positive 5min returns times selected volume SD / daily mean volume'),
 'UP_VOLSD_VAR':('report','variance','Rolling variance of daily UP_VOLSD'),
 'VP_MED_SKEW':('report','mean','3(mean-median)/SD of volume-weighted price-decile label'),
 'VP_REVERSAL':('report','mean','Report-like POC/close-decile weighted intraday return'),
 'MOM_RV':('sohu','mean','Recent return divided by trailing return SD'),
 'MOM_SIGN':('sohu','mean','Sign of recent return'),
 'VOL_TARGET':('sohu','mean','Sign(recent return)*0.4/annualized return SD, clipped'),
 'RV_RATIO':('sohu','mean','Short realized volatility / long realized volatility'),
 'REV_5':('sohu','mean','Negative 5-session close return'),
 'AMIHUD_ILLIQ':('sohu','mean','Absolute daily return / daily dollar volume (scaled)'),
 'ST_REV_20':('ghz','mean','Negative 20-session return; short-term reversal proxy'),
 'MAXRET_20':('ghz','mean','Maximum daily return in last 20 sessions'),
 'VOL_20':('ghz','mean','SD of daily close return in last 20 sessions'),
 'DOLVOL_20':('ghz','mean','Mean daily dollar volume in last 20 sessions, log1p'),
 'ILLIQ_20':('ghz','mean','Mean |daily return|/dollar volume over 20 sessions'),
 'MID_AVG':('baseline','mean','Original noise-band position average'),
 'SKEW_DIFF':('baseline','mean','Original noise-band mean-minus-median proxy'),
}

def _div(a,b):
    return float(a/b) if np.isfinite(a) and np.isfinite(b) and b>0 else np.nan

def daily_intraday_features(path: str,panel: MarketPanel,cfg: StudyConfig) -> pd.DataFrame:
    raw=pd.read_csv(path,parse_dates=['datetime'])
    raw['day']=raw.datetime.dt.strftime('%Y-%m-%d')
    raw['clock']=raw.datetime.dt.strftime('%H:%M')
    close_map=dict(zip(panel.dates,panel.close))
    records=[]
    for day,g in raw.groupby('day',sort=True):
        if day in cfg.early_closes: continue
        g=g.sort_values('datetime')
        clock=g.clock.to_numpy(); v=g.volume.to_numpy(float)
        p=g.open.to_numpy(float)
        if len(g)!=78 or np.any(v<0) or not np.isfinite(p).all():
            raise ValueError(f'Invalid full session: {day}')
        logv=np.log1p(v); lv_sd=logv.std(ddof=1)
        lv_skew=float(np.mean(((logv-logv.mean())/lv_sd)**3)) if lv_sd>0 else np.nan
        om=(clock>='09:30')&(clock<'10:00')
        pm=(clock>='13:00')&(clock<'13:30')
        mc=(clock>='11:30')&(clock<'12:00')
        pc=(clock>='15:30')&(clock<'16:00')
        total=v.sum(); mean=v.mean()
        nextp=np.r_[p[1:],close_map[day]]
        barret=np.log(nextp/p)
        up=(barret>0)&(barret>barret.mean()+barret.std(ddof=1))
        up_sd=barret[up].std(ddof=1) if up.sum()>=2 else np.nan
        # Price deciles formed from the 78 observed bar opens; each bar's
        # volume is allocated to its opening-price decile (a coarse POC proxy).
        edges=np.quantile(p,np.linspace(0,1,11))
        labels=np.searchsorted(edges[1:-1],p,side='right')
        hist=np.bincount(labels,weights=v,minlength=10)
        prob=hist/total if total>0 else np.full(10,np.nan)
        mu=np.sum(prob*np.arange(10))/10
        med=np.searchsorted(np.cumsum(prob),.5)/10
        sd=np.sqrt(np.sum(prob*(np.arange(10)/10-mu)**2))
        smooth=np.r_[2*hist[0]+hist[1],hist[:-2]+hist[2:],hist[-2]+2*hist[-1]]
        poc=np.argmax(smooth)/10
        cbin=np.clip(np.searchsorted(edges[1:-1],close_map[day],side='right'),0,9)/10
        dayret=close_map[day]/p[0]-1
        rev=((poc-cbin+1) if dayret<0 else (cbin-poc+1))*dayret
        records.append({'day':day,'LOGVOL_STD':lv_sd,'LOGVOL_SKEW':lv_skew,
            'OPEN_VOL_SHARE':_div(v[om].sum(),total),
            'OPEN_VOL_SD':_div(v[om].std(ddof=1),total),
            'AM_PM_OPEN_RATIO':_div(v[om].sum(),v[pm].sum()),
            'PM_MID_CLOSE_RATIO':_div(v[pc].sum(),v[mc].sum()),
            'UP_RVOL':_div(up_sd*v[up].mean(),mean) if up.sum()>=2 else np.nan,
            'UP_VOLSD':_div(up_sd*v[up].std(ddof=1),mean) if up.sum()>=2 else np.nan,
            'VP_MED_SKEW':3*(mu-med)/sd if sd>1e-12 else np.nan,
            'VP_REVERSAL':rev})
    features=pd.DataFrame(records).set_index('day')
    # Stability factors are derived from their unsmoothed daily parent below.
    close=pd.Series(panel.close,index=panel.dates)
    ret=close.pct_change(fill_method=None)
    vol=pd.Series(raw.assign(dol=raw.open*raw.volume).groupby('day').dol.sum()).reindex(panel.dates)
    f=pd.DataFrame(index=features.index)
    for name in ['LOGVOL_STD','LOGVOL_SKEW','OPEN_VOL_SHARE','OPEN_VOL_SD',
                 'AM_PM_OPEN_RATIO','PM_MID_CLOSE_RATIO','UP_RVOL','UP_VOLSD',
                 'VP_MED_SKEW','VP_REVERSAL']:
        f[name]=features[name]
    f['MOM_RV']=(close.pct_change(10,fill_method=None)/ret.rolling(20).std()).reindex(f.index)
    f['MOM_SIGN']=np.sign(close.pct_change(10,fill_method=None)).reindex(f.index)
    f['VOL_TARGET']=(np.sign(close.pct_change(10,fill_method=None))*.4/
                     (ret.rolling(20).std()*np.sqrt(252))).clip(-3,3).reindex(f.index)
    f['RV_RATIO']=(ret.rolling(5).std()/ret.rolling(20).std()).reindex(f.index)
    f['REV_5']=(-close.pct_change(5,fill_method=None)).reindex(f.index)
    f['AMIHUD_ILLIQ']=(ret.abs()/vol*1e9).reindex(f.index)
    f['ST_REV_20']=(-close.pct_change(20,fill_method=None)).reindex(f.index)
    f['MAXRET_20']=ret.rolling(20).max().reindex(f.index)
    f['VOL_20']=ret.rolling(20).std().reindex(f.index)
    f['DOLVOL_20']=np.log1p(vol.rolling(20).mean()).reindex(f.index)
    f['ILLIQ_20']=(ret.abs()/vol*1e9).rolling(20).mean().reindex(f.index)
    f['MID_AVG']=panel.factors.MID_AVG
    f['SKEW_DIFF']=panel.factors.SKEW_DIFF
    return f.replace([np.inf,-np.inf],np.nan)

def aggregate(daily: pd.DataFrame,name: str,k: int) -> pd.Series:
    kind=FACTOR_CATALOG[name][1]
    if kind=='variance':
        parent=name.removesuffix('_VAR')
        return daily[parent].rolling(k,min_periods=k).var(ddof=0)
    return daily[name].rolling(k,min_periods=k).mean()

def metrics(symbol: str,panel: MarketPanel,name: str,k: int,version: str,
            series: pd.Series,cfg: StudyConfig):
    positions=panel.dates.get_indexer(series.index)
    bucket=trailing_bucket_ids(series,cfg,5)
    rows=[]
    for execution,ex_label in EXECUTIONS.items():
        for h in HORIZONS:
            for scope in ('available','common120D'):
                candidates=positions if scope=='available' else positions[positions+max(HORIZONS)+1<len(panel.dates)]
                eligible,entry,exit_=event_prices(panel,candidates,h,execution)
                xs=series.to_numpy()[np.searchsorted(positions,eligible)]
                bins=bucket[np.searchsorted(positions,eligible)]
                y=exit_/entry-1
                mask=np.isfinite(xs)&np.isfinite(y)
                x=xs[mask]; yr=y[mask]; bb=bins[mask]; pp=eligible[mask]
                pear=float(np.corrcoef(x,yr)[0,1]) if len(x)>2 and np.std(x)>0 and np.std(yr)>0 else None
                rank=float(pd.Series(x).corr(pd.Series(yr),method='spearman')) if len(x)>2 and np.std(x)>0 and np.std(yr)>0 else None
                bm=[float(yr[bb==b].mean()) if np.any(bb==b) else None for b in range(1,6)]
                bn=[int(np.sum(bb==b)) for b in range(1,6)]
                bi=[nonoverlap_count(pp[bb==b],h,execution) for b in range(1,6)]
                complete=all(z is not None for z in bm)
                years={}
                for year in sorted(set(panel.dates[p][:4] for p in pp)):
                    select=np.array([panel.dates[p].startswith(year) for p in pp])
                    xx,yy=x[select],yr[select]
                    years[year]={'n':len(xx),
                        'pearson_ic':float(np.corrcoef(xx,yy)[0,1]) if len(xx)>2 and np.std(xx)>0 and np.std(yy)>0 else None,
                        'bucket_counts':[int(np.sum(select&(bb==b))) for b in range(1,6)]}
                rows.append({'symbol':symbol,'factor':name,'source':FACTOR_CATALOG[name][0],
                    'aggregation':k,'version':version,'execution':execution,'horizon':h,'scope':scope,
                    'n_ic':len(x),'pearson_ic':pear,'rank_ic':rank,
                    'hac_slope_t':nw_slope_t(x,yr,h),'mean_return':float(yr.mean()) if len(yr) else None,
                    'up_rate':float(np.mean(yr>0)) if len(yr) else None,
                    'bucket_means':bm,'bucket_counts':bn,'bucket_nonoverlap':bi,
                    'bucket_high_low':bm[-1]-bm[0] if bm[0] is not None and bm[-1] is not None else None,
                    'bucket_up_steps':sum(bm[i+1]>bm[i] for i in range(4)) if complete else None,
                    'yearly':years})
    return rows

def run(spy: str,qqq: str,cfg: StudyConfig):
    result=[]; panels={}; factor_series={}
    for symbol,path in {'SPY':spy,'QQQ':qqq}.items():
        panel=load_market(path,cfg)
        daily=daily_intraday_features(path,panel,cfg)
        panels[symbol]=panel
        factor_series[symbol]={}
        for name in FACTOR_CATALOG:
            for k in WINDOWS:
                if FACTOR_CATALOG[name][1]=='variance' and k<2: continue
                raw=aggregate(daily,name,k)
                residual=orthogonalize_past_k(raw,panel,k,cfg)
                factor_series[symbol][(name,k)]=raw
                result+=metrics(symbol,panel,name,k,'raw',raw,cfg)
                result+=metrics(symbol,panel,name,k,'raw_matched',raw.where(residual.notna()),cfg)
                result+=metrics(symbol,panel,name,k,'orthog',residual,cfg)
    return result,panels,factor_series

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spy',required=True);p.add_argument('--qqq',required=True)
    p.add_argument('--output-dir',required=True)
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    rows,_,_=run(a.spy,a.qqq,StudyConfig())
    (out/'extended_factor_metrics.json').write_text(json.dumps(rows,ensure_ascii=False,allow_nan=False,indent=2))
    (out/'factor_catalog.json').write_text(json.dumps(FACTOR_CATALOG,ensure_ascii=False,indent=2))
    print('factor definitions',len(FACTOR_CATALOG),'metric rows',len(rows),out)
