"""Reusable SPY/QQQ noise-band factor event and portfolio backtest.

Example:
    python index_timing_framework.py --spy SPY_5min.csv --qqq QQQ_5min.csv \
        --output factor_results.json

Source timestamps must be New York local session time, with bars labelled by
their starting time. A signal is known at the end of trading day t. Both
execution modes first trade on t+1:

    next_open  : buy at O[t+1]; h=1 exit C[t+1], h=5 exit C[t+5].
    next_close : buy at C[t+1]; h=1 exit C[t+2], h=5 exit C[t+6].

The source dataset's five-minute 'close' is not necessarily a true auction
close. Results are price-only and cannot establish real close-auction fills.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import warnings

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StudyConfig:
    horizons: tuple[int, ...] = (1, 5, 10, 20, 30, 60, 90, 120)
    windows: tuple[int, ...] = (1, 5, 10, 20, 30, 60)
    quantiles: tuple[int, ...] = (1, 3, 5, 8, 10, 90, 92, 95, 97, 99)
    noise_lookback: int = 14
    threshold_lookback: int = 120
    threshold_min_periods: int = 80
    orthog_lookback: int = 120
    orthog_min_periods: int = 60
    layer_factor_ids: tuple[str, ...] = ("SKEW_DIFF",)
    bucket_counts: tuple[int, ...] = (5, 10)
    checkpoints: tuple[str, ...] = tuple(
        f"{(630 + 30*k)//60:02d}:{(630 + 30*k)%60:02d}" for k in range(11)
    )
    early_closes: tuple[str, ...] = (
        "2024-07-03", "2024-11-29", "2024-12-24",
        "2025-07-03", "2025-11-28", "2025-12-24",
    )
    fee_bp: float = 1.0
    portfolio_year: int = 2025


FACTOR_NAMES = {
    "MID_MED": "带中点距离_日内中位数",
    "MID_AVG": "带中点距离_日内均值",
    "SKEW_DIFF": "带内位置_日内均值减中位数",
    "MID_END": "带中点距离_尾盘值",
    "OUT_AVG": "带外超额距离_日内均值",
}
EXECUTIONS = {"next_open": "T+0：t+1开盘成交", "next_close": "T+1：t+1收盘成交"}
VERSIONS = {"raw":"原始", "raw_matched":"原始_同日期", "orthog":"正交残差"}


def event_direction(factor_id: str,low: bool) -> str:
    if factor_id=="SKEW_DIFF":
        return "低位看空" if low else "高位看多"
    return "低位看多" if low else "高位看空"


def event_wins(factor_id: str,low: bool,returns: np.ndarray) -> np.ndarray:
    bullish=(factor_id!="SKEW_DIFF")==low
    return returns>0 if bullish else returns<0


@dataclass
class MarketPanel:
    dates: pd.Index
    daily_open: np.ndarray
    close: np.ndarray
    factors: pd.DataFrame
    input_bars: int


def load_market(path: str, cfg: StudyConfig) -> MarketPanel:
    raw = pd.read_csv(path, parse_dates=["datetime"])
    required = {"datetime", "open", "high", "low", "close", "volume"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing columns {required - set(raw.columns)} in {path}")
    if raw.datetime.isna().any() or raw.datetime.duplicated().any():
        raise ValueError(f"Invalid or duplicate bar timestamps in {path}")
    if not raw.datetime.is_monotonic_increasing:
        raw = raw.sort_values("datetime", kind="stable").reset_index(drop=True)
    if (raw[["open", "high", "low", "close"]].le(0).any().any() or
        raw[["open", "high", "low", "close"]].isna().any().any()):
        raise ValueError("Prices must be finite and positive")
    raw["day"] = raw.datetime.dt.strftime("%Y-%m-%d")
    raw["clock"] = raw.datetime.dt.strftime("%H:%M")
    dates = pd.Index(raw.day.drop_duplicates().to_list(), name="day")
    if not dates.is_monotonic_increasing:
        raise ValueError("Trading dates must be ordered")
    opens = raw[raw.clock.eq("09:30")].set_index("day").open.reindex(dates)
    if opens.isna().any():
        raise ValueError("Missing 09:30 bar opens")
    eod_clock = pd.Series(np.where(dates.isin(cfg.early_closes),"12:55","15:55"),index=dates)
    eod = raw[raw.clock.isin(["12:55","15:55"])].set_index(["day","clock"]).close
    close = np.array([eod.loc[(d,eod_clock.loc[d])] for d in dates], dtype=float)
    full = raw[~raw.day.isin(cfg.early_closes)]
    by_clock = full.pivot(index="day", columns="clock", values="open")
    if not set(cfg.checkpoints).issubset(by_clock.columns):
        raise ValueError("Missing configured checkpoint columns")
    checkpoints = by_clock.loc[:, list(cfg.checkpoints)].copy()
    if checkpoints.isna().any().any():
        raise ValueError("Missing checkpoint on a full session")
    day_open = opens.reindex(checkpoints.index).to_numpy()
    prior_close = pd.Series(close,index=dates).shift(1).reindex(checkpoints.index).to_numpy()
    noise = (checkpoints.div(day_open,axis=0).sub(1).abs()
             .rolling(cfg.noise_lookback,min_periods=cfg.noise_lookback).mean().shift(1))
    b = noise.to_numpy()
    upper = np.maximum(day_open,prior_close)[:,None]*(1+b)
    lower = np.minimum(day_open,prior_close)[:,None]*(1-b)
    z = (checkpoints.to_numpy()-(upper+lower)/2)/np.maximum((upper-lower)/2,1e-9)
    factors = daily_factors_from_z(z,checkpoints.index)
    return MarketPanel(dates,opens.to_numpy(float),close,factors,len(raw))


def daily_factors_from_z(z: np.ndarray,index: pd.Index) -> pd.DataFrame:
    """Daily checkpoint summaries; SKEW_DIFF is mean minus median of z."""
    factors = pd.DataFrame(index=index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        factors["MID_MED"] = np.nanmedian(z,axis=1)
        factors["MID_AVG"] = np.nanmean(z,axis=1)
        factors["SKEW_DIFF"] = factors["MID_AVG"]-factors["MID_MED"]
        factors["MID_END"] = z[:,-1]
        factors["OUT_AVG"] = np.nanmean(np.sign(z)*np.maximum(np.abs(z)-1,0),axis=1)
    return factors


def orthogonalize_past_k(series: pd.Series,panel: MarketPanel,k: int,
                         cfg: StudyConfig) -> pd.Series:
    """One-step-ahead residual against the ETF's trailing K-session return.

    At signal t, alpha/beta use up to 120 valid (F_s,R_s) observations with
    s <= t-1. R_t=C_t/C_{t-K}-1 is observable at t close. Insufficient training
    history or zero variance in the control yields a missing residual.
    """
    prices=pd.Series(panel.close,index=panel.dates)
    past_return=prices.div(prices.shift(k)).sub(1).reindex(series.index)
    valid=series.notna()&past_return.notna()
    x=past_return.where(valid,0.0)
    y=series.where(valid,0.0)
    def rolling_prior(s):
        return s.shift(1,fill_value=0.0).rolling(cfg.orthog_lookback,min_periods=1).sum()
    count=rolling_prior(valid.astype(float))
    sx,sy=rolling_prior(x),rolling_prior(y)
    sxx,sxy=rolling_prior(x*x),rolling_prior(x*y)
    cov=sxy-sx*sy/count.replace(0,np.nan)
    variance=sxx-sx*sx/count.replace(0,np.nan)
    beta=cov/variance.where(variance>1e-12)
    alpha=(sy-beta*sx)/count.replace(0,np.nan)
    residual=series-alpha-beta*past_return
    return residual.where(valid&(count>=cfg.orthog_min_periods))


def make_factor_cache(panel: MarketPanel,cfg: StudyConfig):
    """Compute factor versions and past-only thresholds once per instrument."""
    cache = {}
    for factor_id in FACTOR_NAMES:
        for window in cfg.windows:
            raw=panel.factors[factor_id].rolling(window,min_periods=window).mean()
            orthog=orthogonalize_past_k(raw,panel,window,cfg)
            versions={"raw":raw,"raw_matched":raw.where(orthog.notna()),"orthog":orthog}
            for version,series in versions.items():
                past=series.shift(1)
                bounds={q:past.rolling(cfg.threshold_lookback,
                        min_periods=cfg.threshold_min_periods).quantile(q/100)
                        for q in cfg.quantiles}
                cache[(factor_id,window,version)] = (series,bounds)
    return cache


def event_prices(panel: MarketPanel, signal_positions: np.ndarray,
                 horizon: int, execution: str):
    """Return executable entry and horizon-exit arrays for all signal dates."""
    entry_i = signal_positions+1
    exit_i = signal_positions+horizon+(execution=="next_close")
    valid = exit_i<len(panel.dates)
    positions = signal_positions[valid]
    entries = (panel.daily_open if execution=="next_open" else panel.close)[entry_i[valid]]
    exits = panel.close[exit_i[valid]]
    valid_price = np.isfinite(entries)&(entries>0)&np.isfinite(exits)
    return positions[valid_price], entries[valid_price], exits[valid_price]


def nonoverlap_count(signal_positions: np.ndarray,h: int,execution: str) -> int:
    last_exit = -1
    count = 0
    for i in signal_positions:
        entry,exit_ = i+1,i+h+(execution=="next_close")
        if entry>=last_exit:
            count += 1
            last_exit = exit_
    return count


def trailing_bucket_ids(series: pd.Series, cfg: StudyConfig,
                        n_buckets: int) -> np.ndarray:
    """Historical-percentile bucket; never uses factor observations at t+1+."""
    values=series.to_numpy(float)
    labels=np.zeros(len(values),dtype=np.int16)  # 0 denotes insufficient history
    for t,value in enumerate(values):
        if not np.isfinite(value): continue
        prior=values[max(0,t-cfg.threshold_lookback):t]
        prior=prior[np.isfinite(prior)]
        if len(prior)<cfg.threshold_min_periods: continue
        # Midrank for ties; fixed, causal edges from the preceding signal history.
        percentile=(np.count_nonzero(prior<value)+0.5*np.count_nonzero(prior==value))/len(prior)
        labels[t]=min(n_buckets,1+int(percentile*n_buckets))
    return labels


def nonoverlapping_returns(positions: np.ndarray, returns: np.ndarray,
                           horizon: int, execution: str) -> np.ndarray:
    keep=[]
    last_exit=-1
    for i,(pos,ret) in enumerate(zip(positions,returns)):
        entry,exit_=pos+1,pos+horizon+(execution=="next_close")
        if entry>=last_exit:
            keep.append(ret)
            last_exit=exit_
    return np.array(keep,float)


def bucket_study(symbol: str,panel: MarketPanel,cfg: StudyConfig,cache):
    """Point-in-time quintile/decile layers; separate observed and paired samples."""
    layers,summaries,years=[],[],[]
    n=len(panel.dates)
    max_h=max(cfg.horizons)
    for (factor_id,window,version),(series,_) in cache.items():
      if factor_id not in cfg.layer_factor_ids: continue
      positions=panel.dates.get_indexer(series.index)
      for n_bins in cfg.bucket_counts:
        bucket=trailing_bucket_ids(series,cfg,n_bins)
        labelled=positions[bucket>0]
        for execution,label in EXECUTIONS.items():
          for horizon in cfg.horizons:
            scopes=("各期限", "共同120D") if n_bins==5 else ("各期限",)
            for scope in scopes:
              candidates=labelled if scope=="各期限" else labelled[labelled+max_h+1<n]
              valid,entries,exits=event_prices(panel,candidates,horizon,execution)
              returns=exits/entries-1
              group_labels=bucket[np.searchsorted(positions,valid)]
              baseline=float(returns.mean()) if len(returns) else None
              means=[]
              counts=[]
              nonoverlap_counts=[]
              base={"标的":symbol,"执行版本":label,"因子版本":VERSIONS[version],
                    "因子ID":factor_id,"窗口":window,"期限":horizon,
                    "分桶数":n_bins,"样本范围":scope}
              for b in range(1,n_bins+1):
                selected=group_labels==b
                p,ret=valid[selected],returns[selected]
                independent=nonoverlapping_returns(p,ret,horizon,execution)
                mean=float(ret.mean()) if len(ret) else None
                means.append(mean); counts.append(len(ret));nonoverlap_counts.append(len(independent))
                layers.append({**base,"桶":b,"触发日数":len(ret),
                    "非重叠次数":len(independent),"平均原始收益":mean,
                    "中位原始收益":float(np.median(ret)) if len(ret) else None,
                    "上涨比例":float(np.mean(ret>0)) if len(ret) else None,
                    "相对全体收益":mean-baseline if mean is not None and baseline is not None else None,
                    "非重叠平均收益":float(independent.mean()) if len(independent) else None,
                    "全体平均收益":baseline})
              complete=all(m is not None for m in means)
              bucket_rank=pd.Series(means).rank().to_numpy() if complete else None
              monotonic=(float(np.corrcoef(np.arange(n_bins),bucket_rank)[0,1])
                         if complete and np.std(bucket_rank)>0 else None)
              summaries.append({**base,"全体可评估日":len(valid),"最少桶日数":min(counts),
                  "最少桶非重叠次数":min(nonoverlap_counts),
                  "高减低收益":means[-1]-means[0] if means[0] is not None and means[-1] is not None else None,
                  "桶均值Spearman":monotonic,
                  "相邻上升段数":sum(means[j+1]>means[j] for j in range(n_bins-1)) if complete else None,
                  "总相邻段数":n_bins-1 if complete else None})
              if n_bins==5 and scope=="各期限":
                signal_years=np.array([panel.dates[i][:4] for i in valid])
                for year in sorted(set(signal_years)):
                    in_year=signal_years==year
                    market_year=returns[in_year]
                    for b in range(1,n_bins+1):
                        selected=in_year&(group_labels==b)
                        ret=returns[selected]
                        years.append({**base,"信号年份":int(year),"桶":b,
                          "触发日数":len(ret),"非重叠次数":nonoverlap_count(valid[selected],horizon,execution),
                          "平均原始收益":float(ret.mean()) if len(ret) else None,
                          "相对同年全体收益":float(ret.mean()-market_year.mean()) if len(ret) else None})
    return layers,summaries,years


def event_study(symbol: str,panel: MarketPanel,cfg: StudyConfig,cache):
    common,available,coverage = [],[],[]
    n=len(panel.dates)
    max_h=max(cfg.horizons)
    for (factor_id,window,version),(series,bounds) in cache.items():
        positions=panel.dates.get_indexer(series.index)
        values=series.to_numpy()
        for execution,execution_label in EXECUTIONS.items():
            # Same common signal dates for both modes to permit paired comparison.
            eligible=np.isfinite(values)&np.isfinite(bounds[cfg.quantiles[0]].to_numpy())&(
                positions+max_h+1<n)
            coverage.append({"标的":symbol,"执行版本":execution_label,"因子版本":VERSIONS[version],"因子ID":factor_id,
                "窗口":window,"共同120D可评估信号日":int(eligible.sum()),
                "起点":series.index[eligible][0] if eligible.any() else None,
                "终点":series.index[eligible][-1] if eligible.any() else None})
            for q,threshold in bounds.items():
                low=q<=10
                triggered=np.isfinite(values)&np.isfinite(threshold.to_numpy())&(
                    values<=threshold.to_numpy() if low else values>=threshold.to_numpy())
                triggered_positions=positions[triggered]
                base={"标的":symbol,"执行版本":execution_label,"执行代码":execution,
                      "因子版本":VERSIONS[version],
                      "因子ID":factor_id,"因子名称":FACTOR_NAMES[factor_id],
                      "窗口":window,"分位数":q,"方向":event_direction(factor_id,low)}
                common_row=base.copy()
                available_row=base.copy()
                common_positions=triggered_positions[triggered_positions+max_h+1<n]
                common_row["触发次数"]=len(common_positions)
                common_row["120D非重叠次数"]=nonoverlap_count(common_positions,max_h,execution)
                common_row["首个信号日"]=panel.dates[common_positions[0]] if len(common_positions) else None
                common_row["最后信号日"]=panel.dates[common_positions[-1]] if len(common_positions) else None
                for h in cfg.horizons:
                    for row,selected,per_h in ((common_row,common_positions,False),
                                                (available_row,triggered_positions,True)):
                        valid,entry,exit_=event_prices(panel,selected,h,execution)
                        returns=exit_/entry-1
                        win=event_wins(factor_id,low,returns)
                        row[f"{h}D胜率"]=float(win.mean()) if len(win) else None
                        row[f"{h}D收益率"]=float(returns.mean()) if len(returns) else None
                        if per_h:
                            row[f"{h}D触发次数"]=len(returns)
                            row[f"{h}D非重叠次数"]=nonoverlap_count(valid,h,execution)
                common_row["样本标识"]=("触发<20" if len(common_positions)<20 else
                                       "120D非重叠<3" if common_row["120D非重叠次数"]<3 else "探索性")
                common.append(common_row)
                available.append(available_row)
    return common,available,coverage


def target_position(signal_date: str,previous: float,series: pd.Series,
                    q10: pd.Series,q90: pd.Series,rule: str) -> float:
    if signal_date not in series.index or not np.isfinite(series.loc[signal_date]):
        return previous
    value=series.loc[signal_date]
    if rule=="正值持有":
        return float(value>0)
    if np.isfinite(q10.loc[signal_date]) and np.isfinite(q90.loc[signal_date]):
        if rule=="低10%买入_高10%卖出":
            if value<=q10.loc[signal_date]: return 1.0
            if value>=q90.loc[signal_date]: return 0.0
        if rule=="高10%买入_低10%卖出":
            if value>=q90.loc[signal_date]: return 1.0
            if value<=q10.loc[signal_date]: return 0.0
    return previous


def portfolio_study(symbol: str,panel: MarketPanel,cfg: StudyConfig,cache):
    records=[]
    for factor_id in FACTOR_NAMES:
      for window in cfg.windows:
        for version,version_label in VERSIONS.items():
          series,bounds=cache[(factor_id,window,version)]
          rules=("正值持有","低10%买入_高10%卖出")
          if factor_id=="SKEW_DIFF": rules+=("高10%买入_低10%卖出",)
          for rule in rules:
            for execution,label in EXECUTIONS.items():
                prev_position=0.0
                pnl=[]
                for j in range(1,len(panel.dates)):
                    target=target_position(panel.dates[j-1],prev_position,
                                           series,bounds[10],bounds[90],rule)
                    overnight=panel.daily_open[j]/panel.close[j-1]-1
                    intraday=panel.close[j]/panel.daily_open[j]-1
                    market=panel.close[j]/panel.close[j-1]-1
                    turnover=abs(target-prev_position)
                    if execution=="next_open":
                        # Gap is held at the old position; today's signal is traded
                        # at today's open and earns today's open-to-close return.
                        gross=(1+prev_position*overnight)*(1+target*intraday)-1
                        active=target
                    else:
                        # Trade at today's close; new position first earns price
                        # return over tomorrow's close-to-close interval.
                        gross=prev_position*market
                        active=prev_position
                    net=gross-cfg.fee_bp*1e-4*turnover
                    pnl.append((panel.dates[j],net,market,active,turnover))
                    prev_position=target
                df=pd.DataFrame(pnl,columns=["date","return","market","position","turnover"])
                df=df[df.date.str.startswith(str(cfg.portfolio_year))]
                if df.empty: continue
                r=df["return"].to_numpy()
                curve=np.cumprod(1+r)
                high=np.maximum.accumulate(np.r_[1.0,curve])[1:]
                sd=np.std(r,ddof=1)
                records.append({"标的":symbol,"执行版本":label,"执行代码":execution,
                    "因子版本":version_label,
                    "因子ID":factor_id,"窗口":window,"规则":rule,
                    "收益年份":cfg.portfolio_year,"收益日数":len(df),
                    "持仓比例":float(df.position.mean()),"换手单位":float(df.turnover.sum()),
                    "费率每腿bp":cfg.fee_bp,"组合收益率":float(curve[-1]-1),
                    "年化Sharpe":float(r.mean()/sd*np.sqrt(252)) if sd>0 else None,
                    "最大回撤":float((curve/high-1).min()),
                    "同期原始买入持有收益":float(np.prod(1+df.market.to_numpy())-1)})
    return records


def nw_slope_t(x: np.ndarray,y: np.ndarray,max_lag: int) -> float | None:
    """OLS slope t-stat with Bartlett Newey-West covariance, intercept included."""
    n=len(x)
    if n<30 or np.std(x)==0: return None
    design=np.column_stack((np.ones(n),x))
    inv=np.linalg.inv(design.T@design)
    beta=inv@(design.T@y)
    scores=design*(y-design@beta)[:,None]
    meat=scores.T@scores
    for lag in range(1,min(max_lag,n-1)+1):
        cross=scores[lag:].T@scores[:-lag]
        meat+=(1-lag/(max_lag+1))*(cross+cross.T)
    var=(inv@meat@inv)[1,1]
    return float(beta[1]/np.sqrt(var)) if var>0 else None


def predictive_metrics(symbol: str,panel: MarketPanel,cfg: StudyConfig,cache):
    """Time-series Pearson/rank IC and overlap-aware slope t-stat, no fitting."""
    rows=[]
    for (factor_id,window,version),(series,_) in cache.items():
        positions=panel.dates.get_indexer(series.index)
        values=series.to_numpy()
        for execution,label in EXECUTIONS.items():
            for h in cfg.horizons:
                valid,entries,exits=event_prices(panel,positions,h,execution)
                xs=values[np.searchsorted(positions,valid)]
                ys=exits/entries-1
                mask=np.isfinite(xs)&np.isfinite(ys)
                xs,ys=xs[mask],ys[mask]
                pearson=float(np.corrcoef(xs,ys)[0,1]) if len(xs)>2 and np.std(xs)>0 and np.std(ys)>0 else None
                ranks_x=pd.Series(xs).rank().to_numpy()
                ranks_y=pd.Series(ys).rank().to_numpy()
                rank_ic=float(np.corrcoef(ranks_x,ranks_y)[0,1]) if len(xs)>2 and np.std(ranks_x)>0 and np.std(ranks_y)>0 else None
                rows.append({"标的":symbol,"执行版本":label,"因子版本":VERSIONS[version],"因子ID":factor_id,
                    "窗口":window,"期限":h,"可评估信号日":len(xs),"Pearson_IC":pearson,
                    "Spearman_RankIC":rank_ic,"HAC斜率t值":nw_slope_t(xs,ys,h),
                    "平均原始收益":float(ys.mean()) if len(ys) else None})
    return rows


def yearly_predictive_metrics(symbol: str,panel: MarketPanel,cfg: StudyConfig,cache):
    """Diagnostic IC by signal year; future exits may fall in the next year."""
    rows=[]
    for (factor_id,window,version),(series,_) in cache.items():
        positions=panel.dates.get_indexer(series.index)
        for execution,label in EXECUTIONS.items():
            for h in cfg.horizons:
                valid,entries,exits=event_prices(panel,positions,h,execution)
                values=series.to_numpy()[np.searchsorted(positions,valid)]
                returns=exits/entries-1
                years=np.array([panel.dates[i][:4] for i in valid])
                for year in sorted(set(years)):
                    take=(years==year)&np.isfinite(values)&np.isfinite(returns)
                    x,y=values[take],returns[take]
                    corr=float(np.corrcoef(x,y)[0,1]) if len(x)>2 and np.std(x)>0 and np.std(y)>0 else None
                    ranks_x=pd.Series(x).rank().to_numpy()
                    ranks_y=pd.Series(y).rank().to_numpy()
                    rank_corr=float(np.corrcoef(ranks_x,ranks_y)[0,1]) if len(x)>2 and np.std(ranks_x)>0 and np.std(ranks_y)>0 else None
                    rows.append({"标的":symbol,"执行版本":label,"因子版本":VERSIONS[version],
                                 "因子ID":factor_id,"窗口":window,"期限":h,
                                 "信号年份":int(year),"可评估信号日":len(x),
                                 "Pearson_IC":corr,"Spearman_RankIC":rank_corr})
    return rows


def orthog_diagnostics(symbol: str,panel: MarketPanel,cfg: StudyConfig,cache):
    prices=pd.Series(panel.close,index=panel.dates)
    rows=[]
    for factor_id in FACTOR_NAMES:
        for k in cfg.windows:
            raw=cache[(factor_id,k,"raw_matched")][0]
            residual=cache[(factor_id,k,"orthog")][0]
            past=prices.div(prices.shift(k)).sub(1).reindex(raw.index)
            valid=raw.notna()&residual.notna()&past.notna()
            rows.append({"标的":symbol,"因子ID":factor_id,"窗口":k,
                "匹配信号日":int(valid.sum()),
                "原始因子与过去K日收益相关":float(raw[valid].corr(past[valid])) if valid.sum()>2 else None,
                "滚动残差与过去K日收益相关":float(residual[valid].corr(past[valid])) if valid.sum()>2 else None,
                "残差首日":raw.index[valid][0] if valid.any() else None,
                "残差末日":raw.index[valid][-1] if valid.any() else None})
    return rows


def run(inputs: dict[str,str],cfg: StudyConfig) -> dict:
    if not 0<cfg.orthog_min_periods<=cfg.orthog_lookback:
        raise ValueError("orthog_min_periods must be positive and <= orthog_lookback")
    if any(fid not in FACTOR_NAMES for fid in cfg.layer_factor_ids):
        raise ValueError("layer_factor_ids must exist in FACTOR_NAMES")
    if any(n<2 for n in cfg.bucket_counts):
        raise ValueError("bucket_counts must be >= 2")
    common,available,coverage,portfolios,predictive,yearly,diagnostics=[],[],[],[],[],[],[]
    layers,layer_summary,layer_years=[],[],[]
    source_counts={}
    for symbol,path in inputs.items():
        panel=load_market(path,cfg)
        cache=make_factor_cache(panel,cfg)
        c,a,v=event_study(symbol,panel,cfg,cache)
        common.extend(c);available.extend(a);coverage.extend(v)
        portfolios.extend(portfolio_study(symbol,panel,cfg,cache))
        predictive.extend(predictive_metrics(symbol,panel,cfg,cache))
        yearly.extend(yearly_predictive_metrics(symbol,panel,cfg,cache))
        diagnostics.extend(orthog_diagnostics(symbol,panel,cfg,cache))
        l,s,y=bucket_study(symbol,panel,cfg,cache)
        layers.extend(l);layer_summary.extend(s);layer_years.extend(y)
        source_counts[symbol]={"bars":panel.input_bars,"sessions":len(panel.dates),
                               "start":panel.dates[0],"end":panel.dates[-1]}
    ranked=sorted(common,key=lambda r:(r["90D胜率"] if r["90D胜率"] is not None else -1,
                                     r["120D胜率"] if r["120D胜率"] is not None else -1,
                                     r["触发次数"]),reverse=True)
    return {"source":"https://github.com/Aryan092/fds-indx-data",
        "signal_time":"end of t; checkpoints <= 15:30 on full sessions",
        "next_open":"entry O[t+1], exit C[t+h]; h=1 earns on t+1",
        "next_close":"entry C[t+1], exit C[t+1+h]; h=1 earns on t+2",
        "paired_common":"all horizons observable in both execution modes; t+1+max_h < N",
        "orthogonalization":"F[t] - (alpha[t-1]+beta[t-1]*(C[t]/C[t-K]-1)); OLS intercept/slope from up to 120 preceding full-session pairs, minimum 60; residual thresholds recalibrated on preceding residuals",
        "data_coverage":source_counts,"horizons":cfg.horizons,"windows":cfg.windows,
        "quantiles":cfg.quantiles,"coverage":coverage,"sensitivity":common,
        "available_horizon":available,"ranked":ranked,"portfolio":portfolios,
        "predictive":predictive,"yearly_predictive":yearly,
        "orthog_diagnostics":diagnostics,
        "bucket_layers":layers,"bucket_summary":layer_summary,"bucket_yearly":layer_years}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spy",required=True)
    parser.add_argument("--qqq",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--fee-bp",type=float,default=1.0)
    parser.add_argument("--orthog-lookback",type=int,default=120)
    parser.add_argument("--orthog-min-periods",type=int,default=60)
    parser.add_argument("--layer-factor",action="append",default=None,
                        help="factor ID to layer; repeat for multiple IDs (default SKEW_DIFF)")
    parser.add_argument("--early-close",action="append",default=None,
                        help="YYYY-MM-DD; repeat for all early closes if overriding defaults")
    args=parser.parse_args()
    overrides={"fee_bp":args.fee_bp,"orthog_lookback":args.orthog_lookback,
               "orthog_min_periods":args.orthog_min_periods}
    if args.early_close is not None: overrides["early_closes"]=tuple(args.early_close)
    if args.layer_factor is not None: overrides["layer_factor_ids"]=tuple(args.layer_factor)
    cfg=StudyConfig(**overrides)
    result=run({"SPY":args.spy,"QQQ":args.qqq},cfg)
    with open(args.output,"w",encoding="utf8") as out:
        json.dump(result,out,ensure_ascii=False,indent=2,allow_nan=False)
    print(args.output,"common rows",len(result["sensitivity"]),"portfolio rows",len(result["portfolio"]))
