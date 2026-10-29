"""日更拥挤度(docs/PROPOSAL_crowding_daily.md §1)。
每天开盘前用截至 t−1 收盘的数据,把最近 61×21 个交易日切成"月块",按 Feng(2027)的月频公式计算;MOM 用 French 已发布部分 + MTUM/SPY 拟合的实时尾部拼接。
python3 crowding_daily.py   → 自检 + data/_crowding_daily.pkl
"""
from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crowding_index as CX

RAW, DATA = CX.RAW, CX.DATA
B, NB, NRUN = 21, 61, 12           # 块长、块数(60 对 + 1)、run-up 块数
W_STD, W_STATE = 2520, 1260
PUB_LAG = pd.Timedelta(days=30)


def load_daily():
    rd = lambda f: (lambda d: d.set_axis(pd.to_datetime(d.index).tz_localize(None)))(pd.read_parquet(RAW / "yahoo" / f))
    ndx = rd("IDX_NDX.parquet")["close"]; cal = ndx.index
    lr = np.log(ndx).diff()
    vix = pd.read_parquet(RAW / "cboe" / "VIX_long.parquet")["close"]; vix.index = pd.to_datetime(vix.index).tz_localize(None); vix = vix.reindex(cal).ffill()
    tsm = np.sign(lr.rolling(252).sum().shift(1)) * lr
    mom = CX._french("F-F_Momentum_Factor_daily.csv", ["MOM"]).MOM.reindex(cal)
    mtum = rd("MTUM.parquet")["close"].reindex(cal).pct_change(); spy = rd("SPY.parquet")["close"].reindex(cal).pct_change()
    return pd.DataFrame({"TSM": tsm, "MKT": lr, "MOM": mom, "MTUM": mtum, "SPY": spy, "VIX": vix}, index=cal)


def _stats(f, v):
    """f: 至少 NB*B 个日收益(末尾 = t−1);v: 对应日的 VIX。返回 (run, ar, beta)。"""
    f, v = f[-NB * B:], v[-NB * B:]
    if len(f) < NB * B or not (np.isfinite(f).all() and np.isfinite(v).all()): return np.nan, np.nan, np.nan
    fb = f.reshape(NB, B).sum(1); vb = v.reshape(NB, B)[:, -1]; dvb = np.diff(vb); fb1 = fb[1:]
    run = fb1[-NRUN:].sum(); ar = CX._ols_slope(fb1[1:], fb1[:-1])
    beta = CX._ols_slope(fb1, dvb); return run, ar, beta


def stat_series(D, name, eval_idx, vintage=None):
    """对每个评估日(日历下标 i,用 [:i] 即截至 t−1 的数据)计算三个统计量。vintage(i) 返回该日可得的 f 序列(MOM 用)。"""
    f_all = D[name].to_numpy(float); v_all = D.VIX.to_numpy(float); out = np.full((len(eval_idx), 3), np.nan)
    for j, i in enumerate(eval_idx):
        f = vintage(i) if vintage is not None else f_all[:i]
        n = len(f); out[j] = _stats(f, v_all[:n])                     # f 覆盖日历 [0, n),VIX 同步对齐
    return out


def mom_vintage_factory(D):
    cal = D.index; mom = D.MOM.to_numpy(float); mt = D.MTUM.to_numpy(float); sp = D.SPY.to_numpy(float)
    me = pd.DatetimeIndex(sorted(set(cal + pd.offsets.MonthEnd(0))))
    known_pos = np.where(np.isfinite(mom))[0]; diag = []
    def vint(i):
        t = cal[i]; E = me[me + PUB_LAG <= t]
        if len(E) == 0: return mom[:0]
        e_pos = int(np.searchsorted(cal, E[-1], side="right"))          # French 已知:[0, e_pos)
        e_pos = min(e_pos, i)
        f = mom[:e_pos].copy()
        if e_pos >= i: return f[:i]
        lo = max(0, e_pos - 504); ok = np.arange(lo, e_pos); ok = ok[np.isfinite(mom[ok]) & np.isfinite(mt[ok]) & np.isfinite(sp[ok])]
        tail = np.arange(e_pos, i)
        if len(ok) < 120 or not (np.isfinite(mt[tail]).all() and np.isfinite(sp[tail]).all()): return f       # 无法补齐 → 截止到 E
        A = np.column_stack([np.ones(len(ok)), mt[ok], sp[ok]]); beta = np.linalg.lstsq(A, mom[ok], rcond=None)[0]
        pred = beta[0] + beta[1] * mt[tail] + beta[2] * sp[tail]; diag.append((tail, pred))
        # 拼接时 VIX 必须同步对齐到同样的长度:返回的 f 长度 = i
        return np.r_[f, pred]
    return vint, diag


def finish(stats, eval_dates):
    S = pd.DataFrame(stats, index=eval_dates, columns=["run", "ar", "beta"]); n = len(S); Z = {}
    for c in ("run", "ar", "beta"):
        x = S[c].to_numpy(); z = np.full(n, np.nan)
        for t in range(W_STD, n):
            h = x[t - W_STD:t]
            if np.isfinite(h).all() and np.isfinite(x[t]) and h.std(ddof=1) > 0: z[t] = (x[t] - h.mean()) / h.std(ddof=1)
        Z[c] = z
    Cv = (Z["run"] + Z["ar"] - Z["beta"]) / 3.0; st = np.full(n, np.nan)
    for t in range(W_STATE, n):
        h = Cv[t - W_STATE:t]
        if np.isfinite(h).all() and np.isfinite(Cv[t]):
            q1, q2 = np.quantile(h, [1 / 3, 2 / 3]); st[t] = 0 if Cv[t] <= q1 else (2 if Cv[t] >= q2 else 1)
    S["C"], S["state"] = Cv, st
    return S


def build(D=None, start="1994-01-01"):
    D = load_daily() if D is None else D; cal = D.index; ev = np.where(cal >= start)[0]
    out = {}
    for nm in ("TSM", "MKT"):
        out[nm] = finish(stat_series(D, nm, ev), cal[ev])
    vint, diag = mom_vintage_factory(D)
    out["MOM"] = finish(stat_series(D, "MOM", ev, vintage=vint), cal[ev])
    return D, out, diag


def main():
    t0 = time.time(); pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:8.3f}")
    D, CD, diag = build(); print(f"[构造] {time.time()-t0:.0f}s;评估日 {len(CD['TSM'])}")
    # 拼接质量:拟合尾部 vs 随后发布的 French 日度 MOM(只用最后一次出现的预测,避免重复计数)
    last = {}
    for tail, pred in diag:
        for k, p in zip(tail, pred): last[k] = p
    k = np.array(sorted(last)); p = np.array([last[x] for x in k]); m_ = D.MOM.to_numpy()[k]; ok = np.isfinite(m_)
    print(f"[拼接] MTUM/SPY 拟合的实时尾部 vs 随后发布的 French MOM:{ok.sum()} 天,相关 {np.corrcoef(p[ok], m_[ok])[0, 1]:.3f},拟合/实际标准差比 {p[ok].std()/m_[ok].std():.2f}")
    # 投毒:切点之后所有输入 ×(−3)
    worst = 0.0
    for cut in ("2016-06-30", "2022-09-30"):
        D2 = D.copy(); m = D2.index > cut; D2.loc[m, ["TSM", "MKT", "MOM", "MTUM", "SPY"]] *= -3.0; D2.loc[m, "VIX"] = D2.loc[m, "VIX"] * 2 + 7
        _, CD2, _ = build(D2); nxt = D.index[D.index > cut][0]
        for nm in CD:
            a, b_ = CD[nm].loc[:nxt], CD2[nm].loc[:nxt]
            worst = max(worst, float(np.nanmax(np.abs(a.to_numpy() - b_.to_numpy()))), float((a.isna() != b_.isna()).to_numpy().sum()))
    print(f"[投毒自检] 2 个切点:截至切点次日(该日只用到切点收盘)的所有列最大差 {worst:.2e}(须为 0)"); assert worst == 0.0
    # 与月频的一致率与切换提前量
    Mo, CI = CX.build()
    for nm, lag in (("TSM", 0), ("MKT", 0), ("MOM", 1)):
        dd = CD[nm].loc["2010":]; mon = CX.to_daily(CI[nm], dd.index, extra_lag=lag).state
        ok = dd.state.notna() & mon.notna(); agree = float((dd.state[ok] == mon[ok]).mean())
        hi_d, hi_m = (dd.state == 2), (mon == 2); on_d = dd.index[(hi_d & ~hi_d.shift(1, fill_value=False))]; on_m = dd.index[(hi_m & ~hi_m.shift(1, fill_value=False))]
        leads = [((on_m[on_m >= x][0] - x).days) for x in on_d if (on_m >= x).any() and (on_m[on_m >= x][0] - x).days < 120]
        print(f"[{nm}] 2010+ 日更 vs 月频状态一致率 {agree:.1%};日更进入高拥挤 {len(on_d)} 次(月频 {len(on_m)} 次);日更比月频提前进入高拥挤的中位天数 {np.median(leads) if leads else np.nan:.0f}(n={len(leads)})")
    print("日更 C 之间的相关(2010+):"); print(pd.DataFrame({k: v.C for k, v in CD.items()}).loc["2010":].corr().round(2).to_string())
    pd.to_pickle(dict(D=D, CD=CD), DATA / "_crowding_daily.pkl"); print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
