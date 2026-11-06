"""FOMC 决议日怎么交易最好 / 交易时点要不要偏移 —— docs/PROPOSAL_fomc_day.md(预注册)。python3 fomc_day_study.py"""
from __future__ import annotations

import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lowvol_avoidance_study as L
import sleeve_anatomy as SA
import trend_rv_sleeve_study as T
from audit import reimpl_noise as RI
from event_calendar import FOMC_DECISION
from etft.intraday.bars import load_bars

TAB, OUT = L.TAB, L.OUT
COST = 1.0                    # bp / 边(QQQ 规则)
NPERM = 2000


def seg_table(b, idx, cols, fo, ctrl):
    """分段收益 bp:返回 DataFrame(index=段, 列=FOMC 均值/区间/|ret|/对照均值/…)。"""
    c = b.c.to_numpy(); O = b.day_open.to_numpy(); cl = b.day_close.to_numpy()
    P = lambda t: c[:, idx[t]]
    prev = pd.Series(cl, index=b.days).shift(1).to_numpy()
    segs = {"隔夜(昨收→开)": (prev, O), "开盘→13:55": (O, P("13:55")), "13:55→14:05(冲击)": (P("13:55"), P("14:05")), "14:05→14:30": (P("14:05"), P("14:30")),
            "14:30→15:30": (P("14:30"), P("15:30")), "15:30→收盘": (P("15:30"), cl), "昨收→13:55(公告前漂移)": (prev, P("13:55")), "全日": (prev, cl)}
    rg = np.random.default_rng(0); rows = []
    for nm, (a, z) in segs.items():
        r = (z / a - 1) * 1e4; r = pd.Series(r, index=b.days)
        x, y = r[fo].dropna(), r[ctrl].dropna()
        bs = [x.iloc[rg.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        rows.append(dict(段=nm, FOMC均值=x.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), FOMC_abs=x.abs().mean(), 对照周三均值=y.mean(), 对照_abs=y.abs().mean(),
                         abs比=x.abs().mean() / y.abs().mean(), n=len(x)))
    return pd.DataFrame(rows)


def rule_pnls(b, idx):
    """所有日子的规则净 bp(成本 2×COST,往返)。返回 {配置: Series}。"""
    c, o = b.c.to_numpy(), b.o.to_numpy(); O = b.day_open.to_numpy(); cl = b.day_close.to_numpy(); days = b.days
    P = lambda t: c[:, idx[t]]; nxt = lambda t: o[:, idx[t] + 1]
    r1 = np.sign(P("14:05") / P("13:55") - 1); r2 = np.sign(P("14:30") / P("14:05") - 1)
    g = lambda s, a, z: s * (z / a - 1) * 1e4 - 2 * COST
    out = {"B1 公告后动量": g(r1, nxt("14:05"), cl), "B2 公告后回转": g(-r1, nxt("14:05"), cl), "B3 二次动量": g(r2, nxt("14:30"), cl), "B4 二次回转": g(-r2, nxt("14:30"), cl),
           "B5 公告前多头": g(np.ones(len(days)), O, P("13:55"))}
    return {k: pd.Series(v, index=days) for k, v in out.items()}


def sleeve_variant(b, days, fomc_set, scale, check_filter, phases=range(30), keep_only=True, sides="both"):
    """FOMC 日 Sleeve 的某个检查点方案;返回 (30 相位均值日收益 Series, {相位: F})。keep_only: 只模拟 FOMC 日(skip 其余日)。"""
    nonf = [d for d in days if d not in fomc_set]; rs, Fs = [], {}
    for off in phases:
        checks = [t for t in SA.chk(off) if check_filter(t)]
        if not checks:
            rs.append(pd.Series(0.0, index=days)); Fs[off] = pd.DataFrame(); continue
        rr, F = RI.simulate(b, checks=checks, sizing="voltarget", max_lev=4.0, reentry="hold", half_spread=0.005, commission=0.0045, skip_days=nonf, sides=sides, **SA.BASE)
        if len(F):
            F = F.copy(); F["day"] = pd.to_datetime(F.day); c_old = F.groupby("day").commission.sum().reindex(days).fillna(0.0)
            rr = rr + c_old - c_old * scale.reindex(days).fillna(1.0)
        rs.append(rr); Fs[off] = F
    return pd.concat(rs, axis=1).mean(axis=1), Fs


def trades_from_F(F):
    rows = []
    for d, g in F.groupby("day", sort=False):
        cur = None
        for _, r in g.iterrows():
            k = str(r.kind)
            if k.startswith("ENTRY"): cur = r
            elif cur is not None:
                rows.append(dict(day=d, side=int(np.sign(cur.shares)), t_in=str(cur.clock), t_out=str(r.clock), kind=k, pnl=float(cur.shares * (r.price - cur.price)) * 1e4)); cur = None
    return pd.DataFrame(rows)


def main():
    b = load_bars("QQQ"); days = b.days; cols = list(b.c.columns); idx = {t: cols.index(t) for t in ("13:55", "14:05", "14:30", "15:30", "14:00")}
    fomc = [d for d in FOMC_DECISION if d in set(days)]; fset = set(fomc); full = (b.last_idx.to_numpy() >= 389)
    fo = pd.Series([d in fset for d in days], index=days) & pd.Series(full, index=days)
    ctrl = pd.Series((days.dayofweek == 2), index=days) & ~pd.Series([d in fset for d in days], index=days) & pd.Series(full, index=days)
    yrs = pd.Series(days.year, index=days)
    print(f"FOMC 日 {int(fo.sum())}(dev {int((fo&(yrs<2023)).sum())} / test {int((fo&(yrs>=2023)).sum())});对照周三 {int(ctrl.sum())}")
    pd.set_option("display.width", 220, "display.float_format", lambda x: f"{x:8.2f}")
    # ---------------- A
    A = seg_table(b, idx, cols, fo, ctrl); print("\nA) 日内分段收益(bp):FOMC 日 vs 非 FOMC 周三"); print(A.to_string(index=False)); A.to_csv(TAB / "fomc_segments.csv", index=False)
    # ---------------- B
    R = rule_pnls(b, idx); rg = np.random.default_rng(1); pool = np.where(ctrl.to_numpy())[0]; nF = int(fo.sum())
    draws = rg.choice(pool, (NPERM, nF)); rows = []; nulls = {}
    for k, s in R.items():
        v = s.to_numpy(); nulls[k] = np.nanmean(v[draws], axis=1)
    famax = np.max(np.vstack(list(nulls.values())), axis=0)
    for k, s in R.items():
        x = s[fo].dropna(); xd, xt = s[fo & (yrs < 2023)].dropna(), s[fo & (yrs >= 2023)].dropna(); bs = [x.iloc[rg.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        rows.append(dict(配置=k, 净bp均值=x.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), dev=xd.mean(), test=xt.mean(), 胜率=(x > 0).mean(), SR=x.mean() / x.std() * np.sqrt(8) if x.std() > 0 else np.nan,
                         对照周三均值=s[ctrl].mean(), p_单=float(np.mean(nulls[k] >= x.mean())), p_族内max=float(np.mean(famax >= x.mean())), n=len(x)))
    B = pd.DataFrame(rows); print("\nB) FOMC 日 QQQ 规则(每边 1bp;p=相对非 FOMC 周三的伪事件置换,单侧)"); print(B.to_string(index=False)); B.to_csv(TAB / "fomc_rules.csv", index=False)
    # ---------------- C
    import yfinance as yf
    pt = yf.download("TQQQ", start="2018-10-01", end="2026-10-12", auto_adjust=False, progress=False, threads=False)["Close"]
    pt = pt.iloc[:, 0] if hasattr(pt, "columns") else pt; pt.index = pd.to_datetime(pt.index).tz_localize(None)
    sp = yf.Ticker("TQQQ").splits; sp.index = pd.to_datetime(sp.index).tz_localize(None); fsp = pd.Series(1.0, index=pt.index)
    for d, r in sp.items(): fsp[pt.index < d] *= r
    scale = (b.day_close / (pt * fsp).reindex(days).ffill()) / 3.0
    variants = {"V0 实盘": lambda t: True, "V2 去掉[13:55,14:30)": lambda t: not ("13:55" <= t < "14:30"), "V3 只≥14:30": lambda t: t >= "14:30", "V4 只≥15:00": lambda t: t >= "15:00"}
    res, Fv = {}, {}
    for nm, f in variants.items(): res[nm], Fv[nm] = sleeve_variant(b, days, fset, scale, f)
    res["V5 探索:FOMC日只做多"], Fv["V5 探索:FOMC日只做多"] = sleeve_variant(b, days, fset, scale, lambda t: True, sides="long")
    res["V1 FOMC日关闭"] = pd.Series(0.0, index=days)
    ret = pd.read_pickle(L.OUT / "sleeve_anatomy_curve.pkl"); ret.index = pd.to_datetime(ret.index); ret = ret.reindex(days).fillna(0.0)
    fm = pd.Series([d in fset for d in days], index=days)
    print(f"\n校验:V0 重算 vs sleeve_anatomy_curve 在 FOMC 日的最大绝对差 {float((res['V0 实盘'][fm] - ret[fm]).abs().max())*1e4:.4f} bp")
    td = pd.Series(days.dayofweek != 0, index=days); rg2 = np.random.default_rng(2)
    rows = []
    for nm, s in res.items():
        d = (s - res["V0 实盘"])[fm]; bs = [d.iloc[rg2.integers(0, len(d), len(d))].mean() for _ in range(3000)]
        full_s = ret.copy(); full_s[fm] = s[fm]
        dsr, dist = T.boot_dsr(full_s[td], ret[td], nb=2000, L=10, seed=0)
        rows.append(dict(方案=nm, FOMC日均bp=s[fm].mean() * 1e4, 总贡献bp=s[fm].sum() * 1e4, 配对差bp=d.mean() * 1e4, lo=np.percentile(bs, 2.5) * 1e4, hi=np.percentile(bs, 97.5) * 1e4,
                         dev差=d[fm & (yrs < 2023)].mean() * 1e4, test差=d[fm & (yrs >= 2023)].mean() * 1e4, 整条ΔSR=dsr, dSR_lo=np.percentile(dist, 2.5), dSR_hi=np.percentile(dist, 97.5)))
    C_ = pd.DataFrame(rows); print("\nC) Sleeve 在 FOMC 日的检查点方案(30 相位均值;配对差相对 V0,按 FOMC 日自助)"); print(C_.to_string(index=False)); C_.to_csv(TAB / "fomc_sleeve_variants.csv", index=False)
    # 公告时刻是否持仓
    T0 = pd.concat([trades_from_F(F).assign(phase=o) for o, F in Fv["V0 实盘"].items() if len(F)])
    T0["类别"] = np.where((T0.t_in < "14:00") & (T0.t_out >= "14:00"), "跨过公告(14:00 前入、后出)", np.where(T0.t_out < "14:00", "公告前已平仓", "公告后入场"))
    G = T0.groupby("类别").pnl.agg(["count", "mean", "sum"]).rename(columns={"count": "笔数(30相位合计)", "mean": "均bp(资本)", "sum": "合计bp"}); G["均bp/相位平均日"] = G["合计bp"] / 30
    print("\nV0 的成交按『是否跨过 14:00 公告』分类(30 相位的合计):"); print(G.to_string())
    print("跨过公告的成交按出场类型:"); print(T0[T0.类别.str.startswith("跨过")].groupby("kind").pnl.agg(["count", "mean"]).to_string())
    print("按入场方向(跨过公告):", T0[T0.类别.str.startswith("跨过")].groupby("side").pnl.agg(["count", "mean"]).to_dict())
    pd.to_pickle(dict(A=A, B=B, C=C_, res=res), OUT / "fomc_day.pkl")


if __name__ == "__main__":
    main()
