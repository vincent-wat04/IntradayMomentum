"""事件日:不入场 / 推迟入场 是否普遍更好?(对所有事件与结果分类)+ CPI 缺口回补 overlay 能加多少利润。
探索性(事后),见 docs/PROPOSAL_event_delay.md。python3 event_delay_study.py"""
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
import disp_pred_study as DP
import lowvol_avoidance_study as L
import macro_events_study as M
import round3_common as C
import trend_rv_sleeve_study as T
from event_calendar import FOMC_DECISION, event_frame
from etft.intraday.bars import load_bars
from sens_rv_weights import pct_w

OUT, TAB = L.OUT, L.TAB
STARTS = ["10:00", "10:30", "11:00", "11:30", "12:00", "13:00", "14:00", "14:30", "15:00"]
NB = 3000


def curves(b):
    p = OUT / "sleeve_start_curves.pkl"
    if p.exists():
        d = pd.read_pickle(p)
        if set(STARTS) <= set(d): return d
    scale = DP.tqqq_scale(b); d = {}
    for s in STARTS:
        d[s] = DP.sleeve_start(b, scale, s); print(f"  ≥{s}: 日均 {d[s].mean()*1e4:.2f}bp", flush=True)
    pd.to_pickle(d, p); return d


def main():
    pd.set_option("display.width", 260, "display.max_rows", 400, "display.float_format", lambda x: f"{x:8.2f}")
    b = load_bars("QQQ"); days = b.days; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    Cv = curves(b); base = pd.read_pickle(OUT / "sleeve_anatomy_curve.pkl"); base.index = pd.to_datetime(base.index); base = base.reindex(days).fillna(0.0)
    print(f"校验:≥10:00 曲线 vs 实盘基线 最大差 {float((Cv['10:00'] - base).abs().max())*1e4:.5f}bp")
    EV, adp_d, _ = M.load_events(days); E0 = event_frame(days, b.last_idx.to_numpy())
    win = (days >= "2019-01-01") & (days <= "2025-12-31")                 # 宏观日历覆盖期
    sets = {k: pd.DatetimeIndex(v.index) for k, v in EV.items()}
    sets["FOMC"] = pd.DatetimeIndex([d for d in FOMC_DECISION if d in set(days)])
    sets["ADP"] = pd.DatetimeIndex([d for d in adp_d if d in set(days)])
    for c_, nm in (("e_opex", "OPEX"), ("e_quad", "季度OPEX"), ("e_month_end", "月末两日"), ("e_pre_holiday", "节前"), ("e_post_holiday", "节后"), ("e_fomc_eve", "FOMC前一日"), ("e_fomc_post", "FOMC后一日")):
        sets[nm] = days[E0[c_].to_numpy() == 1]
    macro = set().union(*[set(sets[k]) for k in ("CPI", "NFP", "RETAIL", "ISM", "ADP", "FOMC")]); sets["任一宏观(CPI/NFP/零售/ISM/ADP/FOMC)"] = pd.DatetimeIndex(sorted(macro))
    anyev = macro | set().union(*[set(sets[k]) for k in ("OPEX", "月末两日", "节前", "节后", "FOMC前一日", "FOMC后一日")])
    ok = td & full & pd.Series(win, index=days)
    ctrl = ok & ~pd.Series([d in anyev for d in days], index=days)
    print("事件天数(周二至周五、完整日、2019–2025):", {k: int(ok[v].sum()) for k, v in sets.items()}, "| 非事件对照日", int(ctrl.sum()))
    rg = np.random.default_rng(0); yrs = pd.Series(days.year, index=days)
    variants = {**{f"推迟≥{s}": Cv[s] for s in STARTS[1:]}, "不入场": pd.Series(0.0, index=days)}
    # =========================================================== 1) 事件 × 方案 矩阵(相对实盘 V0 的日均差 bp)
    def diff_stats(dts):
        dts = [d for d in dts if ok[d]]; out = {"n": len(dts), "V0日均bp": 1e4 * base.loc[dts].mean()}
        wd = pd.Series(pd.DatetimeIndex(dts).dayofweek).value_counts()
        for vn, s in variants.items():
            dd = (s - base).loc[dts]; bs = [dd.iloc[rg.integers(0, len(dd), len(dd))].mean() for _ in range(1000)]
            # 『推迟本身』的一般代价:同星期几的非事件日上同样操作的日均差
            cm = sum((s - base)[ctrl & (days.dayofweek == w)].mean() * n for w, n in wd.items()) / wd.sum()
            out[vn] = 1e4 * dd.mean(); out[vn + "_lo"] = 1e4 * np.percentile(bs, 2.5); out[vn + "_hi"] = 1e4 * np.percentile(bs, 97.5); out[vn + "_对照"] = 1e4 * cm
            dv = [d for d in dts if d.year <= 2022]; tv = [d for d in dts if d.year >= 2023]
            out[vn + "_dev"] = 1e4 * (s - base).loc[dv].mean() if dv else np.nan; out[vn + "_test"] = 1e4 * (s - base).loc[tv].mean() if tv else np.nan
        return out
    rows = [dict(事件=k, **diff_stats(v)) for k, v in sets.items()]
    rows.append(dict(事件="非事件日(对照)", **diff_stats(list(days[ctrl]))))
    Mx = pd.DataFrame(rows); Mx.to_csv(TAB / "event_delay_matrix.csv", index=False)
    show = ["事件", "n", "V0日均bp"] + [v for v in variants]
    print("\n1) 事件日 × 方案:相对实盘的日均差(bp;正 = 该方案更好)。最后一行 = 非事件日上做同样操作的差(推迟本身的代价)"); print(Mx[show].to_string(index=False))
    print("\n   『不入场』的 95% 区间与 dev/test:"); print(Mx[["事件", "n", "V0日均bp", "不入场", "不入场_lo", "不入场_hi", "不入场_dev", "不入场_test"]].to_string(index=False))
    best = []
    for _, r in Mx.iterrows():
        vals = {v: r[v] - r[v + "_对照"] for v in variants}; vb = max(vals, key=vals.get)
        best.append(dict(事件=r.事件, n=r.n, 最优方案=vb, 相对V0=r[vb], 扣除推迟一般代价后=vals[vb], lo=r[vb + "_lo"], hi=r[vb + "_hi"], dev=r[vb + "_dev"], test=r[vb + "_test"]))
    print("\n   各事件的事后最优方案(事后挑选,须谨慎):"); print(pd.DataFrame(best).to_string(index=False))
    # =========================================================== 2) 按结果分类
    rows = []
    for nm in ("CPI", "NFP", "RETAIL", "ISM"):
        ev = EV[nm]
        for kind, kn in (("sf", "对预期"), ("sp", "对前值")):
            lb = ev[kind].map(lambda s: M.lab(nm, kind, s))
            for cls in sorted(lb.unique()):
                dts = [d for d in lb.index[lb == cls] if ok.get(d, False)]
                if len(dts) < 5: continue
                r_ = dict(事件=nm, 方向=kn, 类别=cls, n=len(dts), V0日均bp=1e4 * base.loc[dts].mean())
                for vn in ("推迟≥10:30", "推迟≥11:00", "推迟≥12:00", "不入场"): r_[vn] = 1e4 * (variants[vn] - base).loc[dts].mean()
                rows.append(r_)
    Kc = pd.DataFrame(rows); print("\n2) 按新闻结果分类:相对实盘的日均差(bp)"); print(Kc.to_string(index=False)); Kc.to_csv(TAB / "event_delay_by_class.csv", index=False)
    # =========================================================== 3) 组合层:几条『事件规则』能加多少(2019–2025)
    bh = pd.read_pickle(OUT / "strategy_comparison_bh.pkl").dropna(); bh.index = pd.to_datetime(bh.index)
    px_all = C.load_close("yahoo/QQQ.parquet"); p_all = pct_w(px_all.pct_change().rolling(21).std(), 504)
    port = lambda sl: T.build(bh, sl.reindex(bh.index).fillna(0.0), (1.0, 1.0), (("lin", .5, 1.5), None, 1.0), 1.0, px_all, p_all)[0]
    pwin = lambda x: x[(x.index >= "2019-01-01") & (x.index <= "2025-12-31")]
    base_p = port(base)
    def apply(rule):
        s = base.copy()
        for dts, vn in rule:
            m = pd.Index(dts).intersection(days); s.loc[m] = variants[vn].loc[m] if vn != "V0" else base.loc[m]
        return s
    # CPI overlay(缺口回补:热→多,冷→空;开盘入场持有到收盘;成本 1bp/边)
    O = b.day_open.to_numpy(); cl = b.day_close.to_numpy(); oc = pd.Series(cl / O - 1, index=days)
    cpi = EV["CPI"]; sg = np.sign(cpi.sf); sg = sg[sg != 0]
    ov_unit = pd.Series(0.0, index=days); mm = sg.index.intersection(days); ov_unit.loc[mm] = sg.loc[mm] * oc.loc[mm] - 2e-4
    rules = {"实盘": [], "FOMC 日推迟≥14:30": [(sets["FOMC"], "推迟≥14:30")], "FOMC 日不入场": [(sets["FOMC"], "不入场")],
             "CPI 日不入场": [(sets["CPI"], "不入场")], "任一宏观日不入场": [(sets["任一宏观(CPI/NFP/零售/ISM/ADP/FOMC)"], "不入场")],
             "任一宏观日推迟≥11:00": [(sets["任一宏观(CPI/NFP/零售/ISM/ADP/FOMC)"], "推迟≥11:00")]}
    rows, pan = [], []
    for rn, rule in rules.items():
        s = apply(rule)
        for f in ((0.0,) if rn not in ("实盘", "CPI 日不入场", "FOMC 日推迟≥14:30") else (0.0, 0.25, 0.5, 1.0)):
            p = port(s); ov = (f * ov_unit).reindex(p.index).fillna(0.0); x = pwin(p + ov); x0 = pwin(base_p)
            d, dist = T.boot_dsr(x, x0, nb=1500, L=10, seed=0)
            nm = rn + (f" + CPI回补overlay {f:g}×" if f > 0 else "")
            rows.append(dict(方案=nm, 年化增加bp=1e4 * 252 * (x.mean() - x0.mean()), CAGR=DP.panel(x)["CAGR"], ΔCAGR_pp=100 * (DP.panel(x)["CAGR"] - DP.panel(x0)["CAGR"]), Sharpe=L.sr(x), ΔSR=d, lo=np.percentile(dist, 2.5), hi=np.percentile(dist, 97.5),
                             MDD=DP.panel(x)["MDD"], 最差日bp=DP.panel(x)["最差日bp"], dev年化bp=1e4 * 252 * (x[x.index.year <= 2022].mean() - x0[x0.index.year <= 2022].mean()), test年化bp=1e4 * 252 * (x[x.index.year >= 2023].mean() - x0[x0.index.year >= 2023].mean())))
            pan.append(DP.panel(x, nm))
    R3 = pd.DataFrame(rows); print("\n3) 组合层(L1 RV[0.5,1.5] + Sleeve;2019–2025):事件规则与 CPI 回补 overlay 能加多少"); print(R3.to_string(index=False)); R3.to_csv(TAB / "event_rules_portfolio.csv", index=False)
    Pn = pd.DataFrame(pan); print("\n   分布面板:"); print(Pn.to_string(index=False))
    # CPI overlay 单独账本
    x = ov_unit[ov_unit != 0]; print(f"\n4) CPI 回补 overlay 单独(1× 名义,每笔扣 2bp 往返):{len(x)} 笔,每笔均 {1e4*x.mean():+.1f}bp,中位 {1e4*x.median():+.1f}bp,胜率 {np.mean(x>0):.2f},最差 {1e4*x.min():+.0f}bp,最好 {1e4*x.max():+.0f}bp;"
          f"年化(÷7 年)≈ {1e4*x.sum()/7:+.0f}bp/年;去掉最好 3 笔后每笔 {1e4*x.drop(x.nlargest(3).index).mean():+.1f}bp")
    print("   分年(笔数, 合计 bp):", {int(y): (int(g.size), round(float(1e4 * g.sum()), 0)) for y, g in x.groupby(x.index.year)})
    print("   CPI 日实盘 Sleeve 均 bp:", round(1e4 * base.loc[mm].mean(), 1), "| overlay 与 Sleeve 在 CPI 日的相关:", round(float(np.corrcoef(ov_unit.loc[mm], base.loc[mm])[0, 1]), 2))


if __name__ == "__main__":
    main()
