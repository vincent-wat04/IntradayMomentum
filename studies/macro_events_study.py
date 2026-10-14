"""CPI / NFP / 零售 / ISM 日内检验与新闻方向 —— docs/PROPOSAL_macro_events.md(预注册)。python3 macro_events_study.py"""
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
import paths as _P
from event_calendar import FOMC_DECISION, event_frame
from etft.intraday.bars import load_bars

TAB, OUT = L.TAB, L.OUT
COST = 1.0; NPERM = 2000
EC = _P.RAW / "econ_cal" / "ec.csv"


def load_events(days):
    d = pd.read_csv(EC); d = d[d.currency == "USD"].copy()
    et = pd.to_datetime(d.datetime).dt.tz_localize("UTC").dt.tz_convert("America/New_York"); d["date"] = et.dt.tz_localize(None).dt.normalize()
    for c in ("actual", "forecast", "previous"): d[c] = pd.to_numeric(d[c], errors="coerce")
    d["sf"] = np.sign(d.actual - d.forecast); d["sp"] = np.sign(d.actual - d.previous)
    dset = set(days); out = {}
    def pick(name): return d[d.event == name].set_index("date")
    cpi, core = pick("cpi-yoy"), pick("core-cpi-yoy"); idx = cpi.index.intersection(core.index)
    E = pd.DataFrame(index=idx); E["sf"] = cpi.loc[idx, "sf"] + core.loc[idx, "sf"]; E["sp"] = cpi.loc[idx, "sp"] + core.loc[idx, "sp"]; out["CPI"] = E
    for nm, ev in (("NFP", "nonfarm-payrolls"), ("RETAIL", "retail-sales-mom"), ("ISM", "ism-manufacturing-pmi")):
        x = pick(ev); out[nm] = pd.DataFrame({"sf": x.sf, "sp": x.sp})
    adp, clm = pick("adp-employment-change"), pick("initial-jobless-claims")
    for k in out: out[k] = out[k][~out[k].index.duplicated()].dropna(); out[k] = out[k][out[k].index.isin(dset) & (out[k].index >= "2019-01-01") & (out[k].index <= "2025-12-31")]
    return out, adp.index.unique(), clm.index.unique()


def lab(nm, kind, s):
    if nm == "CPI":
        if kind == "sf": return "hot" if s > 0 else ("cool" if s < 0 else "inline")
        return "rising" if s > 0 else ("falling" if s < 0 else "flat")
    if kind == "sf": return "strong/beat" if s > 0 else ("weak/miss" if s < 0 else "inline")
    return "up" if s > 0 else ("down" if s < 0 else "flat")


def mapped_sign(nm, s):            # S1/S2 预注册映射:CPI 热/上升→空;其余强/上→多
    return -np.sign(s) if nm == "CPI" else np.sign(s)


def main():
    pd.set_option("display.width", 240, "display.max_rows", 300, "display.float_format", lambda x: f"{x:8.2f}")
    b = load_bars("QQQ"); days = b.days; cols = list(b.c.columns); idx = {t: cols.index(t) for t in ("09:55", "10:00", "10:05")}
    c, o = b.c.to_numpy(), b.o.to_numpy(); O = b.day_open.to_numpy(); cl = b.day_close.to_numpy(); prev = b.day_close.shift(1).to_numpy(); full = b.last_idx.to_numpy() >= 389
    EV, adp_d, clm_d = load_events(days); yrs = pd.Series(days.year, index=days)
    fomc = set(FOMC_DECISION); allev = set().union(*[set(v.index) for v in EV.values()]) | fomc | set(adp_d)
    isev = pd.Series([d in allev for d in days], index=days); ok_ctrl = (~isev) & pd.Series(full, index=days) & (days >= "2019-01-01") & (days <= "2025-12-31")
    P = lambda t: c[:, idx[t]]
    seg = {"隔夜(昨收→开)": (prev, O), "开→10:00": (O, P("10:00")), "10:00→收盘": (P("10:00"), cl), "全日(开→收)": (O, cl), "昨收→收盘": (prev, cl)}
    S = pd.DataFrame({k: (z / a - 1) * 1e4 for k, (a, z) in seg.items()}, index=days); S["09:55→10:05"] = (P("10:05") / P("09:55") - 1) * 1e4; S["10:05→收盘"] = (cl / P("10:05") - 1) * 1e4
    print("事件天数(2019–2025):", {k: len(v) for k, v in EV.items()}, "| ADP", int(pd.Index(adp_d).isin(days).sum()), "| 对照日", int(ok_ctrl.sum()))
    rg = np.random.default_rng(11)
    # ===================================================== A) 形态
    rowsA = []
    for nm, ev in EV.items():
        dts = ev.index; wd = pd.Series(dts.dayofweek).value_counts()
        for sg in (["隔夜(昨收→开)", "开→10:00", "10:00→收盘", "全日(开→收)"] + (["09:55→10:05", "10:05→收盘"] if nm == "ISM" else [])):
            x = S.loc[dts, sg].dropna(); ctrl = S.loc[ok_ctrl, sg].dropna(); cw = ctrl.groupby(ctrl.index.dayofweek)
            cm = sum(cw.mean().get(w, np.nan) * n for w, n in wd.items()) / wd.sum(); ca = sum(cw.apply(lambda s: s.abs().mean()).get(w, np.nan) * n for w, n in wd.items()) / wd.sum()
            bs = [x.iloc[rg.integers(0, len(x), len(x))].mean() for _ in range(2000)]
            rowsA.append(dict(事件=nm, 段=sg, n=len(x), 均值=x.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), 对照均值=cm, abs=x.abs().mean(), 对照abs=ca, abs比=x.abs().mean() / ca))
    A = pd.DataFrame(rowsA); print("\nA) 形态:事件日 vs 同星期非事件日(bp)"); print(A.to_string(index=False)); A.to_csv(TAB / "macro_segments.csv", index=False)
    # ===================================================== A2) 分类
    rowsC = []
    for nm, ev in EV.items():
        for kind, kn in (("sf", "对预期"), ("sp", "对前值")):
            lbls = ev[kind].map(lambda s: lab(nm, kind, s))
            for cls in sorted(lbls.unique()):
                dts = lbls.index[lbls == cls]; r_ = {"事件": nm, "方向": kn, "类别": cls, "n": len(dts)}
                for sg in ("隔夜(昨收→开)", "开→10:00", "10:00→收盘", "全日(开→收)"):
                    x = S.loc[dts, sg].dropna(); bs = [x.iloc[rg.integers(0, len(x), len(x))].mean() for _ in range(1000)] if len(x) > 3 else [np.nan, np.nan]
                    r_[sg] = x.mean(); r_[sg + "_lo"] = np.nanpercentile(bs, 2.5); r_[sg + "_hi"] = np.nanpercentile(bs, 97.5)
                rowsC.append(r_)
    Cc = pd.DataFrame(rowsC); print("\nA2) 按新闻方向分类的均值(bp;区间为 95%)")
    print(Cc[["事件", "方向", "类别", "n", "隔夜(昨收→开)", "隔夜(昨收→开)_lo", "隔夜(昨收→开)_hi", "开→10:00", "10:00→收盘", "全日(开→收)", "全日(开→收)_lo", "全日(开→收)_hi"]].to_string(index=False)); Cc.to_csv(TAB / "macro_classes.csv", index=False)
    # ===================================================== B) 规则
    rows, nulls, obsz = [], {}, {}
    dev_m = yrs < 2023
    for nm, ev in EV.items():
        dts = ev.index
        if nm == "ISM": entry = o[:, idx["10:05"] + 1]; gate = np.sign(S["09:55→10:05"].to_numpy())
        else: entry = O; gate = np.sign((O / prev - 1))
        gross_unit = pd.Series((cl / entry - 1) * 1e4, index=days)            # 做多一单位的毛 bp
        gate = pd.Series(gate, index=days)
        defs = {"G1 延续": gate, "G2 回转": -gate, "S1 方向映射(对预期)": pd.Series(mapped_sign(nm, 1) * np.sign(ev.sf), index=ev.index).reindex(days).fillna(0.0),
                "S2 方向映射(对前值)": pd.Series(mapped_sign(nm, 1) * np.sign(ev.sp), index=ev.index).reindex(days).fillna(0.0)}
        wd = pd.Series(dts.dayofweek).value_counts(); pool = {w: np.where(ok_ctrl.to_numpy() & (days.dayofweek == w))[0] for w in wd.index}
        for rn, sig in defs.items():
            sgn = sig.loc[dts].to_numpy(); g = gross_unit.loc[dts].to_numpy(); trade = (sgn != 0) & np.isfinite(g)
            pnl = np.where(trade, sgn * g - 2 * COST, 0.0); stat = pnl.sum() / len(dts)           # 每个事件日的平均(不交易 = 0)
            if rn.startswith("G"):
                null = np.array([np.nanmean(np.concatenate([np.where(sig.to_numpy()[ii] != 0, sig.to_numpy()[ii] * gross_unit.to_numpy()[ii] - 2 * COST, 0.0) for ii in [rg.choice(pool[w], n) for w, n in wd.items()]])) for _ in range(NPERM)])
            else:
                null = np.empty(NPERM)
                for k_ in range(NPERM):
                    p_ = rg.permutation(sgn); t_ = (p_ != 0) & np.isfinite(g); null[k_] = np.where(t_, p_ * g - 2 * COST, 0.0).sum() / len(dts)
            key = (nm, rn); nulls[key] = (null - null.mean()) / null.std(); obsz[key] = (stat - null.mean()) / null.std()
            x = pnl[trade]; bs = [pnl[rg.integers(0, len(pnl), len(pnl))].mean() for _ in range(2000)]
            dv = dts.year < 2023
            rows.append(dict(事件=nm, 规则=rn, 事件日=len(dts), 交易日=int(trade.sum()), 每事件日净bp=stat, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), 每笔净bp=x.mean() if len(x) else np.nan, 胜率=float(np.mean(x > 0)) if len(x) else np.nan,
                             dev每事件日=pnl[dv].sum() / max(dv.sum(), 1), test每事件日=pnl[~dv].sum() / max((~dv).sum(), 1), 单测p=float(np.mean(null >= stat))))
    famz = np.max(np.vstack([v for v in nulls.values()]), axis=0)
    Bt = pd.DataFrame(rows); Bt["族内p(16)"] = [float(np.mean(famz >= obsz[(a, r)])) for a, r in zip(Bt.事件, Bt.规则)]
    print("\nB) 规则(每事件日净 bp;成本每边 1bp;p 为相对随机日/置换标签的单侧;族内 max 16 个)"); print(Bt.to_string(index=False)); Bt.to_csv(TAB / "macro_rules.csv", index=False)
    # ===================================================== C) Sleeve 与波动
    ret = pd.read_pickle(L.OUT / "sleeve_anatomy_curve.pkl"); ret.index = pd.to_datetime(ret.index); ret = ret.reindex(days).fillna(0.0); td = pd.Series(days.dayofweek != 0, index=days)
    print("\nC1) Sleeve 日均 bp:事件日 vs 同星期非事件日(Monday 排除);括号内为置换 p(双侧)")
    rowsS = []; rv_ = ret.to_numpy()
    for nm, ev in EV.items():
        dts = [d for d in ev.index if td[d]]; wd = pd.Series(pd.DatetimeIndex(dts).dayofweek).value_counts(); m = ok_ctrl & td
        pool = {w: np.where(m.to_numpy() & (days.dayofweek == w))[0] for w in wd.index}; obs = ret.loc[dts].mean()
        cm = sum(ret[m & (days.dayofweek == w)].mean() * n for w, n in wd.items()) / wd.sum()
        null = np.array([np.mean(rv_[np.concatenate([rg.choice(pool[w], n) for w, n in wd.items()])]) for _ in range(NPERM)])
        p = float(np.mean(np.abs(null - null.mean()) >= abs(obs - null.mean())))
        r_ = dict(事件=nm, n=len(dts), 事件日均bp=1e4 * obs, 对照均bp=1e4 * cm, 差bp=1e4 * (obs - cm), 置换p=p)
        for kind, kn in (("sf", "对预期"), ("sp", "对前值")):
            lbls = ev[kind].map(lambda s: lab(nm, kind, s)); r_.update({f"{kn}:{cls}": f"{1e4*ret.loc[[d for d in lbls.index[lbls==cls] if td[d]]].mean():+.1f}({int(sum(td[d] for d in lbls.index[lbls==cls]))})" for cls in sorted(lbls.unique())})
        rowsS.append(r_)
    RS = pd.DataFrame(rowsS); print(RS.to_string(index=False)); RS.to_csv(TAB / "macro_sleeve.csv", index=False)
    print("\nC2) 波动模型:加入各事件日历虚拟变量后的 OOS ΔR²(2021–2025,月聚类自助)")
    X, Y, A_, B_ = L.build(b); E0 = event_frame(days, b.last_idx.to_numpy()); EC0 = list(E0.columns)
    M = pd.DataFrame(0.0, index=days, columns=["m_cpi", "m_nfp", "m_retail", "m_ism", "m_adp"])
    for col, nm in (("m_cpi", "CPI"), ("m_nfp", "NFP"), ("m_retail", "RETAIL"), ("m_ism", "ISM")): M.loc[M.index.intersection(EV[nm].index), col] = 1.0
    M.loc[M.index.intersection(adp_d), "m_adp"] = 1.0
    X = pd.concat([X, E0, M], axis=1); MC = list(M.columns); L.OOS_YEARS = range(2021, 2026)
    sets = {"A+B+E": (A_ + B_ + EC0, "ridge"), "A+B+E+macro": (A_ + B_ + EC0 + MC, "ridge")}
    pred, base_rv, _ = L.walk_forward(X, Y, sets)
    oos = X.index[(X.index.year >= 2021) & (X.index.year <= 2025) & X[A_ + B_ + EC0].notna().all(1) & Y.notna().all(1)]; dev, test = oos[oos.year <= 2022], oos[oos.year >= 2023]
    sse_b = pd.Series(((Y.loc[oos, "y_rv"] - base_rv.loc[oos]) ** 2).to_numpy(), index=oos)
    e0 = pd.Series(((Y.loc[oos, "y_rv"] - pred[("A+B+E", "rv")].loc[oos]) ** 2).to_numpy(), index=oos); e1 = pd.Series(((Y.loc[oos, "y_rv"] - pred[("A+B+E+macro", "rv")].loc[oos]) ** 2).to_numpy(), index=oos)
    rowsV = []
    for pn, ii_ in (("全", oos), ("dev", dev), ("test", test)):
        pt, lo, hi = L.month_boot(ii_, lambda jj, ii_=ii_: (e0.loc[ii_].iloc[jj].sum() - e1.loc[ii_].iloc[jj].sum()) / sse_b.loc[ii_].iloc[jj].sum(), nb=1500); rowsV.append(dict(期间=pn, 基线R2=1 - e0.loc[ii_].sum() / sse_b.loc[ii_].sum(), dR2=pt, lo=lo, hi=hi))
    print(pd.DataFrame(rowsV).round(4).to_string(index=False))
    res = (Y.y_rv - pred[("A+B+E", "rv")]).loc[oos]; print("A+B+E 残差(ln RV)在各事件日的均值(正 = 模型低估):", {c_: (round(float(res[X.loc[oos, c_] == 1].mean()), 3), int((X.loc[oos, c_] == 1).sum())) for c_ in MC + ["e_fomc"]})
    pd.to_pickle(dict(A=A, Cc=Cc, Bt=Bt, RS=RS), OUT / "macro_events.pkl")


if __name__ == "__main__":
    main()
