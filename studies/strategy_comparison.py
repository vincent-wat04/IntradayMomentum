"""几个候选版本的完整对照：PnL 曲线 + Sharpe / 年化 / Calmar / 分边胜率与盈亏比。

对照的四个因子（可自由组合）：
  A 止损设计   vwap + 硬尾k=1.0（原始 SOTA，当前实盘）   vs   band + 硬尾k=0.5
  B 方向       多空（论文原版）                        vs   只做多
  C 星期       全周                                   vs   排除周一
  D 空头止盈   关（SOTA）                              vs   开（k=0.75，仅空头）

组合规则：D 只对空头生效，所以 long-only 版本不含 D。
  多空：2(止损) × 2(星期) × 2(止盈) = 8
  只做多：2(止损) × 2(星期) = 4
  合计 12 个版本。

口径（与本项目之前所有实验一致，不另起炉灶）：
  * 30 个相位（检查时点整体平移 0~29 分钟）取均值 —— 避免"相位运气"
    （PART10：论文那个相位是 30 个里的第 90 百分位）
  * 成本换算到 TQQQ 单股口径（实测单腿 0.1323bp）
  * dev 2019-2022 / test 2023-2026 分开报，另报全样本
  * 同时报**作为 B&H 叠加层的 ΔSharpe** —— 这才是部署口径（PART9 §3、PART12 §4），
    独立 Sharpe 会系统性地偏向"削弱空头腿"的版本

分边统计（逐笔配对 fills 得到，成本已换算到 TQQQ）：
  胜率 = 净收益>0 的比例；盈亏比 = 平均盈利 / |平均亏损|
  注意：30 个相位复用同一批交易日（两两相关 0.880），所以笔数是叠加的，
  比例类指标无偏，但**不要**把笔数当独立样本去算显著性。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etft.intraday.bars import load_bars
from etft.intraday.strategy import buy_hold

from audit import reimpl_noise as RI
from stop_significance import sr

OUT = Path("outputs/full/intraday")
OUT.mkdir(parents=True, exist_ok=True)
DEV, TEST = ("2019", "2022"), ("2023", "2026")
NPH = 30

STOPS = {
    "vwap+尾1.0": dict(stop_long="vwap", stop_short="vwap", tail_k=1.0,
                       tail_every_bar=True),
    "band+尾0.5": dict(stop_long="band+tail", stop_short="band+tail", tail_k=0.5,
                       tail_every_bar=True),
}
TP = dict(tp_k=0.75, tp_side="short", tp_rearm="rearm")


def chk(off):
    base = [(h, m) for h in range(10, 16) for m in (0, 30)][:-1]
    return [(pd.Timestamp(f"2000-01-01 {h:02d}:{m:02d}") +
             pd.Timedelta(minutes=off)).strftime("%H:%M") for h, m in base]


def metrics(x, bh):
    """x = 日收益序列（含被跳过日子的 0）。"""
    x = pd.Series(x).dropna()
    if len(x) < 40 or x.std() == 0:
        return {}
    eq = (1 + x).cumprod(); dd = eq / eq.cummax() - 1
    ann = float(eq.iloc[-1] ** (252 / len(x)) - 1)
    mdd = float(dd.min())
    nz = x[x != 0]
    h = bh.reindex(x.index).dropna()
    x2 = x.reindex(h.index).fillna(0)
    return dict(sharpe=sr(x), ann=ann, mdd=mdd,
                calmar=(ann / abs(mdd) if mdd < 0 else np.nan),
                日胜率=float((nz > 0).mean()) if len(nz) else np.nan,
                叠加贡献=sr(h + x2) - sr(h))


def trade_table(F, scale):
    """把 fills 配对成逐笔交易，成本换算到 TQQQ，返回分边统计。"""
    if not len(F):
        return {}
    F = F.copy(); F["day"] = pd.to_datetime(F.day)
    recs = []
    for d, g in F.groupby("day", sort=False):
        sc = float(scale.get(d, 1.0))
        g = g.reset_index(drop=True)
        cur = None
        for _, r in g.iterrows():
            kind = str(r.kind)
            if kind.startswith("ENTRY"):
                cur = dict(sgn=(1.0 if r.shares > 0 else -1.0), sh=float(r.shares),
                           px=float(r.price), cm=float(r.commission) * sc)
            elif cur is not None:
                gross = cur["sh"] * (float(r.price) - cur["px"])
                net = gross - cur["cm"] - float(r.commission) * sc
                recs.append(dict(side=("多" if cur["sgn"] > 0 else "空"),
                                 net=net, kind=kind))
                cur = None
    T = pd.DataFrame(recs)
    out = {}
    for side in ("多", "空"):
        s = T[T.side == side]
        if not len(s):
            out[side] = dict(n=0)
            continue
        w = s.net[s.net > 0]; l = s.net[s.net <= 0]
        out[side] = dict(n=len(s), 胜率=float((s.net > 0).mean()),
                         均值bp=float(s.net.mean() * 1e4),
                         盈利bp=float(w.mean() * 1e4) if len(w) else np.nan,
                         亏损bp=float(l.mean() * 1e4) if len(l) else np.nan,
                         盈亏比=float(w.mean() / abs(l.mean()))
                         if len(w) and len(l) and l.mean() != 0 else np.nan)
    return out


def main():
    b = load_bars("QQQ"); days = b.days; bh = buy_hold(b)
    pt = yf.download("TQQQ", start="2018-10-01", end="2026-10-05", auto_adjust=False,
                     progress=False, threads=False)["Close"]
    pt = pt.iloc[:, 0] if hasattr(pt, "columns") else pt
    pt.index = pd.to_datetime(pt.index).tz_localize(None)
    sp = yf.Ticker("TQQQ").splits; sp.index = pd.to_datetime(sp.index).tz_localize(None)
    fsp = pd.Series(1.0, index=pt.index)
    for d, r in sp.items():
        fsp[pt.index < d] *= r
    PT = (pt * fsp).reindex(days).ffill()
    scale = (b.day_close / PT) / 3.0
    sc_map = scale.to_dict()

    # ---------- 版本清单
    cfgs = []
    for sname, skw in STOPS.items():
        for mon, mkw in (("全周", {}), ("无周一", dict(skip_dows=(0,)))):
            for tpn, tkw in (("", {}), ("+空头止盈", TP)):
                cfgs.append((f"多空 {sname} {mon}{tpn}", {**skw, **mkw, **tkw}))
    for sname, skw in STOPS.items():
        for mon, mkw in (("全周", {}), ("无周一", dict(skip_dows=(0,)))):
            cfgs.append((f"只做多 {sname} {mon}",
                         {**skw, **mkw, "sides": "long"}))

    curves, rows = {}, []
    for name, kw in cfgs:
        acc, tstats = [], []
        for off in range(NPH):
            rr, F = RI.simulate(b, checks=chk(off), sizing="voltarget", max_lev=4.0,
                                reentry="hold", half_spread=0.005, commission=0.0045,
                                **kw)
            if len(F):
                F = F.copy(); F["day"] = pd.to_datetime(F.day)
                c_old = F.groupby("day").commission.sum().reindex(days).fillna(0.0)
                rr = rr + c_old - c_old * scale.reindex(days).fillna(1.0)
                tstats.append(trade_table(F, sc_map))
            acc.append(rr)
        r = pd.DataFrame(acc).T.mean(axis=1)
        curves[name] = r

        # 分边统计：把 30 个相位的逐笔指标取均值
        agg = {}
        for side in ("多", "空"):
            vals = [t[side] for t in tstats if t.get(side, {}).get("n", 0) > 0]
            if not vals:
                continue
            for k2 in ("n", "胜率", "均值bp", "盈亏比"):
                agg[f"{side}_{k2}"] = float(np.mean([v[k2] for v in vals]))

        rec = dict(版本=name)
        for per, pl in ((DEV, "dev"), (TEST, "test"), (None, "全")):
            x = r if per is None else r.loc[per[0]:per[1]]
            m = metrics(x.dropna(), bh)
            for k2, v in m.items():
                rec[f"{pl}_{k2}"] = v
        rec.update(agg)
        rows.append(rec)
        print(f"  {name:34s} dev SR={rec.get('dev_sharpe',float('nan')):+.3f} "
              f"test SR={rec.get('test_sharpe',float('nan')):+.3f} "
              f"全 Calmar={rec.get('全_calmar',float('nan')):+.2f} "
              f"叠加(test)={rec.get('test_叠加贡献',float('nan')):+.3f}", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "strategy_comparison.csv", index=False)
    pd.DataFrame(curves).to_pickle(OUT / "strategy_comparison_curves.pkl")
    bh.to_pickle(OUT / "strategy_comparison_bh.pkl")

    pd.set_option("display.width", 250)
    print("\n" + "=" * 150)
    print("【风险收益】")
    c1 = ["版本", "dev_sharpe", "test_sharpe", "全_sharpe", "全_ann", "全_mdd",
          "全_calmar", "全_日胜率", "dev_叠加贡献", "test_叠加贡献"]
    print(R[c1].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    print("\n" + "=" * 150)
    print("【分边逐笔】（30 相位均值；成本已按 TQQQ 换算）")
    c2 = ["版本", "多_n", "多_胜率", "多_均值bp", "多_盈亏比",
          "空_n", "空_胜率", "空_均值bp", "空_盈亏比"]
    print(R[[c for c in c2 if c in R.columns]].to_string(
        index=False, float_format=lambda v: f"{v:,.3f}"))
    print(f"\n表格 -> {OUT}/strategy_comparison.csv")


if __name__ == "__main__":
    main()
