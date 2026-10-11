"""GEX 能不能告诉我们"今天该不该止盈"？

用户问的是"基于 GEX 算出来的 wall 去止盈"。先说数据结论：
  我手上的 GEX 是 etf_timing/data/raw/squeeze_dix_gex.parquet，
  字段只有 (price, dix, gex) —— **每天一个标量**，SPX 口径，2011-05-02 起 3878 行。
  而"gamma wall"需要**逐行权价**的 gamma 分布（取 OI×gamma 后按 strike 聚合，
  找 call wall / put wall / gamma flip）。单个标量算不出任何 wall。
  所以"GEX wall 止盈"**在现有数据上无法回测**，这不是方法问题而是数据粒度问题。

但 GEX 的**机制**仍然可以用一个更弱、可测的形式来检验：
  净正 gamma 越大 -> 做市商越是"涨了卖、跌了买" -> 日内波动被压制、价格趋于钉住
  -> 行情更容易衰竭 -> **止盈应该在高 GEX 日更有效**。
  这是一个可以用标量 GEX 直接测的可falsify命题，而且它用的正是之前唯一复现成功的
  GEX 结果（GEX 对日内波动幅度有预测力：四分位 |open→close| 比值 0.42~0.56，
  样本外 R² 0.009->0.035，DM t 最高 +3.88）。

若"高 GEX 日止盈更有效"成立，则 wall 版本值得去买逐行权价数据；
若不成立，则连机制都不支持，不必再投入。

口径：GEX 取 t−1（合法最小滞后），按 dev 期分位切组避免用到未来信息。
"""
from __future__ import annotations

import paths as _P
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
from stop_significance import block_boot_pair, sr

OUT = Path("outputs/full/intraday")
DEV, TEST = ("2019", "2022"), ("2023", "2026")
NPH = 30
BASE = dict(stop_long="vwap", stop_short="vwap", tail_k=1.0, tail_every_bar=True)
GEXP = (_P.RAW / "squeeze_dix_gex.parquet")


def chk(off):
    base = [(h, m) for h in range(10, 16) for m in (0, 30)][:-1]
    return [(pd.Timestamp(f"2000-01-01 {h:02d}:{m:02d}") +
             pd.Timedelta(minutes=off)).strftime("%H:%M") for h, m in base]


def main():
    b = load_bars("QQQ"); days = b.days; bh = buy_hold(b)
    G = pd.read_parquet(GEXP)
    print(f"GEX 数据字段：{list(G.columns)}  行数 {len(G)}  "
          f"范围 {G.index.min().date()} ~ {G.index.max().date()}")
    print("-> 只有标量 gex，没有逐行权价分布，故无法构造 gamma wall（见文件头说明）\n")

    g = G.gex.reindex(days).ffill().shift(1)            # t−1，合法最小滞后
    # 分位阈值只用 dev 期计算，避免用到 test 期信息
    gdev = g.loc[DEV[0]:DEV[1]].dropna()
    q33, q67 = gdev.quantile([1 / 3, 2 / 3])
    hi = set(days[(g >= q67).reindex(days).fillna(False).to_numpy()])
    lo = set(days[(g <= q33).reindex(days).fillna(False).to_numpy()])
    mid = set(days) - hi - lo
    print(f"按 dev 期三分位切组（阈值 {q33:.3e} / {q67:.3e}）："
          f"高GEX {len(hi)} 天、中 {len(mid)} 天、低GEX {len(lo)} 天")

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

    def run_cfg(**kw):
        acc = []
        for off in range(NPH):
            rr, F = RI.simulate(b, checks=chk(off), sizing="voltarget", max_lev=4.0,
                                reentry="hold", half_spread=0.005, commission=0.0045,
                                **BASE, **kw)
            if len(F):
                F = F.copy(); F["day"] = pd.to_datetime(F.day)
                c_old = F.groupby("day").commission.sum().reindex(days).fillna(0.0)
                rr = rr + c_old - c_old * scale.reindex(days).fillna(1.0)
            acc.append(rr)
        return pd.DataFrame(acc).T.mean(axis=1)

    TP = dict(tp_k=0.75, tp_side="short", tp_rearm="rearm")
    cfgs = [("基线（无止盈）", {}),
            ("止盈·全部日子", TP),
            ("止盈·仅高GEX日", {**TP, "tp_days": hi}),
            ("止盈·仅低GEX日", {**TP, "tp_days": lo}),
            ("止盈·仅中GEX日", {**TP, "tp_days": mid})]

    curves, rows = {}, []
    for name, kw in cfgs:
        r = run_cfg(**kw); curves[name] = r
        rec = dict(配置=name)
        for per, pl in ((DEV, "dev"), (TEST, "test")):
            x = r.loc[per[0]:per[1]].dropna()
            h = bh.reindex(x.index).dropna(); x2 = x.reindex(h.index).fillna(0)
            rec[f"{pl}_sharpe"] = sr(x)
            rec[f"{pl}_叠加贡献"] = sr(h + x2) - sr(h)
        rows.append(rec)
        print(f"  {name:18s} dev={rec['dev_sharpe']:+.3f} test={rec['test_sharpe']:+.3f}"
              f"  叠加 dev={rec['dev_叠加贡献']:+.3f} test={rec['test_叠加贡献']:+.3f}",
              flush=True)

    R = pd.DataFrame(rows); R.to_csv(OUT / "takeprofit_gex.csv", index=False)
    print("\n" + "=" * 110); print(R.to_string(index=False))

    print("\n" + "=" * 110)
    print("配对自助：各 GEX 分组的止盈 vs 基线（只在该组日子上生效）")
    base = curves["基线（无止盈）"]
    for name in ("止盈·仅高GEX日", "止盈·仅中GEX日", "止盈·仅低GEX日", "止盈·全部日子"):
        line = f"  {name:18s}"
        for per, pl in ((DEV, "dev"), (TEST, "test")):
            a = curves[name].loc[per[0]:per[1]].dropna()
            bb = base.loc[per[0]:per[1]].dropna()
            real, dist = block_boot_pair(a, bb, seed=abs(hash(name)) % 9999)
            line += (f" | {pl} ΔSR={real:+.3f} p={float((dist<=0).mean()):.3f}")
        print(line)

    print("\n" + "=" * 110)
    print("判读：若 gamma 机制为真，『仅高GEX日』应明显优于『仅低GEX日』")
    print("（高正 gamma -> 做市商逆势对冲 -> 行情衰竭 -> 止盈更该有效）")
    pd.DataFrame(curves).to_pickle(OUT / "takeprofit_gex_curves.pkl")


if __name__ == "__main__":
    main()
