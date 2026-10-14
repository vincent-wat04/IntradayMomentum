"""Reproduce the README headline row (多空 band+尾0.5 无周一) by REUSING strategy_comparison.main() internals.
Only the version list is restricted to the one live configuration; every other line is the repo's own code."""
import os, sys, types
sys.path.insert(0, ".")
import strategy_comparison as SC
import pandas as pd, numpy as np, yfinance as yf
from etft.intraday.bars import load_bars
from etft.intraday.strategy import buy_hold
from audit import reimpl_noise as RI

b = load_bars("QQQ"); days = b.days; bh = buy_hold(b)
pt = yf.download("TQQQ", start="2018-10-01", end="2026-10-12", auto_adjust=False, progress=False, threads=False)["Close"]
pt = pt.iloc[:, 0] if hasattr(pt, "columns") else pt
pt.index = pd.to_datetime(pt.index).tz_localize(None)
sp = yf.Ticker("TQQQ").splits; sp.index = pd.to_datetime(sp.index).tz_localize(None)
fsp = pd.Series(1.0, index=pt.index)
for d, r in sp.items(): fsp[pt.index < d] *= r
PT = (pt * fsp).reindex(days).ffill()
scale = (b.day_close / PT) / 3.0
print("TQQQ splits seen:", {str(k.date()): v for k, v in sp.items()})
print("scale (QQQ-price / TQQQ-quoted-price / 3): " + " | ".join(f"{x[:7]} {float(scale.loc[:x].iloc[-1]):.3f}" for x in ("2019-06-03", "2023-01-03", "2025-06-02", "2026-09-30")))
kw = {**SC.STOPS["band+尾0.5"], "skip_dows": (0,)}
acc = []
for off in range(SC.NPH):
    rr, F = RI.simulate(b, checks=SC.chk(off), sizing="voltarget", max_lev=4.0, reentry="hold", half_spread=0.005, commission=0.0045, **kw)
    if len(F):
        F = F.copy(); F["day"] = pd.to_datetime(F.day)
        c_old = F.groupby("day").commission.sum().reindex(days).fillna(0.0)
        rr = rr + c_old - c_old * scale.reindex(days).fillna(1.0)
    acc.append(rr)
r = pd.DataFrame(acc).T.mean(axis=1)
r.to_pickle(os.path.join(os.environ["ETFT_DATA"], "_repro_live_cfg.pkl"))
for per, nm in ((SC.DEV, "dev"), (SC.TEST, "test"), (None, "全")):
    x = r if per is None else r.loc[per[0]:per[1]]
    m = SC.metrics(x.dropna(), bh); print(f"  {nm:4s} sharpe {m['sharpe']:.3f}  ann {m['ann']:.4f}  mdd {m['mdd']:.4f}  calmar {m['calmar']:.3f}  叠加贡献 {m['叠加贡献']:+.3f}")
print("\nREADME / strategy_comparison.csv expected row: dev 0.704, test 1.203, 全 0.955, ann 0.122, mdd -0.116")
