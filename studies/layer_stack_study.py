"""分层叠加表(ARCHITECTURE_layers.md):逐层累加已登记的配置,报告每层相对前一层的增量。不做新的选择。
L1 = RV[0.5,1.5](repo 既有);L0 = SMA200 / 信用 A2_C(regime 第二轮,e_off=0.5);L2 = Sleeve(实盘配置 30 相位均值);L4 = FOMC 日关闭 Sleeve(lowvol 第二轮探索性)。"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
import lowvol_avoidance_study as L, trend_rv_sleeve_study as T, regime_timing_study as R1, regime_features as RF
import round3_common as C
from sens_rv_weights import pct_w
from event_calendar import event_frame
from etft.intraday.bars import load_bars
b = load_bars("QQQ"); days = b.days
P = RF.load_P(); px = P["QQQ"].dropna(); r = px.pct_change().fillna(0.0); p_all = pct_w(r.rolling(21).std(), 504)
E2 = pd.read_pickle(L.OUT / "regime_timing_r2.pkl")["E2"]
ret = pd.read_pickle(L.OUT / "sleeve_anatomy_curve.pkl"); ret.index = pd.to_datetime(ret.index); sleeve = ret.reindex(px.index).fillna(0.0)
E = event_frame(days, b.last_idx.to_numpy()); fomc = E.e_fomc.reindex(px.index).fillna(0.0)
w_rv = T.weights(px, p_all, (("lin", .5, 1.5), None), 1.0).fillna(1.0)
e_sma = R1.trend_exp(px, "SMA", 200, 0.5); e_cred = E2["A2_C|e0.5|alone"]; e_cs = E2["A2_C|e0.5|+SMA200"]
def build(e=None, rv=False, sl=None):
    e = pd.Series(1.0, index=px.index) if e is None else e
    w = e * (w_rv if rv else 1.0)
    x = w * r - 2e-4 * w.diff().abs().fillna(0.0)
    return x + (sl if sl is not None else 0.0)
stack = [("0 静态 QQQ 100%", build()), ("1 +L1 RV[0.5,1.5]", build(rv=True)), ("2 +L0 SMA200(e=0.5)", build(e_sma, True)), ("3 +L0 信用 A2_C(e=0.5)", build(e_cred, True)),
         ("4 +L0 信用与SMA200平均", build(e_cs, True)), ("5 =4 +L2 Sleeve", build(e_cs, True, sleeve)), ("6 =5 +L4 FOMC日关闭Sleeve(探索性)", build(e_cs, True, sleeve * (1 - fomc))),
         ("A  1 +L2 Sleeve(无L0)", build(rv=True, sl=sleeve)), ("B  2 +L2 Sleeve", build(e_sma, True, sleeve)), ("C  3 +L2 Sleeve", build(e_cred, True, sleeve))]
W = {"2004-06~2026": ("2004-06-01", "2026-10-09"), "2019~2026": ("2019-01-02", "2026-10-09"), "2019~2022(dev)": ("2019-01-02", "2022-12-31"), "2023~2026(test)": ("2023-01-02", "2026-10-09")}
pd.set_option("display.width", 220)
for wn, (a, z) in W.items():
    print(f"\n=== {wn} ===  (无 Sleeve 的行在 2019 前也可算;含 Sleeve 的行仅 2019+)")
    rows = []
    for nm, x in stack:
        has_sl = nm.startswith(("5", "6", "A", "B", "C"))
        if has_sl and a < "2019": continue
        s = x.loc[a:z]; rows.append(dict(层=nm, Sharpe=R1.sr(s), CAGR=R1.cagr(s), MDD=R1.mdd(s), Calmar=R1.cagr(s) / abs(R1.mdd(s))))
    D = pd.DataFrame(rows); print(D.round(3).to_string(index=False))
# 相邻层增量 ΔSR 的块自助区间(2019~2026)
print("\n=== 相邻层增量(2019~2026,配对块自助 L=20)===")
names = dict(stack); pairs = [("1 +L1 RV[0.5,1.5]", "0 静态 QQQ 100%"), ("2 +L0 SMA200(e=0.5)", "1 +L1 RV[0.5,1.5]"), ("3 +L0 信用 A2_C(e=0.5)", "1 +L1 RV[0.5,1.5]"), ("4 +L0 信用与SMA200平均", "1 +L1 RV[0.5,1.5]"),
         ("5 =4 +L2 Sleeve", "4 +L0 信用与SMA200平均"), ("6 =5 +L4 FOMC日关闭Sleeve(探索性)", "5 =4 +L2 Sleeve"), ("A  1 +L2 Sleeve(无L0)", "1 +L1 RV[0.5,1.5]"), ("C  3 +L2 Sleeve", "3 +L0 信用 A2_C(e=0.5)")]
for n1, n0 in pairs:
    a_, b_ = names[n1].loc["2019-01-02":"2026-10-09"], names[n0].loc["2019-01-02":"2026-10-09"]
    d, dist = T.boot_dsr(a_, b_, nb=3000, L=20, seed=0); lo, hi = np.percentile(dist, [2.5, 97.5])
    print(f"{n1:42s} − {n0:34s} ΔSharpe {d:+.3f} [{lo:+.2f},{hi:+.2f}]  ΔMDD {100*(R1.mdd(a_)-R1.mdd(b_)):+.1f}pp  ΔCAGR {100*(R1.cagr(a_)-R1.cagr(b_)):+.1f}pp")
