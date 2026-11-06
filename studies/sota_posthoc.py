"""事后补充:SOTA 的『跳过』组件在 test 失败(dev 选 q=0.3)后,报告不含跳过的版本与 q=0.2 版本;以及推荐 B&H 腿 + Sleeve 版本的完整面板。"""
import os, sys, warnings; warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS", "1"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
import sota_stack_study as S, lowvol_avoidance_study as L, regime_features as RF, round3_common as C, trend_rv_sleeve_study as T
from etft.intraday.bars import load_bars
b = load_bars("QQQ"); days = b.days; yrs = days.year; td = pd.Series(days.dayofweek != 0, index=days)
D = pd.read_pickle(L.OUT / "sota_sleeve.pkl"); V = D["VER"]; S0, S1, S3 = V["S0 实盘"], V["S1 +FOMC≥14:30"], V["S3 SOTA(+CPI 回补 1×)"]
cand, R, dates = S.build_candidates(b); rk = S.roll_rank(cand["ML-开盘"]).reindex(days); ov = S.cpi_overlay(b)
VV = {"S0 实盘": S0, "S1 +FOMC≥14:30": S1, "A: S1 + CPI 1×(不跳过)": S1 + ov, "B: S1 + 跳过 q=0.2 + CPI 1×(q 事后)": S1 * (~(rk <= 0.2).fillna(False)).astype(float) + ov,
      "预注册 SOTA: S1 + 跳过 q=0.3 + CPI 1×": S3}
px = C.load_close("yahoo/QQQ.parquet"); r = px.pct_change().fillna(0.0); P = RF.load_P(); rf = (P["^IRX"].reindex(px.index).ffill() / 100 / 252).fillna(0.0)
A2C = pd.read_pickle(L.OUT / "regime_timing_r2.pkl")["E2"]["A2_C|e0.5|alone"]
W = S.bh_weights(px, r, {"FFA-SD": cand["FFA-SD"], "ML-开盘": cand["ML-开盘"], "HAR": cand["HAR"], "成分CARR": cand["成分CARR"]}, A2C)
legs = {"静态": ("静态", "无趋势"), "RV21[0.5,1.5](现用)": ("RV21[0.5,1.5]", "无趋势"), "RV21[0.5,1.5]+SMA200 e0.5": ("RV21[0.5,1.5]", "SMA200|e=0.5"),
        "RV21[0.5,1.5]+信用A2_C": ("RV21[0.5,1.5]", "信用A2_C|e=0.5"), "VT[FFA-SD,1.5]+信用A2_C": ("VT[FFA-SD,cap1.5]", "信用A2_C|e=0.5"), "VT[FFA-SD,1.5]+SMA200 e0.5": ("VT[FFA-SD,cap1.5]", "SMA200|e=0.5")}
rows = []
for ln, key in legs.items():
    bhx = S.bh_returns(W[key], r, rf, False)
    for vn, s in VV.items():
        x = bhx + s.reindex(px.index).fillna(0.0)
        for pn, (a, z) in (("2019~2026", ("2019-01-01", "2026-10-01")), ("dev 2019~22", ("2019-01-01", "2022-12-31")), ("test 2023~26", ("2023-01-01", "2026-10-01"))):
            m = (x.index >= a) & (x.index <= z); p = S.panel(x[m]); rows.append(dict(B_H腿=ln, Sleeve=vn, 期间=pn, **{k: p[k] for k in ("Sharpe", "Sortino", "CAGR", "MDD", "Calmar", "CVaR5%bp", "最差日bp")}))
X = pd.DataFrame(rows); X.to_csv(L.TAB / "sota_posthoc.csv", index=False)
pd.set_option("display.width", 260, "display.max_rows", 300, "display.float_format", lambda x: f"{x:7.3f}")
print("Sleeve 层(周二至周五):"); 
for vn, s in VV.items():
    xs = s[td & (yrs >= 2019)]; dv = L.sr(s[td & (yrs >= 2019) & (yrs <= 2022)]); tt = L.sr(s[td & (yrs >= 2023)])
    print(f"  {vn:40s} Sharpe {L.sr(xs):.3f}(dev {dv:.3f} / test {tt:.3f}) 均值 {1e4*xs.mean():.2f}bp  分年%: " + " ".join(f"{y}:{100*((1+s[td&(yrs==y)]).prod()-1):+.1f}" for y in range(2019, 2027)))
for pn in ("2019~2026", "dev 2019~22", "test 2023~26"):
    print(f"\n=== {pn}:Sharpe / CAGR / MDD ==="); t_ = X[X.期间 == pn].pivot_table(index="B_H腿", columns="Sleeve", values=["Sharpe", "CAGR", "MDD"], sort=False); print(t_.round(3).to_string())
