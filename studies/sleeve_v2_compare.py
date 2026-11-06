"""新 Sleeve(实盘 + FOMC≥14:30 + CPI 回补 1× + 跳过 FFA-P 预测最低 20% 振幅日)vs 实盘 Sleeve:
Sleeve 层 dev/test/全周期对比,以及叠加不同 B&H 腿的 dev/test/全周期对比。
注意:三个组件都是在 2019–2026 数据上事后选出(FFA-P q=0.2 来自 sota_stack A2 的 39 个配置),结果为样本内。
python3 sleeve_v2_compare.py"""
import os, sys, warnings
warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS", "1"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
import sota_stack_study as S, lowvol_avoidance_study as L, regime_features as RF, round3_common as C, trend_rv_sleeve_study as T
from etft.intraday.bars import load_bars

PER = {"dev 2019–22": ("2019-01-01", "2022-12-31"), "test 2023–26": ("2023-01-01", "2026-10-01"), "全 2019–26": ("2019-01-01", "2026-10-01")}


def main():
    pd.set_option("display.width", 260, "display.max_rows", 400, "display.float_format", lambda x: f"{x:8.3f}")
    b = load_bars("QQQ"); days = b.days; td = pd.Series(days.dayofweek != 0, index=days)
    D = pd.read_pickle(L.OUT / "sota_sleeve.pkl"); V = D["VER"]; S0, S1 = V["S0 实盘"], V["S1 +FOMC≥14:30"]
    cand, R, dates = S.build_candidates(b); ov = S.cpi_overlay(b)
    rk = S.roll_rank(cand["FFA-P"]).reindex(days); skip = (rk <= 0.2).fillna(False)
    NEW = S1 * (~skip).astype(float) + ov
    assert int((NEW[NEW.index.dayofweek == 0] != 0).sum()) == 0, "新 Sleeve 必须排除周一"
    VER = {"实盘 Sleeve": S0, "新 Sleeve(FOMC+CPI+FFA-P跳过q=0.2)": NEW,
           "  组件:实盘+FOMC≥14:30": S1, "  组件:+CPI回补1×": S1 + ov, "  组件:+FFA-P跳过(无CPI)": S1 * (~skip).astype(float)}
    print(f"FFA-P 跳过比例(2019+ 周二至周五):{float(skip[td & (days.year >= 2019)].mean()):.1%};CPI overlay 交易笔数 {int((ov != 0).sum())}")
    # ---------------- Sleeve 层
    rows = []
    for vn, s in VER.items():
        for pn, (a, z) in PER.items():
            m = td & (days >= a) & (days <= z); p = S.panel(s[m]); rows.append(dict(Sleeve=vn, 期间=pn, **{k: p[k] for k in ("均值bp", "Sharpe", "Sortino", "CAGR", "MDD", "Calmar", "偏度", "CVaR5%bp", "前5%日占利润", "最差日bp")}))
    SL = pd.DataFrame(rows); SL.to_csv(L.TAB / "sleeve_v2_sleeve.csv", index=False)
    print("\n1) Sleeve 层(周二至周五)"); 
    for pn in PER: print(f"\n--- {pn} ---"); print(SL[SL.期间 == pn].drop(columns="期间").to_string(index=False))
    yr = pd.DataFrame({k: (1 + v[td]).groupby(v[td].index.year).prod() - 1 for k, v in VER.items()}).mul(100).round(1).loc[2019:]
    print("\n   分年 Sleeve 收益(%)"); print(yr.to_string())
    # 新 vs 实盘:Sleeve 层 ΔSR 区间
    for pn, (a, z) in PER.items():
        m = td & (days >= a) & (days <= z); d, dist = T.boot_dsr(NEW[m], S0[m], nb=2000, L=10, seed=0)
        print(f"   Sleeve ΔSR(新 − 实盘)[{pn}] = {d:+.3f} [{np.percentile(dist,2.5):+.3f}, {np.percentile(dist,97.5):+.3f}]")
    # ---------------- 叠加 B&H 腿
    px = C.load_close("yahoo/QQQ.parquet"); r = px.pct_change().fillna(0.0); P = RF.load_P(); rf = (P["^IRX"].reindex(px.index).ffill() / 100 / 252).fillna(0.0)
    A2C = pd.read_pickle(L.OUT / "regime_timing_r2.pkl")["E2"]["A2_C|e0.5|alone"]
    W = S.bh_weights(px, r, {"FFA-SD": cand["FFA-SD"], "ML-开盘": cand["ML-开盘"], "HAR": cand["HAR"], "成分CARR": cand["成分CARR"]}, A2C)
    legs = {"静态": ("静态", "无趋势"), "静态+SMA200 e0.5": ("静态", "SMA200|e=0.5"), "RV21[0.5,1.5](现用)": ("RV21[0.5,1.5]", "无趋势"),
            "RV21+SMA200 e0.5": ("RV21[0.5,1.5]", "SMA200|e=0.5"), "RV21+信用A2_C": ("RV21[0.5,1.5]", "信用A2_C|e=0.5"),
            "VT[FFA-SD,1.5]": ("VT[FFA-SD,cap1.5]", "无趋势"), "VT[FFA-SD,1.5]+SMA200 e0.5": ("VT[FFA-SD,cap1.5]", "SMA200|e=0.5"), "VT[FFA-SD,1.5]+信用A2_C": ("VT[FFA-SD,cap1.5]", "信用A2_C|e=0.5")}
    rows = []
    for ln, key in legs.items():
        bhx = S.bh_returns(W[key], r, rf, False)
        for sn, s in (("无 Sleeve", None), ("实盘 Sleeve", S0), ("新 Sleeve", NEW)):
            x = bhx + (s.reindex(px.index).fillna(0.0) if s is not None else 0.0)
            for pn, (a, z) in PER.items():
                m = (x.index >= a) & (x.index <= z); p = S.panel(x[m])
                row = dict(B_H腿=ln, Sleeve=sn, 期间=pn, **{k: p[k] for k in ("Sharpe", "Sortino", "CAGR", "MDD", "Calmar", "年化波动", "CVaR5%bp", "最差日bp")})
                if sn == "新 Sleeve":
                    x0 = bhx + S0.reindex(px.index).fillna(0.0); d, dist = T.boot_dsr(x[m], x0[m], nb=1500, L=10, seed=0)
                    row.update(ΔSR_vs实盘=d, lo=np.percentile(dist, 2.5), hi=np.percentile(dist, 97.5))
                rows.append(row)
    X = pd.DataFrame(rows); X.to_csv(L.TAB / "sleeve_v2_portfolio.csv", index=False)
    print("\n2) 叠加不同 B&H 腿(成本 2bp/边;原口径)")
    for pn in PER:
        print(f"\n--- {pn} ---")
        t_ = X[X.期间 == pn].pivot_table(index="B_H腿", columns="Sleeve", values=["Sharpe", "CAGR", "MDD", "Calmar"], sort=False)[["Sharpe", "CAGR", "MDD", "Calmar"]]
        t_ = t_.reindex(columns=["无 Sleeve", "实盘 Sleeve", "新 Sleeve"], level=1); print(t_.round(3).to_string())
        print("   新 − 实盘 的组合 ΔSR [95%]:"); print(X[(X.期间 == pn) & (X.Sleeve == "新 Sleeve")][["B_H腿", "ΔSR_vs实盘", "lo", "hi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
