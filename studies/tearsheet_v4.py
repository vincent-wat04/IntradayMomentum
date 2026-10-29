"""Tearsheet v4:在 tearsheet_v3 的基础上,Sleeve 换成 v4 = K1。
v4 Sleeve = v3 的全部组件(FOMC≥14:30、CPI 回补 1×、FFA-P 跳过 20%、排除周一、15:40 平仓)
            + 入场带按"无成交量配合的波动"放宽(只放宽入场带,止损带不变;s = clip(√((RV/S_RV)/(V/S_V)), 1, 2))
            + 仓位模型:线性链接 k = clip(1+θᵀz, 0, 2),特征 CLEAN 7 个 + dur_rate,均值–方差组合效用,按各自组合走步重训(2021 起生效)。
① = RV21[0.5,1.5]×SMA200 + Sleeve;② = VT[FFA-SD, cap1.5]×SMA200 + Sleeve(与 tearsheet_v3 相同的 QQQ 腿)。
注意:v4 的三项改动是在 2019–2026 上多轮挑选出来的(docs/PROPOSAL_intraday_vol_entry_r3.md §5–8),没有通过预注册判定;本表是描述性的。
python3 tearsheet_v4.py
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_pred_study as DP
import ffa_vol_study as FV
import liquidity_inverse_study as LI
import r2_rules_study as RR
import r3_study as R3
import round3_common as C
import sizing_models_study as M
import sizing_sigma_feed as SF
import sota_stack_study as S
import sota_unified_stack as SU
import tearsheet as TS
import unified_sizing_posthoc as P
from cjkfont import use_cjk
from etft.intraday.bars import load_bars

OUT = SF.OUT; REPO_FIG = Path("../outputs/figs"); REPO_TAB = Path("../outputs/tables")
PERIODS = (("dev 19-22", ("2019", "2022")), ("test 23-26", ("2023", "2026")), ("全 19-26", None))
REF_V3 = {("①", "dev 19-22"): 1.5871, ("①", "test 23-26"): 2.4610, ("①", "全 19-26"): 2.0224,
          ("②", "dev 19-22"): 1.7889, ("②", "test 23-26"): 2.3549, ("②", "全 19-26"): 2.0709}


def ffa_sd():
    p = OUT / "ffa_sd_only.pkl"
    if p.exists(): return pd.read_pickle(p)
    q, R, r = FV.load(); y = np.log(R.to_numpy()); s = pd.Series(FV.ffa_causal(y, R.to_numpy(), sd_state=True)[1], index=R.index); pd.to_pickle(s, p); return s


def main():
    pd.set_option("display.width", 260, "display.max_rows", 200, "display.float_format", lambda v: f"{v:9.4f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    I = SF.build_inputs(b); X = I["X"].copy(); bh = I["bh"]
    dn = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days)
    X["dur_rate"] = M.family_features(b, I["X"])["dur_rate"]; assert X.loc[sel, "dur_rate"].notna().all()
    # ---- QQQ 腿(与 tearsheet_v3 相同)
    px = C.load_close("yahoo/QQQ.parquet"); r = px.pct_change().fillna(0.0)
    W = S.bh_weights(px, r, {"FFA-SD": ffa_sd()}, pd.Series(1.0, index=px.index))
    legs = {"①": SU.leg_ret(W[("RV21[0.5,1.5]", "SMA200|e=0.5")], bh).reindex(days).fillna(0.0),
            "②": SU.leg_ret(W[("VT[FFA-SD,cap1.5]", "SMA200|e=0.5")], bh).reindex(days).fillna(0.0)}
    leg_rv = SU.leg_ret(W[("RV21[0.5,1.5]", "无趋势")], bh).reindex(days).fillna(0.0)
    assert float((legs["①"] - I["leg1"]).abs().max()) < 1e-12
    # ---- 成交:v3(原入场带)与 v4(入场带放宽)
    _, S3m, _ = RR.band_multipliers(b)
    PHb, PHf = LI.run_phases_fills(b), SU.fomc_late_PH(b); PHbE, PHfE = R3.run_PH_entry(b, S3m), R3.run_PH_entry(b, S3m, fomc_only=True)
    Lv3, Lv4 = R3.legs_k(PHb, PHf, b, I["keep"]), R3.legs_k(PHbE, PHfE, b, I["keep"])
    sleeve = {}
    for cn, lg in legs.items():
        I2 = dict(I); I2["leg1"] = lg
        k3 = R3.sizing(X, P.CLEAN, Lv3, I2, sel, days, "mv", "sig"); k4 = R3.sizing(X, P.CLEAN + ["dur_rate"], Lv4, I2, sel, days, "mv", "lin")
        sleeve[("v3", cn)] = R3.assemble(k3, Lv3, I2, om)["sleeve"]; sleeve[("v4", cn)] = R3.assemble(k4, Lv4, I2, om)["sleeve"]
    for v in sleeve.values(): assert int((v[v.index.dayofweek == 0] != 0).sum()) == 0, "Sleeve 必须排除周一"
    live = (I["base_live"]).reindex(days).fillna(0.0)
    N = {"live": "实盘 Sleeve", "v3s": "v3 Sleeve(①版)", "v4s": "v4 Sleeve(①版)", "bh": "QQQ 买入持有", "rv": "组合+RV(现用:RV21+实盘)",
         "a3": "① RV21+SMA200+v3Sleeve", "a4": "① RV21+SMA200+v4Sleeve", "b3": "② VT[FFA-SD]+SMA200+v3Sleeve", "b4": "② VT[FFA-SD]+SMA200+v4Sleeve"}
    series = {N["live"]: live, N["v3s"]: sleeve[("v3", "①")], N["v4s"]: sleeve[("v4", "①")], N["bh"]: bh, N["rv"]: leg_rv + live,
              N["a3"]: legs["①"] + sleeve[("v3", "①")], N["a4"]: legs["①"] + sleeve[("v4", "①")],
              N["b3"]: legs["②"] + sleeve[("v3", "②")], N["b4"]: legs["②"] + sleeve[("v4", "②")]}
    # ---- 复现检查:本环境的 v3 组合 vs repo 的 tearsheet_v3
    print("[复现] 本环境 v3 组合 vs repo tearsheet_v3(差异来自公开源重建的日线/VIX,见 data/MANIFEST_sandbox_2026-10.md5):")
    for cn, key in (("①", N["a3"]), ("②", N["b3"])):
        print("   " + cn + "  " + "  ".join(f"{pn} {TS.metrics(series[key] if per is None else series[key].loc[per[0]:per[1]])['Sharpe（绝对）']:.3f}(repo {REF_V3[(cn, pn)]:.3f})" for pn, per in PERIODS))
    # ---- 指标
    T = pd.DataFrame({nm: TS.metrics(x, bench=None if nm == N["bh"] else bh) for nm, x in series.items()})
    REPO_TAB.mkdir(parents=True, exist_ok=True); T.to_csv(REPO_TAB / "tearsheet_v4_metrics.csv")
    rows = []
    for pn, per in PERIODS:
        for nm, x in series.items():
            m = TS.metrics(x if per is None else x.loc[per[0]:per[1]])
            rows.append(dict(期间=pn, 序列=nm, Sharpe=m["Sharpe（绝对）"], CAGR=m["年化收益 CAGR"], MDD=m["最大回撤 MDD"], Calmar=m["Calmar"], Sortino=m["Sortino"]))
    PD = pd.DataFrame(rows); PD.to_csv(REPO_TAB / "tearsheet_v4_periods.csv", index=False)
    print("\n【全样本指标】"); print(T.to_string())
    print("\n【分期】"); print(PD.pivot_table(index="序列", columns="期间", values=["Sharpe", "CAGR", "MDD"], sort=False).round(3).to_string())
    yr = pd.DataFrame({nm: {y: TS.metrics(x.loc[str(y)])["Sharpe（绝对）"] for y in range(2019, 2027)} for nm, x in series.items()})
    print("\n【分年 Sharpe】"); print(yr.round(2).to_string()); yr.to_csv(REPO_TAB / "tearsheet_v4_yearly.csv")
    # ================= 作图 =================
    use_cjk(plt)
    fig = plt.figure(figsize=(24, 16)); gs = fig.add_gridspec(3, 3, hspace=.42, wspace=.22, width_ratios=[1.0, 1.0, 1.15], height_ratios=[1.25, 0.9, 0.9])
    COL = {N["live"]: "tab:blue", N["v3s"]: "tab:purple", N["v4s"]: "magenta", N["bh"]: "black", N["rv"]: "tab:cyan",
           N["a3"]: "tab:green", N["a4"]: "darkgreen", N["b3"]: "tab:orange", N["b4"]: "saddlebrown"}
    LS = {N["bh"]: ":", N["v3s"]: "--", N["v4s"]: "--", N["a3"]: "-.", N["b3"]: "-."}
    LW = {N["a4"]: 2.6, N["b4"]: 2.6, N["a3"]: 1.6, N["b3"]: 1.6}
    SPLIT = pd.Timestamp("2023-01-01")
    A = fig.add_subplot(gs[0, :2])
    for nm, x in series.items():
        eq = (1 + x).cumprod(); te = TS.metrics(x.loc["2023":"2026"])
        A.plot(eq, color=COL[nm], lw=LW.get(nm, 1.5), ls=LS.get(nm, "-"), label=f"{nm}  (test Sharpe {te['Sharpe（绝对）']:.2f}, test MDD {te['最大回撤 MDD']:.1%})")
    A.axvline(SPLIT, color="gray", lw=1.1, ls="--"); A.axvline(pd.Timestamp("2021-01-01"), color="gray", lw=0.8, ls=":")
    A.set_yscale("log"); A.set_ylabel("累计净值(对数轴)"); A.set_title("累计净值(2021 起仓位模型走步生效;2023 起 test)", loc="left", fontsize=13); A.legend(fontsize=8.8, loc="upper left"); A.grid(alpha=.3, which="both")
    B = fig.add_subplot(gs[0:2, 2]); B.axis("off")
    cols = [N["live"], N["v3s"], N["v4s"], N["a3"], N["a4"], N["b3"], N["b4"]]; short = ["实盘\nSleeve", "v3\nSleeve", "v4\nSleeve", "①v3", "①v4", "②v3", "②v4"]
    prow = []
    for pn, per in PERIODS:
        for met, lab in (("Sharpe（绝对）", "Sharpe"), ("年化收益 CAGR", "CAGR"), ("最大回撤 MDD", "MDD"), ("Calmar", "Calmar")):
            prow.append([f"{pn.split()[0]} {lab}"] + [(f"{TS.metrics(series[c] if per is None else series[c].loc[per[0]:per[1]])[met]:.2f}" if lab in ("Sharpe", "Calmar")
                                                      else f"{TS.metrics(series[c] if per is None else series[c].loc[per[0]:per[1]])[met]:.1%}") for c in cols])
    tb = B.table(cellText=prow, colLabels=["期间·指标"] + short, cellLoc="center", loc="upper center"); tb.auto_set_font_size(False); tb.set_fontsize(8.6); tb.scale(1, 1.55)
    for (r_, c_), cell in tb.get_celld().items(): cell.set_width(0.2 if c_ == 0 else 0.115)
    for j in range(len(cols) + 1): tb[0, j].set_facecolor("#d9d9d9")
    for i in range(1, len(prow) + 1): tb[i, 3].set_facecolor("#fce4f6"); tb[i, 5].set_facecolor("#e8f5e9"); tb[i, 7].set_facecolor("#fff3e0"); tb[i, 0].set_text_props(ha="left")
    B.set_title("分期对照(dev / test / 全期)\nv4 = v3 + 入场带(含成交量)+ dur_rate + 线性链接", fontsize=11, loc="left")
    Cx = fig.add_subplot(gs[1, :2])
    for nm in (N["bh"], N["a3"], N["a4"], N["b3"], N["b4"]):
        eq = (1 + series[nm]).cumprod(); Cx.plot(eq / eq.cummax() - 1, color=COL[nm], lw=LW.get(nm, 1.3), ls=LS.get(nm, "-"), label=nm)
    Cx.axvline(SPLIT, color="gray", lw=1, ls="--"); Cx.set_title("组合回撤", loc="left", fontsize=12); Cx.legend(fontsize=8.5, loc="lower left"); Cx.grid(alpha=.3)
    D = fig.add_subplot(gs[2, :2]); yy = yr[[N["bh"], N["a3"], N["a4"], N["b3"], N["b4"]]]; w = 0.16
    for j, nm in enumerate(yy.columns):
        D.bar(np.arange(len(yy)) + (j - 2) * w, yy[nm].to_numpy(), w, color=COL[nm], label=nm)
    D.set_xticks(np.arange(len(yy))); D.set_xticklabels(yy.index); D.axhline(0, color="k", lw=.8); D.set_title("分年 Sharpe(组合)", loc="left", fontsize=12); D.legend(fontsize=8, ncol=5, loc="upper left"); D.grid(alpha=.3, axis="y")
    E = fig.add_subplot(gs[2, 2])
    for nm in (N["v3s"], N["v4s"], N["live"]):
        x = series[nm]; rs = x.rolling(252).mean() / x.rolling(252).std() * np.sqrt(252); E.plot(rs.loc["2020":], color=COL[nm], lw=1.4, ls=LS.get(nm, "-"), label=nm)
    E.axhline(0, color="k", lw=.7); E.axvline(SPLIT, color="gray", lw=1, ls="--"); E.set_title("Sleeve 滚动 252 日 Sharpe", loc="left", fontsize=12); E.legend(fontsize=8.5); E.grid(alpha=.3)
    fig.suptitle("Tearsheet v4(描述性;v4 的改动为多轮事后选择,未通过预注册判定)", fontsize=14, x=0.01, ha="left")
    REPO_FIG.mkdir(parents=True, exist_ok=True); fig.savefig(REPO_FIG / "tearsheet_v4.png", dpi=110, bbox_inches="tight")
    print("\n图:outputs/figs/tearsheet_v4.png;表:outputs/tables/tearsheet_v4_{metrics,periods,yearly}.csv")


if __name__ == "__main__":
    main()
