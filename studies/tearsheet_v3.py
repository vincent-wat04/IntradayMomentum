"""Tearsheet v3:①/② 的 Sleeve 换成 v3 = SOTA(FOMC≥14:30 + CPI 回补 1× + FFA-P 跳过)+ 统一仓位(按各自组合重训)+ 15:40 平仓。
以下为 v2 原说明:
Tearsheet v2:在 tearsheet.py(上一版)的 4 条线基础上,叠加
  ① RV21[0.5,1.5] + SMA200(e=0.5) 的 QQQ 腿 + 新 Sleeve
  ② VT[FFA-SD, cap1.5] + SMA200(e=0.5) 的 QQQ 腿 + 新 Sleeve
新 Sleeve = 实盘 + FOMC 日只在 ≥14:30 开仓 + CPI 缺口回补 1× + 跳过 FFA-P 预测最低 20% 振幅日。
三个组件均在 2019–2026 上事后选出(样本内),见 docs/PROPOSAL_sota_stack.md。不改动 tearsheet.py。
python3 tearsheet_v3.py"""
from __future__ import annotations

import os
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cjkfont import use_cjk
import tearsheet as TS
import sota_stack_study as S
import lowvol_avoidance_study as L
import regime_features as RF
import round3_common as C
from etft.intraday.bars import load_bars

use_cjk(plt)
OUT = TS.OUT; FIG = TS.FIG; FIG.mkdir(parents=True, exist_ok=True)
REPO_FIG = Path("../outputs/figs"); REPO_TAB = Path("../outputs/tables")
N_OLD = "实盘 Sleeve"; N_NEW = "v3 Sleeve(SOTA+统一仓位+15:40平仓,①版)"; N_BH = "QQQ 买入持有"; N_C11 = "组合 1:1(QQQ+实盘)"
N_RV = "组合+RV(现用:RV21+实盘)"; N_A = "① RV21+SMA200+v3Sleeve"; N_B = "② VT[FFA-SD]+SMA200+v3Sleeve"
PERIODS = (("dev 19-22", ("2019", "2022")), ("test 23-26", ("2023", "2026")), ("全 19-26", None))


def leg_from(w, bh):
    w = w.reindex(bh.index).ffill().fillna(1.0); return w * bh - 2e-4 * w.diff().abs().fillna(0.0), w


def main():
    cur = pd.read_pickle(OUT / "strategy_comparison_curves.pkl")
    bh = pd.read_pickle(OUT / "strategy_comparison_bh.pkl").dropna(); bh.index = pd.to_datetime(bh.index)
    s_old = cur[TS.VER].copy(); s_old.index = pd.to_datetime(s_old.index); s_old = s_old.reindex(bh.index).fillna(0.0)
    # ---- 新 Sleeve(同 sleeve_v2_compare.py)
    b = load_bars("QQQ"); days = b.days
    V = pd.read_pickle(L.OUT / "sota_sleeve.pkl")["VER"]; S0, S1 = V["S0 实盘"], V["S1 +FOMC≥14:30"]
    assert float((S0.reindex(bh.index).fillna(0) - s_old).abs().max()) < 1e-12
    cand, _, _ = S.build_candidates(b); ov = S.cpi_overlay(b); skip = (S.roll_rank(cand["FFA-P"]).reindex(days) <= 0.2).fillna(False)
    s_new_v2_unused = (S1 * (~skip).astype(float) + ov).reindex(bh.index).fillna(0.0)
    s_v2 = (S1 * (~skip).astype(float) + ov).reindex(bh.index).fillna(0.0)
    # ---- v3:按组合重训的统一仓位(sota_unified_stack.py)+ 15:40 平仓(letf_posthoc.py 的 G)
    import sota_unified_stack as SU, letf_posthoc as LP, liquidity_inverse_study as LI, disp_pred_study as DP
    from event_calendar import FOMC_DECISION
    Uu = pd.read_pickle(L.OUT / "sota_unified.pkl"); L2, S2, ovu = Uu["L2"], Uu["S2"], Uu["ov"]
    Xf = pd.read_pickle(L.OUT / "unified_sizing.pkl")["X"]; yrs = days.year
    dn = DP.targets(DP.Arr(b), 30).dn; fullm = pd.Series(b.last_idx.to_numpy() >= 389, index=days); tdm = pd.Series(days.dayofweek != 0, index=days)
    om = pd.Series((tdm & fullm & Xf.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy() & (yrs >= 2021), index=days)
    colsb = list(b.c.columns); ob = b.o.to_numpy()
    GLb, GSb = LP.G_series(LI.run_phases_fills(b), days, ob, colsb, "15:40"); GLf, GSf = LP.G_series(SU.fomc_late_PH(b), days, ob, colsb, "15:40")
    fm = pd.Series(days.isin(pd.DatetimeIndex(FOMC_DECISION)), index=days); GL, GS = GLb.where(~fm, GLf), GSb.where(~fm, GSf)
    keep = (~skip).astype(float)
    v3 = {}
    for cn in ("① RV21+SMA200", "② VT[FFA-SD]+SMA200"):
        k = Uu["kB"][cn]; kl = k.kl.where(om, 1.0).fillna(1.0); ks = k.ks.where(om, 1.0).fillna(1.0)
        sz = kl * L2 + ks * S2 + ovu
        chk = float((sz + Uu["series"][cn + " | SOTA"].sub(Uu["L2"] + Uu["S2"] + Uu["ov"]) - Uu["series"][cn + " | SOTA + 统一仓位(按组合重训)"]).abs().max()) * 1e4
        v3[cn] = (sz - (kl * GL + ks * GS) * keep).reindex(bh.index).fillna(0.0)
        assert int((v3[cn][v3[cn].index.dayofweek == 0] != 0).sum()) == 0, "v3 Sleeve 必须排除周一"
        print(f"校验({cn}):统一仓位重建 vs sota_unified 最大差 {chk:.6f}bp;15:40 平仓日均影响 {1e4*float((v3[cn]-sz.reindex(bh.index).fillna(0.0)).mean()):+.3f}bp")
    s_new = v3["① RV21+SMA200"]
    assert int((s_new[s_new.index.dayofweek == 0] != 0).sum()) == 0, "新 Sleeve 必须排除周一"
    # ---- QQQ 腿
    px = C.load_close("yahoo/QQQ.parquet"); r = px.pct_change().fillna(0.0)
    A2C = pd.read_pickle(L.OUT / "regime_timing_r2.pkl")["E2"]["A2_C|e0.5|alone"]
    W = S.bh_weights(px, r, {"FFA-SD": cand["FFA-SD"]}, A2C)
    leg_rv, w_rv = leg_from(W[("RV21[0.5,1.5]", "无趋势")], bh)
    leg_a, w_a = leg_from(W[("RV21[0.5,1.5]", "SMA200|e=0.5")], bh)
    leg_b, w_b = leg_from(W[("VT[FFA-SD,cap1.5]", "SMA200|e=0.5")], bh)
    old_rv_leg, _ = TS.qqq_leg(bh, q=1.0)
    print(f"校验:本脚本的 RV21 腿 vs 上一版 tearsheet 的 qqq_leg 最大日差 {float((leg_rv - old_rv_leg).abs().max())*1e4:.3f}bp")
    series = {N_OLD: s_old, N_NEW: s_new, N_BH: bh, N_C11: bh + s_old, N_RV: old_rv_leg + s_old, N_A: leg_a + v3["① RV21+SMA200"], N_B: leg_b + v3["② VT[FFA-SD]+SMA200"], "_①v2": leg_a + s_v2, "_②v2": leg_b + s_v2}
    # ---- 指标
    ref = {k: series.pop(k) for k in ("_①v2", "_②v2")}
    print("\n【v2 → v3 对照(全样本)】")
    for a_, b_ in (("_①v2", N_A), ("_②v2", N_B)):
        for nm_, x_ in ((a_, ref[a_]), (b_, series[b_])):
            m_ = TS.metrics(x_, bench=bh); print(f"  {nm_:32s} Sharpe {m_['Sharpe（绝对）']:.3f}  CAGR {m_['年化收益 CAGR']:.2%}  MDD {m_['最大回撤 MDD']:.2%}  Calmar {m_['Calmar']:.2f}  最差日 {m_['最差单日']:.2%}")
    T = pd.DataFrame({nm: TS.metrics(x, bench=bh if nm != N_BH else None) for nm, x in series.items()})
    T.loc["平均 QQQ 权重"] = [np.nan, np.nan, 1.0, 1.0, float(w_rv.mean()), float(w_a.mean()), float(w_b.mean())]
    T.to_csv(OUT / "tearsheet_v3_metrics.csv"); REPO_TAB.mkdir(parents=True, exist_ok=True); T.to_csv(REPO_TAB / "tearsheet_v3_metrics.csv")
    PT = {}
    for pn, per in PERIODS:
        sel = (lambda x: x) if per is None else (lambda x, per=per: x.loc[per[0]:per[1]])
        PT[pn] = {nm: TS.metrics(sel(x)) for nm, x in series.items()}
    rows = []
    for pn in PT:
        for nm in series: m = PT[pn][nm]; rows.append(dict(期间=pn, 序列=nm, Sharpe=m["Sharpe（绝对）"], CAGR=m["年化收益 CAGR"], MDD=m["最大回撤 MDD"], Calmar=m["Calmar"], Sortino=m["Sortino"]))
    PD = pd.DataFrame(rows); PD.to_csv(REPO_TAB / "tearsheet_v3_periods.csv", index=False)
    pd.set_option("display.width", 250, "display.float_format", lambda x: f"{x:9.4f}")
    print("\n【全样本指标】"); print(T.to_string())
    print("\n【分期】"); print(PD.pivot_table(index="序列", columns="期间", values=["Sharpe", "CAGR", "MDD"], sort=False).round(3).to_string())
    # ================= 作图 =================
    fig = plt.figure(figsize=(24, 17))
    gs = fig.add_gridspec(4, 4, hspace=.48, wspace=.24, width_ratios=[1.0, 1.0, 0.95, 1.25], height_ratios=[1.15, 0.95, 0.95, 0.95])
    COL = {N_OLD: "tab:blue", N_NEW: "tab:purple", N_BH: "black", N_C11: "tab:red", N_RV: "tab:cyan", N_A: "tab:green", N_B: "tab:orange"}
    LS = {N_BH: ":", N_C11: "-", N_OLD: "-", N_NEW: "--"}
    LW = {N_A: 2.6, N_B: 2.6, N_RV: 2.2, N_C11: 1.6}
    SPLIT = pd.Timestamp("2023-01-01")
    # (1) 净值
    A = fig.add_subplot(gs[0, :3])
    for nm, x in series.items():
        eq = (1 + x).cumprod(); m = TS.metrics(x)
        A.plot(eq, color=COL[nm], lw=LW.get(nm, 1.6), ls=LS.get(nm, "-"), label=f"{nm}  (Sharpe {m['Sharpe（绝对）']:.2f}, CAGR {m['年化收益 CAGR']:.1%}, MDD {m['最大回撤 MDD']:.1%})")
    A.axvline(SPLIT, color="gray", lw=1.1, ls="--"); A.set_yscale("log"); A.set_ylabel("累计净值(对数轴,初始=1)")
    A.set_title("累计净值曲线", fontsize=13, loc="left"); A.legend(fontsize=9.3, loc="upper left"); A.grid(alpha=.3, which="both")
    A.text(SPLIT, A.get_ylim()[0] * 1.04, " test 起点 2023", color="gray", fontsize=9)
    # (2) 指标表
    B = fig.add_subplot(gs[0:2, 3]); B.axis("off")
    cols = [N_OLD, N_NEW, N_RV, N_A, N_B]; short = ["实盘Sleeve", "v3 Sleeve\n(①版)", "组合+RV(现用)", "①RV21+SMA\n+v3Sleeve", "②VT+SMA\n+v3Sleeve"]
    keys = [k for k in T.index if k != "平均 QQQ 权重"] + ["平均 QQQ 权重"]
    cells = [[k] + [("—" if T.loc[k, c] != T.loc[k, c] else (f"{T.loc[k, c]:.2f}" if k == "平均 QQQ 权重" else TS.fmt(k, T.loc[k, c]))) for c in cols] for k in keys]
    tb = B.table(cellText=cells, colLabels=["指标"] + short, cellLoc="right", colLoc="center", loc="upper center")
    tb.auto_set_font_size(False); tb.set_fontsize(7.8); tb.scale(1, 1.38)
    for (r_, c_), cell in tb.get_celld().items(): cell.set_width(0.24 if c_ == 0 else 0.152)
    for j in range(len(cols) + 1): tb[0, j].set_facecolor("#d9d9d9"); tb[0, j].set_text_props(weight="bold", fontsize=7.4)
    for i in range(1, len(cells) + 1):
        tb[i, 0].set_text_props(ha="left"); tb[i, 4].set_facecolor("#e8f5e9"); tb[i, 5].set_facecolor("#fff3e0"); tb[i, 3].set_facecolor("#e6f7fa")
    B.set_title("完整指标(全样本 2019-01 ~ 2026-10)\nβ/α/相关性为对 QQQ 回归;组合含 QQQ 腿", fontsize=10.5, loc="left")
    # (3) 分期表
    P2 = fig.add_subplot(gs[1, 2]); P2.axis("off")
    show = [N_OLD, N_NEW, N_BH, N_RV, N_A, N_B]; sh2 = ["实盘Sl", "v3Sl", "QQQ", "现用组合", "①v3", "②v3"]
    prow = []
    for pn in PT:
        for met, lab in (("Sharpe（绝对）", "Sharpe"), ("年化收益 CAGR", "CAGR"), ("最大回撤 MDD", "MDD")):
            prow.append([f"{pn.split()[0]} {lab}"] + [(f"{PT[pn][nm][met]:.2f}" if lab == "Sharpe" else f"{PT[pn][nm][met]:.1%}") for nm in show])
    t2 = P2.table(cellText=prow, colLabels=["期间·指标"] + sh2, cellLoc="center", loc="center")
    t2.auto_set_font_size(False); t2.set_fontsize(8.2); t2.scale(1, 1.45)
    for (r_, c_), cell in t2.get_celld().items(): cell.set_width(0.22 if c_ == 0 else 0.13)
    for j in range(len(show) + 1): t2[0, j].set_facecolor("#d9d9d9")
    for i in range(1, len(prow) + 1): t2[i, 5].set_facecolor("#e8f5e9"); t2[i, 6].set_facecolor("#fff3e0"); t2[i, 0].set_text_props(ha="left")
    P2.set_title("分期对照(dev / test / 全)", fontsize=11.5, loc="left", pad=14)
    # (4) 回撤
    Cx = fig.add_subplot(gs[1, :2])
    for nm in (N_BH, N_RV, N_A, N_B, N_NEW):
        x = series[nm]; eq = (1 + x).cumprod(); dd = eq / eq.cummax() - 1
        Cx.plot(dd * 100, color=COL[nm], lw=1.5, ls=LS.get(nm, "-"), label=f"{nm}  (MDD {dd.min():.1%})")
        if nm in (N_A, N_B): Cx.fill_between(dd.index, dd * 100, 0, color=COL[nm], alpha=.10)
    Cx.set_ylabel("回撤 (%)"); Cx.set_title("水下回撤曲线", fontsize=13, loc="left"); Cx.legend(fontsize=8.3, loc="lower right", ncol=2, framealpha=.92); Cx.set_ylim(-48, 2); Cx.grid(alpha=.3)
    # (5) 年度收益柱状
    Y = fig.add_subplot(gs[2, :2])
    yr = pd.DataFrame({nm: (1 + series[nm]).groupby(series[nm].index.year).prod() - 1 for nm in (N_BH, N_RV, N_A, N_B)}) * 100
    xs = np.arange(len(yr)); wbar = 0.2
    for i, nm in enumerate(yr.columns):
        Y.bar(xs + (i - 1.5) * wbar, yr[nm], wbar, color=COL[nm], label=nm)
        for xx, v in zip(xs, yr[nm]): Y.text(xx + (i - 1.5) * wbar, v + (1.5 if v >= 0 else -4.5), f"{v:.0f}", ha="center", fontsize=6.8)
    Y.set_xticks(xs); Y.set_xticklabels([str(y) + ("*" if y == 2026 else "") for y in yr.index]); Y.axhline(0, color="k", lw=.8)
    Y.set_ylabel("年度收益 (%)"); Y.set_title("年度收益(2026* 截至 10 月)", fontsize=13, loc="left"); Y.legend(fontsize=8.5, ncol=2); Y.grid(alpha=.3, axis="y")
    # (6) Sleeve 对比:年度
    Z = fig.add_subplot(gs[2, 2])
    sy = pd.DataFrame({nm: (1 + series[nm]).groupby(series[nm].index.year).prod() - 1 for nm in (N_OLD, N_NEW)}) * 100
    for i, nm in enumerate(sy.columns): Z.bar(np.arange(len(sy)) + (i - .5) * .38, sy[nm], .38, color=COL[nm], label=nm)
    Z.set_xticks(range(len(sy))); Z.set_xticklabels(sy.index, fontsize=8.5); Z.axhline(0, color="k", lw=.8)
    Z.set_title("Sleeve 年度收益 (%):实盘 vs 新", fontsize=11.5, loc="left"); Z.legend(fontsize=8.5); Z.grid(alpha=.3, axis="y")
    # (7) QQQ 权重
    Wg = fig.add_subplot(gs[2, 3])
    for nm, w in ((N_RV, w_rv), (N_A, w_a), (N_B, w_b)): Wg.plot(w.rolling(5).mean(), color=COL[nm], lw=1.2, label=f"{nm.split('+v3')[0]}  均值 {w.mean():.2f}")
    Wg.axhline(1, color="k", lw=.8, ls=":"); Wg.set_title("QQQ 腿权重(5 日均)", fontsize=11.5, loc="left"); Wg.legend(fontsize=8.2); Wg.grid(alpha=.3)
    # (8) 月度热力图(① 组合)
    E = fig.add_subplot(gs[3, :2])
    mo = series[N_A].resample("ME").apply(lambda z: (1 + z).prod() - 1)
    M = pd.DataFrame({"y": mo.index.year, "m": mo.index.month, "v": mo.values}); Pm = M.pivot(index="y", columns="m", values="v") * 100
    vmax = float(np.nanmax(np.abs(Pm.to_numpy()))); im = E.imshow(Pm.to_numpy(), cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    E.set_xticks(range(12)); E.set_xticklabels([f"{i}月" for i in range(1, 13)], fontsize=9); E.set_yticks(range(len(Pm.index))); E.set_yticklabels(Pm.index, fontsize=9)
    for i in range(Pm.shape[0]):
        for j in range(Pm.shape[1]):
            v = Pm.to_numpy()[i, j]
            if v == v: E.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.6)
    E.set_title("① RV21+SMA200+v3Sleeve 月度收益 (%)", fontsize=13, loc="left"); fig.colorbar(im, ax=E, fraction=.025, pad=.01)
    # (9) 新 Sleeve 日收益分布
    Dh = fig.add_subplot(gs[3, 2])
    for nm in (N_OLD, N_NEW):
        x = series[nm]; Dh.hist(x[x != 0] * 1e4, bins=70, color=COL[nm], alpha=.45, label=f"{nm}(VaR95 {x.quantile(.05)*1e4:.0f}bp)")
    Dh.axvline(0, color="k", lw=1); Dh.set_xlabel("日收益 (bp)"); Dh.set_ylabel("天数"); Dh.set_title("Sleeve 日收益分布(剔除 0)", fontsize=11, loc="left"); Dh.legend(fontsize=8); Dh.grid(alpha=.3)
    # (10) 滚动 Sharpe
    F = fig.add_subplot(gs[3, 3])
    for nm in (N_BH, N_RV, N_A, N_B):
        x = series[nm]; rs = x.rolling(252).mean() / x.rolling(252).std() * np.sqrt(252); F.plot(rs, color=COL[nm], lw=1.4, ls=LS.get(nm, "-"), label=nm)
    F.axhline(0, color="k", lw=1); F.set_ylabel("滚动 12 个月 Sharpe"); F.set_title("滚动 Sharpe(252 日)", fontsize=12, loc="left"); F.legend(fontsize=8); F.grid(alpha=.3)
    note = ("口径:Sleeve = QQQ 日内噪声区间动量(30 相位均值、TQQQ 执行、排除周一);QQQ 腿成本 2bp/边;忽略融资利息(QQQ 权重最高 1.5,需要杠杆)。\n"
            "v3 Sleeve = 实盘 + FOMC 决议日只在 ≥14:30 开仓 + 跳过 FFA-P 预测振幅最低 20% 的日子 + CPI 缺口回补 1×(开盘入场、收盘平仓)"
            " + 统一仓位(多/空两腿各自 k = 2·sigmoid(θ·z),7 个去污染特征,以对应组合的均值–方差效用 walk-forward 重训,2021 起生效,此前 k=1)+ Sleeve 仓位改在 15:40 平仓(原为 MOC)。\n"
            "① QQQ 腿 = RV21 百分位线性权重 [0.5,1.5] × SMA200(低于 200 日线 ×0.5);② = min(1.5, c / FFA-SD 预测波动) × 同一 SMA200;① / ② 的统一仓位分别按各自组合重训,所以两者的 v3 Sleeve 略有不同(图中『v3 Sleeve』为①版)。\n"
            "【重要】新 Sleeve 的全部组件(FOMC/CPI/跳过/统一仓位/15:40 平仓)与 SMA200/VT 腿都是在 2019–2026 这段数据上比较后选出的(统一仓位为 walk-forward,其余为事后选择),"
            "dev/test 只是时间切分,不是样本外;各组件的增量区间均含 0。15:40 平仓按与 MOC 相同的价差计费,实际 MOC 竞价通常更便宜,故该项优势略被高估。2022 年统一仓位与提前平仓均为负贡献。")
    fig.text(.012, .004, note, fontsize=8.6, va="bottom", color="#333333")
    fig.suptitle("QQQ 日内噪声区间动量策略 — 业绩总览 v3(① / ② = QQQ 腿 + SOTA + 统一仓位 + 15:40 平仓)", fontsize=16, x=0.36, y=.988)
    fig.subplots_adjust(left=.04, right=.985, top=.945, bottom=.085)
    p = FIG / "tearsheet_v3.png"; fig.savefig(p, dpi=115)
    REPO_FIG.mkdir(parents=True, exist_ok=True); shutil.copy(p, REPO_FIG / "tearsheet_v3.png")
    print(f"\n图 -> {p}(副本 {REPO_FIG/'tearsheet_v3.png'});指标 -> {REPO_TAB/'tearsheet_v3_metrics.csv'}、tearsheet_v3_periods.csv")


if __name__ == "__main__":
    main()
