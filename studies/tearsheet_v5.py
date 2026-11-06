"""Tearsheet v5 = v4 + (1) 买力上限 (2) 核心仓不交易带 (3) 等风险换配比 ρ = 1.5(固定)。定义在运行前写定(docs/PROPOSAL_noise_core_dd.md §6)。
  组合_t = c_t·f_t·w_t·r_QQQ,t − 成本 + c_t·ρ·f_t·Sleeve_v4,t
  c_t = clip(σ252(v4 组合)/σ252(核心仓 + ρ·Sleeve), 0.5, 1.5),前一日为止(PIT)
  f_t = min(1, 2.57 / (c·w + e·max(k)·c·ρ/3))      2.57 = 0.85 × 买力 $1,513,400 / CAPITAL_BASE $500,000
  核心仓执行权重:与目标偏离 ≥ 0.10 才调整;若不调会使占用超过 2.57 则强制调整
仓位模型 k 沿用 v4(不针对新核心仓重训)。规则无训练参数,2019 起全样本生效(c 需 126 日历史,之前取 1)。
① = RV21[0.5,1.5]×SMA200;② = VT[FFA-SD,cap1.5]×SMA200。敏感性 ρ ∈ {1, 2, 3} 只作参照。
python3 tearsheet_v5.py
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
import r2_rules_study as RR
import r3_study as R3
import round3_common as C
import sizing_models_study as M
import sizing_sigma_feed as SF
import sota_stack_study as S
import sota_unified_stack as SU
import tearsheet as TS
import tearsheet_v4 as T4
import unified_sizing_posthoc as P
from cjkfont import use_cjk
from etft.intraday.bars import load_bars

CAP = 0.85 * 1_513_400 / 500_000; RHO = 1.5; BAND = 0.10
REPO_FIG, REPO_TAB = Path("../outputs/figs"), Path("../outputs/tables")
PERIODS = (("dev 19-22", ("2019", "2022")), ("test 23-26", ("2023", "2026")), ("全 19-26", None))


def build_v5(w_core, sleeve, ev, kmax, bh, ref_port, rho=RHO, cap=CAP, band=BAND, use_ratio=True, use_band=True, use_cap=True):
    """返回 dict(port, core_w, sleeve_mult, usage_pre, f)。全部数组按交易日对齐。"""
    n = len(bh); refs = pd.Series(ref_port)
    if use_ratio:
        mix = pd.Series(w_core * bh + rho * sleeve)
        c = (refs.rolling(252, min_periods=126).std() / mix.rolling(252, min_periods=126).std()).shift(1).clip(0.5, 1.5).fillna(1.0).to_numpy()
    else:
        c = np.ones(n); rho = 1.0
    tgt = c * w_core; sm = c * rho; slv = ev * kmax / 3.0
    usage_pre = tgt + slv * sm
    f = np.minimum(1.0, cap / np.maximum(usage_pre, 1e-12)) if use_cap else np.ones(n)
    tgt = tgt * f; sm = sm * f
    w = np.empty(n); cur = tgt[0]
    for i in range(n):
        if (not use_band) or abs(tgt[i] - cur) >= band or (use_cap and cur + slv[i] * sm[i] > cap + 1e-12): cur = tgt[i]
        w[i] = cur
    cost = 2e-4 * np.abs(np.diff(np.r_[w[0], w]))
    port = w * bh - cost + sm * sleeve
    return dict(port=port, core_w=w, sleeve_mult=sm, usage_pre=usage_pre, usage_post=w + slv * sm, f=f, c=c)


def main():
    pd.set_option("display.width", 260, "display.max_rows", 200, "display.float_format", lambda v: f"{v:9.4f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    I = SF.build_inputs(b); X = I["X"].copy(); bh = I["bh"]; bhv = bh.to_numpy(); ev = I["e_vt"].fillna(0.0).to_numpy()
    dn = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days)
    X["dur_rate"] = M.family_features(b, I["X"])["dur_rate"]
    px = C.load_close("yahoo/QQQ.parquet"); r = px.pct_change().fillna(0.0)
    W = S.bh_weights(px, r, {"FFA-SD": T4.ffa_sd()}, pd.Series(1.0, index=px.index))
    al = lambda w: w.reindex(bh.index).ffill().fillna(1.0).reindex(days).fillna(1.0).to_numpy()
    wc = {"①": al(W[("RV21[0.5,1.5]", "SMA200|e=0.5")]), "②": al(W[("VT[FFA-SD,cap1.5]", "SMA200|e=0.5")])}
    _, S3m, _ = RR.band_multipliers(b)
    PHb, PHf = R3.LI.run_phases_fills(b), SU.fomc_late_PH(b); PHbE, PHfE = R3.run_PH_entry(b, S3m), R3.run_PH_entry(b, S3m, fomc_only=True)
    Lv3, Lv4 = R3.legs_k(PHb, PHf, b, I["keep"]), R3.legs_k(PHbE, PHfE, b, I["keep"])
    out = {}; diag = {}
    for cn, w in wc.items():
        I2 = dict(I); I2["leg1"] = pd.Series(w * bhv - 2e-4 * np.abs(np.diff(np.r_[w[0], w])), index=days)
        k3 = R3.sizing(X, P.CLEAN, Lv3, I2, sel, days, "mv", "sig"); k4 = R3.sizing(X, P.CLEAN + ["dur_rate"], Lv4, I2, sel, days, "mv", "lin")
        s3 = R3.assemble(k3, Lv3, I2, om)["sleeve"].to_numpy(); s4 = R3.assemble(k4, Lv4, I2, om)["sleeve"].to_numpy()
        kmax = np.maximum(k4.kl.where(om, 1.0).fillna(1.0), k4.ks.where(om, 1.0).fillna(1.0)).to_numpy()
        p3, p4 = I2["leg1"].to_numpy() + s3, I2["leg1"].to_numpy() + s4
        out[("v3", cn)], out[("v4", cn)] = p3, p4; out[("s3", cn)], out[("s4", cn)] = s3, s4
        steps = {"v4 + 买力上限": dict(use_ratio=False, use_band=False), "v4 + 买力上限 + 不交易带": dict(use_ratio=False), "v5(+ 等风险配比 ρ=1.5)": dict()}
        for nm, kw in steps.items():
            res = build_v5(w, s4, ev, kmax, bhv, p4, **kw); out[(nm, cn)] = res["port"]; diag[(nm, cn)] = res
        for rho in (1.0, 2.0, 3.0):
            res = build_v5(w, s4, ev, kmax, bhv, p4, rho=rho); out[(f"敏感性 ρ={rho:g}", cn)] = res["port"]; diag[(f"敏感性 ρ={rho:g}", cn)] = res
        diag[("v4", cn)] = dict(core_w=w, sleeve_mult=np.ones(len(days)), usage_post=w + ev * kmax / 3.0)
    live = I["base_live"].reindex(days).fillna(0.0)
    S_ = lambda a: pd.Series(a, index=days)
    series = {"实盘 Sleeve": live, "QQQ 买入持有": bh, "① v3": S_(out[("v3", "①")]), "① v4": S_(out[("v4", "①")]), "① v5": S_(out[("v5(+ 等风险配比 ρ=1.5)", "①")]),
              "② v4": S_(out[("v4", "②")]), "② v5": S_(out[("v5(+ 等风险配比 ρ=1.5)", "②")])}
    print("[复现] ① v4 OOS 2021–26(周二至周五样本日)SR", f"{(lambda x: x.mean()/x.std()*np.sqrt(252))(out[('v4','①')][om.to_numpy()]):.3f}(tearsheet_v4 1.911)")
    # ---- 指标
    T = pd.DataFrame({nm: TS.metrics(x, bench=None if nm == "QQQ 买入持有" else bh) for nm, x in series.items()})
    REPO_TAB.mkdir(parents=True, exist_ok=True); T.to_csv(REPO_TAB / "tearsheet_v5_metrics.csv")
    rows = []
    for pn, per in PERIODS:
        for nm, x in series.items():
            m = TS.metrics(x if per is None else x.loc[per[0]:per[1]])
            rows.append(dict(期间=pn, 序列=nm, Sharpe=m["Sharpe（绝对）"], CAGR=m["年化收益 CAGR"], 波动=m["年化波动"], MDD=m["最大回撤 MDD"], Calmar=m["Calmar"], Sortino=m["Sortino"]))
    PD = pd.DataFrame(rows); PD.to_csv(REPO_TAB / "tearsheet_v5_periods.csv", index=False)
    print("\n【全样本指标 2019–2026】"); print(T.to_string())
    print("\n【分期】"); print(PD.pivot_table(index="序列", columns="期间", values=["Sharpe", "CAGR", "MDD"], sort=False).round(3).to_string())
    # ---- 分解与敏感性(①)
    rows = []
    for nm in ("v4", "v4 + 买力上限", "v4 + 买力上限 + 不交易带", "v5(+ 等风险配比 ρ=1.5)", "敏感性 ρ=1", "敏感性 ρ=2", "敏感性 ρ=3"):
        x = S_(out[(nm, "①")]); d = diag[(nm, "①")]; o = (days >= "2019-01-01")
        row = dict(配置=nm)
        for pn, per in PERIODS:
            m = TS.metrics(x if per is None else x.loc[per[0]:per[1]]); row[f"{pn} SR"] = m["Sharpe（绝对）"]; row[f"{pn} MDD"] = m["最大回撤 MDD"]
        row.update(核心仓平均权重=float(d["core_w"][o].mean()), Sleeve平均乘数=float(d["sleeve_mult"][o].mean()),
                   核心仓年化换手=float(np.abs(np.diff(d["core_w"][o])).sum() / (o.sum() / 252)), 最坏占用最大=float(d["usage_post"][o].max()),
                   超过2_57天数比例=float((d["usage_post"][o] > CAP + 1e-9).mean()))
        rows.append(row)
    DT = pd.DataFrame(rows); print("\n【① 逐步分解与敏感性(2019–26;占用单位 = CAPITAL_BASE,护栏 2.57)】"); print(DT.to_string(index=False)); DT.to_csv(REPO_TAB / "tearsheet_v5_steps.csv", index=False)
    yr = pd.DataFrame({nm: {y: TS.metrics(x.loc[str(y)])["Sharpe（绝对）"] for y in range(2019, 2027)} for nm, x in series.items()})
    print("\n【分年 Sharpe】"); print(yr.round(2).to_string()); yr.to_csv(REPO_TAB / "tearsheet_v5_yearly.csv")
    # ================= 图 =================
    use_cjk(plt)
    fig = plt.figure(figsize=(24, 17)); gs = fig.add_gridspec(3, 3, hspace=.42, wspace=.22, width_ratios=[1.0, 1.0, 1.15], height_ratios=[1.25, 0.9, 0.9])
    COL = {"实盘 Sleeve": "tab:blue", "QQQ 买入持有": "black", "① v3": "tab:olive", "① v4": "tab:green", "① v5": "crimson", "② v4": "tab:orange", "② v5": "purple"}
    LS = {"QQQ 买入持有": ":", "① v3": "-.", "① v4": "--", "② v4": "--"}; LW = {"① v5": 2.8, "② v5": 2.4}
    SPLIT = pd.Timestamp("2023-01-01")
    A = fig.add_subplot(gs[0, :2])
    for nm, x in series.items():
        eq = (1 + x).cumprod(); te = TS.metrics(x.loc["2023":"2026"])
        A.plot(eq, color=COL[nm], lw=LW.get(nm, 1.5), ls=LS.get(nm, "-"), label=f"{nm}  (test Sharpe {te['Sharpe（绝对）']:.2f}, test MDD {te['最大回撤 MDD']:.1%})")
    A.axvline(SPLIT, color="gray", lw=1.1, ls="--"); A.set_yscale("log"); A.set_title("累计净值(对数轴;2023 起 test)", loc="left", fontsize=13); A.legend(fontsize=9, loc="upper left"); A.grid(alpha=.3, which="both")
    B = fig.add_subplot(gs[0:2, 2]); B.axis("off")
    cols = ["QQQ 买入持有", "① v3", "① v4", "① v5", "② v4", "② v5"]; short = ["QQQ", "①v3", "①v4", "①v5", "②v4", "②v5"]; prow = []
    for pn, per in PERIODS:
        for met, lab in (("Sharpe（绝对）", "Sharpe"), ("年化收益 CAGR", "CAGR"), ("年化波动", "波动"), ("最大回撤 MDD", "MDD"), ("Calmar", "Calmar")):
            prow.append([f"{pn.split()[0]} {lab}"] + [(f"{TS.metrics(series[c_] if per is None else series[c_].loc[per[0]:per[1]])[met]:.2f}" if lab in ("Sharpe", "Calmar")
                                                      else f"{TS.metrics(series[c_] if per is None else series[c_].loc[per[0]:per[1]])[met]:.1%}") for c_ in cols])
    tb = B.table(cellText=prow, colLabels=["期间·指标"] + short, cellLoc="center", loc="upper center"); tb.auto_set_font_size(False); tb.set_fontsize(8.6); tb.scale(1, 1.42)
    for (r_, c_), cell in tb.get_celld().items(): cell.set_width(0.2 if c_ == 0 else 0.13)
    for j in range(len(cols) + 1): tb[0, j].set_facecolor("#d9d9d9")
    for i in range(1, len(prow) + 1): tb[i, 4].set_facecolor("#fde0e0"); tb[i, 6].set_facecolor("#efe0f7"); tb[i, 0].set_text_props(ha="left")
    B.set_title("分期对照\nv5 = v4 + 买力上限 2.57 + 核心仓不交易带 0.10 + 等风险配比 ρ=1.5", fontsize=11, loc="left")
    Cx = fig.add_subplot(gs[1, :2])
    for nm in ("QQQ 买入持有", "① v4", "① v5", "② v5"):
        eq = (1 + series[nm]).cumprod(); Cx.plot(eq / eq.cummax() - 1, color=COL[nm], lw=LW.get(nm, 1.3), ls=LS.get(nm, "-"), label=nm)
    Cx.axvline(SPLIT, color="gray", lw=1, ls="--"); Cx.set_title("组合回撤", loc="left", fontsize=12); Cx.legend(fontsize=9, loc="lower left"); Cx.grid(alpha=.3)
    D = fig.add_subplot(gs[2, :2]); yy = yr[["QQQ 买入持有", "① v3", "① v4", "① v5", "② v5"]]; wd = 0.16
    for j, nm in enumerate(yy.columns): D.bar(np.arange(len(yy)) + (j - 2) * wd, yy[nm].to_numpy(), wd, color=COL[nm], label=nm)
    D.set_xticks(np.arange(len(yy))); D.set_xticklabels(yy.index); D.axhline(0, color="k", lw=.8); D.set_title("分年 Sharpe", loc="left", fontsize=12); D.legend(fontsize=8.5, ncol=5, loc="upper left"); D.grid(alpha=.3, axis="y")
    E = fig.add_subplot(gs[2, 2]); d4, d5 = diag[("v4", "①")], diag[("v5(+ 等风险配比 ρ=1.5)", "①")]
    E.plot(days, pd.Series(d4["usage_post"]).rolling(5).max(), color="tab:green", lw=1, label="① v4 最坏占用")
    E.plot(days, pd.Series(d5["usage_post"]).rolling(5).max(), color="crimson", lw=1.2, label="① v5 最坏占用")
    E.plot(days, d5["core_w"], color="k", lw=.8, alpha=.6, label="① v5 核心仓权重")
    E.axhline(CAP, color="red", ls="--", lw=1, label="护栏 2.57"); E.set_title("最坏情形买力占用(× CAPITAL_BASE,5 日滚动最大)", loc="left", fontsize=11); E.legend(fontsize=8); E.grid(alpha=.3)
    fig.suptitle("Tearsheet v5(v4 与 v5 的新增规则均为多轮研究后选定,未经前向验证;ρ = 1.5 为运行前固定)", fontsize=14, x=0.01, ha="left")
    REPO_FIG.mkdir(parents=True, exist_ok=True); fig.savefig(REPO_FIG / "tearsheet_v5.png", dpi=110, bbox_inches="tight")
    print("\n图:outputs/figs/tearsheet_v5.png;表:outputs/tables/tearsheet_v5_{metrics,periods,steps,yearly}.csv")


if __name__ == "__main__":
    main()
