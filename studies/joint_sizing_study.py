"""联合仓位研究 —— 预注册(运行前写定,2026-11-06)

问题(用户提出):
  Q1  把核心仓(B&H 腿)也放进 PPP,SMA、σ 等作为特征一起优化,会怎样?
  Q2  Sleeve 仍由 PPP 决定,但不再固定 ρ = 1.5:把核心仓和 Sleeve 当两个资产,在总风险约束下用凸优化解出两者的权重。换手怎么处理?

基线
  v4      核心仓规则权重 w_rule(RV21 分位 × SMA200),Sleeve 用 v4 的 PPP(线性链接 + dur_rate + 入场带);无买力上限、无配比
  v5pit   v5(ρ=1.5、买力上限 2.57、不交易带 0.10),买力缩放用开盘前可知的 k 上界 2(PR #2 修正后的口径)
  Sleeve 部分与 v4 完全相同(同一批成交、同一套特征、同样的 λ 网格);核心仓头为零时必须逐位复现 v4 的 k,否则中止。

配置(共 6 个,不增删)
  J1  联合 PPP:w_t = clip(w_rule,t + θ_Cᵀz_t, 0, 1.5);核心仓特征 = 收盘前可知的 CLEAN 子集(lvix_z、dlvix、rvm_ratio)
  J2  同上,再加 SMA200 下行哑变量、RV21 分位、log σ14、21 日动量(用户提出的 SMA / σ 特征)
      (gap_n、r30_n、b30_n、rvm_panic、dur_rate 含 10:00 之前才有的信息,核心仓是收盘到收盘持有,**不能**用)
      目标 = 与 v4 相同的均值-方差效用,γ = 基线组合均值/方差;θ 岭惩罚向规则收缩;λ ∈ {0.1,0.3,1,3,10} 留一年交叉验证;每年初重训。
      J1/J2 各报两个版本:raw(无上限无带)和 impl(买力上限 + 不交易带 0.10,PIT 的 k = 2 上界)。
  O1  静态凸优化:每个 OOS 年初,用此前全部数据(2019 起)估计核心仓(规则净收益)和 Sleeve(k = 1 的代理)的 μ、Σ,
      解 max μᵀa s.t. aᵀΣa ≤ T²,a ≥ 0,T = 训练期 1:1 混合的波动;a_core 乘规则权重,a_sleeve 乘 v4 的 Sleeve;
      μ 向"夏普相等"收缩的强度 s ∈ {0, 0.5, 1}(3 个配置:O1-s0、O1-s05、O1-s1);再加买力上限 + 不交易带(impl)。
  O2  动态版:252 日滚动窗口,s = 0.5,每日重解,a 再做半衰期 63 日的指数平滑;买力上限 + 不交易带。
  换手:核心仓年化换手、成本(2bp/单位)、Sleeve 乘数均值,每个配置都报;不交易带 0.10 沿用 v5。
事后补充(看到 J1/J2 的结果后加,标记为 post-hoc,不参与"优于"判定的族内校正):J1s、J2s —— 核心仓头的岭惩罚乘以 (std(核心收益)/平均 std(Sleeve 腿))²,
  因为同一个 λ 对收益尺度大 5–6 倍的核心仓头几乎不起作用(J1/J2 的 λ 在 2023–26 年全部选到网格上限 10)。
判据(相对 v5pit,OOS 2021–26,配对块自助 L = 10、2000 次):ΔSR > 0 且 95% 下界 > −0.05,
  且 2023–26 的 ΔSR > 0,且核心仓换手 ≤ 1.5 × v5pit,三条同时满足才记为"优于 v5pit"。否则只记 E1。
另报:分布面板、MDD、年份 Sharpe、各年选出的 λ / a。
python3 joint_sizing_study.py   (在 studies/ 下运行;约 3–5 分钟)
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "audit")); sys.path.insert(0, str(HERE.parent))
import review_checks as RC
import r3_study as R3, noise_core_dd as NC, round3_common as C, unified_sizing_posthoc as P, unified_sizing_study as U, sizing_models_study as M
import ivol_objective_study as AO
from quantkit.backtest import daily, portfolio
from quantkit.stats import bootstrap, perf
from quantkit.stats.cv import walk_forward_tuned

TAB = HERE.parent / "outputs" / "tables"; WMAX = 1.5; CAP = portfolio.CAP_DEFAULT
CORE_BASE = ["lvix_z", "dlvix", "rvm_ratio"]; CORE_EXTRA = ["sma_dn", "rv_pct", "lsd14", "mom21"]


# ------------------------------------------------------------------ 联合 PPP
def joint_fit(Zs, Zc, Lb, Sb, Eb, rc, wr, lam, use_core=True, core_pen=1.0):
    """k_leg = clip(1 + Zsθ, 0, 2)(线性链接,同 v4);w = clip(w_rule + Zcθ_C, 0, WMAX)。R、L、S、rc 以 bp 计。
    R = kl·L + ks·S + Eb + (w − w_rule)·rc;Eb = 规则核心仓净收益 + CPI 回补。θ_C = 0 时 R 与 v4 的 R 逐位相同。"""
    n, p1 = Zs.shape; pc = Zc.shape[1] if use_core else 0; B = Lb + Sb + Eb; gam = B.mean() / B.var()
    def f(th):
        kl, dl = M.link(Zs @ th[:p1], "lin"); ks, ds = M.link(Zs @ th[p1:2 * p1], "lin")
        if use_core:
            raw = wr + Zc @ th[2 * p1:]; w = np.clip(raw, 0, WMAX); dw = ((raw > 0) & (raw < WMAX)).astype(float)
        else:
            w = wr; dw = np.zeros(n)
        R = kl * Lb + ks * Sb + Eb + (w - wr) * rc; Rc = R - R.mean()
        dRl = (Lb * dl)[:, None] * Zs; dRs = (Sb * ds)[:, None] * Zs
        g = [dRl.mean(0) - gam * (Rc[:, None] * dRl).mean(0), dRs.mean(0) - gam * (Rc[:, None] * dRs).mean(0)]
        if use_core:
            dRc = (rc * dw)[:, None] * Zc; g.append(dRc.mean(0) - gam * (Rc[:, None] * dRc).mean(0))
        pw = np.r_[np.ones(2 * p1), np.full(pc, core_pen)]                       # 核心仓头的岭惩罚权重(默认 1 = 与 Sleeve 头相同)
        g = np.concatenate(g) - 2 * lam * pw * th
        return -(R.mean() - 0.5 * gam * R.var() - lam * np.sum(pw * th ** 2)), -g
    r = minimize(f, np.zeros(2 * p1 + pc), jac=True, method="L-BFGS-B", options={"maxiter": 500}); return r.x, gam


def run_joint(X, feats_s, feats_c, Lb, Sb, Eb, rc, wr, sel, days, om_years=range(2021, 2027), use_core=True, scale_pen=False):
    """走步:每个 OOS 年用此前全部样本训练,λ 由内层留一年 CV 选;返回 (DataFrame[kl, ks, w], 各年 λ)。"""
    def design(idx, ref, feats):
        mu, sd = X.loc[ref, feats].mean(), X.loc[ref, feats].std().replace(0, 1)
        return np.column_stack([np.ones(int(idx.sum())), ((X.loc[idx, feats] - mu) / sd).to_numpy()])
    def fit(mask, lam):
        Zs, Zc = design(mask, mask, feats_s), design(mask, mask, feats_c)
        cp = (np.std(rc[mask]) / np.mean([np.std(Lb[mask]), np.std(Sb[mask])])) ** 2 if scale_pen else 1.0
        th, gam = joint_fit(Zs, Zc, Lb[mask], Sb[mask], Eb[mask], rc[mask], wr[mask], lam, use_core, cp)
        return dict(th=th, gam=gam, mask=mask)
    def pred(model, idx):
        m = model["mask"]; Zs, Zc = design(idx, m, feats_s), design(idx, m, feats_c); p1 = Zs.shape[1]; th = model["th"]
        kl, _ = M.link(Zs @ th[:p1], "lin"); ks, _ = M.link(Zs @ th[p1:2 * p1], "lin")
        w = np.clip(wr[idx] + Zc @ th[2 * p1:], 0, WMAX) if use_core else wr[idx]
        return np.column_stack([kl, ks, w])
    def score(model, held):
        a = pred(model, held); gam = model["gam"]; R = a[:, 0] * Lb[held] + a[:, 1] * Sb[held] + Eb[held] + (a[:, 2] - wr[held]) * rc[held]
        B = Lb[held] + Sb[held] + Eb[held]; return U.utility(R, gam) - U.utility(B, gam)
    preds, chosen = walk_forward_tuned(days, sel, U.LAMS, fit, pred, score, om_years.start, om_years.stop - 1)
    out = pd.DataFrame({"kl": np.nan, "ks": np.nan, "w": np.nan}, index=days)
    for yr, a in preds.items():
        te = sel & (days.year == yr); out.loc[te, ["kl", "ks", "w"]] = a
    return out, chosen


# ------------------------------------------------------------------ 凸优化权重
def mv_weights(mu, Sigma, T, shrink):
    """max μᵀa s.t. aᵀΣa ≤ T², a ≥ 0。μ 向"夏普相等"收缩:μ_i ← (1−s)μ_i + s·σ_i·mean(SR)。两资产,闭式解 + 角解。"""
    sig = np.sqrt(np.diag(Sigma)); sr = mu / sig; mu_s = (1 - shrink) * mu + shrink * sig * sr.mean()
    a = np.linalg.solve(Sigma, mu_s)
    if (a < 0).any():
        j = int(np.argmax(mu_s / sig)); a = np.zeros(2); a[j] = 1.0 / sig[j]
        return a * T
    return a * T / np.sqrt(a @ Sigma @ a)


def static_scales(core, slv, days, shrink, years=range(2021, 2027)):
    a = pd.DataFrame(np.nan, index=days, columns=["ac", "as"]); info = {}
    for yr in years:
        tr = (days.year < yr) & (days >= "2019-01-01"); x = np.vstack([core[tr], slv[tr]]); S = np.cov(x); mu = x.mean(1)
        T = np.sqrt(np.ones(2) @ S @ np.ones(2)); w = mv_weights(mu, S, T, shrink); a.loc[days.year == yr, :] = w; info[yr] = w
    return a.ffill().fillna(1.0), info


def rolling_scales(core, slv, days, shrink, win=252, half_life=63):
    n = len(days); A = np.ones((n, 2))
    for i in range(win, n):
        x = np.vstack([core[i - win:i], slv[i - win:i]]); S = np.cov(x); mu = x.mean(1); T = np.sqrt(np.ones(2) @ S @ np.ones(2)); A[i] = mv_weights(mu, S, T, shrink)
    A = pd.DataFrame(A, index=days).ewm(halflife=half_life, adjust=False).mean().to_numpy(); return pd.DataFrame(A, index=days, columns=["ac", "as"])


# ------------------------------------------------------------------ 主程序
def main():
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", lambda v: f"{v:9.4f}")
    E = RC.setup(); days, om, b, I, X, sel = E["days"], E["om"], E["b"], E["I"], E["X"].copy(), E["sel"]; o = om.to_numpy(); yrs = days.year; n = len(days)
    base = RC.v4(E); legs = base["legs"]; s4 = base["sleeve"].to_numpy(); p4 = base["port"].to_numpy()
    px = C.load_close("yahoo/QQQ.parquet"); r = px.pct_change()
    w_rule = NC.core_weights(px)[0]["base"].reindex(I["bh"].index).ffill().fillna(1.0).reindex(days).fillna(1.0).to_numpy()
    bh = I["bh"].to_numpy(); ev = I["e_vt"].fillna(0).to_numpy(); leg1 = I["leg1"].to_numpy()
    al = lambda s: s.shift(1).reindex(days, method="ffill")
    X["sma_dn"] = al(daily.trend_down(px, "SMA", 200).astype(float)); X["rv_pct"] = al(daily.causal_percentile(r.rolling(21).std(), 504))
    X["lsd14"] = np.log(I["sd14"]); X["mom21"] = al(px / px.shift(21) - 1)
    need = P.CLEAN + ["dur_rate"] + CORE_BASE + CORE_EXTRA; assert X.loc[sel, need].notna().all().all(), X.loc[sel, need].isna().sum()
    Lb, Sb = (legs["L2"] * 1e4).to_numpy(), (legs["S2"] * 1e4).to_numpy(); Eb = ((I["leg1"] + I["ov"]) * 1e4).to_numpy(); rc = bh * 1e4; wr = w_rule
    feats_s = P.CLEAN + ["dur_rate"]
    # ---- 门槛:核心仓头为零时复现 v4 的 k
    k0, lam0 = run_joint(X, feats_s, CORE_BASE, Lb, Sb, Eb, rc, wr, sel, days, use_core=False)
    d = np.nanmax(np.abs(k0[["kl", "ks"]].to_numpy() - base["k"][["kl", "ks"]].to_numpy())); print(f"[门槛] 核心仓头关闭时 k 与 v4 最大差 {d:.2e}"); assert d < 1e-8
    # ---- 联合 PPP
    cfg = {}; info = {}
    for nm, fc, sp in (("J1", CORE_BASE, False), ("J2", CORE_BASE + CORE_EXTRA, False), ("J1s", CORE_BASE, True), ("J2s", CORE_BASE + CORE_EXTRA, True)):
        out, lam = run_joint(X, feats_s, fc, Lb, Sb, Eb, rc, wr, sel, days, scale_pen=sp); info[nm] = lam
        kl = out.kl.where(om, 1.0).fillna(1.0); ks = out.ks.where(om, 1.0).fillna(1.0); wc = out.w.where(om, pd.Series(wr, index=days)).fillna(pd.Series(wr, index=days)).to_numpy()
        sl = (kl * legs["L2"] + ks * legs["S2"] + I["ov"] - (kl * legs["GL"] + ks * legs["GS"]) * legs["keep"]).to_numpy()
        raw = wc * bh - 2e-4 * np.abs(np.diff(np.r_[wc[0], wc])) + sl
        impl = portfolio.build_v5(wc, sl, ev, None, bh, p4, use_ratio=False, pit_cap=True)
        cfg[f"{nm} raw"] = dict(port=raw, core_w=wc, sleeve_mult=np.ones(n), f=np.ones(n)); cfg[f"{nm} impl"] = impl
        print(f"  {nm}: 选出的 λ {lam};OOS 平均 k 多 {kl[om].mean():.3f} 空 {ks[om].mean():.3f};核心仓权重 {wc[o].mean():.3f}(规则 {wr[o].mean():.3f}),与规则相关 {np.corrcoef(wc[o], wr[o])[0,1]:.3f},min {wc[o].min():.2f} max {wc[o].max():.2f}")
    # ---- 凸优化权重
    k1 = pd.DataFrame({"kl": 1.0, "ks": 1.0}, index=days); s_k1 = R3.assemble(k1, legs, I, om)["sleeve"].to_numpy()
    core_net = leg1; scal = {}
    for s in (0.0, 0.5, 1.0):
        A, inf = static_scales(core_net, s_k1, days, s); scal[f"O1-s{s:g}"] = A; info[f"O1-s{s:g}"] = {y: np.round(v, 2).tolist() for y, v in inf.items()}
    scal["O2"] = rolling_scales(core_net, s_k1, days, 0.5)
    for nm, A in scal.items():
        res = portfolio.build_v5(wr, s4, ev, None, bh, p4, pit_cap=True, scales=(A.ac.to_numpy(), A["as"].to_numpy())); cfg[f"{nm} impl"] = res
        raw_port = A.ac.to_numpy() * leg1 + A["as"].to_numpy() * s4 if False else None
    # ---- 基线
    v5 = portfolio.build_v5(wr, s4, ev, None, bh, p4, rho=1.5, pit_cap=True); v5o = portfolio.build_v5(wr, s4, ev, np.maximum(base["k"].kl.where(om, 1.0).fillna(1.0), base["k"].ks.where(om, 1.0).fillna(1.0)).to_numpy(), bh, p4, rho=1.5)
    refs = {"v4": dict(port=p4, core_w=wr, sleeve_mult=np.ones(n), f=np.ones(n)), "v5pit": v5, "v5 原口径": v5o}
    allc = {**refs, **cfg}
    # ---- 评估
    yrs_oos = yrs[o]; te = (yrs[o] >= 2023); ref = v5["port"][o]; rows = []; years = o.sum() / 252
    for nm, c in allc.items():
        x = c["port"][o]; dv5 = bootstrap.paired_delta_sharpe(x, ref, 2000, 10, 0); d4 = bootstrap.paired_delta_sharpe(x, p4[o], 2000, 10, 0)
        dte = perf.sharpe(x[te]) - perf.sharpe(ref[te]); w = c["core_w"]; turn = np.abs(np.diff(w[o])).sum() / years
        pan = perf.distribution_panel(pd.Series(x))
        rows.append(dict(配置=nm, SR=perf.sharpe(x), SR_23_26=perf.sharpe(x[te]), 波动=x.std() * np.sqrt(252), MDD=pan["mdd"], Calmar=pan["calmar"], CVaR5_bp=pan["cvar5_bp"], 最差日bp=pan["worst_day_bp"],
                         dSR_vs_v5pit=dv5["delta"], lo=dv5["lo"], hi=dv5["hi"], dSR23_26=dte, dSR_vs_v4=d4["delta"], lo4=d4["lo"], hi4=d4["hi"],
                         核心均权=float(w[o].mean()), Sleeve乘数=float(c["sleeve_mult"][o].mean()), 核心年换手=float(turn), f小于1占比=float((c["f"][o] < 1 - 1e-9).mean())))
    R = pd.DataFrame(rows); R["优于v5pit"] = (R.dSR_vs_v5pit > 0) & (R.lo > -0.05) & (R.dSR23_26 > 0) & (R.核心年换手 <= 1.5 * R.loc[R.配置 == "v5pit", "核心年换手"].iloc[0])
    R["优于v5pit"] = R["优于v5pit"].astype(object); R.loc[R.配置.isin(["v4", "v5pit", "v5 原口径"]), "优于v5pit"] = "基线"
    print("\n=== 结果(OOS 2021–26;ΔSR 相对 v5pit/v4,配对块自助 L=10)==="); print(R.to_string(index=False)); R.to_csv(TAB / "joint_sizing.csv", index=False)
    yr = pd.DataFrame({nm: pd.Series(c["port"][o]).groupby(yrs_oos).apply(lambda v: perf.sharpe(v.to_numpy())) for nm, c in allc.items()}); print("\n分年 Sharpe"); print(yr.round(2).T.to_string()); yr.to_csv(TAB / "joint_sizing_yearly.csv")
    print("\n各年选择:"); [print(f"  {k}: {v}") for k, v in info.items()]
    pd.to_pickle(dict(info=info, scal=scal), HERE.parent / "outputs" / "tables" / "joint_sizing_info.pkl")


if __name__ == "__main__":
    main()
