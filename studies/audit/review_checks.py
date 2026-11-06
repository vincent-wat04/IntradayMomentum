"""研究链复盘检查(2026-10-30,预注册:运行前写定判据;handbook 第 2/14/15/17/18/31 讲)。
目的不是找新的 alpha,而是核对"当前 v4/v5 的结论有多少是流程造成的":
 R1 增量族 DSR/PBO:只含仓位模型变体(去掉 ρ 维度),增量 = 候选 − 等波动 v3。报告,不设门槛。
 R2 嵌入式禁运(embargo):v4 内层留一年 CV 时剔除被留出年份前后 5/10/21 个交易日。
    判据:embargo=21 的 OOS 组合 SR 相对 embargo=0 下降不超过 0.10,否则说明 λ 选择/仓位对相邻日泄漏敏感。
 R3a 简化 Sleeve 的跳过规则:把 FFA-P 滚动分位 ≤0.2 的硬跳过换成 EWMA0.94(r²)、EWMA0.94(Parkinson 区间²)、20 日 RV 的同分位规则(因果:t 日只用 ≤t−1 的收益)。
    判据:某个简单规则的 ΔSR(相对 v4)≥ −0.05 且块自助 95% 区间含 0 → 记为"FFA-P 无独立价值的证据",只写进文档,不改生产(生产变更另需前瞻证据)。
 R3c 分布面板:v4 / 不跳过 / EWMA0.94(r²) 跳过(OOS 2021–26),因为 R3a 只看 SR 不够(见全局约定:凸性策略必须看分布)。
 R3d 去掉整个仓位层:k≡1(多空腿都是 1×,其余同 v4:入场带、FFA-P 跳过、CPI 回补、σ14 波动目标)。判据同 R3b:ΔSR ≥ −0.05 且区间含 0 → 仓位层(PPP)在组合 SR 上无独立证据。
 R3b 简化 dur_rate:(i) v4 去掉 dur_rate;(ii) 用 log(开盘 30 分钟已实现波动 / σ14) 代替。同一判据。另报 dur_rate 对 CLEAN 其余特征的 R²(>0.7 视为冗余)。
 R4 漂移与归因:CLEAN+dur_rate 的 PSI(2019–22 vs 2025–26;<0.10 稳定 / 0.10–0.25 预警 / ≥0.25 严重);
    v5−v3、v4−v3(等波动)对 [QQQ 收益, QQQ 收益², 高波动三分位哑变量, 2022 哑变量] 的 Barra-lite 回归(HAC lag 5),报告 α 年化与 R²。
 R5 特征单调性与显著性:对每个特征的五分位 vs Sleeve 日收益(L+S 总腿)的 MonoScore(Spearman,≥0.8 记为单调)与回归斜率 HAC-t(|t|≥3 为 HLZ 门槛)。
所有检查只读已有研究产物;不增删候选;不改 v4/v5 的配置。
python3 audit/review_checks.py   (在 studies/ 下运行)
"""
from __future__ import annotations

import itertools
import os
import sys
import warnings

warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import disp_pred_study as DP, r2_rules_study as RR, r3_study as R3, sizing_models_study as M, sizing_sigma_feed as SF
import unified_sizing_posthoc as P, noise_core_dd as NC, tearsheet_v5 as T5, round3_common as C, sota_stack_study as S
import ivol_objective_study as AO
from etft.intraday.bars import load_bars
import overfit_audit as OA

TAB = HERE.parent.parent / "outputs" / "tables"; ANN = np.sqrt(252)


def srv(x): x = np.asarray(x, float); return x.mean() / x.std(ddof=1) * ANN


def setup():
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    I = SF.build_inputs(b); X = I["X"].copy(); keep = I["keep"]
    dn = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days)
    X["dur_rate"] = M.family_features(b, I["X"])["dur_rate"]; _, S3m, A = RR.band_multipliers(b)
    PHb, PHf = R3.run_PH_entry(b, S3m), R3.run_PH_entry(b, S3m, fomc_only=True)
    return dict(b=b, days=days, yrs=yrs, I=I, X=X, keep=keep, sel=sel, om=om, PHb=PHb, PHf=PHf)


def v4(E, keep=None, feats=None, embargo=0):
    I, X, sel, om, days, b = (E[k] for k in ("I", "X", "sel", "om", "days", "b"))
    kp = E["keep"] if keep is None else keep
    L = R3.legs_k(E["PHb"], E["PHf"], b, kp); feats = P.CLEAN + ["dur_rate"] if feats is None else feats
    Lb, Sb, Eb = (L["L2"] * 1e4).to_numpy(), (L["S2"] * 1e4).to_numpy(), ((I["leg1"] + I["ov"]) * 1e4).to_numpy()
    Q = pd.DataFrame({"L": Lb, "S": Sb, "E": Eb}, index=days)
    k, lam = AO.wf_obj(X, feats, Lb, Sb, Eb, Q, sel, days, "mv", "lin", embargo=embargo)
    r = R3.assemble(k, L, I, om); r["lam"] = lam; r["legs"] = L; return r


def boot_delta(x, ref, om, n=2000, L=10):
    oi = np.where(om)[0]; BI = M.block_idx(len(oi), L, n, 0); x = np.asarray(x)[oi]; ref = np.asarray(ref)[oi]
    d = np.array([srv(x[r]) - srv(ref[r]) for r in BI]); return srv(x) - srv(ref), np.percentile(d, 2.5), np.percentile(d, 97.5)


def nw_ols(y, Xm, lag=5):
    import statsmodels.api as sm
    r = sm.OLS(y, sm.add_constant(Xm)).fit(cov_type="HAC", cov_kwds={"maxlags": lag}); return r


def psi(ref, cur, nb=10):
    q = np.unique(np.quantile(ref, np.linspace(0, 1, nb + 1))); q[0], q[-1] = -np.inf, np.inf
    a = np.histogram(ref, q)[0] / len(ref); c = np.histogram(cur, q)[0] / len(cur); a, c = np.clip(a, 1e-4, None), np.clip(c, 1e-4, None)
    return float(((c - a) * np.log(c / a)).sum())


def main():
    pd.set_option("display.width", 240, "display.max_columns", 40, "display.float_format", lambda v: f"{v:8.3f}")
    F, oo = OA.build_family(); Fo = F[oo]; K = OA.registry_K(); out = {}
    # ---------------------------------------------------------------- R1
    print("=" * 30, "R1 增量族 DSR/PBO(去掉 ρ 维度)")
    sizing_cols = [c for c in F.columns if not c.startswith("v5")]
    inc = pd.DataFrame({c: Fo[c].to_numpy() - Fo["v3"].to_numpy() * (Fo[c].std() / Fo["v3"].std()) for c in sizing_cols if c != "v3"})
    srs = inc.apply(OA.sr_daily); var = float(srs.var(ddof=1))
    rows = []
    for Kk in (inc.shape[1], 50, K):
        sr0 = OA.expected_max_sr(var, Kk)
        for c in ("v4(E1+lin+dur)", "O2-lin", "E1+R2+dur"):
            x = inc[c].to_numpy(); rows.append(dict(K=Kk, 候选=c, 增量年化SR=OA.sr_daily(x) * ANN, PSR0=OA.psr(x), SR0年化=sr0 * ANN, DSR=OA.psr(x, sr0)))
    R1 = pd.DataFrame(rows); print(R1.to_string(index=False)); R1.to_csv(TAB / "review_R1_increment_dsr.csv", index=False)
    pbo, lam = OA.cscv_pbo(Fo[sizing_cols].to_numpy(), S=16)
    print(f"仓位模型变体族(N={len(sizing_cols)})CSCV-PBO = {pbo:.3f},logit 中位 {np.median(lam):+.2f}")
    pbo2, lam2 = OA.cscv_pbo(inc.to_numpy(), S=16); print(f"增量族(N={inc.shape[1]})CSCV-PBO = {pbo2:.3f},logit 中位 {np.median(lam2):+.2f}")
    blocks = np.array_split(np.arange(len(Fo)), 16); Mx = Fo[sizing_cols].to_numpy(); win = {}
    for tr in itertools.combinations(range(16), 8):
        it = np.concatenate([blocks[b_] for b_ in tr]); n = sizing_cols[int(np.argmax(Mx[it].mean(0) / Mx[it].std(0, ddof=1)))]; win[n] = win.get(n, 0) + 1
    print("样本内最优者分布(仓位族):", dict(sorted(win.items(), key=lambda kv: -kv[1])))
    rho = pd.DataFrame({c: dict(SR=srv(Fo[c]), MDD=float((np.cumprod(1 + Fo[c].to_numpy()) / np.maximum.accumulate(np.cumprod(1 + Fo[c].to_numpy())) - 1).min())) for c in F.columns if c.startswith("v5")}).T
    print("ρ 曲线(v5):\n" + rho.to_string())
    # ---------------------------------------------------------------- 搭建
    E = setup(); days, om, yrs = E["days"], E["om"], E["yrs"]; o = om.to_numpy()
    base = v4(E); chk = abs(srv(base["port"].to_numpy()[o]) - srv(Fo["v4(E1+lin+dur)"])); print(f"\n[复现] v4 OOS SR {srv(base['port'].to_numpy()[o]):.3f}(缓存族 {srv(Fo['v4(E1+lin+dur)']):.3f},差 {chk:.4f})"); assert chk < 0.01
    # ---------------------------------------------------------------- R2
    print("=" * 30, "R2 embargo")
    rows = []; kb = base["k"]
    for e in (0, 5, 10, 21):
        r = base if e == 0 else v4(E, embargo=e); x = r["port"].to_numpy(); d, lo, hi = boot_delta(x, base["port"].to_numpy(), o)
        rows.append(dict(embargo=e, OOS_SR=srv(x[o]), ΔSR=d, lo=lo, hi=hi, k多均=float(r["k"].kl[om].mean()), k空均=float(r["k"].ks[om].mean()),
                         k多相关=float(np.corrcoef(r["k"].kl[om], kb.kl[om])[0, 1]), λ=" ".join(f"{r['lam'][y]:g}" for y in sorted(r["lam"]))))
    R2 = pd.DataFrame(rows); print(R2.to_string(index=False)); R2.to_csv(TAB / "review_R2_embargo.csv", index=False)
    # ---------------------------------------------------------------- R3a
    print("=" * 30, "R3a 简化跳过规则")
    b = E["b"]; I = E["I"]
    r_d = (b.day_close / b.prev_close_adj - 1).reindex(days).fillna(0.0).to_numpy()
    hl = pd.read_parquet(Path(os.environ["ETFT_DATA"]) / "raw" / "yahoo" / "QQQ.parquet")[["high", "low"]]; hl.index = pd.to_datetime(hl.index).tz_localize(None)
    park = (np.log(hl.high / hl.low) ** 2 / (4 * np.log(2))).reindex(days).ffill().to_numpy()
    def ewma_fc(sq, lamb=0.94):
        e = np.full(len(sq), np.nan); cur = np.nanmean(sq[:20])
        for i in range(len(sq)):
            e[i] = np.sqrt(cur)                       # 预测 t 日:只用 ≤ t−1 的信息
            cur = lamb * cur + (1 - lamb) * sq[i]
        return pd.Series(e, index=days)
    cand = {"EWMA0.94(r²)": ewma_fc(r_d ** 2), "EWMA0.94(Parkinson)": ewma_fc(park),
            "RV20": pd.Series(pd.Series(r_d, index=days).shift(1).rolling(20).std().to_numpy(), index=days)}
    rows = []; kfa = E["keep"].to_numpy()
    for nm, fc in cand.items():
        kp = (~(S.roll_rank(fc) <= 0.2).fillna(False)).astype(float); r = v4(E, keep=kp); x = r["port"].to_numpy(); d, lo, hi = boot_delta(x, base["port"].to_numpy(), o)
        kk = kp.to_numpy(); skA, skB = kfa == 0, kk == 0
        rows.append(dict(跳过规则=nm, OOS_SR=srv(x[o]), ΔSR=d, lo=lo, hi=hi, 跳过天数=int(skB[o].sum()), FFA跳过天数=int(skA[o].sum()), 重合=int((skA & skB)[o].sum()),
                         简单规则无差异=bool(d >= -0.05 and lo <= 0 <= hi)))
    nk = v4(E, keep=pd.Series(1.0, index=days)); x = nk["port"].to_numpy(); d, lo, hi = boot_delta(x, base["port"].to_numpy(), o)
    rows.append(dict(跳过规则="不跳过", OOS_SR=srv(x[o]), ΔSR=d, lo=lo, hi=hi, 跳过天数=0, FFA跳过天数=int((kfa == 0)[o].sum()), 重合=0, 简单规则无差异=bool(d >= -0.05 and lo <= 0 <= hi)))
    R3a = pd.DataFrame(rows); print(R3a.to_string(index=False)); R3a.to_csv(TAB / "review_R3a_skip_rules.csv", index=False)
    print("=" * 30, "R3c 分布面板(OOS 2021–26)")
    kp_e = (~(S.roll_rank(cand["EWMA0.94(r²)"]) <= 0.2).fillna(False)).astype(float)
    pan = {"v4(FFA-P 跳过)": SF.panel(pd.Series(base["port"].to_numpy()[o])), "不跳过": SF.panel(pd.Series(nk["port"].to_numpy()[o])),
           "EWMA 跳过": SF.panel(pd.Series(v4(E, keep=kp_e)["port"].to_numpy()[o])),
           "Sleeve 单独 v4": SF.panel(pd.Series(base["sleeve"].to_numpy()[o])), "Sleeve 单独 不跳过": SF.panel(pd.Series(nk["sleeve"].to_numpy()[o]))}
    R3c = pd.DataFrame(pan).T; print(R3c.to_string()); R3c.to_csv(TAB / "review_R3c_panel.csv")
    # ---------------------------------------------------------------- R3b
    print("=" * 30, "R3b 简化 dur_rate")
    c30 = b.c.to_numpy()[:, :30]; o30 = b.day_open.to_numpy()
    lr = np.diff(np.log(np.column_stack([o30, c30])), axis=1); rv30 = np.sqrt(np.nansum(lr ** 2, axis=1))
    X2 = E["X"].copy(); X2["rv30_s14"] = np.log(rv30 / I["sd14"].to_numpy())
    rows = []
    for nm, feats in (("v4 基线(含 dur_rate)", P.CLEAN + ["dur_rate"]), ("去掉 dur_rate", P.CLEAN), ("dur_rate→rv30/σ14", P.CLEAN + ["rv30_s14"])):
        Ev = dict(E, X=X2); r = base if nm.startswith("v4") else v4(Ev, feats=feats); x = r["port"].to_numpy(); d, lo, hi = boot_delta(x, base["port"].to_numpy(), o)
        rows.append(dict(方案=nm, OOS_SR=srv(x[o]), ΔSR=d, lo=lo, hi=hi, 简单规则无差异=bool(d >= -0.05 and lo <= 0 <= hi)))
    R3b = pd.DataFrame(rows); print(R3b.to_string(index=False)); R3b.to_csv(TAB / "review_R3b_durrate.csv", index=False)
    Z = E["X"].loc[E["sel"], P.CLEAN]; y = E["X"].loc[E["sel"], "dur_rate"]
    import statsmodels.api as sm
    r2 = sm.OLS(y.to_numpy(), sm.add_constant(Z.to_numpy())).fit().rsquared; print(f"dur_rate 对 CLEAN({len(P.CLEAN)} 个)的 R² = {r2:.3f}(>0.7 视为冗余)")
    print("=" * 30, "R3d 去掉仓位层(k≡1)")
    one = pd.DataFrame({"kl": 1.0, "ks": 1.0}, index=days); k1 = R3.assemble(one, base["legs"], I, om); x = k1["port"].to_numpy(); d, lo, hi = boot_delta(x, base["port"].to_numpy(), o)
    R3d = pd.DataFrame([dict(方案="k≡1(无 PPP)", OOS_SR=srv(x[o]), ΔSR_vs_v4=d, lo=lo, hi=hi, 无差异=bool(d >= -0.05 and lo <= 0 <= hi))]); print(R3d.to_string(index=False))
    d3, lo3, hi3 = boot_delta(Fo["v3"].reindex(days).fillna(0).to_numpy() if False else F["v3"].to_numpy(), x, o); print(f"v3 相对 k≡1(同入场带为 v3 的旧入场带,非严格同口径):ΔSR {d3:+.3f} [{lo3:+.3f}, {hi3:+.3f}]")
    pk = pd.DataFrame({"v4": SF.panel(pd.Series(base["port"].to_numpy()[o])), "k≡1": SF.panel(pd.Series(x[o]))}).T; print(pk.to_string()); R3d.to_csv(TAB / "review_R3d_no_ppp.csv", index=False)
    # ---------------------------------------------------------------- R4
    print("=" * 30, "R4 PSI 与 Barra-lite 归因")
    feats = P.CLEAN + ["dur_rate"]; Xs = E["X"].loc[E["sel"], feats]; yy = Xs.index.year
    rows = [dict(特征=f, PSI=psi(Xs.loc[yy <= 2022, f].to_numpy(), Xs.loc[yy >= 2025, f].to_numpy())) for f in feats]
    R4a = pd.DataFrame(rows); R4a["状态"] = pd.cut(R4a.PSI, [-1, 0.10, 0.25, 99], labels=["稳定", "预警", "严重"]); print(R4a.to_string(index=False)); R4a.to_csv(TAB / "review_R4_psi.csv", index=False)
    bh = I["bh"].to_numpy(); hv = (I["sd14"] > I["sd14"][o].quantile(2 / 3)).astype(float).to_numpy(); d22 = (yrs == 2022).astype(float)
    Xf = np.column_stack([bh, bh ** 2, hv, d22])[o]; rows = []
    for nm, x in (("v4−v3", Fo["v4(E1+lin+dur)"]), ("v5ρ1.5−v3", Fo["v5 ρ=1.5"])):
        d = x.to_numpy() - Fo["v3"].to_numpy() * (x.std() / Fo["v3"].std()); rr = nw_ols(d, Xf)
        rows.append(dict(增量=nm, α年化=rr.params[0] * 252, α_t=rr.tvalues[0], β_QQQ=rr.params[1], β_QQQt=rr.tvalues[1], 凸性=rr.params[2], 凸性t=rr.tvalues[2], 高波动=rr.params[3] * 252, 高波动t=rr.tvalues[3], d2022=rr.params[4] * 252, d2022t=rr.tvalues[4], R2=rr.rsquared,
                         总均值年化=d.mean() * 252))
    R4b = pd.DataFrame(rows); print(R4b.to_string(index=False)); R4b.to_csv(TAB / "review_R4_barra_lite.csv", index=False)
    # ---------------------------------------------------------------- R5
    print("=" * 30, "R5 单调性 / HAC-t")
    Lg = base["legs"]; tot = ((Lg["L2"] + Lg["S2"]) * 1e4).to_numpy()
    rows = []
    for f in feats:
        v = E["X"][f].to_numpy(); m = o & np.isfinite(v); z = (v[m] - v[m].mean()) / v[m].std(); yv = tot[m]
        q = pd.qcut(pd.Series(v[m]), 5, labels=False, duplicates="drop"); qm = pd.Series(yv).groupby(q.to_numpy()).mean()
        mono = spearmanr(qm.index, qm.to_numpy())[0]; rr = nw_ols(yv, z)
        rows.append(dict(特征=f, Q1=qm.iloc[0], Q3=qm.iloc[len(qm) // 2], Q5=qm.iloc[-1], MonoScore=mono, 斜率bps=rr.params[1], HAC_t=rr.tvalues[1], 单调=bool(abs(mono) >= 0.8), 过HLZ=bool(abs(rr.tvalues[1]) >= 3)))
    R5 = pd.DataFrame(rows); print(R5.to_string(index=False)); R5.to_csv(TAB / "review_R5_monotonic.csv", index=False)


if __name__ == "__main__":
    main()
