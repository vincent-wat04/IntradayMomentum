"""过拟合审计(handbook 第 2 讲;Bailey & López de Prado 2012/2014;Bailey, Borwein, López de Prado & Zhu 2014)。
  1) PSR(SR > 0)与 PSR(SR > 基准):偏度/峰度修正后的夏普显著性
  2) DSR:按 docs/TRIAL_REGISTRY.md 的试验次数 K 和候选族的夏普横截面方差,计算"在 K 次试验里挑最大值"的期望 SR0,再算 PSR(SR0)
  3) CSCV-PBO:对 v4/v5 的候选族(仓位模型变体 + 组合层变体,共 N 条组合收益序列)做组合对称交叉验证,
     估计"样本内最优者在样本外排名落到中位数以下"的概率
候选族收益序列从已缓存的研究结果重建,不重新挑选。
python3 audit/overfit_audit.py
"""
from __future__ import annotations

import itertools
import os
import sys
import re
import warnings

warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
EULER = 0.5772156649


def sr_daily(x):
    x = np.asarray(x, float); return x.mean() / x.std(ddof=1)


def psr(x, sr_star_daily=0.0):
    x = np.asarray(x, float); T = len(x); s = sr_daily(x); g3 = skew(x); g4 = kurtosis(x, fisher=False)
    den = np.sqrt(max(1 - g3 * s + (g4 - 1) / 4 * s ** 2, 1e-12))
    return float(norm.cdf((s - sr_star_daily) * np.sqrt(T - 1) / den))


def expected_max_sr(var_sr_daily, K):
    return float(np.sqrt(var_sr_daily) * ((1 - EULER) * norm.ppf(1 - 1 / K) + EULER * norm.ppf(1 - 1 / (K * np.e))))


def cscv_pbo(M, S=16):
    """M: (T, N) 收益矩阵。按时间切 S 块,所有 C(S, S/2) 种训练/测试组合;返回 PBO 与 logit 分布。"""
    T, N = M.shape; blocks = np.array_split(np.arange(T), S); lam = []
    for tr in itertools.combinations(range(S), S // 2):
        te = [b for b in range(S) if b not in tr]
        it = np.concatenate([blocks[b] for b in tr]); io = np.concatenate([blocks[b] for b in te])
        sr_in = M[it].mean(0) / M[it].std(0, ddof=1); sr_out = M[io].mean(0) / M[io].std(0, ddof=1)
        n_star = int(np.argmax(sr_in)); rk = (sr_out < sr_out[n_star]).sum() + 0.5 * ((sr_out == sr_out[n_star]).sum() - 1)
        w = (rk + 1) / (N + 1); lam.append(np.log(w / (1 - w)))
    lam = np.array(lam); return float((lam <= 0).mean()), lam


def registry_K(path=HERE.parent.parent / "docs" / "TRIAL_REGISTRY.md"):
    m = re.search(r"\*\*合计\*\*.*?\*\*≈ ?(\d+)\*\*", path.read_text()); return int(m.group(1))


def build_family():
    """v4/v5 候选族:与 v3 同口径的组合收益(① RV21×SMA200 腿 + Sleeve),全部来自已写定的研究脚本。"""
    import disp_pred_study as DP, r2_rules_study as RR, r3_study as R3, sizing_models_study as M, sizing_sigma_feed as SF
    import unified_sizing_posthoc as P, noise_core_dd as NC, tearsheet_v5 as T5, round3_common as C
    from etft.intraday.bars import load_bars
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    I = SF.build_inputs(b); X = I["X"].copy(); keep = I["keep"]; bhv = I["bh"].to_numpy(); ev = I["e_vt"].fillna(0).to_numpy()
    dn = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days)
    X["dur_rate"] = M.family_features(b, I["X"])["dur_rate"]; _, S3m, A = RR.band_multipliers(b)
    X["rv_surp_1000"] = pd.Series(np.log(A["cRV"][:, 29] / A["S_rv"][:, 29]), index=days).fillna(0.0); CR2 = [("rv_surp_1000" if f == "b30_n" else f) for f in P.CLEAN]
    Lb = R3.legs_k(R3.LI.run_phases_fills(b), R3.SU.fomc_late_PH(b), b, keep); LE = R3.legs_k(R3.run_PH_entry(b, S3m), R3.run_PH_entry(b, S3m, fomc_only=True), b, keep)
    spec = {"v3": (Lb, P.CLEAN, "mv", "sig"), "线性": (Lb, P.CLEAN, "mv", "lin"), "线性+dur": (Lb, P.CLEAN + ["dur_rate"], "mv", "lin"), "O2": (Lb, P.CLEAN, "semi", "sig"),
            "O2-lin": (Lb, P.CLEAN, "semi", "lin"), "O2-lin+dur": (Lb, P.CLEAN + ["dur_rate"], "semi", "lin"), "E1": (LE, P.CLEAN, "mv", "sig"),
            "E1+R2": (LE, CR2, "mv", "sig"), "E1+R2+dur": (LE, CR2 + ["dur_rate"], "mv", "sig"), "E1+R2+dur+lin": (LE, CR2 + ["dur_rate"], "mv", "lin"),
            "v4(E1+lin+dur)": (LE, P.CLEAN + ["dur_rate"], "mv", "lin"), "E1+lin+dur+O2": (LE, P.CLEAN + ["dur_rate"], "semi", "lin"), "R2": (Lb, CR2, "mv", "sig")}
    fam = {}; v4k = None
    for nm, (L, ft, obj, lk) in spec.items():
        r = R3.assemble(R3.sizing(X, ft, L, I, sel, days, obj, lk), L, I, om); fam[nm] = r["port"].to_numpy()
        if nm == "v4(E1+lin+dur)": v4k = r
    w = NC.core_weights(C.load_close("yahoo/QQQ.parquet"))[0]["base"].reindex(I["bh"].index).ffill().fillna(1.0).reindex(days).fillna(1.0).to_numpy()
    kmax = np.maximum(v4k["k"].kl.where(om, 1.0).fillna(1.0), v4k["k"].ks.where(om, 1.0).fillna(1.0)).to_numpy(); s4 = v4k["sleeve"].to_numpy()
    for rho in (1.0, 1.5, 2.0, 3.0):
        fam[f"v5 ρ={rho:g}"] = T5.build_v5(w, s4, ev, kmax, bhv, fam["v4(E1+lin+dur)"], rho=rho)["port"]
    return pd.DataFrame(fam, index=days), om.to_numpy()


def main():
    pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:8.4f}")
    K = registry_K(); print(f"[登记簿] 全项目累计试验 K ≈ {K}(下界)")
    F, oo = build_family(); Fo = F[oo]; T = len(Fo); ann = np.sqrt(252)
    srs = Fo.apply(sr_daily); var_sr = float(srs.var(ddof=1))
    print(f"[候选族] {F.shape[1]} 条组合序列,OOS 2021–26 共 {T} 个样本日;年化 SR 范围 {srs.min()*ann:.3f}–{srs.max()*ann:.3f},横截面 SR 标准差(年化){np.sqrt(var_sr)*ann:.3f}")
    # 基准:QQQ 买入持有同期
    import sizing_sigma_feed as SF
    from etft.intraday.bars import load_bars
    bh = SF.build_inputs(load_bars("QQQ"))["bh"].to_numpy()[oo]
    rows = []
    for K_ in (F.shape[1], 50, K):
        sr0 = expected_max_sr(var_sr, K_)
        for nm in ("v3", "v4(E1+lin+dur)", "v5 ρ=1.5"):
            x = Fo[nm].to_numpy()
            rows.append(dict(K=K_, 序列=nm, 年化SR=sr_daily(x) * ann, 偏度=skew(x), 峰度=kurtosis(x, fisher=False), PSR_SR大于0=psr(x), PSR_SR大于QQQ=psr(x, sr_daily(bh)),
                             期望最大SR0_年化=sr0 * ann, DSR=psr(x, sr0)))
    R = pd.DataFrame(rows); print("\n=== PSR / DSR(OOS 2021–26)===\n注:DSR 用候选族的 SR 横截面方差;K = 族内 / 50 / 全项目登记簿"); print(R.to_string(index=False))
    # 相对 v3 的增量是否显著(差值序列的 PSR)
    for nm in ("v4(E1+lin+dur)", "v5 ρ=1.5"):
        d = Fo[nm].to_numpy() - Fo["v3"].to_numpy() * (Fo[nm].std() / Fo["v3"].std())
        print(f"  {nm} 相对 v3(等波动)增量序列:年化 SR {sr_daily(d)*ann:.3f},PSR(>0) {psr(d):.3f}")
    pbo, lam = cscv_pbo(Fo.to_numpy(), S=16)
    print(f"\n=== CSCV-PBO(N = {F.shape[1]} 个候选,S = 16 块,C(16,8) = {len(lam)} 种切分)===\nPBO = {pbo:.3f};logit λ 中位 {np.median(lam):+.2f};样本内最优者在样本外的平均相对排名 {np.mean(1/(1+np.exp(-lam))):.2f}")
    # 样本内最优者是谁(频率)
    blocks = np.array_split(np.arange(T), 16); M_ = Fo.to_numpy(); win = {}
    for tr in itertools.combinations(range(16), 8):
        it = np.concatenate([blocks[b] for b in tr]); n = F.columns[int(np.argmax(M_[it].mean(0) / M_[it].std(0, ddof=1)))]; win[n] = win.get(n, 0) + 1
    print("样本内最优者分布:", dict(sorted(win.items(), key=lambda kv: -kv[1])))
    R.to_csv(HERE.parent.parent / "outputs" / "tables" / "overfit_audit.csv", index=False)


if __name__ == "__main__":
    main()
