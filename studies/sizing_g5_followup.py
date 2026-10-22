"""事后追加(看到 sizing_models_study 结果后,2026-10-22):F1 线性链接 + G5 价格持续期特征;G5 的特征重要性。
**事后**:F1 与 G5 都是在 11 个配置的结果里挑出来的 → 本配置的任何正结果只能当前向假设,族内计数 +1(累计 12)。
协议与 sizing_models_study 完全相同(同一数据门槛、B 复现、走步、λ 网格、评估);只换链接函数与特征集。

重要性(三种,都只用 OOS,不参与任何选择):
  1) 系数:各折标准化 θ 的均值与符号一致率(每条腿分开)
  2) 置换重要性:固定已训练模型,在 OOS 年内打乱某一特征 50 次,组合 OOS Sharpe 的平均下降
  3) 删一法:去掉某一特征整体重训,组合 OOS ΔSR 的变化
另报 G5 三个特征与 CLEAN 特征的相关(判断是否只是开盘振幅的重复)。
python3 sizing_g5_followup.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_pred_study as DP
import sizing_models_study as M
import sizing_sigma_feed as SF
import tearsheet as TS
import unified_sizing_posthoc as P
import unified_sizing_study as U
from etft.intraday.bars import load_bars

G5 = M.FAM["G5 价格持续期"]


def fit_theta(Z, Lb, Sb, Eb, lam, kind):
    """同 M.ppp_fit,另外返回 θ。"""
    n, p1 = Z.shape; B = Lb + Sb + Eb; gam = B.mean() / B.var()
    def f(th):
        kl, dl = M.link(Z @ th[:p1], kind); ks, ds = M.link(Z @ th[p1:], kind)
        R = kl * Lb + ks * Sb + Eb; Rc = R - R.mean()
        dRl = (Lb * dl)[:, None] * Z; dRs = (Sb * ds)[:, None] * Z
        g = np.r_[dRl.mean(0) - gam * (Rc[:, None] * dRl).mean(0), dRs.mean(0) - gam * (Rc[:, None] * dRs).mean(0)] - 2 * lam * th
        return -(R.mean() - 0.5 * gam * R.var() - lam * np.sum(th ** 2)), -g
    r = minimize(f, np.zeros(2 * p1), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return r.x.reshape(2, p1), gam


def wf_theta(X, feats, Lb, Sb, Eb, sel, days, kind):
    """与 M.wf 相同的 λ 选择;返回 OOS k、各折 (θ, μ, σ) 以便做置换重要性。"""
    fitf = lambda Z, l_, s_, e_, lam: M.ppp_fit(Z, l_, s_, e_, lam, kind)
    k, lam_sel = M.wf(X, feats, Lb, Sb, Eb, sel, days, fitf, U.LAMS)
    yrs = days.year; folds = {}
    for yr in U.OOS_YEARS:
        tr = sel & (yrs < yr); mu, sd = X.loc[tr, feats].mean(), X.loc[tr, feats].std().replace(0, 1)
        Z = np.column_stack([np.ones(int(tr.sum())), ((X.loc[tr, feats] - mu) / sd).to_numpy()])
        th, _ = fit_theta(Z, Lb[tr], Sb[tr], Eb[tr], lam_sel[yr], kind); folds[yr] = (th, mu, sd)
    return k, folds, lam_sel


def k_from_folds(X, feats, folds, sel, days, kind, perm=None, rng=None):
    yrs = days.year; out = pd.DataFrame(np.nan, index=days, columns=["kl", "ks"])
    for yr, (th, mu, sd) in folds.items():
        te = sel & (yrs == yr); Z = ((X.loc[te, feats] - mu) / sd).to_numpy()
        if perm is not None: j = feats.index(perm); Z[:, j] = Z[rng.permutation(len(Z)), j]
        Z = np.column_stack([np.ones(len(Z)), Z])
        out.loc[te, "kl"], out.loc[te, "ks"] = M.link(Z @ th[0], kind)[0], M.link(Z @ th[1], kind)[0]
    return out


def main():
    t0 = time.time(); pd.set_option("display.width", 260, "display.max_rows", 200, "display.float_format", lambda v: f"{v:8.3f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    I = SF.build_inputs(b); L2, S2, GL, GS, keep, ov, X, leg1 = (I[k] for k in ("L2", "S2", "GL", "GS", "keep", "ov", "X", "leg1"))
    dn = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days)
    X = X.join(M.family_features(b, X))
    Lb, Sb, Eb = (L2 * 1e4).to_numpy(), (S2 * 1e4).to_numpy(), ((leg1 + ov) * 1e4).to_numpy()
    o = om.to_numpy(); oi = np.where(o)[0]; dev = yrs[oi] <= 2022; BI = M.block_idx(len(oi), 10, 2000, 0)
    srv = lambda x: x.mean() / x.std() * np.sqrt(252)
    def port(k):
        kl = k.kl.where(om, 1.0).fillna(1.0); ks = k.ks.where(om, 1.0).fillna(1.0)
        return (leg1 + kl * L2 + ks * S2 + ov - (kl * GL + ks * GS) * keep).to_numpy()[oi]
    # ---- 配置:B(sig, CLEAN)、F1(lin, CLEAN)、G5(sig, CLEAN+G5)、F1+G5(lin, CLEAN+G5)
    cfg = {"B 基线 sigmoid": ("sig", P.CLEAN), "F1 线性": ("lin", P.CLEAN), "G5 sigmoid+持续期": ("sig", P.CLEAN + G5), "F1+G5 线性+持续期(事后)": ("lin", P.CLEAN + G5)}
    res = {nm: wf_theta(X, f, Lb, Sb, Eb, sel, days, kd) for nm, (kd, f) in cfg.items()}
    ref = port(res["B 基线 sigmoid"][0]); refb = np.array([srv(ref[r]) for r in BI])
    print(f"[复现] B 组合 OOS Sharpe {srv(ref):.3f}(sizing_models_study:1.766)")
    rows = []
    for nm, (k, folds, lam) in res.items():
        x = port(k); db = np.array([srv(x[r]) for r in BI]) - refb
        r_ = dict(方案=nm, 组合SR=srv(x), ΔSR全=srv(x) - srv(ref), lo=np.percentile(db, 2.5), hi=np.percentile(db, 97.5),
                  ΔSR_dev=srv(x[dev]) - srv(ref[dev]), ΔSR_test=srv(x[~dev]) - srv(ref[~dev]), λ=str(list(lam.values())))
        r_.update({str(y): srv(x[yrs[oi] == y]) - srv(ref[yrs[oi] == y]) for y in range(2021, 2027)}); rows.append(r_)
    R = pd.DataFrame(rows); print("\n=== 组合层 OOS 2021–26,ΔSR 相对 B ==="); print(R.to_string(index=False))
    Pn = pd.DataFrame({nm: SF.panel(pd.Series(port(res[nm][0]))) for nm in res}).T; print("\n=== 分布面板 ==="); print(Pn.to_string())
    # ---- 重要性(两个含 G5 的模型)
    imp_rows = []
    for nm in ("G5 sigmoid+持续期", "F1+G5 线性+持续期(事后)"):
        kd, feats = cfg[nm]; k, folds, lam = res[nm]; base_sr = srv(port(k))
        TH = np.stack([folds[y][0] for y in folds])                        # (折, 2, 1+p)
        rg = np.random.default_rng(0)
        for j, f in enumerate(feats):
            th_l, th_s = TH[:, 0, j + 1], TH[:, 1, j + 1]
            pdrop = np.mean([base_sr - srv(port(k_from_folds(X, feats, folds, sel, days, kd, perm=f, rng=rg))) for _ in range(50)])
            kd1, _ = M.wf(X, [g for g in feats if g != f], Lb, Sb, Eb, sel, days, lambda Z, l_, s_, e_, lm: M.ppp_fit(Z, l_, s_, e_, lm, kd), U.LAMS)
            ddrop = base_sr - srv(port(kd1))
            imp_rows.append(dict(模型=nm, 特征=f, 是G5=f in G5, θ多_均值=th_l.mean(), θ多_同号率=max((th_l > 0).mean(), (th_l < 0).mean()),
                                 θ空_均值=th_s.mean(), θ空_同号率=max((th_s > 0).mean(), (th_s < 0).mean()), 置换重要性_ΔSR=pdrop, 删一法_ΔSR=ddrop))
        print(f"  [{nm}] 重要性完成 {time.time()-t0:.0f}s", flush=True)
    IM = pd.DataFrame(imp_rows); print("\n=== 特征重要性(OOS;ΔSR = 打乱/删除后组合 Sharpe 下降,正 = 有用)===")
    print(IM.sort_values(["模型", "置换重要性_ΔSR"], ascending=[True, False]).to_string(index=False))
    C = X.loc[sel, P.CLEAN + G5].corr().loc[G5, P.CLEAN]; print("\n=== G5 与 CLEAN 特征的相关(样本日)==="); print(C.to_string())
    R.to_csv(M.TAB / "sizing_g5_followup.csv", index=False); IM.to_csv(M.TAB / "sizing_g5_importance.csv", index=False); Pn.to_csv(M.TAB / "sizing_g5_panel.csv")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
