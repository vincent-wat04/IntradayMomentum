"""事后追加(2026-10-22,看到 sizing_g5_followup 之后):
  1) 事件数的直观示例:某日 09:31–10:00 的价格持续期事件清单,以及 dur_rate 的分布
  2) 配置 F1+dur_rate:线性链接 + CLEAN + dur_rate(事后,累计 +1)
  3) 诊断(机理,非选择,累计 +1):sigmoid 链接、λ 网格 ×1/4 —— 检验"线性优于 sigmoid"是否只是有效正则强度不同
     (在 θ≈0 附近 2σ(a) ≈ 1 + a/2,同样的 k 偏离 sigmoid 需要 2 倍的 θ,惩罚 λ‖θ‖² 因而大 4 倍)
  4) λ 诊断:各折留一年交叉验证曲线、所选 λ、k 的年内标准差、‖θ‖;线性 k 撞到边界 0/2 的比例
python3 sizing_durrate_followup.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b0_entry_features as B0
import disp_pred_study as DP
import duration_events as DE
import sizing_g5_followup as G
import sizing_models_study as M
import sizing_sigma_feed as SF
import unified_sizing_posthoc as P
import unified_sizing_study as U
from etft.intraday.bars import load_bars


def cv_curve(X, feats, Lb, Sb, Eb, sel, days, kind, lams):
    """与 M.wf 内层完全相同的留一年交叉验证,返回 {年: {λ: 平均效用增量}}。"""
    yrs = days.year; out = {}
    for yr in U.OOS_YEARS:
        tr = sel & (yrs < yr); cv = {}
        for lam in lams:
            sc = []
            for hy in sorted(set(yrs[tr])):
                a_, h_ = tr & (yrs != hy), tr & (yrs == hy)
                if a_.sum() < 150 or h_.sum() < 50: continue
                mu, sd = X.loc[a_, feats].mean(), X.loc[a_, feats].std().replace(0, 1)
                Z = lambda idx: np.column_stack([np.ones(int(idx.sum())), ((X.loc[idx, feats] - mu) / sd).to_numpy()])
                pr, g = M.ppp_fit(Z(a_), Lb[a_], Sb[a_], Eb[a_], lam, kind); kl, ks = pr(Z(h_))
                sc.append(U.utility(kl * Lb[h_] + ks * Sb[h_] + Eb[h_], g) - U.utility(Lb[h_] + Sb[h_] + Eb[h_], g))
            cv[lam] = np.mean(sc)
        out[yr] = cv
    return out


def main():
    t0 = time.time(); pd.set_option("display.width", 260, "display.max_rows", 200, "display.float_format", lambda v: f"{v:8.3f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    # ---------------- 1. 事件数示例
    sd_in, _ = B0.intraday_sd(b); ev = DE.day_events(b, sd_in, B0.K_EV)
    cols = list(b.c.columns); c = b.c.to_numpy(); O = b.day_open.to_numpy()
    n10 = np.array([np.sum(idx <= 29) for idx, sg, d in ev]); rate = n10 / (30 / 390) / B0.K_EV
    ok = (days >= "2019-01-01") & np.isfinite(sd_in)
    print("=== 1. 事件数:09:31–10:00 的事件个数(按 K=26 每天预期约 2.0 个)===")
    print(pd.Series(n10[ok]).describe(percentiles=[.1, .25, .5, .75, .9]).round(2).to_string())
    print("dur_rate 分位:", dict(zip(["10%", "25%", "50%", "75%", "90%"], np.round(np.quantile(rate[ok], [.1, .25, .5, .75, .9]), 2))))
    for tag, want in (("事件多的一天", np.quantile(n10[ok], 0.9)), ("事件少的一天", np.quantile(n10[ok], 0.1))):
        cand = np.where(ok & (n10 == int(want)) & (days >= "2026-01-01"))[0]; i = cand[-1]
        idx, sg, delta = ev[i]; m = idx <= 29
        print(f"\n--- 示例({tag}):{days[i].date()}  开盘 {O[i]:.2f},阈值 δ = {delta*1e4:.1f}bp(价格 ≈ {O[i]*delta:.2f} 美元),"
              f"10:00 前 {int(m.sum())} 个事件,dur_rate = {rate[i]:.2f};09:31–10:00 振幅 {(b.h.to_numpy()[i,:30].max()/b.l.to_numpy()[i,:30].min()-1)*1e4:.0f}bp")
        ref = O[i]
        for k_, s_ in zip(idx[m], sg[m]):
            print(f"    {cols[k_]}  收盘 {c[i,k_]:.2f}  相对上一事件价 {ref:.2f} 变动 {np.log(c[i,k_]/ref)*1e4:+.1f}bp → {'上' if s_ > 0 else '下'}"); ref = c[i, k_]
    # ---------------- 输入
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
    LQ = tuple(l / 4 for l in U.LAMS)
    cfg = {"B 基线 sigmoid": ("sig", P.CLEAN, U.LAMS), "F1 线性": ("lin", P.CLEAN, U.LAMS),
           "F1+G5 线性+3 个持续期": ("lin", P.CLEAN + M.FAM["G5 价格持续期"], U.LAMS),
           "F1+dur_rate 线性(事后)": ("lin", P.CLEAN + ["dur_rate"], U.LAMS),
           "诊断:sigmoid, λ×1/4": ("sig", P.CLEAN, LQ),
           "无仓位模型 k≡1": None}
    K = {}
    for nm, v in cfg.items():
        if v is None: K[nm] = pd.DataFrame(1.0, index=days, columns=["kl", "ks"]); continue
        kd, f, lams = v; K[nm] = M.wf(X, f, Lb, Sb, Eb, sel, days, lambda Z, l_, s_, e_, lam, kd=kd: M.ppp_fit(Z, l_, s_, e_, lam, kd), lams)
    ref = port(K["B 基线 sigmoid"][0]); refb = np.array([srv(ref[r]) for r in BI]); print(f"\n[复现] B 组合 OOS Sharpe {srv(ref):.3f}(上轮 1.766)")
    rows = []
    for nm, kk in K.items():
        k = kk if isinstance(kk, pd.DataFrame) else kk[0]; lam = {} if isinstance(kk, pd.DataFrame) else kk[1]
        x = port(k); db = np.array([srv(x[r]) for r in BI]) - refb
        r_ = dict(方案=nm, 组合SR=srv(x), ΔSR全=srv(x) - srv(ref), lo=np.percentile(db, 2.5), hi=np.percentile(db, 97.5),
                  ΔSR_dev=srv(x[dev]) - srv(ref[dev]), ΔSR_test=srv(x[~dev]) - srv(ref[~dev]), 所选λ=str([round(v, 3) for v in lam.values()]))
        r_.update({str(y): srv(x[yrs[oi] == y]) - srv(ref[yrs[oi] == y]) for y in range(2021, 2027)}); rows.append(r_)
    R = pd.DataFrame(rows); print("\n=== 2/3. 组合层 OOS 2021–26,ΔSR 相对 B ==="); print(R.to_string(index=False))
    Pn = pd.DataFrame({nm: SF.panel(pd.Series(port(kk if isinstance(kk, pd.DataFrame) else kk[0]))) for nm, kk in K.items()}).T
    print("\n=== 分布面板 ==="); print(Pn.to_string())
    # ---------------- 4. λ 与 k 诊断
    print("\n=== 4a. 各年 k 的标准差(多头 / 空头;k 越接近常数 1 = 仓位模型越接近没开)===")
    sd_rows = []
    for nm in ("B 基线 sigmoid", "F1 线性", "F1+G5 线性+3 个持续期", "F1+dur_rate 线性(事后)", "诊断:sigmoid, λ×1/4"):
        k = K[nm][0]; sd_rows.append({"方案": nm, **{str(y): f"{k.kl[om & (yrs == y)].std():.2f}/{k.ks[om & (yrs == y)].std():.2f}" for y in range(2021, 2027)}})
    print(pd.DataFrame(sd_rows).to_string(index=False))
    for nm in ("F1 线性", "F1+G5 线性+3 个持续期", "F1+dur_rate 线性(事后)"):
        kd, f, lams = cfg[nm]; cv = cv_curve(X, f, Lb, Sb, Eb, sel, days, kd, lams)
        print(f"\n=== 4b. 留一年交叉验证曲线:{nm}(各 λ 的留出年平均效用增量,bp/天;每年取最大者)===")
        print(pd.DataFrame(cv).T.round(3).to_string())
    for nm in ("F1 线性", "F1+dur_rate 线性(事后)"):
        k = K[nm][0]
        print(f"[4c] {nm}:OOS 中 k 撞到边界的比例 多头 0:{(k.kl[om] <= 1e-9).mean():.1%} 2:{(k.kl[om] >= 2 - 1e-9).mean():.1%};"
              f" 空头 0:{(k.ks[om] <= 1e-9).mean():.1%} 2:{(k.ks[om] >= 2 - 1e-9).mean():.1%}")
    for nm in ("B 基线 sigmoid", "F1 线性", "诊断:sigmoid, λ×1/4"):
        k = K[nm][0]; print(f"[4d] {nm}:k 多头分位 5/50/95% = {np.round(np.quantile(k.kl[om], [.05, .5, .95]), 2)};空头 {np.round(np.quantile(k.ks[om], [.05, .5, .95]), 2)}")
    R.to_csv(M.TAB / "sizing_durrate_followup.csv", index=False); Pn.to_csv(M.TAB / "sizing_durrate_panel.csv")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
