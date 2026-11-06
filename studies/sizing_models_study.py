"""仓位模型的函数形式、复杂度与特征 —— 预注册(运行前写定,2026-10-22)

基线 B = v3 ① 的统一仓位:k_leg = 2σ(θᵀz),z = CLEAN 7 特征,目标 = 组合效用(QQQ ① 腿 + CPI overlay + Sleeve),
λ ∈ {0.1,0.3,1,3,10} 留一年交叉验证,每年初重训,OOS 2021–26;k 乘在波动目标 min(4, 2%/σ14) 之上(上一轮已验证这个结构)。
Sleeve 其余部分(FOMC≥14:30、CPI 回补、FFA-P 跳过、15:40 平仓)不变。

新配置(共 11 个,族内校正按 11 个):
 函数形式  F1 线性 k = clip(1+θᵀz, 0, 2)(Brandt–Santa-Clara–Valkanov 原文形式)
           F2 对数线性 k = min(2, exp(θᵀz))
 复杂度    M1 神经网络策略:7→8(tanh)→2,k = 2σ(·),输出层零初始化(起点 = 基线),同一组合效用端到端,λ 同网格,5 个种子平均
           M2 二次 PPP:z + z² + 两两交互(35 维),sigmoid,λ ∈ {0.1,…,10,30,100}
           M3 RAVEN:repo `raven_paper_model.RAVEN` 原样(论文结构,64 日序列,7 通道 = CLEAN 日序列),每条腿一个网络,
              损失 = −组合效用 + 0.1·entropy + 0.01·diversity(repo 的正则项),AdamW 1e-3/1e-2、余弦、60 epoch(repo 默认),3 个种子平均
 特征族    在 CLEAN 上各加一族(sigmoid PPP,其余同 B),特征全部沿用 repo 定义(lowvol_avoidance_study.build / duration_events):
           G1 隐含波动期限结构:log VIX9D − log VIX、log VXN − log VIX
           G2 VVIX:log VVIX 的 252 日 z 分数(t−1)
           G3 做市商 gamma:gex_z(SqueezeMetrics,t−1)
           G4 开盘 30 分钟微观结构:相对成交量、路径效率、跳跃占比、带外比例(b_rel_vol, b_eff, b_jump, b_outband)
           G5 价格持续期(10:00):事件率、方向不平衡、持续期相对季节的对数比(K=26,与 B0 相同)
           G6 星期效应:周三/周四/周五哑变量(周二为基准;周一不交易)
判定(与上一轮相同):全 OOS ΔSR > 0、dev(21–22) 与 test(23–26) 同为正、块自助 95% 下界 > −0.05,三者同时满足才算"优于 B";
另报 Romano–Wolf 单步族内 p(11 个配置共用同一组块自助下标)、分年、分布面板。不做任何参数扫描,不因结果增删配置。
数据门槛:分钟线 md5 必须等于服务器 /opt/etf_timing_data/min1(b1d6e929…);其余为公开源重建(清单 data/MANIFEST_sandbox_2026-10.md5);
B 必须复现 tearsheet_v3 ① 的 dev/test/全 Sharpe 到 ±0.03,否则中止。
python3 sizing_models_study.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b0_entry_features as B0
import disp_pred_study as DP
import duration_events as DE
import lowvol_avoidance_study as LV
import paths as _P
import sizing_sigma_feed as SF
import tearsheet as TS
import unified_sizing_posthoc as P
import unified_sizing_study as U
from etft.intraday.bars import load_bars

TAB = LV.TAB
MIN1_MD5 = "b1d6e92974b80a4a2e8a99fb5ba270a3"
REF = {"dev 19-22": 1.5870580454062146, "test 23-26": 2.461017077099429, "全 19-26": 2.0224162644821404}
LAMS2 = tuple(U.LAMS) + (30.0, 100.0)


# ============================================================ PPP:任意链接函数
def link(a, kind):
    if kind == "sig":
        k = 2 * expit(a); return k, k * (1 - k / 2)
    if kind == "lin":
        k = 1 + a; inside = (k > 0) & (k < 2); return np.clip(k, 0, 2), inside.astype(float)
    e = np.exp(np.clip(a, -20, 1.0)); inside = e < 2; return np.minimum(e, 2.0), np.where(inside, e, 0.0)


def ppp_fit(Z, Lb, Sb, Eb, lam, kind):
    """与 unified_sizing_posthoc.ppp_fit 相同的目标与 γ;Z 已含常数列。kind='sig' 时与 P.ppp_fit 逐位一致。"""
    n, p1 = Z.shape; B = Lb + Sb + Eb; gam = B.mean() / B.var()
    def f(th):
        kl, dl = link(Z @ th[:p1], kind); ks, ds = link(Z @ th[p1:], kind)
        R = kl * Lb + ks * Sb + Eb; Rc = R - R.mean()
        dRl = (Lb * dl)[:, None] * Z; dRs = (Sb * ds)[:, None] * Z
        g = np.r_[dRl.mean(0) - gam * (Rc[:, None] * dRl).mean(0), dRs.mean(0) - gam * (Rc[:, None] * dRs).mean(0)] - 2 * lam * th
        return -(R.mean() - 0.5 * gam * R.var() - lam * np.sum(th ** 2)), -g
    r = minimize(f, np.zeros(2 * p1), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    th = r.x.reshape(2, p1)
    return (lambda Zt: (link(Zt @ th[0], kind)[0], link(Zt @ th[1], kind)[0])), gam


def quad_expand(Z):
    cols = [Z, Z ** 2] + [Z[:, [i]] * Z[:, [j]] for i, j in combinations(range(Z.shape[1]), 2)]
    return np.column_stack(cols)


# ============================================================ M1:神经网络策略
def mlp_fit(Z, Lb, Sb, Eb, lam, seeds=(0, 1, 2, 3, 4), hidden=8, steps=400):
    import torch
    torch.set_num_threads(4)
    B = Lb + Sb + Eb; gam = float(B.mean() / B.var())
    tZ = torch.tensor(Z[:, 1:], dtype=torch.float64); tL, tS, tE = (torch.tensor(x, dtype=torch.float64) for x in (Lb, Sb, Eb))
    nets = []
    for sd in seeds:
        torch.manual_seed(sd)
        net = torch.nn.Sequential(torch.nn.Linear(tZ.shape[1], hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 2)).double()
        torch.nn.init.zeros_(net[2].weight); torch.nn.init.zeros_(net[2].bias)
        opt = torch.optim.Adam(net.parameters(), lr=0.01)
        for _ in range(steps):
            opt.zero_grad(); k = 2 * torch.sigmoid(net(tZ)); R = k[:, 0] * tL + k[:, 1] * tS + tE
            pen = sum((p ** 2).sum() for nm, p in net.named_parameters() if "weight" in nm)
            loss = -(R.mean() - 0.5 * gam * R.var(unbiased=False)) + lam * pen; loss.backward(); opt.step()
        nets.append(net.eval())
    def pred(Zt):
        with torch.no_grad():
            t = torch.tensor(Zt[:, 1:], dtype=torch.float64); k = torch.stack([2 * torch.sigmoid(n_(t)) for n_ in nets]).mean(0).numpy()
        return k[:, 0], k[:, 1]
    return pred, gam


# ============================================================ 通用走步(与 P.wf 同协议)
def wf(X, feats, Lb, Sb, Eb, sel, days, fit, lams, expand=None, embargo=0):
    """embargo:内层留一年交叉验证时,在被留出年份的前后各剔除 embargo 个交易日的训练样本(handbook 第 18 讲;波动聚集使相邻日不独立)。默认 0 = 原行为。"""
    yrs = days.year; out = pd.DataFrame(np.nan, index=days, columns=["kl", "ks"]); lam_sel = {}; pos = np.arange(len(days))
    def design(idx, ref):
        mu, sd = X.loc[ref, feats].mean(), X.loc[ref, feats].std().replace(0, 1)
        Z = ((X.loc[idx, feats] - mu) / sd).to_numpy()
        if expand is not None:
            Zr = ((X.loc[ref, feats] - mu) / sd).to_numpy(); Er = expand(Zr); m2, s2 = Er.mean(0), Er.std(0); s2[s2 == 0] = 1
            Z = (expand(Z) - m2) / s2
        return np.column_stack([np.ones(len(Z)), Z])
    for yr in U.OOS_YEARS:
        tr = sel & (yrs < yr); te = sel & (yrs == yr); cv = {}
        for lam in lams:
            sc = []
            for hy in sorted(set(yrs[tr])):
                a_, h_ = tr & (yrs != hy), tr & (yrs == hy)
                if embargo and h_.any(): a_ = a_ & ~((pos >= pos[h_].min() - embargo) & (pos <= pos[h_].max() + embargo))
                if a_.sum() < 150 or h_.sum() < 50: continue
                pr, g = fit(design(a_, a_), Lb[a_], Sb[a_], Eb[a_], lam)
                kl, ks = pr(design(h_, a_))
                sc.append(U.utility(kl * Lb[h_] + ks * Sb[h_] + Eb[h_], g) - U.utility(Lb[h_] + Sb[h_] + Eb[h_], g))
            cv[lam] = np.mean(sc) if sc else -np.inf
        lam = max(cv, key=cv.get); lam_sel[yr] = lam
        pr, _ = fit(design(tr, tr), Lb[tr], Sb[tr], Eb[tr], lam)
        out.loc[te, "kl"], out.loc[te, "ks"] = pr(design(te, tr))
    return out, lam_sel


# ============================================================ M3:RAVEN(repo 原样网络 + 效用损失)
def raven_wf(X, feats, Lb, Sb, Eb, sel, days, seeds=(7, 8, 9), epochs=60, lookback=64):
    import torch
    import raven_paper_model as RP
    torch.set_num_threads(4)
    yrs = days.year; out = pd.DataFrame(np.nan, index=days, columns=["kl", "ks"]); pos = np.arange(len(days))
    ok = sel & (pos >= lookback - 1)
    for yr in U.OOS_YEARS:
        tr = ok & (yrs < yr); te = ok & (yrs == yr)
        mu, sd = X.loc[tr, feats].mean(), X.loc[tr, feats].std().replace(0, 1)
        Zs = ((X[feats] - mu) / sd).fillna(0.0).to_numpy(np.float32)
        seq = lambda idx: np.stack([Zs[i - lookback + 1:i + 1] for i in np.where(idx)[0]])
        xtr, xte = torch.tensor(seq(tr)), torch.tensor(seq(te))
        tL, tS, tE = (torch.tensor(v[tr], dtype=torch.float32) for v in (Lb, Sb, Eb))
        B = Lb[tr] + Sb[tr] + Eb[tr]; gam = float(B.mean() / B.var())
        ks_all = []
        for sd_ in seeds:
            torch.manual_seed(sd_); mL, mS = RP.RAVEN(len(feats)), RP.RAVEN(len(feats))
            params = list(mL.parameters()) + list(mS.parameters())
            opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            gen = torch.Generator().manual_seed(sd_)
            for _ in range(epochs):
                mL.train(); mS.train()
                for idx in torch.randperm(len(xtr), generator=gen).split(128):
                    opt.zero_grad(set_to_none=True)
                    aL, eL, dL, _ = mL(xtr[idx]); aS, eS, dS, _ = mS(xtr[idx])
                    R = 2 * torch.sigmoid(aL) * tL[idx] + 2 * torch.sigmoid(aS) * tS[idx] + tE[idx]
                    loss = -(R.mean() - 0.5 * gam * R.var(unbiased=False)) + .1 * (eL + eS) + .01 * (dL + dS)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                sch.step()
            mL.eval(); mS.eval()
            with torch.no_grad():
                ks_all.append(np.column_stack([2 * torch.sigmoid(mL(xte)[0]).numpy(), 2 * torch.sigmoid(mS(xte)[0]).numpy()]))
        k = np.mean(ks_all, axis=0); out.loc[te, "kl"], out.loc[te, "ks"] = k[:, 0], k[:, 1]
        print(f"    RAVEN {yr}: 训练 {int(tr.sum())} 天,OOS 平均 k 多 {k[:, 0].mean():.3f} 空 {k[:, 1].mean():.3f}", flush=True)
    return out


# ============================================================ 新特征(repo 定义)
def family_features(b, X):
    days = b.days; Xd, _, _, _ = LV.build(b)
    F = pd.DataFrame(index=days)
    F["ts_9d"] = Xd.log_vix9d - Xd.log_vix; F["vxn_vix"] = Xd.log_vxn - Xd.log_vix
    # VVIX:与 unified_sizing_study.build_features 的 lvix_z 完全同口径(全历史滚动 252 日 z,再 t−1 对齐)
    vv = pd.read_parquet(_P.RAW / "cboe" / "VVIX.parquet")["close"]; vv.index = pd.to_datetime(vv.index).tz_localize(None); vv = vv[~vv.index.duplicated()].sort_index()
    lvv = np.log(vv); F["lvvix_z"] = ((lvv - lvv.rolling(252, min_periods=126).mean()) / lvv.rolling(252, min_periods=126).std()).shift(1).reindex(days, method="ffill")
    # gex_z:repo 定义(LV.build,在分钟线日历上滚动)→ 2019 上半年为预热期 NaN,置 0(中性);只影响 2021 折的训练样本(偏离记录)
    F["gex_z"] = Xd.gex_z.fillna(0.0)
    for c in ("b_rel_vol", "b_eff", "b_jump", "b_outband"): F[c] = Xd[c]
    sd_in, _ = B0.intraday_sd(b); ev = DE.day_events(b, sd_in, B0.K_EV); season = DE.seasonal_duration(ev, B0.K_EV)
    D = DE.trade_features(ev, season, B0.K_EV, np.arange(len(days)), np.full(len(days), 29), np.ones(len(days)))
    F["dur_rate"], F["dur_imb"], F["dur_m"] = D.ev_rate.to_numpy(), D.ev_imb.to_numpy(), D.ev_dur_m.fillna(0.0).to_numpy()
    for d in (2, 3, 4): F[f"dow{d}"] = Xd[f"dow{d}"]
    return F


FAM = {"G1 期限结构": ["ts_9d", "vxn_vix"], "G2 VVIX": ["lvvix_z"], "G3 GEX": ["gex_z"], "G4 开盘微观结构": ["b_rel_vol", "b_eff", "b_jump", "b_outband"],
       "G5 价格持续期": ["dur_rate", "dur_imb", "dur_m"], "G6 星期效应": ["dow2", "dow3", "dow4"]}


# ============================================================ 评估
def block_idx(n, L, nb, seed):
    rg = np.random.default_rng(seed); out = np.empty((nb, n), dtype=int); nblk = int(np.ceil(n / L))
    for i in range(nb):
        st = rg.integers(0, n - L + 1, nblk); out[i] = (st[:, None] + np.arange(L)[None, :]).ravel()[:n]
    return out


def main():
    t0 = time.time(); pd.set_option("display.width", 280, "display.max_rows", 200, "display.float_format", lambda v: f"{v:8.3f}")
    md5 = hashlib.md5(open(_P.RAW / "min1" / "QQQ.csv", "rb").read()).hexdigest()
    assert md5 == MIN1_MD5, f"分钟线与服务器不一致:{md5}"; print(f"[数据门槛] QQQ 分钟线 md5 = 服务器 {md5[:8]}…")
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    I = SF.build_inputs(b)
    L2, S2, GL, GS, keep, ov, X, leg1 = (I[k] for k in ("L2", "S2", "GL", "GS", "keep", "ov", "X", "leg1"))
    dn = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days)
    X = X.join(family_features(b, X))
    allc = sorted(set(c for v in FAM.values() for c in v)); nan_ = X.loc[sel, allc].isna().sum()
    assert nan_.sum() == 0, f"新特征在样本日上有缺失:{nan_[nan_ > 0].to_dict()}"
    Lb, Sb, Eb = (L2 * 1e4).to_numpy(), (S2 * 1e4).to_numpy(), ((leg1 + ov) * 1e4).to_numpy()
    Ls, Ss, Es = (pd.Series(v, index=days) for v in (Lb, Sb, Eb))

    def v3_from(k):
        kl = k.kl.where(om, 1.0).fillna(1.0); ks = k.ks.where(om, 1.0).fillna(1.0)
        return kl * L2 + ks * S2 + ov - (kl * GL + ks * GS) * keep

    # ---- B 与一致性
    kB, lamB = P.wf(X, Ls, Ss, Es, sel, days, P.CLEAN)[0], None
    portB = leg1 + v3_from(kB)
    for nm, (a, z) in (("dev 19-22", ("2019", "2022")), ("test 23-26", ("2023", "2026")), ("全 19-26", ("2019", "2026"))):
        s_ = TS.metrics(portB.loc[a:z])["Sharpe（绝对）"]; print(f"[复现] B {nm} Sharpe {s_:.3f}(tearsheet_v3 {REF[nm]:.3f},差 {s_-REF[nm]:+.3f})")
        assert abs(s_ - REF[nm]) <= 0.03, "复现超出 ±0.03,中止"
    kchk, _ = wf(X, P.CLEAN, Lb, Sb, Eb, sel, days, lambda Z, l_, s_, e_, lam: ppp_fit(Z, l_, s_, e_, lam, "sig"), U.LAMS)
    d = float(np.nanmax(np.abs(kchk.to_numpy() - kB.to_numpy()))); print(f"[一致性] 通用走步(sigmoid)vs repo P.wf:k 最大差 {d:.2e}"); assert d < 1e-6
    portG = leg1 + v3_from(kchk); dsr_g = TS.metrics(portG.loc["2021":"2026"])["Sharpe（绝对）"] - TS.metrics(portB.loc["2021":"2026"])["Sharpe（绝对）"]
    print(f"[一致性] 通用走步 vs repo P.wf:组合 OOS Sharpe 差 {dsr_g:+.2e}")

    K = {}; CK = SF.OUT / "sizing_models_cache"; CK.mkdir(exist_ok=True)
    sig = lambda Z, l_, s_, e_, lam: ppp_fit(Z, l_, s_, e_, lam, "sig")
    jobs = {"F1 线性 clip(1+θᵀz)": lambda: wf(X, P.CLEAN, Lb, Sb, Eb, sel, days, lambda Z, l_, s_, e_, lam: ppp_fit(Z, l_, s_, e_, lam, "lin"), U.LAMS)[0],
            "F2 对数线性 exp(θᵀz)": lambda: wf(X, P.CLEAN, Lb, Sb, Eb, sel, days, lambda Z, l_, s_, e_, lam: ppp_fit(Z, l_, s_, e_, lam, "exp"), U.LAMS)[0],
            "M1 神经网络策略": lambda: wf(X, P.CLEAN, Lb, Sb, Eb, sel, days, mlp_fit, U.LAMS)[0],
            "M2 二次 PPP": lambda: wf(X, P.CLEAN, Lb, Sb, Eb, sel, days, sig, LAMS2, expand=quad_expand)[0],
            "M3 RAVEN": lambda: raven_wf(X, P.CLEAN, Lb, Sb, Eb, sel, days)}
    for fam, cols in FAM.items(): jobs[fam] = (lambda cols=cols: wf(X, P.CLEAN + cols, Lb, Sb, Eb, sel, days, sig, U.LAMS)[0])
    budget = float(os.environ.get("BUDGET_S", "1e9")); pending = []
    for i, (nm, fn) in enumerate(jobs.items()):
        f = CK / f"cfg{i:02d}.pkl"
        if f.exists(): K[nm] = pd.read_pickle(f); continue
        if time.time() - t0 > budget: pending.append(nm); continue
        K[nm] = fn(); K[nm].to_pickle(f); print(f"  [{nm}] 完成 {time.time()-t0:.0f}s", flush=True)
    if pending: print(f"[未完成] {pending} —— 重跑同一命令续跑"); return

    # ---- 评估
    o = om.to_numpy(); oi = np.where(o)[0]; ref = portB.to_numpy()[oi]; dev = (yrs[oi] <= 2022); BI = block_idx(len(oi), 10, 2000, 0)
    srv = lambda x: x.mean() / x.std() * np.sqrt(252)
    def sr_rows(x): return np.array([srv(x[r]) for r in BI])
    refb = sr_rows(ref); rows, deltas, boots = [], {}, {}
    for nm, k in K.items():
        x = (leg1 + v3_from(k)).to_numpy()[oi]; dsr = srv(x) - srv(ref); db = sr_rows(x) - refb
        deltas[nm], boots[nm] = dsr, db
        r_ = dict(方案=nm, 组合SR=srv(x), ΔSR全=dsr, lo=np.percentile(db, 2.5), hi=np.percentile(db, 97.5),
                  ΔSR_dev=srv(x[dev]) - srv(ref[dev]), ΔSR_test=srv(x[~dev]) - srv(ref[~dev]),
                  k多=float(k.kl[om].mean()), k空=float(k.ks[om].mean()), k多sd=float(k.kl[om].std()))
        r_.update({str(y): srv(x[yrs[oi] == y]) - srv(ref[yrs[oi] == y]) for y in range(2021, 2027)})
        r_["优于B"] = bool(r_["ΔSR全"] > 0 and r_["ΔSR_dev"] > 0 and r_["ΔSR_test"] > 0 and r_["lo"] > -0.05)
        rows.append(r_)
    cen = np.column_stack([boots[n] - deltas[n] for n in K]); mx = cen.max(axis=1)
    R = pd.DataFrame(rows); R["p族内(RW)"] = [float(np.mean(mx >= deltas[n])) for n in K]
    print(f"\n=== 组合层(QQQ ① 腿 + v3 Sleeve),OOS 2021–26;ΔSR 相对 B(组合 SR {srv(ref):.3f});块自助 L=10 ×2000 ===")
    print(R.to_string(index=False))
    pan = {"B 基线": SF.panel(pd.Series(ref))}; pan.update({n: SF.panel(pd.Series((leg1 + v3_from(K[n])).to_numpy()[oi])) for n in K})
    Pn = pd.DataFrame(pan).T; print("\n=== 分布面板(组合,OOS 2021–26)==="); print(Pn.to_string())
    R.to_csv(TAB / "sizing_models.csv", index=False); Pn.to_csv(TAB / "sizing_models_panel.csv")
    pd.to_pickle(dict(K=K, kB=kB), SF.OUT / "sizing_models.pkl"); print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
