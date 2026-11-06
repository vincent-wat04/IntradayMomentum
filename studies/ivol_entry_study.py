"""B 部分:日内波动时间分布特征(IVT)→ 入场筛选(docs/PROPOSAL_intraday_vol_entry.md §2,预注册)。
目标 y(单笔净回报 bp)与 y_ra(y / 持仓期已实现波动);特征 IVT 9 个;自检、单变量 IC、增量 ΔIC、跳过 20% 的经济检验 + 族内置换。
python3 ivol_entry_study.py
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b0_entry_features as B0
import liquidity_inverse_study as LI
import trade_filter_study as TF
from b0_econ import causal_rank, panel
from b0_validate import Ctx
from etft.intraday.bars import load_bars

DATA, TAB = B0.DATA, B0.TAB
IVT = ["vr5", "vr10", "rv_surp", "rv_com_dev", "rv_ent", "jump_k", "rs_sgn", "rskew_sgn", "eff_k"]


# ============================================================ 特征
def day_arrays(c, O):
    """c: (n_days, 390) 已前向填充的 1 分钟收盘;O: 开盘。返回逐日累计量(每行独立计算)与 20 日季节均值(只用此前的日子)。"""
    c = np.ascontiguousarray(c, dtype=float); O = np.ascontiguousarray(O, dtype=float)   # 固定内存布局:否则跨行求和的舍入顺序随布局变化(投毒自检发现 1e-16 差)
    lp = np.column_stack([np.log(O), np.log(c)])                         # lp[:, m+1] = log c_m,lp[:, 0] = log O
    r = np.diff(lp, axis=1)                                              # r[:, m] = 第 m 根 bar 的对数收益
    r = np.nan_to_num(r)
    r2 = r ** 2; m_ = np.arange(390)[None, :]
    A = dict(lp=lp, r=r, cRV=np.cumsum(r2, 1), cAbs=np.cumsum(np.abs(r), 1), cR3=np.cumsum(r ** 3, 1),
             cRSp=np.cumsum(np.where(r > 0, r2, 0.0), 1), cMR=np.cumsum(m_ * r2, 1),
             cBV=np.cumsum(np.column_stack([np.zeros(len(r)), np.abs(r[:, 1:]) * np.abs(r[:, :-1])]), 1))
    com = A["cMR"] / np.maximum(m_, 1) / np.where(A["cRV"] > 0, A["cRV"], np.nan)
    A["com"] = com
    A["S_rv"] = _season(A["cRV"]); A["S_com"] = _season(com)
    return A


def _season(X, w=20, minp=10):
    """第 i 行 = 第 i−w..i−1 行的均值(逐行独立求和,确定性;不含当天)。"""
    out = np.full_like(X, np.nan)
    for i in range(len(X)):
        seg = X[max(0, i - w):i]
        if len(seg) >= minp:
            ok = np.isfinite(seg); cnt = ok.sum(0)
            out[i] = np.where(cnt >= minp, np.where(ok, seg, 0.0).sum(0) / np.maximum(cnt, 1), np.nan)
    return out


def trade_features(A, ii, kk, side):
    n = len(ii); F = {f: np.full(n, np.nan) for f in IVT}
    lp, r = A["lp"], A["r"]
    for j in range(n):
        i, k, s = ii[j], kk[j], side[j]
        rv = A["cRV"][i, k]
        if rv <= 0: continue
        for q, nm in ((5, "vr5"), (10, "vr10")):
            nb = (k + 1) // q
            if nb >= 3:
                pos = (k + 1) - q * np.arange(nb + 1)                    # lp 下标(从 k+1 往回)
                rq = -np.diff(lp[i, pos]); lo = (k + 1) - q * nb
                rv1 = A["cRV"][i, k] - (A["cRV"][i, lo - 1] if lo >= 1 else 0.0)
                F[nm][j] = np.sum(rq ** 2) / rv1 if rv1 > 0 else np.nan
        F["rv_surp"][j] = np.log(rv / A["S_rv"][i, k]) if np.isfinite(A["S_rv"][i, k]) and A["S_rv"][i, k] > 0 else np.nan
        F["rv_com_dev"][j] = A["com"][i, k] - A["S_com"][i, k]
        b5 = np.add.reduceat(r[i, :k + 1] ** 2, np.arange(0, k + 1, 5)); p = b5 / b5.sum(); p = p[p > 0]
        F["rv_ent"][j] = -np.sum(p * np.log(p)) / np.log(len(b5)) if len(b5) > 1 else np.nan
        rv_1 = rv - r[i, 0] ** 2
        F["jump_k"][j] = np.clip(1 - (np.pi / 2) * A["cBV"][i, k] / rv_1, 0, 1) if rv_1 > 0 else np.nan
        F["rs_sgn"][j] = s * (2 * A["cRSp"][i, k] - rv) / rv
        F["rskew_sgn"][j] = s * np.sqrt(k + 1) * A["cR3"][i, k] / rv ** 1.5
        F["eff_k"][j] = abs(lp[i, k + 1] - lp[i, 0]) / A["cAbs"][i, k] if A["cAbs"][i, k] > 0 else np.nan
    return pd.DataFrame(F)


def hold_rv(c, O, o, ii, kin, kout, exit_px):
    """持仓期已实现方差:入场(k_in+1 开盘)→ 各 bar 收盘 → 出场价。"""
    out = np.empty(len(ii))
    for j in range(len(ii)):
        i, a, z = ii[j], kin[j], kout[j]
        seg = [np.log(c[i, a + 1] / o[i, a + 1])]
        if z >= a + 2: seg.extend(np.log(c[i, a + 2:z + 1] / c[i, a + 1:z]))
        seg.append(np.log(exit_px[j] / c[i, z]))
        out[j] = np.sum(np.square(seg))
    return out


# ============================================================ 评估工具
def boot_ic(p, y, dayid, nb=300, seed=0, base=None):
    ud, inv = np.unique(dayid, return_inverse=True); order = np.argsort(inv, kind="stable")
    st = np.searchsorted(inv[order], np.arange(len(ud))); en = np.append(st[1:], len(order)); rg = np.random.default_rng(seed); out = []
    for _ in range(nb):
        pk = rg.integers(0, len(ud), len(ud)); ix = np.concatenate([order[st[q]:en[q]] for q in pk])
        v = spearmanr(p[ix], y[ix])[0]
        if base is not None: v -= spearmanr(base[ix], y[ix])[0]
        out.append(v)
    return np.percentile(out, [2.5, 97.5])


def main():
    t0 = time.time(); pd.set_option("display.width", 260, "display.max_rows", 300, "display.float_format", lambda v: f"{v:8.3f}")
    b = load_bars("QQQ"); days = b.days; cols = list(b.c.columns); tix = {t: i for i, t in enumerate(cols)}; dpos = {d: i for i, d in enumerate(days)}
    TR = pd.read_pickle(DATA / "_TR.pkl").sort_values(["day", "phase", "k_dec"]).reset_index(drop=True)
    PH = LI.run_phases_fills(b)
    T = pd.concat([LI.trades_from_fills(F, off) for off, (rr, F) in PH.items() if len(F)], ignore_index=True)
    T["k_dec"] = T.t_in.map(tix); T["k_out"] = T.t_out.map(tix); T["day"] = pd.to_datetime(T.day)
    TR = TR.merge(T[["day", "phase", "k_dec", "side", "k_out"]], on=["day", "phase", "k_dec", "side"], how="left"); assert TR.k_out.notna().all()
    ii = TR.day.map(dpos).to_numpy(); kk = TR.k_dec.to_numpy(); ko = TR.k_out.to_numpy().astype(int); side = TR.side.to_numpy().astype(float)
    c = b.c.ffill(axis=1).to_numpy(); O = b.day_open.to_numpy(); o = b.o.to_numpy()
    A = day_arrays(c, O); F = trade_features(A, ii, kk, side); print(f"[特征] {len(F)} 笔 × {len(IVT)} 个  ({time.time()-t0:.0f}s)")
    rvh = hold_rv(c, O, o, ii, kk, ko, TR.exit_px.to_numpy())
    y = TR.y.to_numpy(); y_ra = np.clip(y / (1e4 * np.sqrt(np.maximum(rvh, 1e-12))), -10, 10)
    print(f"[目标] y 中位 {np.median(y):.1f}bp;y_ra 分位 1/50/99% = {np.round(np.quantile(y_ra, [.01, .5, .99]), 2)};截断比例 {(np.abs(y_ra) >= 10).mean():.2%};corr(y, y_ra) = {spearmanr(y, y_ra)[0]:.3f}")
    # ---- 自检 1:未来投毒
    rg = np.random.default_rng(0); worst = 0.0
    for j in rg.choice(len(TR), 12, replace=False):
        i, k = ii[j], kk[j]; c2 = c.copy(); O2 = O.copy(); c2[i, k + 1:] *= 7.3; c2[i + 1:] *= 7.3; O2[i + 1:] *= 7.3
        A2 = day_arrays(c2, O2); f2 = trade_features(A2, ii[[j]], kk[[j]], side[[j]]).to_numpy()[0]
        worst = max(worst, float(np.nanmax(np.abs(f2 - F.iloc[j].to_numpy()))) if np.isfinite(f2).any() else 0.0)
    print(f"[投毒自检] 12 笔:k 之后及之后各日价格 ×7.3,IVT 最大差 {worst:.2e}(须为 0)"); assert worst == 0.0
    # ---- 自检 2:阳性对照
    fut5 = side * np.log(c[ii, np.minimum(kk + 6, 389)] / c[ii, kk + 1])
    print(f"[阳性对照] 入场后 5 分钟收益 IC:y {spearmanr(fut5, y)[0]:.3f};y_ra {spearmanr(fut5, y_ra)[0]:.3f}")
    # ---- 1. 单变量 IC
    yrs = TR.day.dt.year.to_numpy(); dayid = TR.day.dt.strftime("%Y-%m-%d").to_numpy(); rows = []
    for f in IVT:
        x = F[f].to_numpy()
        for tn, yy in (("y", y), ("y_ra", y_ra)):
            r_ = dict(特征=f, 目标=tn)
            for pn, m in (("2019–22", yrs <= 2022), ("2023–26", yrs >= 2023)):
                ok = m & np.isfinite(x); ic = spearmanr(x[ok], yy[ok])[0]; lo, hi = boot_ic(x[ok], yy[ok], dayid[ok], nb=200)
                r_[f"IC {pn}"] = ic; r_[f"{pn} 显著"] = bool(lo > 0 or hi < 0)
            r_["同号且都显著"] = bool(r_["2019–22 显著"] and r_["2023–26 显著"] and np.sign(r_["IC 2019–22"]) == np.sign(r_["IC 2023–26"])); rows.append(r_)
    R1 = pd.DataFrame(rows); print("\n=== 1. 单变量 Spearman IC(按交易日聚类自助 95% 判显著)==="); print(R1.to_string(index=False))
    print("\nIVT 特征之间及与 F1 关键特征的相关:")
    ALL = pd.read_pickle(DATA / "_B0_features2.pkl"); ALL = pd.concat([ALL.reset_index(drop=True), F], axis=1)
    print(ALL[IVT + ["rv_ratio15", "range15", "z_open", "bo_strength", "rv_near_share"]].corr(method="spearman").loc[IVT].round(2).to_string())
    # ---- 2. 增量 ΔIC(walk-forward)
    okrow = ALL[TF.FEATS_F1 + IVT].notna().all(1).to_numpy(); preds = {}
    for tn, yy in (("y", y), ("y_ra", y_ra)):
        for mdl in ("ridge", "lgbm"):
            for sn, cl in (("F1", TF.FEATS_F1), ("F1+IVT", TF.FEATS_F1 + IVT)):
                pr = np.full(len(TR), np.nan)
                for yr in range(2021, 2027):
                    tr = (yrs < yr) & okrow; te = (yrs == yr) & okrow
                    pr[te] = TF.fit_predict(ALL.loc[tr, cl], yy[tr], ALL.loc[te, cl], mdl)
                preds[(tn, mdl, sn)] = pr
    print(f"  走步完成 ({time.time()-t0:.0f}s)", flush=True)
    rows = []; oos = np.isfinite(preds[("y", "ridge", "F1")])
    for tn, yy in (("y", y), ("y_ra", y_ra)):
        for mdl in ("ridge", "lgbm"):
            p0, p1 = preds[(tn, mdl, "F1")][oos], preds[(tn, mdl, "F1+IVT")][oos]; yo = yy[oos]
            lo, hi = boot_ic(p1, yo, dayid[oos], base=p0)
            rows.append(dict(目标=tn, 模型=mdl, IC_F1=spearmanr(p0, yo)[0], IC_F1IVT=spearmanr(p1, yo)[0], ΔIC=spearmanr(p1, yo)[0] - spearmanr(p0, yo)[0], lo=lo, hi=hi,
                             ΔIC_dev=spearmanr(p1[yrs[oos] <= 2022], yo[yrs[oos] <= 2022])[0] - spearmanr(p0[yrs[oos] <= 2022], yo[yrs[oos] <= 2022])[0],
                             ΔIC_test=spearmanr(p1[yrs[oos] >= 2023], yo[yrs[oos] >= 2023])[0] - spearmanr(p0[yrs[oos] >= 2023], yo[yrs[oos] >= 2023])[0]))
    R2 = pd.DataFrame(rows); print(f"\n=== 2. 增量 IC(OOS 2021–26,{int(oos.sum())} 笔;区间 = 按交易日聚类自助 95%)==="); print(R2.to_string(index=False))
    # ---- 3. 经济:跳过 20%
    ctx = Ctx(); assert (ctx.TR.day.to_numpy() == TR.day.to_numpy()).all() and (ctx.TR.k_dec.to_numpy() == kk).all()
    def cell(pr):
        rk = causal_rank(pr, ctx.day_arr); cc = ctx.rule_curve(pr, rk, "skip", 0.20)
        return cc, (ctx.dsr(cc, "dev"), ctx.dsr(cc, "test"), ctx.dsr(cc, "full"), ctx.dsr(cc, "full", port=True))
    rows = []; curves = {}
    for tn in ("y", "y_ra"):
        for mdl in ("ridge", "lgbm"):
            for sn in ("F1", "F1+IVT"):
                cc, (d, t_, f, pf) = cell(preds[(tn, mdl, sn)]); curves[(tn, mdl, sn)] = cc
                rows.append(dict(目标=tn, 模型=mdl, 特征=sn, Sleeve_ΔSR_dev=d, Sleeve_ΔSR_test=t_, Sleeve_ΔSR_全=f, 组合ΔSR_全=pf))
    R3 = pd.DataFrame(rows)
    cells = [(tn, mdl, "F1+IVT") for tn in ("y", "y_ra") for mdl in ("ridge", "lgbm")]
    obs = {cl: R3[(R3.目标 == cl[0]) & (R3.模型 == cl[1]) & (R3.特征 == cl[2])].Sleeve_ΔSR_全.iloc[0] for cl in cells}
    NP = int(os.environ.get("NPERM", "500")); rg = np.random.default_rng(20261022); mx = np.empty(NP); year = ctx.year
    for p_ in range(NP):
        perm = np.arange(len(TR))
        for yv in np.unique(year):
            ix = np.where(year == yv)[0]; perm[ix] = rg.permutation(ix)
        mx[p_] = max(cell(preds[cl][perm])[1][2] for cl in cells)
    R3["p族内(4格max)"] = [float((np.sum(mx >= obs[(r.目标, r.模型, r.特征)]) + 1) / (NP + 1)) if r.特征 == "F1+IVT" else np.nan for r in R3.itertuples()]
    def judge(r):
        if r["特征"] != "F1+IVT": return ""
        ref = R3[(R3.目标 == r["目标"]) & (R3.模型 == r["模型"]) & (R3.特征 == "F1")].Sleeve_ΔSR_全.iloc[0]
        dv, ts, fl = r["Sleeve_ΔSR_dev"], r["Sleeve_ΔSR_test"], r["Sleeve_ΔSR_全"]
        ok = dv > 0 and ts > 0 and fl >= 0.10 and r["p族内(4格max)"] < 0.05 and fl > ref
        return "有证据" if ok else ("线索" if (dv > 0 and ts > 0) else "无")
    R3["判定"] = [judge(r) for r in R3.to_dict("records")]
    print(f"\n=== 3. 跳过预测最差 20%(Sleeve ΔSR 相对不过滤;dev 2021–22 / test 2023–26;组合 = L1 RV + Sleeve;置换 {NP} 次)==="); print(R3.to_string(index=False))
    m = ctx.per_mask["full"]; pan = {"不过滤": panel(pd.Series(ctx.base[m]))}
    for cl in cells: pan[" | ".join(cl)] = panel(pd.Series(curves[cl][m]))
    Pn = pd.DataFrame(pan).T; print("\n=== Sleeve 分布面板(2021–26)==="); print(Pn.round(3).to_string())
    R1.to_csv(TAB / "ivol_entry_univariate.csv", index=False); R2.to_csv(TAB / "ivol_entry_incremental.csv", index=False)
    R3.to_csv(TAB / "ivol_entry_econ.csv", index=False); Pn.to_csv(TAB / "ivol_entry_panel.csv")
    F.assign(y=y, y_ra=y_ra).to_pickle(DATA / "_IVT_features.pkl"); print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
