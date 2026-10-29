"""第二轮 B 部分:高频因子库(HFZ 10 个)+ 跨资产(XA 4 个)→ 逐笔入场筛选(docs/PROPOSAL_intraday_vol_entry_r2.md §3,预注册)。
框架与 ivol_entry_study 相同(实盘 Sleeve 30 相位逐笔;y / y_ra;单变量、增量、跳过 20% + 8 格族内置换)。
python3 r2_factor_zoo.py
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
import ivol_entry_study as IE
import trade_filter_study as TF
from b0_econ import causal_rank
from b0_validate import Ctx
from etft.intraday.bars import load_bars

DATA, TAB = IE.DATA, IE.TAB
HFZ = ["smart_q", "pv_corr", "rv_corr", "fog", "tide", "surge_ret", "rkurt", "com_gap", "vol_surp", "rv_per_vol"]
XA = ["xa_rel_spy", "xa_iwm", "xa_spy_rvsurp", "xa_iwm_lead15"]


def prep(bq, bs, bi):
    P = {}
    for nm, b in (("q", bq), ("s", bs), ("i", bi)):
        c = np.ascontiguousarray(b.c.ffill(axis=1).to_numpy(), dtype=float); O = np.ascontiguousarray(b.day_open.to_numpy(), dtype=float)
        P[nm] = dict(c=c, O=O, A=IE.day_arrays(c, O))
    v = np.ascontiguousarray(np.nan_to_num(bq.v.to_numpy()), dtype=float); cV = np.cumsum(v, 1)
    P["q"].update(v=v, cV=cV, S_V=IE._season(cV), cMV=np.cumsum(np.arange(390)[None, :] * v, 1))
    return P


def features(P, ii, kk, side, sd14):
    n = len(ii); F = {f: np.full(n, np.nan) for f in HFZ + XA}
    q, s_, i_ = P["q"], P["s"], P["i"]; c, v, A = q["c"], q["v"], q["A"]
    for j in range(n):
        d, k, sg, sd = ii[j], kk[j], side[j], sd14[j]
        r = A["r"][d, :k + 1]; vv = v[d, :k + 1]; cc = c[d, :k + 1]; rv = A["cRV"][d, k]
        if rv <= 0 or vv.sum() <= 0 or not np.isfinite(sd): continue
        Sm = np.where(vv > 0, np.abs(r) / np.sqrt(np.maximum(vv, 1.0)), 0.0)
        top = Sm >= np.quantile(Sm, 0.8)
        F["smart_q"][j] = sg * np.log((cc[top] * vv[top]).sum() / vv[top].sum() / ((cc * vv).sum() / vv.sum())) if vv[top].sum() > 0 else 0.0
        F["pv_corr"][j] = sg * np.corrcoef(cc, vv)[0, 1] if cc.std() > 0 and vv.std() > 0 else 0.0
        F["rv_corr"][j] = np.corrcoef(np.abs(r), vv)[0, 1] if r.std() > 0 and vv.std() > 0 else 0.0
        rs = pd.Series(r).rolling(5).std().to_numpy()[4:]
        F["fog"][j] = rs.std() / rs.mean() if len(rs) > 2 and rs.mean() > 0 else 0.0
        vs = pd.Series(vv).rolling(5, min_periods=1).sum().to_numpy(); m_ = int(np.argmax(vs))
        F["tide"][j] = sg * np.log(cc[k] / cc[m_]) / (sd * np.sqrt((k - m_ + 1) / 390))
        thr = vv.mean() + vv.std(); ms = np.where((vv > thr) & (np.arange(k + 1) <= k - 5))[0]
        F["surge_ret"][j] = sg * np.mean(np.log(cc[ms + 5] / cc[ms])) / (sd * np.sqrt(5 / 390)) if len(ms) else 0.0
        F["rkurt"][j] = (k + 1) * np.sum(r ** 4) / rv ** 2
        F["com_gap"][j] = q["cMV"][d, k] / max(k, 1) / q["cV"][d, k] - A["com"][d, k]
        vsurp = np.log(q["cV"][d, k] / q["S_V"][d, k]) if np.isfinite(q["S_V"][d, k]) and q["S_V"][d, k] > 0 else np.nan
        rsurp = np.log(rv / A["S_rv"][d, k]) if np.isfinite(A["S_rv"][d, k]) and A["S_rv"][d, k] > 0 else np.nan
        F["vol_surp"][j] = vsurp; F["rv_per_vol"][j] = rsurp - vsurp
        sc = sd * np.sqrt((k + 1) / 390)
        rq = np.log(c[d, k] / q["O"][d]); rsp = np.log(s_["c"][d, k] / s_["O"][d]); riw = np.log(i_["c"][d, k] / i_["O"][d])
        F["xa_rel_spy"][j] = sg * (rq - rsp) / sc; F["xa_iwm"][j] = sg * riw / sc
        As = s_["A"]; F["xa_spy_rvsurp"][j] = np.log(As["cRV"][d, k] / As["S_rv"][d, k]) if np.isfinite(As["S_rv"][d, k]) and As["S_rv"][d, k] > 0 else np.nan
        F["xa_iwm_lead15"][j] = sg * np.log(i_["c"][d, k] / i_["c"][d, max(k - 15, 0)]) / (sd * np.sqrt(15 / 390))
    return pd.DataFrame(F)


def main():
    t0 = time.time(); pd.set_option("display.width", 260, "display.max_rows", 300, "display.float_format", lambda v: f"{v:8.3f}")
    bq, bs, bi = load_bars("QQQ"), load_bars("SPY"), load_bars("IWM"); days = bq.days; dpos = {d: i for i, d in enumerate(days)}
    TR = pd.read_pickle(DATA / "_TR.pkl").sort_values(["day", "phase", "k_dec"]).reset_index(drop=True)
    IV = pd.read_pickle(DATA / "_IVT_features.pkl"); ALL = pd.concat([pd.read_pickle(DATA / "_B0_features2.pkl").reset_index(drop=True), IV.drop(columns=["y", "y_ra"])], axis=1)
    y, y_ra = IV.y.to_numpy(), IV.y_ra.to_numpy(); assert np.allclose(y, TR.y.to_numpy())
    ii = TR.day.map(dpos).to_numpy(); kk = TR.k_dec.to_numpy(); side = TR.side.to_numpy().astype(float); sd14 = np.exp(ALL.log_sd14.to_numpy())
    P = prep(bq, bs, bi); F = features(P, ii, kk, side, sd14); print(f"[特征] {F.shape},缺失 {F.isna().sum()[F.isna().sum() > 0].to_dict()}  ({time.time()-t0:.0f}s)", flush=True)
    # ---- 投毒(QQQ/SPY/IWM 同时)
    rg = np.random.default_rng(0); worst = 0.0
    for j in rg.choice(len(TR), 10, replace=False):
        d, k = ii[j], kk[j]; P2 = {}
        for nm in ("q", "s", "i"):
            c2 = P[nm]["c"].copy(); O2 = P[nm]["O"].copy(); c2[d, k + 1:] *= 7.3; c2[d + 1:] *= 7.3; O2[d + 1:] *= 7.3
            P2[nm] = dict(c=c2, O=O2, A=IE.day_arrays(c2, O2))
        v2 = P["q"]["v"].copy(); v2[d, k + 1:] *= 5.1; v2[d + 1:] *= 5.1; cV2 = np.cumsum(v2, 1)
        P2["q"].update(v=v2, cV=cV2, S_V=IE._season(cV2), cMV=np.cumsum(np.arange(390)[None, :] * v2, 1))
        f2 = features(P2, ii[[j]], kk[[j]], side[[j]], sd14[[j]]).to_numpy()[0]; f1 = F.iloc[j].to_numpy()
        worst = max(worst, float(np.nanmax(np.abs(f2 - f1))))
    print(f"[投毒自检] 10 笔(价格 ×7.3、成交量 ×5.1,含 SPY/IWM):最大差 {worst:.2e}"); assert worst == 0.0
    # ---- 单变量
    yrs = TR.day.dt.year.to_numpy(); dayid = TR.day.dt.strftime("%Y-%m-%d").to_numpy(); rows = []
    for f in HFZ + XA:
        x = F[f].to_numpy()
        for tn, yy in (("y", y), ("y_ra", y_ra)):
            r_ = dict(族="HFZ" if f in HFZ else "XA", 特征=f, 目标=tn)
            for pn, m in (("2019–22", yrs <= 2022), ("2023–26", yrs >= 2023)):
                ok = m & np.isfinite(x); r_[f"IC {pn}"] = spearmanr(x[ok], yy[ok])[0]; lo, hi = IE.boot_ic(x[ok], yy[ok], dayid[ok], nb=200); r_[f"{pn} 显著"] = bool(lo > 0 or hi < 0)
            r_["同号且都显著"] = bool(r_["2019–22 显著"] and r_["2023–26 显著"] and np.sign(r_["IC 2019–22"]) == np.sign(r_["IC 2023–26"])); rows.append(r_)
    R1 = pd.DataFrame(rows); print("\n=== 1. 单变量 Spearman IC ==="); print(R1.to_string(index=False))
    C = pd.concat([F, ALL[["rv_surp", "relvol", "netvol15", "z_open", "bo_strength", "rv_ratio15"]]], axis=1).corr(method="spearman")
    print("\n与已有特征的最大 |相关|:"); print(C.loc[HFZ + XA, ["rv_surp", "relvol", "netvol15", "z_open", "bo_strength", "rv_ratio15"]].abs().max(1).round(2).to_string())
    # ---- 增量 + 经济
    ALL = pd.concat([ALL, F], axis=1); okrow = ALL[TF.FEATS_F1 + HFZ + XA].notna().all(1).to_numpy(); preds = {}
    for tn, yy in (("y", y), ("y_ra", y_ra)):
        for mdl in ("ridge", "lgbm"):
            for sn, cl in (("F1", TF.FEATS_F1), ("F1+HFZ", TF.FEATS_F1 + HFZ), ("F1+XA", TF.FEATS_F1 + XA)):
                pr = np.full(len(TR), np.nan)
                for yr in range(2021, 2027):
                    tr = (yrs < yr) & okrow; te = (yrs == yr) & okrow; pr[te] = TF.fit_predict(ALL.loc[tr, cl], yy[tr], ALL.loc[te, cl], mdl)
                preds[(tn, mdl, sn)] = pr
    print(f"  走步完成 ({time.time()-t0:.0f}s)", flush=True)
    oos = np.isfinite(preds[("y", "ridge", "F1")]); rows = []
    for tn, yy in (("y", y), ("y_ra", y_ra)):
        for mdl in ("ridge", "lgbm"):
            p0 = preds[(tn, mdl, "F1")][oos]; yo = yy[oos]
            for sn in ("F1+HFZ", "F1+XA"):
                p1 = preds[(tn, mdl, sn)][oos]; lo, hi = IE.boot_ic(p1, yo, dayid[oos], base=p0)
                rows.append(dict(目标=tn, 模型=mdl, 特征集=sn, IC_F1=spearmanr(p0, yo)[0], IC_新=spearmanr(p1, yo)[0], ΔIC=spearmanr(p1, yo)[0] - spearmanr(p0, yo)[0], lo=lo, hi=hi))
    R2 = pd.DataFrame(rows); print(f"\n=== 2. 增量 IC(OOS 2021–26,{int(oos.sum())} 笔)==="); print(R2.to_string(index=False))
    ctx = Ctx(); assert (ctx.TR.k_dec.to_numpy() == kk).all()
    def cell(pr):
        rk = causal_rank(pr, ctx.day_arr); cc = ctx.rule_curve(pr, rk, "skip", 0.20)
        return (ctx.dsr(cc, "dev"), ctx.dsr(cc, "test"), ctx.dsr(cc, "full"), ctx.dsr(cc, "full", port=True))
    rows = []
    for key, pr in preds.items():
        d_, t_, f_, pf = cell(pr); rows.append(dict(目标=key[0], 模型=key[1], 特征集=key[2], Sleeve_ΔSR_dev=d_, Sleeve_ΔSR_test=t_, Sleeve_ΔSR_全=f_, 组合ΔSR_全=pf))
    R3 = pd.DataFrame(rows); cells = [k for k in preds if k[2] != "F1"]
    obs = {k: R3[(R3.目标 == k[0]) & (R3.模型 == k[1]) & (R3.特征集 == k[2])].Sleeve_ΔSR_全.iloc[0] for k in cells}
    NP = int(os.environ.get("NPERM", "500")); rg = np.random.default_rng(20261023); mx = np.empty(NP)
    for p_ in range(NP):
        perm = np.arange(len(TR))
        for yv in np.unique(ctx.year):
            ix = np.where(ctx.year == yv)[0]; perm[ix] = rg.permutation(ix)
        mx[p_] = max(cell(preds[k][perm])[2] for k in cells)
    R3["p族内(8格max)"] = [float((np.sum(mx >= obs[(r["目标"], r["模型"], r["特征集"])]) + 1) / (NP + 1)) if r["特征集"] != "F1" else np.nan for r in R3.to_dict("records")]
    def judge(r):
        if r["特征集"] == "F1": return ""
        ref = R3[(R3.目标 == r["目标"]) & (R3.模型 == r["模型"]) & (R3.特征集 == "F1")].Sleeve_ΔSR_全.iloc[0]
        ok = r["Sleeve_ΔSR_dev"] > 0 and r["Sleeve_ΔSR_test"] > 0 and r["Sleeve_ΔSR_全"] >= 0.10 and r["p族内(8格max)"] < 0.05 and r["Sleeve_ΔSR_全"] > ref
        return "有证据" if ok else ("线索" if (r["Sleeve_ΔSR_dev"] > 0 and r["Sleeve_ΔSR_test"] > 0) else "无")
    R3["判定"] = [judge(r) for r in R3.to_dict("records")]
    print(f"\n=== 3. 跳过预测最差 20%(Sleeve ΔSR 相对不过滤;dev 21–22 / test 23–26;置换 {NP} 次)==="); print(R3.to_string(index=False))
    R1.to_csv(TAB / "r2_zoo_univariate.csv", index=False); R2.to_csv(TAB / "r2_zoo_incremental.csv", index=False); R3.to_csv(TAB / "r2_zoo_econ.csv", index=False)
    F.to_pickle(DATA / "_HFZ_features.pkl"); print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
