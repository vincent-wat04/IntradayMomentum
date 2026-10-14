"""① 统一仓位模型(U3p-clean)叠加到 SOTA Sleeve / 组合上(描述性)。docs/PROPOSAL_letf_rebalance.md §①。
python3 sota_unified_stack.py"""
import os, sys, warnings
warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS", "1"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
import liquidity_inverse_study as LI, unified_sizing_study as U, unified_sizing_posthoc as P
import sota_stack_study as S, sleeve_anatomy as SA, tearsheet as TS, trend_rv_sleeve_study as T, round3_common as C, regime_features as RF
import disp_pred_study as DP, lowvol_avoidance_study as L
from audit import reimpl_noise as RI
from event_calendar import FOMC_DECISION
from etft.intraday.bars import load_bars

OUT, TAB = U.OUT, U.TAB
WIN = {"全 2019–26": ("2019-01-01", "2026-12-31"), "dev 2019–22": ("2019-01-01", "2022-12-31"), "test 2023–26": ("2023-01-01", "2026-12-31"), "OOS 2021–26": ("2021-01-01", "2026-12-31")}
KEYS = ["年化收益 CAGR", "Sharpe（绝对）", "Sortino", "最大回撤 MDD", "Calmar", "最长水下（交易日）", "CVaR95（日）", "最差单日", "α（年化）", "α 的 t 值 (HAC)", "偏度", "超额峰度"]


def legs_from_PH(PH, days, scale):
    offs = list(PH)
    TR = pd.concat([LI.trades_from_fills(F, off) for off, (rr, F) in PH.items() if len(F)], ignore_index=True)
    cm = pd.concat([PH[o][1].groupby("day").commission.sum().reindex(days).fillna(0.0) if len(PH[o][1]) else pd.Series(0.0, index=days) for o in offs], axis=1).astype(float); cm.columns = offs
    adj = (cm - cm.mul(scale.fillna(1.0), axis=0)).mean(axis=1); out, cms = {}, {}
    for sd_, nm in ((1, "L"), (-1, "S")):
        sub = TR[TR.side == sd_]
        out[nm] = sub.pivot_table(index="day", columns="phase", values="pnl_net", aggfunc="sum").reindex(index=days, columns=offs).fillna(0.0).mean(axis=1)
        cms[nm] = sub.assign(c=sub.cm_in + sub.cm_out).pivot_table(index="day", columns="phase", values="c", aggfunc="sum").reindex(index=days, columns=offs).fillna(0.0).mean(axis=1)
    tot = (cms["L"] + cms["S"]).replace(0, np.nan)
    return out["L"] + (adj * cms["L"] / tot).fillna(0.0), out["S"] + (adj * cms["S"] / tot).fillna(0.0)


def fomc_late_PH(b):
    p = OUT / "fomc_late_phase_fills.pkl"
    if p.exists(): return pd.read_pickle(p)
    days = b.days; fs = set(d for d in FOMC_DECISION if d in set(days)); nonf = [d for d in days if d not in fs]; PH = {}
    for off in range(30):
        rr, F = RI.simulate(b, checks=[t for t in SA.chk(off) if t >= "14:30"], sizing="voltarget", max_lev=4.0, reentry="hold",
                            half_spread=0.005, commission=0.0045, skip_days=nonf, **SA.BASE)
        F = F.copy()
        if len(F): F["day"] = pd.to_datetime(F.day)
        PH[off] = (rr, F)
    pd.to_pickle(PH, p); return PH


def leg_ret(w, bh):
    w = w.reindex(bh.index).ffill().fillna(1.0); return w * bh - 2e-4 * w.diff().abs().fillna(0.0)


def main():
    pd.set_option("display.width", 270, "display.max_rows", 300, "display.float_format", lambda x: f"{x:9.4f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    scale = (b.day_close / LI.tqqq_traded_px(days)) / 3.0
    Lb0, Sb0 = legs_from_PH(LI.run_phases_fills(b), days, scale)
    Lf, Sf = legs_from_PH(fomc_late_PH(b), days, scale)
    V = pd.read_pickle(OUT / "sota_sleeve.pkl")["VER"]; base, S1v = V["S0 实盘"].reindex(days).fillna(0), V["S1 +FOMC≥14:30"].reindex(days).fillna(0)
    fm = pd.Series(days.isin(pd.DatetimeIndex([d for d in FOMC_DECISION])), index=days)
    L1 = Lb0.where(~fm, Lf); S1 = Sb0.where(~fm, Sf)
    print(f"[校验1] 实盘腿级 vs 实盘曲线 最大差 {float((Lb0+Sb0-base).abs().max())*1e4:.6f}bp;[校验2] FOMC 推迟版腿级 vs S1 曲线 最大差 {float((L1+S1-S1v).abs().max())*1e4:.6f}bp")
    cand, _, _ = S.build_candidates(b); skip = (S.roll_rank(cand["FFA-P"]).reindex(days) <= 0.2).fillna(False); ov = S.cpi_overlay(b)
    keep = (~skip).astype(float); L2, S2 = L1 * keep, S1 * keep; sota = L2 + S2 + ov
    ref = (S1v * keep + ov)
    print(f"[校验3] SOTA 腿级重建 vs sleeve_v2 口径 最大差 {float((sota-ref).abs().max())*1e4:.6f}bp;周一非零天数 {int((sota[days.dayofweek==0]!=0).sum())}")
    # 特征 / 样本
    D = pd.read_pickle(OUT / "unified_sizing.pkl"); X = D["X"]
    dn = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days)
    # QQQ 腿
    bh = pd.read_pickle(OUT / "strategy_comparison_bh.pkl").dropna(); bh.index = pd.to_datetime(bh.index)
    px = C.load_close("yahoo/QQQ.parquet"); r = px.pct_change().fillna(0.0)
    A2C = pd.read_pickle(OUT / "regime_timing_r2.pkl")["E2"]["A2_C|e0.5|alone"]
    W = S.bh_weights(px, r, {"FFA-SD": cand["FFA-SD"]}, A2C)
    legs = {"现用 RV21": leg_ret(W[("RV21[0.5,1.5]", "无趋势")], bh), "① RV21+SMA200": leg_ret(W[("RV21[0.5,1.5]", "SMA200|e=0.5")], bh),
            "② VT[FFA-SD]+SMA200": leg_ret(W[("VT[FFA-SD,cap1.5]", "SMA200|e=0.5")], bh)}
    legs = {k: v.reindex(days).fillna(0.0) for k, v in legs.items()}
    # A 冻结迁移:U3p-clean(训练目标 = 实盘 Sleeve + 现用 RV21 腿)
    kA, _, _ = P.wf(X, Lb0 * 1e4, Sb0 * 1e4, legs["现用 RV21"] * 1e4, sel, days, P.CLEAN)
    sized = lambda k, Lx, Sx: k.kl.where(om, 1.0).fillna(1.0) * Lx + k.ks.where(om, 1.0).fillna(1.0) * Sx
    sotaA = sized(kA, L2, S2) + ov
    # B 按组合重训
    kB, sotaB, lamB = {}, {}, {}
    for nm, lg in legs.items():
        kB[nm], co, lamB[nm] = P.wf(X, L2 * 1e4, S2 * 1e4, (lg + ov) * 1e4, sel, days, P.CLEAN); sotaB[nm] = sized(kB[nm], L2, S2) + ov
        print(f"  重训({nm}):λ* {lamB[nm]};OOS 平均 k多 {float(kB[nm].kl[om].mean()):.3f}、k空 {float(kB[nm].ks[om].mean()):.3f}")
    print(f"  冻结迁移:OOS 平均 k多 {float(kA.kl[om].mean()):.3f}、k空 {float(kA.ks[om].mean()):.3f}")
    # ---------------- 指标
    bh_d = bh.reindex(days).fillna(0.0)
    series = {"Sleeve | 实盘": base, "Sleeve | SOTA": sota, "Sleeve | SOTA + 统一仓位(冻结)": sotaA}
    for nm, lg in legs.items():
        series[f"{nm} | 实盘 Sleeve"] = lg + base; series[f"{nm} | SOTA"] = lg + sota
        series[f"{nm} | SOTA + 统一仓位(冻结)"] = lg + sotaA; series[f"{nm} | SOTA + 统一仓位(按组合重训)"] = lg + sotaB[nm]
    rows = []
    for wn, (a, z) in WIN.items():
        m = (days >= a) & (days <= z)
        for nm, x in series.items():
            mt = TS.metrics(x[m], bench=bh_d[m]); rows.append(dict(窗口=wn, 序列=nm, **{k: mt[k] for k in KEYS}))
    M = pd.DataFrame(rows); M.to_csv(TAB / "sota_unified_metrics.csv", index=False)
    for wn in WIN:
        print(f"\n===== {wn} ====="); t_ = M[M.窗口 == wn].set_index("序列")[KEYS]
        print(t_.to_string())
    # 增量:相对『SOTA(无乘数)』,OOS 2021–26
    print("\n===== 增量(相对同一组合的 SOTA 无乘数;OOS 2021–26;块自助 L=10;圆周平移 p)=====")
    idx = np.where(om.to_numpy())[0]; rg = np.random.default_rng(3); sh = rg.integers(60, len(idx) - 60, 2000); rows = []
    for nm, lg in legs.items():
        x0 = (lg + sota)[om]
        for tag, k in (("冻结", kA), ("按组合重训", kB[nm])):
            x1 = (lg + sized(k, L2, S2) + ov)[om]; d, dist = T.boot_dsr(x1, x0, nb=2000, L=10, seed=0)
            kl, ks = k.kl.fillna(1).to_numpy()[idx], k.ks.fillna(1).to_numpy()[idx]; Lv, Sv, rest = L2.to_numpy()[idx], S2.to_numpy()[idx], (lg + ov).to_numpy()[idx]
            null = np.array([L.sr(pd.Series(np.roll(kl, s) * Lv + np.roll(ks, s) * Sv + rest)) for s in sh]); p_ = float(np.mean(null >= L.sr(x1)))
            yr = {int(y): round(L.sr(x1[x1.index.year == y]) - L.sr(x0[x0.index.year == y]), 3) for y in range(2021, 2027)}
            rows.append(dict(组合=nm, 方式=tag, ΔSR=d, lo=np.percentile(dist, 2.5), hi=np.percentile(dist, 97.5), p圆周平移=p_,
                             ΔCAGR_pp=100 * (TS.metrics(x1)["年化收益 CAGR"] - TS.metrics(x0)["年化收益 CAGR"]), ΔMDD_pp=100 * (TS.metrics(x1)["最大回撤 MDD"] - TS.metrics(x0)["最大回撤 MDD"]),
                             Δ最差日bp=1e4 * (TS.metrics(x1)["最差单日"] - TS.metrics(x0)["最差单日"]), **{str(k_): v for k_, v in yr.items()}))
    I = pd.DataFrame(rows); print(I.to_string(index=False)); I.to_csv(TAB / "sota_unified_increment.csv", index=False)
    pd.to_pickle(dict(series=series, kA=kA, kB=kB, L2=L2, S2=S2, ov=ov), OUT / "sota_unified.pkl")


if __name__ == "__main__":
    main()
