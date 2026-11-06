"""事后(看到 H1–H3 后追加,见 PROPOSAL_letf_rebalance.md 偏离记录):
1) 收盘前各时段的『顺日内方向』收益,分年 —— 检验『LETF 流被提前交易、收盘反转』的假说;
2) Sleeve 平仓时点网格(15:30/15:40/15:45/15:50/15:55/MOC),分年;
3) 把『15:46 提前平仓』叠加到 SOTA + 统一仓位(按组合重训)的 ① 组合。
python3 letf_posthoc.py"""
import os, sys, warnings
warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS", "1"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, statsmodels.api as sm
import liquidity_inverse_study as LI, lowvol_avoidance_study as L, sota_unified_stack as SU, tearsheet as TS, trend_rv_sleeve_study as T
from etft.intraday.bars import load_bars
from event_calendar import FOMC_DECISION

OUT, TAB = L.OUT, L.TAB


def tstat(x):
    x = pd.Series(x).dropna(); return float(sm.OLS(x.to_numpy(), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": 5}).tvalues[0])


def G_series(PH, days, o, cols, te):
    """te = 平仓决策时点(bar 收盘);返回 (long, short) 两条腿的逐日 G(30 相位均值):持有 te 之后到原平仓价的损益。"""
    k = cols.index(te) + 1; px = pd.Series(o[:, k], index=days)
    TR = pd.concat([LI.trades_from_fills(F, off) for off, (rr, F) in PH.items() if len(F)], ignore_index=True)
    TR = TR[(TR.t_in < te) & (TR.t_out > te)].copy(); TR["G"] = TR.shares * (TR.exit_px - TR.day.map(px))
    out = {}
    for sd_, nm in ((1, "L"), (-1, "S")):
        out[nm] = (TR[TR.side == sd_].groupby("day").G.sum() / 30.0).reindex(days).fillna(0.0)
    return out["L"], out["S"]


def main():
    pd.set_option("display.width", 250, "display.float_format", lambda x: f"{x:8.3f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year; cols = list(b.c.columns); c, o = b.c.to_numpy(), b.o.to_numpy()
    full = (b.last_idx.to_numpy() >= 389) & (days >= "2019-01-01"); prev = b.prev_close_adj.to_numpy(); td = full & (days.dayofweek != 0)
    # ---------- 1) 时段分解
    seg = [("10:00", "14:30"), ("14:30", "15:00"), ("15:00", "15:30"), ("15:30", "15:45"), ("15:45", "16:00")]
    rows = []
    for a, z in seg:
        ka = cols.index(a); kz = cols.index(z); r_dir = np.sign(c[:, ka] / prev - 1); x = pd.Series(r_dir * (c[:, kz] / c[:, ka] - 1) * 1e4, index=days)[full]
        row = dict(时段=f"{a}→{z}", 顺日内方向均值bp=float(x.mean()), t=tstat(x))
        row.update({str(y): round(float(x[x.index.year == y].mean()), 2) for y in range(2019, 2027)}); rows.append(row)
    S1 = pd.DataFrame(rows); print("1) 各时段收益 × sign(前收→时段起点),bp;分年均值:"); print(S1.to_string(index=False)); S1.to_csv(TAB / "letf_posthoc_segments.csv", index=False)
    # ---------- 2) 平仓时点网格(实盘 Sleeve)
    base = pd.read_pickle(OUT / "sleeve_anatomy_curve.pkl"); base.index = pd.to_datetime(base.index); base = base.reindex(days).fillna(0.0)
    PH = LI.run_phases_fills(b); rows = []
    for te in ("15:30", "15:40", "15:45", "15:50", "15:55"):
        GL, GS = G_series(PH, days, o, cols, te); G = GL + GS; x = base - G
        row = dict(平仓时点=te, 放弃的G日均bp=1e4 * float(G[td].mean()), t=tstat(G[td] * 1e4), 多头腿Gbp=1e4 * float(GL[td].mean()), 空头腿Gbp=1e4 * float(GS[td].mean()),
                   Sleeve_ΔSR=L.sr(x[td]) - L.sr(base[td]), dev=L.sr(x[td & (yrs <= 2022)]) - L.sr(base[td & (yrs <= 2022)]), test=L.sr(x[td & (yrs >= 2023)]) - L.sr(base[td & (yrs >= 2023)]))
        row.update({str(y): round(L.sr(x[td & (yrs == y)]) - L.sr(base[td & (yrs == y)]), 3) for y in range(2019, 2027)}); rows.append(row)
    S2 = pd.DataFrame(rows); print("\n2) Sleeve 平仓时点网格(相对 MOC;正 ΔSR = 提前平仓更好):"); print(S2.to_string(index=False)); S2.to_csv(TAB / "letf_posthoc_exit_grid.csv", index=False)
    # ---------- 3) 叠加到 SOTA + 统一仓位(重训)的 ① 组合
    U = pd.read_pickle(OUT / "sota_unified.pkl"); series = U["series"]; kB = U["kB"]["① RV21+SMA200"]
    Df = pd.read_pickle(OUT / "unified_sizing.pkl"); X = Df["X"]
    import disp_pred_study as DP
    dn = DP.targets(DP.Arr(b), 30).dn; sel = (pd.Series(td, index=days) & X.notna().all(1) & dn.notna()).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days)
    GLb, GSb = G_series(PH, days, o, cols, "15:45"); GLf, GSf = G_series(SU.fomc_late_PH(b), days, o, cols, "15:45")
    fm = pd.Series(days.isin(pd.DatetimeIndex(FOMC_DECISION)), index=days); GL = GLb.where(~fm, GLf); GS = GSb.where(~fm, GSf)
    import sota_stack_study as S
    cand, _, _ = S.build_candidates(b); keep = (~(S.roll_rank(cand["FFA-P"]).reindex(days) <= 0.2).fillna(False)).astype(float)
    kl = kB.kl.where(om, 1.0).fillna(1.0); ks = kB.ks.where(om, 1.0).fillna(1.0); Gsota = (kl * GL + ks * GS) * keep
    bh = pd.read_pickle(OUT / "strategy_comparison_bh.pkl").dropna(); bh.index = pd.to_datetime(bh.index); bhd = bh.reindex(days).fillna(0.0)
    rows = []
    for nm in ("① RV21+SMA200 | 实盘 Sleeve", "① RV21+SMA200 | SOTA + 统一仓位(按组合重训)"):
        x0 = series[nm]; x1 = x0 - (Gsota if "SOTA" in nm else (GLb + GSb))
        for tag, x in (("MOC", x0), ("15:46 平仓", x1)):
            for wn, (a, z) in SU.WIN.items():
                m = (days >= a) & (days <= z); mt = TS.metrics(x[m], bench=bhd[m])
                rows.append(dict(序列=nm.replace("统一仓位", "UNI"), 平仓=tag, 窗口=wn, **{k: mt[k] for k in ("Sharpe（绝对）", "年化收益 CAGR", "最大回撤 MDD", "Calmar", "最差单日")}))
    S3 = pd.DataFrame(rows); S3.to_csv(TAB / "letf_posthoc_combo.csv", index=False)
    print("\n3) ① 组合:MOC vs 15:46 提前平仓")
    for k_ in ("Sharpe（绝对）", "年化收益 CAGR", "最大回撤 MDD"):
        print(f"\n  {k_}"); print(S3.pivot_table(index=["序列", "平仓"], columns="窗口", values=k_, sort=False)[list(SU.WIN)].to_string())
    x0 = series["① RV21+SMA200 | SOTA + 统一仓位(按组合重训)"]; x1 = x0 - Gsota; m = days >= "2019-01-01"
    d, dist = T.boot_dsr(x1[m], x0[m], nb=2000, L=10, seed=0); print(f"\n  ΔSR(15:46 − MOC,全 2019–26)= {d:+.3f} [{np.percentile(dist,2.5):+.3f}, {np.percentile(dist,97.5):+.3f}]")


if __name__ == "__main__":
    main()
