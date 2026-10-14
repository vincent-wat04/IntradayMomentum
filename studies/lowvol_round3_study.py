"""lowvol_avoidance 第三轮:已发表的新闻文本指数 N 组(GPR/EPU)。见 docs/PROPOSAL_lowvol_avoidance.md「第三轮」。"""
from __future__ import annotations
import os, sys, warnings
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lowvol_avoidance_study as L
import trend_rv_sleeve_study as T
import round3_common as C
import paths as _P
from event_calendar import event_frame
from etft.intraday.bars import load_bars
from sens_rv_weights import pct_w
OUT, TAB = L.OUT, L.TAB
NEWS = _P.RAW / "newsidx"


def prev_day(s, days):
    s = s.copy(); s.index = pd.DatetimeIndex(s.index) + pd.Timedelta(days=1)       # 日期 d 的值从 d+1 起可用
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(days, method="ffill")


def news_frame(days):
    g = pd.read_excel(NEWS / "data_gpr_daily_recent.xls"); g["date"] = pd.to_datetime(g["date"]); g = g.set_index("date")
    e = pd.read_csv(NEWS / "All_Daily_Policy_Data.csv"); e["date"] = pd.to_datetime(dict(year=e.year, month=e.month, day=e.day)); e = e.set_index("date").daily_policy_index
    lg, le = np.log1p(g.GPRD), np.log1p(e)
    rel = lambda x: x - x.rolling(30, min_periods=20).mean().shift(1)                # 当日减去『此前』30 日均值(再整体滞后 1 天使用)
    F = pd.DataFrame({
        "n_gpr": prev_day(lg, days), "n_gpr_rel": prev_day(rel(lg), days),
        "n_gpr_act": prev_day(np.log1p(g.GPRD_ACT), days), "n_gpr_thr": prev_day(np.log1p(g.GPRD_THREAT), days),
        "n_epu": prev_day(le, days), "n_epu_rel": prev_day(rel(le), days), "n_epu_5d": prev_day(np.log1p(e.rolling(5).mean()), days)}, index=days)
    return F


def main():
    b = load_bars("QQQ"); days = b.days
    X, Y, A, B = L.build(b)
    E = event_frame(days, b.last_idx.to_numpy()); EC = list(E.columns)
    N = news_frame(days); NC = list(N.columns)
    X = pd.concat([X, E, N], axis=1)
    print("N 组缺失占比(OOS 期):", X.loc["2021":, NC].isna().mean().round(3).to_dict())
    sets = {"A·ridge": (A, "ridge"), "A+B+E·ridge": (A + B + EC, "ridge"), "N·ridge": (NC, "ridge"),
            "A+N·ridge": (A + NC, "ridge"), "A+B+E+N·ridge": (A + B + EC + NC, "ridge"), "A+E+N·ridge": (A + EC + NC, "ridge")}
    pred, base_rv, thr_s = L.walk_forward(X, Y, sets)
    oos = X.index[(X.index.year >= 2021) & X[A + B + EC + NC].notna().all(1) & Y.notna().all(1)]
    ylow = (Y.loc[oos, "disp"] <= thr_s.loc[oos]).astype(int).to_numpy()
    sse_b = ((Y.loc[oos, "y_rv"] - base_rv.loc[oos]) ** 2).to_numpy()
    pd.set_option("display.width", 220)
    rows = []
    for (nm, key), p in pred.items():
        p = p.loc[oos]
        if key == "rv":
            e = ((Y.loc[oos, "y_rv"] - p) ** 2).to_numpy(); pt, lo, hi = L.month_boot(oos, lambda ii, e=e: 1 - e[ii].sum() / sse_b[ii].sum())
            rows.append(dict(模型=nm, 指标="OOS R² (log RV)", 值=pt, lo=lo, hi=hi))
        else:
            pp = p.to_numpy(); pt, lo, hi = L.month_boot(oos, lambda ii, pp=pp: L.auc(ylow[ii], pp[ii])); rows.append(dict(模型=nm, 指标="AUC(低波)", 值=pt, lo=lo, hi=hi))
    R1 = pd.DataFrame(rows); print("\n1) 预测力(OOS 2021–2026)"); print(R1.round(4).to_string(index=False))
    inc = []
    for n1, n0 in (("A+N·ridge", "A·ridge"), ("A+B+E+N·ridge", "A+B+E·ridge")):
        e1 = ((Y.loc[oos, "y_rv"] - pred[(n1, "rv")].loc[oos]) ** 2).to_numpy(); e0 = ((Y.loc[oos, "y_rv"] - pred[(n0, "rv")].loc[oos]) ** 2).to_numpy()
        pt, lo, hi = L.month_boot(oos, lambda ii: (e0[ii].sum() - e1[ii].sum()) / sse_b[ii].sum()); inc.append(dict(比较=f"{n1} − {n0}", 指标="ΔR²", 值=pt, lo=lo, hi=hi))
        p1, p0 = pred[(n1, "low")].loc[oos].to_numpy(), pred[(n0, "low")].loc[oos].to_numpy()
        pt, lo, hi = L.month_boot(oos, lambda ii: L.auc(ylow[ii], p1[ii]) - L.auc(ylow[ii], p0[ii])); inc.append(dict(比较=f"{n1} − {n0}", 指标="ΔAUC", 值=pt, lo=lo, hi=hi))
    R1i = pd.DataFrame(inc); print("\n增量:"); print(R1i.round(4).to_string(index=False))
    # 分期增量(是否集中在某些年份/事件期)
    yrs = oos.year
    e1 = ((Y.loc[oos, "y_rv"] - pred[("A+B+E+N·ridge", "rv")].loc[oos]) ** 2); e0 = ((Y.loc[oos, "y_rv"] - pred[("A+B+E·ridge", "rv")].loc[oos]) ** 2)
    print("\n逐年 ΔSSE%(N 组的贡献,>0 = 改善):", {int(y): round(float((e0[yrs == y].sum() - e1[yrs == y].sum()) / e0[yrs == y].sum()), 4) for y in sorted(set(yrs))})
    R1.to_csv(TAB / "lowvol_r3_prediction.csv", index=False); R1i.to_csv(TAB / "lowvol_r3_incremental.csv", index=False)
    # 单变量:N 组与残差(相对 A+B+E)
    res = (Y.y_rv - pred[("A+B+E·ridge", "rv")]).loc[oos]
    from scipy.stats import spearmanr
    print("\nN 组各特征与『A+B+E 残差』的 Spearman:", {c: round(float(spearmanr(X.loc[oos, c], res)[0]), 3) for c in NC})
    # 变现
    S2 = pd.read_pickle(OUT / "lowvol_round2.pkl")
    ret = pd.read_pickle(OUT / "sleeve_anatomy_curve.pkl"); ret.index = pd.to_datetime(ret.index); ret = ret.reindex(days).fillna(0.0)
    td = pd.Series(days.dayofweek != 0, index=days)
    bh = pd.read_pickle(OUT / "strategy_comparison_bh.pkl").dropna(); bh.index = pd.to_datetime(bh.index)
    px_all = C.load_close("yahoo/QQQ.parquet"); p_all = pct_w(px_all.pct_change().rolling(21).std(), 504)
    port = lambda sl: T.build(bh, sl.reindex(bh.index).fillna(0.0), (1.0, 1.0), (("lin", .5, 1.5), None, 1.0), 1.0, px_all, p_all)[0]
    base_p = port(ret); cfgs = dict(S2["cfgs"])
    for mname in ("A+N·ridge", "A+B+E+N·ridge"):
        for key in ("rv", "low"):
            yh = pred[(mname, key)]; yh = -np.log(yh / (1 - yh)) if key == "low" else yh
            for kind, par in (("skip", .2), ("skip", .3), ("cont", .5), ("cont", 1.0)):
                cfgs[f"R3 {mname}[{key}] {kind}{par:g}"] = L.causal_s(yh, kind, par)
    tr_idx = np.where(td.to_numpy() & (days.year >= 2021))[0]; rg = np.random.default_rng(1); perms = [rg.permutation(tr_idx) for _ in range(L.NPERM)]
    base_sr = L.sr(ret.iloc[tr_idx]); rv_ = ret.to_numpy(); rows, nulls = [], {}
    for nm, s in cfgs.items():
        sv = s.reindex(days).fillna(1.0).to_numpy(); obs = L.sr(sv[tr_idx] * rv_[tr_idx]) - base_sr
        nulls[nm] = np.array([L.sr(sv[pi] * rv_[tr_idx]) for pi in perms]) - base_sr
        sser = pd.Series(sv, index=days); xs, xp = ret * sser, port(ret * sser)
        row = dict(配置=nm, 跳过比例=float((sv[tr_idx] == 0).mean()), _obs=obs)
        for k in ("dev-OOS", "test", "全OOS"):
            row[f"Sleeve ΔSR {k}"] = L.sr(L.period(xs[td], k)) - L.sr(L.period(ret[td], k)); row[f"组合 ΔSR {k}"] = L.sr(L.period(xp, k)) - L.sr(L.period(base_p, k))
        row["置换p(单)"] = float(np.mean(nulls[nm] >= obs)); rows.append(row)
    fam = np.max(np.vstack(list(nulls.values())), axis=0)
    R3 = pd.DataFrame(rows); R3[f"置换p(族内max,{len(cfgs)}配置)"] = [float(np.mean(fam >= o)) for o in R3._obs]; R3 = R3.drop(columns="_obs")
    print(f"\n3) Sleeve 日级缩放(三轮共 {len(cfgs)} 个配置;显示第三轮 + 全部配置中 dev/test 同号且 ΔSR(全OOS) 最大的 3 个)")
    ok = R3[(R3["Sleeve ΔSR dev-OOS"] > 0) & (R3["Sleeve ΔSR test"] > 0)].nlargest(3, "Sleeve ΔSR 全OOS")
    print(pd.concat([R3[R3.配置.str.startswith("R3")], ok]).drop_duplicates("配置").round(3).to_string(index=False))
    R3.to_csv(TAB / "lowvol_r3_economic.csv", index=False)
    pd.to_pickle(dict(pred=pred, X=X, Y=Y, thr=thr_s, cfgs=cfgs), OUT / "lowvol_round3.pkl")


if __name__ == "__main__":
    main()
