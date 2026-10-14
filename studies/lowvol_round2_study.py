"""lowvol_avoidance 第二轮:加入确定性事件日历 E 组 —— 见 docs/PROPOSAL_lowvol_avoidance.md「第二轮」。
复用第一轮的构件(lowvol_avoidance_study),不改动它。"""
from __future__ import annotations

import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lowvol_avoidance_study as L
import trend_rv_sleeve_study as T
import round3_common as C
from event_calendar import event_frame
from etft.intraday.bars import load_bars
from sens_rv_weights import pct_w

OUT, TAB = L.OUT, L.TAB


def main():
    b = load_bars("QQQ"); days = b.days
    X, Y, A, B = L.build(b)
    E = event_frame(days, b.last_idx.to_numpy()); E_COLS = list(E.columns)
    X = pd.concat([X, E], axis=1)
    print("事件日历计数:", {c: int(E[c].sum()) for c in E_COLS})
    sets = {
        "A·ridge": (A, "ridge"), "A+B·ridge": (A + B, "ridge"),
        "E·ridge": (E_COLS, "ridge"), "A+E·ridge": (A + E_COLS, "ridge"), "A+B+E·ridge": (A + B + E_COLS, "ridge"),
        "A+E·lgbm": (A + E_COLS, "lgbm"), "A+B+E·lgbm": (A + B + E_COLS, "lgbm"),
    }
    pred, base_rv, thr_s = L.walk_forward(X, Y, sets)
    oos = X.index[(X.index.year >= 2021) & X[A + B + E_COLS].notna().all(1) & Y.notna().all(1)]
    ylow = (Y.loc[oos, "disp"] <= thr_s.loc[oos]).astype(int).to_numpy()
    sse_b = ((Y.loc[oos, "y_rv"] - base_rv.loc[oos]) ** 2).to_numpy()
    pd.set_option("display.width", 220)

    # ---- 1) 预测力
    rows = []
    for (nm, key), p in pred.items():
        p = p.loc[oos]
        if key == "rv":
            e = ((Y.loc[oos, "y_rv"] - p) ** 2).to_numpy()
            pt, lo, hi = L.month_boot(oos, lambda ii, e=e: 1 - e[ii].sum() / sse_b[ii].sum())
            rows.append(dict(模型=nm, 指标="OOS R² (log RV)", 值=pt, lo=lo, hi=hi))
        else:
            pp = p.to_numpy(); pt, lo, hi = L.month_boot(oos, lambda ii, pp=pp: L.auc(ylow[ii], pp[ii]))
            rows.append(dict(模型=nm, 指标="AUC(低波标签)", 值=pt, lo=lo, hi=hi))
    R1 = pd.DataFrame(rows); print("\n" + "=" * 100 + "\n1) 预测力(OOS 2021–2026)\n" + "=" * 100); print(R1.round(4).to_string(index=False))
    inc = []
    for n1, n0 in (("A+E·ridge", "A·ridge"), ("A+B+E·ridge", "A+B·ridge"), ("A+E·lgbm", "A·ridge")):
        e1 = ((Y.loc[oos, "y_rv"] - pred[(n1, "rv")].loc[oos]) ** 2).to_numpy(); e0 = ((Y.loc[oos, "y_rv"] - pred[(n0, "rv")].loc[oos]) ** 2).to_numpy()
        pt, lo, hi = L.month_boot(oos, lambda ii: (e0[ii].sum() - e1[ii].sum()) / sse_b[ii].sum())
        inc.append(dict(比较=f"{n1} − {n0}", 指标="ΔR²", 值=pt, lo=lo, hi=hi))
        p1, p0 = pred[(n1, "low")].loc[oos].to_numpy(), pred[(n0, "low")].loc[oos].to_numpy()
        pt, lo, hi = L.month_boot(oos, lambda ii: L.auc(ylow[ii], p1[ii]) - L.auc(ylow[ii], p0[ii]))
        inc.append(dict(比较=f"{n1} − {n0}", 指标="ΔAUC", 值=pt, lo=lo, hi=hi))
    R1i = pd.DataFrame(inc); print("\n增量:"); print(R1i.round(4).to_string(index=False))
    R1.to_csv(TAB / "lowvol_r2_prediction.csv", index=False); R1i.to_csv(TAB / "lowvol_r2_incremental.csv", index=False)

    # ---- 2) 每个事件的残差(相对 A 模型)——可解释性
    res = (Y.y_rv - pred[("A·ridge", "rv")]).loc[oos]
    ev = []
    for c in E_COLS:
        m = E.loc[oos, c] == 1
        if m.sum() < 5: continue
        r = res[m]; ev.append(dict(事件=c, 天数=int(m.sum()), 残差均值=r.mean(), t=r.mean() / (r.std() / np.sqrt(len(r))),
                                   位移_低波占比=float(np.mean(ylow[m.to_numpy()])), 基准低波占比=float(ylow.mean())))
    EV = pd.DataFrame(ev); print("\n" + "=" * 100 + "\n2) 事件日的 log RV 残差(实际 − A 模型预测;>0 = A 模型低估)\n" + "=" * 100); print(EV.round(3).to_string(index=False))
    EV.to_csv(TAB / "lowvol_r2_events.csv", index=False)

    # ---- 3) 变现(并入第一轮全部配置作族内置换)
    S1 = pd.read_pickle(OUT / "lowvol_study.pkl")
    ret = pd.read_pickle(OUT / "sleeve_anatomy_curve.pkl"); ret.index = pd.to_datetime(ret.index); ret = ret.reindex(days).fillna(0.0)
    td = pd.Series(days.dayofweek != 0, index=days)
    bh = pd.read_pickle(OUT / "strategy_comparison_bh.pkl").dropna(); bh.index = pd.to_datetime(bh.index)
    px_all = C.load_close("yahoo/QQQ.parquet"); p_all = pct_w(px_all.pct_change().rolling(21).std(), 504)
    spec = (("lin", 0.5, 1.5), None, 1.0)
    port = lambda sl: T.build(bh, sl.reindex(bh.index).fillna(0.0), (1.0, 1.0), spec, 1.0, px_all, p_all)[0]
    base_p = port(ret)
    cfgs = dict(S1["cfgs"])                                     # 第一轮 18 个
    for mname in ("A+E·ridge", "A+B+E·ridge"):
        for key in ("rv", "low"):
            yh = pred[(mname, key)]; yh = -np.log(yh / (1 - yh)) if key == "low" else yh
            for kind, par in (("skip", 0.2), ("skip", 0.3), ("cont", 0.5), ("cont", 1.0)):
                cfgs[f"R2 {mname}[{key}] {kind}{par:g}"] = L.causal_s(yh, kind, par)
    tr_idx = np.where(td.to_numpy() & (days.year >= 2021))[0]
    rg = np.random.default_rng(1); perms = [rg.permutation(tr_idx) for _ in range(L.NPERM)]
    base_sr = L.sr(ret.iloc[tr_idx]); rows, nulls = [], {}
    rv_ = ret.to_numpy()
    for nm, s in cfgs.items():
        sv = s.reindex(days).fillna(1.0).to_numpy()
        obs = L.sr(sv[tr_idx] * rv_[tr_idx]) - base_sr
        nulls[nm] = np.array([L.sr(sv[pi] * rv_[tr_idx]) for pi in perms]) - base_sr
        xs, xp = ret * pd.Series(sv, index=days), port(ret * pd.Series(sv, index=days))
        row = dict(配置=nm, 平均s=float(sv[tr_idx].mean()), 跳过比例=float((sv[tr_idx] == 0).mean()), _obs=obs)
        for k in ("dev-OOS", "test", "全OOS"):
            row[f"Sleeve ΔSR {k}"] = L.sr(L.period(xs[td], k)) - L.sr(L.period(ret[td], k))
            row[f"组合 ΔSR {k}"] = L.sr(L.period(xp, k)) - L.sr(L.period(base_p, k))
        row["置换p(单)"] = float(np.mean(nulls[nm] >= obs)); rows.append(row)
    fam = np.max(np.vstack(list(nulls.values())), axis=0)
    R3 = pd.DataFrame(rows); R3["置换p(族内max,%d配置)" % len(cfgs)] = [float(np.mean(fam >= o)) for o in R3._obs]; R3 = R3.drop(columns="_obs")
    print("\n" + "=" * 150 + f"\n3) Sleeve 日级缩放(第一轮 + 第二轮共 {len(cfgs)} 个配置;仅显示第二轮及第一轮 ΔSR 最大者)\n" + "=" * 150)
    show = pd.concat([R3[R3.配置.str.startswith("R2")], R3[~R3.配置.str.startswith("R2")].nlargest(2, "Sleeve ΔSR 全OOS")])
    print(show.round(3).to_string(index=False)); R3.to_csv(TAB / "lowvol_r2_economic.csv", index=False)
    pd.to_pickle(dict(pred=pred, X=X, Y=Y, thr=thr_s, cfgs=cfgs, E=E), OUT / "lowvol_round2.pkl")


if __name__ == "__main__":
    main()
