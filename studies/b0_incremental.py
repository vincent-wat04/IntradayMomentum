"""B0 增量检验:新特征族相对 F1 的 walk-forward 增量 IC(描述性)。
特征集: F1 | F1+DUR | F1+MIC | F1+BRP | F1+VOLO | F1+STATE | F1+DUR+MIC+BRP+VOLO
模型:   ridge(α=100,标准化)、LightGBM(depth 3, 200 棵, lr 0.03, min_child 500, subsample 0.7, colsample 0.8) —— 与 trade_filter 冻结设定相同
OOS:    2021–2026,每年用此前所有年份重估;同一天的多笔(多相位)落在同一年,不跨训练/测试。
指标:   OOS IC;相对 F1 的 ΔIC(配对,按交易日聚类自助);分方向 IC。
"""
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b0_entry_features as B0
import b0_eval as EV
import trade_filter_study as TF

DATA, TAB = B0.DATA, B0.TAB


def main():
    TR = pd.read_pickle(DATA / "_TR.pkl").sort_values(["day", "phase", "k_dec"]).reset_index(drop=True)
    ALL = pd.read_pickle(DATA / "_B0_features2.pkl")
    yrs = TR.day.dt.year.to_numpy(); y = TR.y.to_numpy(); side = TR.side.to_numpy(); day_id = TR.day.dt.strftime("%Y-%m-%d").to_numpy()
    SETS = {"F1": TF.FEATS_F1, "F1+STATE": TF.FEATS_F1 + EV.STATE, "F1+DUR": TF.FEATS_F1 + EV.DUR, "F1+MIC": TF.FEATS_F1 + EV.MIC,
            "F1+BRP": TF.FEATS_F1 + EV.BRP, "F1+VOLO": TF.FEATS_F1 + EV.VOLO,
            "F1+ALL": TF.FEATS_F1 + EV.DUR + EV.MIC + EV.BRP + EV.VOLO}
    allcols = sorted(set(c for v in SETS.values() for c in v))
    okrow = ALL[allcols].notna().all(1).to_numpy()
    print(f"所有集合共同可用的行 {okrow.sum()}/{len(ALL)}(以此定义 OOS,保证各集合可比)")
    preds = {}
    for mdl in ("ridge", "lgbm"):
        for nm, cols in SETS.items():
            pr = np.full(len(TR), np.nan)
            for yr in range(2021, 2027):
                tr = (yrs < yr) & okrow; te = (yrs == yr) & okrow
                if tr.sum() < 3000 or te.sum() == 0: continue
                pr[te] = TF.fit_predict(ALL.loc[tr, cols], y[tr], ALL.loc[te, cols], mdl)
            preds[(nm, mdl)] = pr
    oos = np.isfinite(preds[("F1", "ridge")]) & (yrs >= 2021)
    ud, inv = np.unique(day_id[oos], return_inverse=True); order = np.argsort(inv, kind="stable")
    st = np.searchsorted(inv[order], np.arange(len(ud))); en = np.append(st[1:], len(order))

    def boot(fn, nb=500, seed=0):
        rg = np.random.default_rng(seed); out = []
        for _ in range(nb):
            pick = rg.integers(0, len(ud), len(ud)); out.append(fn(np.concatenate([order[st[p]:en[p]] for p in pick])))
        return np.percentile(out, [2.5, 97.5])

    yo = y[oos]; so = side[oos]; yro = yrs[oos]
    rows = []
    for mdl in ("ridge", "lgbm"):
        base = preds[("F1", mdl)][oos]
        for nm in SETS:
            p = preds[(nm, mdl)][oos]; ic = spearmanr(p, yo)[0]
            lo, hi = boot(lambda ix: spearmanr(p[ix], yo[ix])[0])
            if nm != "F1":
                dlo, dhi = boot(lambda ix: spearmanr(p[ix], yo[ix])[0] - spearmanr(base[ix], yo[ix])[0])
                dic = ic - spearmanr(base, yo)[0]
            else:
                dic = dlo = dhi = np.nan
            rows.append(dict(model=mdl, set=nm, ic=ic, ic_lo=lo, ic_hi=hi, d_ic_vs_F1=dic, d_lo=dlo, d_hi=dhi,
                             ic_long=spearmanr(p[so > 0], yo[so > 0])[0], ic_short=spearmanr(p[so < 0], yo[so < 0])[0],
                             ic_dev=spearmanr(p[yro <= 2022], yo[yro <= 2022])[0], ic_test=spearmanr(p[yro >= 2023], yo[yro >= 2023])[0]))
    R = pd.DataFrame(rows); R.to_csv(TAB / "b0_incremental_ic.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"\nOOS 2021–2026 共 {oos.sum()} 笔(按交易日聚类自助 95%)\n")
    print(R.round(3).to_string(index=False))
    pd.to_pickle({k: v for k, v in preds.items()}, DATA / "_B0_preds.pkl")


if __name__ == "__main__":
    main()
