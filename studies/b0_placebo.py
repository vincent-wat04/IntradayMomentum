"""B0 重训练安慰剂:DUR 特征块是否真的比"同分布的假 DUR"多提供信息?

做法:同年内把交易日随机配对成"捐赠日",每笔取捐赠日的 DUR 行(按笔序号取模,保留日内聚集结构),
其余列(F1)不动;用与 b0_incremental 完全相同的 walk-forward 流程重训 → skip20% 的 Sleeve ΔSR。
重复 NPLC 次得到零分布;与真 DUR(lgbm 0.271 / ridge 0.251)和 F1 单独(lgbm 0.106 / ridge ?)比较。
同时给一个"假特征=F1 列自身的日置换"不需要——只检验增量块。
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
import b0_eval as EV
import trade_filter_study as TF
from b0_econ import causal_rank
from b0_validate import Ctx, TAB, DATA


def donor_permute(ALL, TR, cols, rng, okrow):
    out = ALL[cols].to_numpy().copy(); yrs = TR.day.dt.year.to_numpy(); days = TR.day.to_numpy()
    res = out.copy()
    for y in np.unique(yrs):
        iy = np.where((yrs == y) & okrow)[0]; ud = pd.unique(days[iy]); pos = {d: iy[days[iy] == d] for d in ud}
        don = rng.permutation(len(ud))
        for i, d in enumerate(ud):
            tgt = pos[d]; src = pos[ud[don[i]]]
            res[tgt] = out[src[np.arange(len(tgt)) % len(src)]]
    return pd.DataFrame(res, columns=cols, index=ALL.index)


def wf(ALL, TR, cols, mdl, okrow):
    yrs = TR.day.dt.year.to_numpy(); y = TR.y.to_numpy(); pr = np.full(len(TR), np.nan)
    for yr in range(2021, 2027):
        tr = (yrs < yr) & okrow; te = (yrs == yr) & okrow
        pr[te] = TF.fit_predict(ALL.loc[tr, cols], y[tr], ALL.loc[te, cols], mdl)
    return pr


def main():
    ctx = Ctx(); TR = ctx.TR
    ALL = pd.read_pickle(DATA / "_B0_features2.pkl")
    cols_f1 = TF.FEATS_F1; cols_dur = EV.DUR
    okrow = ALL[sorted(set(cols_f1 + cols_dur + EV.MIC + EV.BRP + EV.VOLO))].notna().all(1).to_numpy()
    NPLC = int(os.environ.get("NPLC", "60")); rng = np.random.default_rng(777); res = []
    # 真值参照
    for mdl in ("ridge", "lgbm"):
        for nm, cols in (("F1", cols_f1), ("F1+DUR", cols_f1 + cols_dur)):
            pr = ctx.preds[(nm, mdl)]; rk = causal_rank(pr, ctx.day_arr); c = ctx.rule_curve(pr, rk, "skip", 0.20)
            res.append(dict(kind="real", model=mdl, set=nm, dev=ctx.dsr(c, "dev"), test=ctx.dsr(c, "test"), full=ctx.dsr(c, "full"),
                            ic=spearmanr(pr[np.isfinite(pr)], TR.y.to_numpy()[np.isfinite(pr)])[0]))
    t0 = time.time()
    for i in range(NPLC):
        fake = ALL.copy(); fake[cols_dur] = donor_permute(ALL, TR, cols_dur, rng, okrow)
        for mdl in ("ridge", "lgbm"):
            pr = wf(fake, TR, cols_f1 + cols_dur, mdl, okrow); rk = causal_rank(pr, ctx.day_arr); c = ctx.rule_curve(pr, rk, "skip", 0.20)
            f = np.isfinite(pr)
            res.append(dict(kind="placebo", model=mdl, set="F1+fakeDUR", dev=ctx.dsr(c, "dev"), test=ctx.dsr(c, "test"), full=ctx.dsr(c, "full"),
                            ic=spearmanr(pr[f], TR.y.to_numpy()[f])[0]))
        if i == 1: print(f"  每轮 {(time.time()-t0)/2:.1f}s,预计 {(time.time()-t0)/2*NPLC/60:.1f} 分钟", flush=True)
    R = pd.DataFrame(res); R.to_csv(TAB / "b0_placebo.csv", index=False)
    pd.set_option("display.width", 200)
    print(R[R.kind == "real"].round(3).to_string(index=False))
    for mdl in ("ridge", "lgbm"):
        pl = R[(R.kind == "placebo") & (R.model == mdl)]; real = R[(R.kind == "real") & (R.model == mdl) & (R.set == "F1+DUR")].iloc[0]
        f1 = R[(R.kind == "real") & (R.model == mdl) & (R.set == "F1")].iloc[0]
        print(f"\n[{mdl}] 假DUR n={len(pl)}: 全OOS ΔSR median {pl.full.median():.3f}, 95% {pl.full.quantile(.95):.3f}, max {pl.full.max():.3f};  真 DUR {real.full:.3f};  F1单独 {f1.full:.3f}")
        print(f"        p(假DUR ≥ 真DUR) 全={((pl.full >= real.full).sum()+1)/(len(pl)+1):.4f}  dev={((pl.dev >= real.dev).sum()+1)/(len(pl)+1):.4f}  test={((pl.test >= real.test).sum()+1)/(len(pl)+1):.4f}")
        print(f"        IC: 假 median {pl.ic.median():.4f} 95% {pl.ic.quantile(.95):.4f};  真 {real.ic:.4f};  F1 {f1.ic:.4f}")


if __name__ == "__main__":
    main()
