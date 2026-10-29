"""第二轮 §5:把 r2_factor_zoo 中满足 Sleeve 层判定的 3 格(D1/D2/D3)带入 v3 组合复测。
python3 r2_zoo_in_v3.py
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
import disp_pred_study as DP
import liquidity_inverse_study as LI
import r2_factor_zoo as Z
import r2_rules_study as RR
import sizing_models_study as M
import sizing_sigma_feed as SF
import sota_unified_stack as SU
import tearsheet as TS
import trade_filter_study as TF
from b0_econ import causal_rank
from etft.intraday.bars import load_bars
from event_calendar import FOMC_DECISION

DATA, TAB = Z.DATA, Z.TAB
CELLS = {"D1 lgbm·HFZ·y": ("y", Z.HFZ), "D2 lgbm·XA·y": ("y", Z.XA), "D3 lgbm·HFZ·y_ra": ("y_ra", Z.HFZ)}


def main():
    t0 = time.time(); pd.set_option("display.width", 270, "display.float_format", lambda v: f"{v:8.3f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    TR = pd.read_pickle(DATA / "_TR.pkl").sort_values(["day", "phase", "k_dec"]).reset_index(drop=True)
    IV = pd.read_pickle(DATA / "_IVT_features.pkl"); HF = pd.read_pickle(DATA / "_HFZ_features.pkl")
    ALL = pd.concat([pd.read_pickle(DATA / "_B0_features2.pkl").reset_index(drop=True), IV.drop(columns=["y", "y_ra"]), HF], axis=1)
    okrow = ALL[TF.FEATS_F1 + Z.HFZ + Z.XA].notna().all(1).to_numpy(); ty = TR.day.dt.year.to_numpy()
    preds = {}
    for nm, (tn, fam) in CELLS.items():
        yy = IV[tn].to_numpy(); pr = np.full(len(TR), np.nan); cl = TF.FEATS_F1 + fam
        for yr in range(2021, 2027):
            tr = (ty < yr) & okrow; te = (ty == yr) & okrow; pr[te] = TF.fit_predict(ALL.loc[tr, cl], yy[tr], ALL.loc[te, cl], "lgbm")
        preds[nm] = pr
    print(f"[预测] 完成 ({time.time()-t0:.0f}s)")
    I = SF.build_inputs(b); X = I["X"]
    dn = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days)
    PHb, PHf = LI.run_phases_fills(b), SU.fomc_late_PH(b)
    base = RR.legs_all(PHb, PHf, b, I); R = {"B v3 现行": RR.evaluate("B", base, I, X, sel, days, om)}
    cols = list(b.c.columns); fm = pd.DatetimeIndex(FOMC_DECISION)
    key = pd.MultiIndex.from_arrays([TR.phase, TR.day, TR.k_dec.map(lambda k: cols[k])])
    for nm, pr in preds.items():
        rk = causal_rank(pr, TR.day.to_numpy()); sk_m = np.isfinite(pr) & (rk <= 0.20) & ~TR.day.isin(fm).to_numpy()
        skip = set(key[sk_m]); legs = RR.legs_all(RR.filter_PH(PHb, skip), PHf, b, I)
        R[nm] = RR.evaluate(nm, legs, I, X, sel, days, om); print(f"  [{nm}] 跳过 OOS 笔数 {sk_m[ty >= 2021].mean():.1%}  ({time.time()-t0:.0f}s)", flush=True)
    oi = np.where(om.to_numpy())[0]; dev = yrs[oi] <= 2022; BI = M.block_idx(len(oi), 10, 2000, 0); srv = lambda v: v.mean() / v.std() * np.sqrt(252)
    ref = R["B v3 现行"]["port"].to_numpy()[oi]; refb = np.array([srv(ref[q]) for q in BI]); rows = []; pans = {}
    for nm, r_ in R.items():
        x_ = r_["port"].to_numpy()[oi]; db = np.array([srv(x_[q]) for q in BI]) - refb
        row = dict(方案=nm, 组合SR_OOS=srv(x_), ΔSR=srv(x_) - srv(ref), lo=np.percentile(db, 2.5), hi=np.percentile(db, 97.5),
                   ΔSR_dev=srv(x_[dev]) - srv(ref[dev]), ΔSR_test=srv(x_[~dev]) - srv(ref[~dev]), Sleeve_SR_OOS=srv(r_["sleeve"].to_numpy()[oi]),
                   test期SR=TS.metrics(r_["port"].loc["2023":"2026"])["Sharpe（绝对）"], test期MDD=TS.metrics(r_["port"].loc["2023":"2026"])["最大回撤 MDD"])
        row.update({str(y): srv(x_[yrs[oi] == y]) - srv(ref[yrs[oi] == y]) for y in range(2021, 2027)})
        row["优于B"] = bool(nm != "B v3 现行" and row["ΔSR"] > 0 and row["ΔSR_dev"] > 0 and row["ΔSR_test"] > 0 and row["lo"] > -0.05); rows.append(row)
        pans[nm] = SF.panel(pd.Series(x_))
    Rt = pd.DataFrame(rows); print("\n=== v3 组合内复测(OOS 2021–26,ΔSR 相对 B)==="); print(Rt.to_string(index=False))
    print("\n=== 分布面板 ==="); print(pd.DataFrame(pans).T.to_string())
    Rt.to_csv(TAB / "r2_zoo_in_v3.csv", index=False); print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
