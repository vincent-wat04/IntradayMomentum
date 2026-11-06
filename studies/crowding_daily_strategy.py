"""日更拥挤度 → 策略(docs/PROPOSAL_crowding_daily.md §2,预注册):H1–H6,相对 v4 ①。
python3 crowding_daily_strategy.py
"""
from __future__ import annotations

import itertools
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crowding_daily as CDY
import disp_pred_study as DP
import liquidity_inverse_study as LI
import r2_rules_study as RR
import r3_study as R3
import round3_common as C
import sizing_models_study as M
import sizing_sigma_feed as SF
import sota_stack_study as S
import sota_unified_stack as SU
import tearsheet as TS
import unified_sizing_posthoc as P
from etft.intraday.bars import load_bars

DELTAS = (0.0, 0.25, 0.5, 0.75, 1.0); MULTS = (1.0, 1.25, 1.5)


def recursive(cands, days, sel, start="2021-01-01"):
    """cands: {参数: 组合日收益数组}。每月按 2019-01 至上月末样本日的扩展窗口夏普选参数;并列时取更接近"不改动"的参数(按 key 排序的最后一个)。"""
    days = pd.DatetimeIndex(days); keys = list(cands); base = cands[keys[-1]]; out = base.copy(); mon = days.to_period("M"); picks = {}
    for m in sorted(set(mon[days >= start])):
        h = sel & (days < m.start_time) & (days >= "2019-01-01")
        sr = {k: cands[k][h].mean() / cands[k][h].std() for k in keys}; best = max(keys, key=lambda k: (round(sr[k], 10), keys.index(k)))
        picks[str(m)] = best; cur = mon == m; out[cur] = cands[best][cur]
    return out, picks


def main():
    t0 = time.time(); pd.set_option("display.width", 280, "display.max_rows", 200, "display.float_format", lambda v: f"{v:8.3f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    I = SF.build_inputs(b); X = I["X"].copy(); bh = I["bh"]; keep = I["keep"]
    dn = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days); oo = om.to_numpy()
    X["dur_rate"] = M.family_features(b, I["X"])["dur_rate"]; F4 = P.CLEAN + ["dur_rate"]
    _, CD, _ = CDY.build()
    cd = {k: v.reindex(days) for k, v in CD.items()}
    for k, v in cd.items(): assert v.loc[sel, ["C", "state"]].notna().all().all(), f"{k} 在样本日缺失"
    X["c_tsm_d"], X["c_mom_d"] = cd["TSM"].C, cd["MOM"].C
    for k, v in cd.items():
        st = v.state.to_numpy()[oo]; print(f"[{k}] OOS 高拥挤日占比 {(st == 2).mean():.1%};状态日均切换次数 {np.mean(st[1:] != st[:-1]):.3f}(≈ 每 {1/max(np.mean(st[1:] != st[:-1]),1e-9):.1f} 天切换一次)")
    PHb, PHf = LI.run_phases_fills(b), SU.fomc_late_PH(b); _, S3m, _ = RR.band_multipliers(b)
    PHbE, PHfE = R3.run_PH_entry(b, S3m), R3.run_PH_entry(b, S3m, fomc_only=True); Lv4 = R3.legs_k(PHbE, PHfE, b, keep)
    R = {"v4": R3.assemble(R3.sizing(X, F4, Lv4, I, sel, days, "mv", "lin"), Lv4, I, om)}
    for nm, f in (("H1 +c_tsm_d", "c_tsm_d"), ("H2 +c_mom_d", "c_mom_d")):
        R[nm] = R3.assemble(R3.sizing(X, F4 + [f], Lv4, I, sel, days, "mv", "lin"), Lv4, I, om)
    print(f"[v4 / H1 / H2] ({time.time()-t0:.0f}s);v4 组合 OOS SR {(R['v4']['port'].to_numpy()[oo].mean()/R['v4']['port'].to_numpy()[oo].std()*np.sqrt(252)):.3f}", flush=True)
    px = C.load_close("yahoo/QQQ.parquet"); W = S.bh_weights(px, px.pct_change().fillna(0.0), {}, pd.Series(1.0, index=px.index))
    w_rs = W[("RV21[0.5,1.5]", "SMA200|e=0.5")].reindex(bh.index).ffill().fillna(1.0).reindex(days).fillna(1.0).to_numpy(); bhv = bh.to_numpy()
    legw = lambda w: w * bhv - 2e-4 * np.abs(np.diff(np.r_[w[0], w])); leg1 = I["leg1"].to_numpy(); assert np.allclose(legw(w_rs)[sel], leg1[sel], atol=1e-12)
    sl4 = R["v4"]["sleeve"].to_numpy(); hiT = (cd["TSM"].state.to_numpy() == 2) & oo; hiK = (cd["MKT"].state.to_numpy() == 2) & oo
    picks = {}; wts = {}
    for nm, hi in (("H3 核心仓 × δ | C_TSM_d 高", hiT), ("H4 核心仓 × δ | C_MKT_d 高", hiK)):
        cands = {d: legw(np.where(hi, d * w_rs, w_rs)) + sl4 for d in DELTAS}
        port, pk = recursive(cands, days, sel); picks[nm] = pk; R[nm] = dict(port=pd.Series(port, index=days), sleeve=R["v4"]["sleeve"])
        dsel = pd.Series(pk); mon = days.to_period("M"); dd = np.array([dsel.get(str(m_), 1.0) for m_ in mon]); wts[nm] = np.where(hi, dd * w_rs, w_rs)
    cands = {m: leg1 + np.where(hiT, m * sl4, sl4) for m in sorted(MULTS, reverse=True)}
    port, pk = recursive(cands, days, sel); picks["H5 Sleeve × m | C_TSM_d 高"] = pk; R["H5 Sleeve × m | C_TSM_d 高"] = dict(port=pd.Series(port, index=days), sleeve=pd.Series(port - leg1, index=days))
    grid = [(d, m) for m in sorted(MULTS, reverse=True) for d in DELTAS]
    cands = {(d, m): legw(np.where(hiT, d * w_rs, w_rs)) + np.where(hiT, m * sl4, sl4) for d, m in grid}
    port, pk = recursive(cands, days, sel); picks["H6 H3 + H5"] = pk; R["H6 H3 + H5"] = dict(port=pd.Series(port, index=days), sleeve=pd.Series(np.nan, index=days))
    for nm, pk in picks.items(): print(f"  [{nm}] 递归参数分布 {pd.Series([str(v) for v in pk.values()]).value_counts().to_dict()}")
    for nm, w in wts.items():
        print(f"  [{nm}] 核心仓年化换手 {np.abs(np.diff(w[oo])).sum() / (oo.sum() / 252):.2f}(v4 {np.abs(np.diff(w_rs[oo])).sum() / (oo.sum() / 252):.2f});平均权重 {w[oo].mean():.3f}(v4 {w_rs[oo].mean():.3f})")
    # ---- 评估
    oi = np.where(oo)[0]; dev = yrs[oi] <= 2022; BI = M.block_idx(len(oi), 10, 2000, 0); srv = lambda v: v.mean() / v.std() * np.sqrt(252)
    ref = R["v4"]["port"].to_numpy()[oi]; refb = np.array([srv(ref[q]) for q in BI]); rows = []; pans = {}
    for nm, r_ in R.items():
        x_ = r_["port"].to_numpy()[oi]; db = np.array([srv(x_[q]) for q in BI]) - refb
        row = dict(方案=nm, 组合SR_OOS=srv(x_), ΔSR=srv(x_) - srv(ref), lo=np.percentile(db, 2.5), hi=np.percentile(db, 97.5), dev=srv(x_[dev]) - srv(ref[dev]), test=srv(x_[~dev]) - srv(ref[~dev]),
                   test期SR=TS.metrics(r_["port"].loc["2023":"2026"])["Sharpe（绝对）"], test期MDD=TS.metrics(r_["port"].loc["2023":"2026"])["最大回撤 MDD"], OOS_MDD=TS.metrics(pd.Series(x_))["最大回撤 MDD"],
                   **{str(y): srv(x_[yrs[oi] == y]) - srv(ref[yrs[oi] == y]) for y in range(2021, 2027)})
        row["优于v4"] = bool(nm != "v4" and row["ΔSR"] > 0 and row["dev"] > 0 and row["test"] > 0 and row["lo"] > -0.05); rows.append(row); pans[nm] = SF.panel(pd.Series(x_))
    T = pd.DataFrame(rows); print("\n=== 组合层 OOS 2021–26(ΔSR 相对 v4)==="); print(T.to_string(index=False))
    print("\n=== 分布面板(组合,OOS 2021–26)==="); print(pd.DataFrame(pans).T.to_string())
    T.to_csv(M.TAB / "crowding_daily_configs.csv", index=False); pd.to_pickle({k: v["port"] for k, v in R.items()}, SF.OUT / "crowding_daily_ports.pkl"); print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
