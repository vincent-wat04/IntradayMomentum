"""日频信号去噪、核心仓回撤控制、Sleeve : 核心仓配比(docs/PROPOSAL_noise_core_dd.md,预注册)。
python3 noise_core_dd.py
"""
from __future__ import annotations

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
import crowding_daily_strategy as CS
import crowding_index as CX
import disp_pred_study as DP
import r2_rules_study as RR
import r3_study as R3
import round3_common as C
import sizing_models_study as M
import sizing_sigma_feed as SF
import trend_rv_sleeve_study as T
import unified_sizing_posthoc as P
from etft.intraday.bars import load_bars
from sens_rv_weights import pct_w, w_linear

DELTAS = (0.0, 0.25, 0.5, 0.75, 1.0); MS = (1.0, 1.25, 1.5, 1.75, 2.0); RHOS = (1.0, 1.25, 1.5, 2.0, 3.0)
CRISES = (("2000-03 至 2002-10", "2000-03-01", "2002-10-31"), ("2007-10 至 2009-03", "2007-10-01", "2009-03-31"), ("2020-02 至 2020-04", "2020-02-19", "2020-04-30"), ("2022", "2022-01-01", "2022-12-31"))


def core_weights(px):
    """全日历(QQQ 日线)上的各核心仓权重(已 shift(1),t 日收盘前可得)。"""
    r = px.pct_change(); rv = r.rolling(21).std(); p = pct_w(rv, 504)
    w_rv = w_linear(p, 0.5, 1.5).shift(1).fillna(1.0)
    D_slow = T.trend_down(px, "SMA", 200).astype(float); D_fast = (px / px.shift(21) - 1 < 0).astype(float).where(px.shift(21).notna(), 0.0)
    T_base = (1 - 0.5 * D_slow).shift(1).fillna(1.0); T_blend = (1 - 0.5 * (0.5 * D_slow + 0.5 * D_fast)).shift(1).fillna(1.0)
    smed = rv.rolling(504, min_periods=252).median()
    w_cvt = pd.Series(np.where((p > 0.2) & (p < 0.8), 1.0, np.clip(smed / rv, 0.5, 1.5)), index=px.index).where(p.notna(), 1.0).shift(1).fillna(1.0)
    W = {"base": w_rv * T_base, "K1": w_rv * T_blend, "K4": w_cvt * T_base}
    def band(w, b=0.10):
        v = w.to_numpy(); out = np.empty_like(v); cur = v[0]
        for i, x in enumerate(v):
            if abs(x - cur) >= b: cur = x
            out[i] = cur
        return pd.Series(out, index=w.index)
    W["K5"] = band(W["base"]); W["K6"] = band(W["K1"])
    return W, r


def leg(w, r):
    return w * r - 2e-4 * w.diff().abs().fillna(0.0)


def hysteresis(x, hist=1260, q_on=2 / 3, q_off=0.5):
    v = x.to_numpy(); out = np.zeros(len(v), bool); on = False
    for t in range(len(v)):
        h = v[max(0, t - hist):t]; h = h[np.isfinite(h)]
        if len(h) < hist // 2 or not np.isfinite(v[t]): out[t] = on; continue
        if not on and v[t] >= np.quantile(h, q_on): on = True
        elif on and v[t] < np.quantile(h, q_off): on = False
        out[t] = on
    return pd.Series(out, index=x.index)


def fast_in_slow_out(c_daily, monthly_hi, hist=1260):
    v = c_daily.to_numpy(); mh = monthly_hi.to_numpy(); out = np.zeros(len(v), bool); on = False
    for t in range(len(v)):
        h = v[max(0, t - hist):t]; h = h[np.isfinite(h)]
        if len(h) < hist // 2 or not np.isfinite(v[t]): on = bool(mh[t]); out[t] = on; continue
        if not on and (mh[t] or v[t] >= np.quantile(h, 0.9)): on = True
        elif on and (not mh[t]) and v[t] < np.quantile(h, 0.5): on = False
        out[t] = on
    return pd.Series(out, index=c_daily.index)


def stats(x):
    x = pd.Series(x).dropna(); eq = (1 + x).cumprod(); dd = eq / eq.cummax() - 1; under = (dd < 0).astype(int)
    run = under.groupby((under != under.shift()).cumsum()).cumsum().max()
    cagr = eq.iloc[-1] ** (252 / len(x)) - 1
    return dict(Sharpe=x.mean() / x.std() * np.sqrt(252), CAGR=cagr, 波动=x.std() * np.sqrt(252), MDD=dd.min(), Calmar=cagr / abs(dd.min()), 最长水下日=int(run))


def main():
    t0 = time.time(); pd.set_option("display.width", 280, "display.max_rows", 200, "display.float_format", lambda v: f"{v:8.3f}")
    px = C.load_close("yahoo/QQQ.parquet"); W, r = core_weights(px)
    # ================= 1. 核心仓长样本(无参数规则)=================
    rows = []; crisis = []
    for nm, w in (("QQQ 买入持有", None), ("基线 RV21×SMA200", W["base"]), ("K1 快慢混合", W["K1"]), ("K4 条件波动目标", W["K4"]), ("K5 不交易带", W["K5"]), ("K6 K1+K5", W["K6"])):
        x = r if w is None else leg(w, r); x = x.loc["2000-01-03":]
        st = stats(x); wv = (pd.Series(1.0, index=x.index) if w is None else w.loc[x.index])
        rows.append(dict(核心仓=nm, **st, 平均权重=wv.mean(), 年化换手=wv.diff().abs().sum() / (len(wv) / 252)))
        crisis.append(dict(核心仓=nm, **{c[0]: float((1 + x.loc[c[1]:c[2]]).prod() - 1) * 100 for c in CRISES}))
    print("=== 1. 核心仓本身,QQQ 日线 2000-01 至 2026-10(成本 2bp/单位换手)==="); print(pd.DataFrame(rows).to_string(index=False))
    print("\n危机区间收益(%):"); print(pd.DataFrame(crisis).to_string(index=False))
    rows = []
    for a, b_ in (("2000", "2009"), ("2010", "2018"), ("2019", "2026")):
        rows.append(dict(期间=f"{a}–{b_}", **{nm: stats(leg(W[k], r).loc[a:b_])["Sharpe"] for nm, k in (("基线", "base"), ("K1", "K1"), ("K4", "K4"), ("K5", "K5"), ("K6", "K6"))},
                         **{nm + " MDD": stats(leg(W[k], r).loc[a:b_])["MDD"] for nm, k in (("基线", "base"), ("K1", "K1"), ("K4", "K4"), ("K6", "K6"))}))
    print("\n分期 Sharpe / MDD:"); print(pd.DataFrame(rows).to_string(index=False))
    # ================= 2. 组合层(OOS 2021–26)=================
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    I = SF.build_inputs(b); X = I["X"].copy(); keep = I["keep"]; bh = I["bh"]
    dn = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days); oo = om.to_numpy()
    X["dur_rate"] = M.family_features(b, I["X"])["dur_rate"]
    _, S3m, _ = RR.band_multipliers(b); Lv4 = R3.legs_k(R3.run_PH_entry(b, S3m), R3.run_PH_entry(b, S3m, fomc_only=True), b, keep)
    v4 = R3.assemble(R3.sizing(X, P.CLEAN + ["dur_rate"], Lv4, I, sel, days, "mv", "lin"), Lv4, I, om)
    sl4 = v4["sleeve"].to_numpy(); leg1 = I["leg1"].to_numpy(); bhv = bh.to_numpy()
    al = lambda w: w.reindex(bh.index).ffill().fillna(1.0).reindex(days).fillna(1.0).to_numpy()
    legd = lambda w: w * bhv - 2e-4 * np.abs(np.diff(np.r_[w[0], w]))
    wb = al(W["base"]); assert np.allclose(legd(wb)[sel], leg1[sel], atol=1e-12), "基线核心仓与 v4 不一致"
    print(f"\n[一致性] 基线核心仓 = v4 的 QQQ 腿 ✓;v4 组合 OOS SR {(lambda v: v.mean()/v.std()*np.sqrt(252))((leg1 + sl4)[oo]):.3f} ({time.time()-t0:.0f}s)", flush=True)
    R = {"v4": leg1 + sl4}; WT = {"v4": wb}; picks = {}
    for k in ("K1", "K4", "K5", "K6"):
        w = al(W[k]); w = np.where(oo, w, wb); R[k] = legd(w) + sl4; WT[k] = w
    _, CD, _ = CDY.build(); cdT = CD["TSM"]
    hiK2 = hysteresis(cdT.C.ewm(halflife=21, min_periods=21).mean()).reindex(days).fillna(False).to_numpy() & oo
    Mo, CI = CX.build(); mhi = (CX.to_daily(CI["TSM"], cdT.index).state == 2)
    hiK3 = fast_in_slow_out(cdT.C, mhi).reindex(days).fillna(False).to_numpy() & oo
    hiH3 = (cdT.state.reindex(days).to_numpy() == 2) & oo
    for nm, hi in (("K2 日更拥挤 EWMA+滞回", hiK2), ("K3 月频+日频 快进慢出", hiK3)):
        cands = {d: legd(np.where(hi, d * wb, wb)) + sl4 for d in DELTAS}; x_, pk = CS.recursive(cands, days, sel); R[nm] = x_; picks[nm] = pk
        mon = days.to_period("M"); dd_ = np.array([pk.get(str(m_), 1.0) for m_ in mon]); WT[nm] = np.where(hi, dd_ * wb, wb)
    for nm, hi in (("(参照)H3 日更原始状态", hiH3), ("K2 日更拥挤 EWMA+滞回", hiK2), ("K3 月频+日频 快进慢出", hiK3)):
        h = hi[oo]; print(f"  [{nm}] OOS 高拥挤占比 {h.mean():.1%};状态切换 {int(np.sum(h[1:] != h[:-1]))} 次(≈ 每 {len(h)/max(np.sum(h[1:] != h[:-1]),1):.0f} 天一次)")
    cands = {m: leg1 + np.where(oo, m * sl4, sl4) for m in MS}; x_, pk = CS.recursive(cands, days, sel); R["S1 Sleeve × m"] = x_; picks["S1 Sleeve × m"] = pk
    v4s = pd.Series(leg1 + sl4, index=days); cands = {}; cscale = {}
    for rho in sorted(RHOS, reverse=True):
        mix = pd.Series(leg1 + rho * sl4, index=days)
        c_ = (v4s.rolling(252, min_periods=126).std() / mix.rolling(252, min_periods=126).std()).shift(1).clip(0.5, 1.5).fillna(1.0).to_numpy()
        c_ = np.where(oo, c_, 1.0); cands[rho] = np.where(oo, c_ * mix.to_numpy(), leg1 + sl4); cscale[rho] = c_
    x_, pk = CS.recursive(cands, days, sel); R["S2 等风险换配比"] = x_; picks["S2 等风险换配比"] = pk
    mon = days.to_period("M"); rr_ = np.array([pk.get(str(m_), 1.0) for m_ in mon]); cc_ = np.array([cscale[rr_[i]][i] for i in range(len(days))])
    for nm, pk in picks.items(): print(f"  [{nm}] 递归参数分布 {pd.Series([str(v) for v in pk.values()]).value_counts().to_dict()}")
    print(f"  [S2] OOS 平均:核心仓乘数 c = {cc_[oo].mean():.3f},Sleeve 乘数 c·ρ = {(cc_ * rr_)[oo].mean():.3f}")
    oi = np.where(oo)[0]; dev = yrs[oi] <= 2022; BI = M.block_idx(len(oi), 10, 2000, 0); srv = lambda v: v.mean() / v.std() * np.sqrt(252)
    ref = R["v4"][oi]; refb = np.array([srv(ref[q]) for q in BI]); rows = []
    for nm, x in R.items():
        xs = x[oi]; db = np.array([srv(xs[q]) for q in BI]) - refb; st = stats(xs); w = WT.get(nm)
        row = dict(方案=nm, 组合SR=srv(xs), ΔSR=srv(xs) - srv(ref), lo=np.percentile(db, 2.5), hi=np.percentile(db, 97.5), dev=srv(xs[dev]) - srv(ref[dev]), test=srv(xs[~dev]) - srv(ref[~dev]),
                   MDD=st["MDD"], Calmar=st["Calmar"], 最长水下日=st["最长水下日"],
                   核心仓年化换手=(np.abs(np.diff(w[oo])).sum() / (oo.sum() / 252)) if w is not None else np.nan, 核心仓平均权重=w[oo].mean() if w is not None else np.nan,
                   **{str(y): srv(xs[yrs[oi] == y]) - srv(ref[yrs[oi] == y]) for y in range(2021, 2027)})
        row["优于v4"] = bool(nm != "v4" and row["ΔSR"] > 0 and row["dev"] > 0 and row["test"] > 0 and row["lo"] > -0.05); rows.append(row)
    Tt = pd.DataFrame(rows); print("\n=== 2. 组合层 OOS 2021–26(ΔSR 相对 v4;核心仓换手/权重只对改核心仓的配置有意义)==="); print(Tt.to_string(index=False))
    Tt.to_csv(M.TAB / "noise_core_dd.csv", index=False); print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
