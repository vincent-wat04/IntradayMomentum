"""拥挤度追问(2026-10-29):
  1) 动量崩盘日与拥挤状态的拆分(为什么"高拥挤时均值低"与"高拥挤崩盘日赚很多"同时成立)
  2) 动量崩盘日 / 时序动量崩盘日 / QQQ 最差月份:B&H、核心仓、Sleeve、组合各自的损益
  3) "拥挤 × VIX 上升 → Sleeve 更赚"的机制:同日 ΔVIX 与位移、按状态拆分
  4) 开仓前可得的 VIX 变化:VIX 开盘相对前收的跳空(9:30 已知,Sleeve 最早 10:00 决策)—— 关系检验 + 进入 v4 仓位模型(事后新增 2 个配置:Q1、Q2)
python3 crowding_followup.py
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
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crowding_index as CX
import disp_pred_study as DP
import liquidity_inverse_study as LI
import r2_rules_study as RR
import r3_study as R3
import sizing_models_study as M
import sizing_sigma_feed as SF
import sota_unified_stack as SU
import tearsheet as TS
import unified_sizing_posthoc as P
from etft.intraday.bars import load_bars


def hac(y, Z):
    f = sm.OLS(y, sm.add_constant(Z)).fit(cov_type="HAC", cov_kwds={"maxlags": 5}); return np.asarray(f.params), np.asarray(f.tvalues), f.rsquared


def main():
    t0 = time.time(); pd.set_option("display.width", 260, "display.max_rows", 200, "display.float_format", lambda v: f"{v:8.2f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    I = SF.build_inputs(b); X = I["X"].copy(); bh = I["bh"].to_numpy(); keep = I["keep"]
    dn = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days)
    X["dur_rate"] = M.family_features(b, I["X"])["dur_rate"]; F4 = P.CLEAN + ["dur_rate"]
    Mo, CI = CX.build(); cr = {"MOM": CX.to_daily(CI["MOM"], days, extra_lag=1), "TSM": CX.to_daily(CI["TSM"], days)}
    PHb, PHf = LI.run_phases_fills(b), SU.fomc_late_PH(b); _, S3m, _ = RR.band_multipliers(b)
    PHbE, PHfE = R3.run_PH_entry(b, S3m), R3.run_PH_entry(b, S3m, fomc_only=True)
    Lv3, Lv4 = R3.legs_k(PHb, PHf, b, keep), R3.legs_k(PHbE, PHfE, b, keep)
    v4 = R3.assemble(R3.sizing(X, F4, Lv4, I, sel, days, "mv", "lin"), Lv4, I, om)
    raw = (Lv3["L2"] + Lv3["S2"] - (Lv3["GL"] + Lv3["GS"]) * keep).to_numpy() + I["ov"].to_numpy()
    leg1 = I["leg1"].to_numpy(); sl4 = v4["sleeve"].to_numpy(); port = v4["port"].to_numpy()
    s = sel; print(f"[输入] ({time.time()-t0:.0f}s)")
    # ---------------- 1. 拆分
    momd = CX._french("F-F_Momentum_Factor_daily.csv", ["MOM"]).MOM.reindex(days).to_numpy(); mm = s & np.isfinite(momd)
    crash = mm & (momd <= np.quantile(momd[mm], 0.05)); st = cr["MOM"].state.to_numpy(); rows = []
    for k_, lab in ((0, "低"), (1, "中"), (2, "高")):
        a = mm & (st == k_)
        rows.append(dict(MOM状态=lab, 全部天数=int(a.sum()), 全部均值bp=raw[a].mean() * 1e4, 崩盘日数=int((a & crash).sum()), 崩盘日均值bp=raw[a & crash].mean() * 1e4,
                         非崩盘日均值bp=raw[a & ~crash].mean() * 1e4, 崩盘日贡献占该状态总损益=raw[a & crash].sum() / raw[a].sum()))
    print("\n=== 1. Sleeve(v3 SOTA,未缩放)按 MOM 拥挤状态 × 是否动量崩盘日(MOM 日收益 ≤ 5% 分位)==="); print(pd.DataFrame(rows).to_string(index=False))
    # ---------------- 2. 崩盘日的 B&H / 核心仓 / Sleeve / 组合
    ndx = pd.read_parquet(CX.RAW / "yahoo" / "IDX_NDX.parquet")["close"]; ndx.index = pd.to_datetime(ndx.index).tz_localize(None); lr = np.log(ndx).diff()
    tsmd = (np.sign(lr.rolling(252).sum().shift(1)) * lr).reindex(days).to_numpy(); tm = s & np.isfinite(tsmd)
    tcrash = tm & (tsmd <= np.quantile(tsmd[tm], 0.05))
    rows = []
    for nm, m in (("动量崩盘日(MOM ≤ 5%)", crash), ("  其中 MOM 高拥挤", crash & (st == 2)), ("时序动量崩盘日(TSM ≤ 5%)", tcrash),
                  ("  其中 TSM 高拥挤", tcrash & (cr["TSM"].state.to_numpy() == 2)), ("其余日", mm & ~crash & ~tcrash)):
        rows.append(dict(日子=nm, 天数=int(m.sum()), QQQ买入持有bp=bh[m].mean() * 1e4, 核心仓bp=leg1[m].mean() * 1e4, v4_Sleeve_bp=sl4[m].mean() * 1e4, v4组合bp=port[m].mean() * 1e4,
                         组合为负比例=float((port[m] < 0).mean()), Sleeve抵消核心仓亏损比例=float(-sl4[m].sum() / leg1[m].sum()) if leg1[m].sum() < 0 else np.nan))
    print("\n=== 2. 崩盘日各部分的日均损益(2019–26,v4 组合 ①)==="); print(pd.DataFrame(rows).to_string(index=False))
    qm = pd.Series(bh, index=days).resample("ME").apply(lambda x: (1 + x).prod() - 1); worst = qm.loc["2019":].nsmallest(8).index; rows = []
    for e in worst:
        m = (days.to_period("M") == e.to_period("M")); f_ = lambda x: float((1 + pd.Series(x[m])).prod() - 1) * 100
        rows.append(dict(月份=str(e.to_period("M")), QQQ=f_(bh), 核心仓=f_(leg1), v4_Sleeve=f_(sl4), v4组合=f_(port),
                         MOM状态=int(cr["MOM"].state.loc[m].iloc[0]), TSM状态=int(cr["TSM"].state.loc[m].iloc[0])))
    print("\n=== 2b. QQQ 最差的 8 个月(%,状态 0 低 / 1 中 / 2 高)==="); print(pd.DataFrame(rows).to_string(index=False))
    # ---------------- 3. 机制
    vx = pd.read_csv(CX.RAW / "cboe" / "VIX_History.csv"); vx["DATE"] = pd.to_datetime(vx.DATE); vx = vx.set_index("DATE")
    dv_close = np.log(vx.CLOSE).diff().reindex(days).to_numpy(); vgap = np.log(vx.OPEN / vx.CLOSE.shift(1)).reindex(days).to_numpy()
    dnv = dn.to_numpy(); c = b.c.to_numpy(); O = b.day_open.to_numpy(); li = b.last_idx.to_numpy(); n_ = np.arange(len(days))
    oc = np.log(c[n_, li] / O); m3 = s & np.isfinite(dv_close) & np.isfinite(dnv)
    q = pd.qcut(pd.Series(dv_close[m3]).rank(method="first"), 5, labels=["Q1 VIX 大跌", "Q2", "Q3", "Q4", "Q5 VIX 大涨"]).to_numpy()
    D = pd.DataFrame({"q": q, "Sleeve_bp": raw[m3] * 1e4, "多头腿bp": Lv3["L2"].to_numpy()[m3] * 1e4, "空头腿bp": Lv3["S2"].to_numpy()[m3] * 1e4,
                      "位移dn": dnv[m3], "QQQ开到收bp": oc[m3] * 1e4})
    print("\n=== 3a. 按同日 ΔlogVIX(收盘对收盘)五分位:Sleeve、腿、位移(开盘后 30 分钟起,÷σ14)==="); print(D.groupby("q", observed=True).mean().to_string())
    rows = []
    for k_, lab in ((0, "TSM 低"), (1, "TSM 中"), (2, "TSM 高")):
        for up, ul in ((True, "VIX 上升"), (False, "VIX 下降")):
            m = m3 & (cr["TSM"].state.to_numpy() == k_) & ((dv_close > 0) == up)
            rows.append(dict(状态=lab, 当日=ul, 天数=int(m.sum()), Sleeve_bp=raw[m].mean() * 1e4, 位移dn=np.nanmean(dnv[m]), QQQ开到收bp=oc[m].mean() * 1e4, 空头腿bp=Lv3["S2"].to_numpy()[m].mean() * 1e4))
    print("\n=== 3b. TSM 状态 × 当日 VIX 方向 ==="); print(pd.DataFrame(rows).to_string(index=False))
    # ---------------- 4. 开仓前的 VIX 跳空
    m4 = s & np.isfinite(vgap); gapq = np.log(O / b.prev_close_adj.to_numpy())
    print(f"\n=== 4. VIX 开盘跳空 log(VIX_open/VIX_prev_close):与 QQQ 跳空相关 {np.corrcoef(vgap[m4], gapq[m4])[0, 1]:+.3f};与同日收盘 ΔVIX 相关 {np.corrcoef(vgap[m4], dv_close[m4])[0, 1]:+.3f};与 X.dlvix(前一日)相关 {np.corrcoef(vgap[m4], X.dlvix.to_numpy()[m4])[0, 1]:+.3f}")
    rows = []
    for nm in ("TSM", "MOM"):
        c_ = cr[nm].C.to_numpy()
        for lab, dvv in (("同日收盘 ΔVIX(事后)", dv_close), ("开盘跳空(开仓前可得)", vgap), ("前一日 ΔVIX", X.dlvix.to_numpy())):
            m = s & np.isfinite(dvv); bb, tt, r2 = hac(raw[m] * 1e4, np.column_stack([c_[m], dvv[m] * 10, c_[m] * dvv[m] * 10]))
            rows.append(dict(指标=f"C_{nm}", ΔVIX=lab, b_C=bb[1], t_C=tt[1], b_ΔVIX=bb[2], t_ΔVIX=tt[2], b_交互=bb[3], t_交互=tt[3], R2=r2))
    print("交互回归(Sleeve bp;ΔVIX 单位 10%):"); print(pd.DataFrame(rows).to_string(index=False))
    m = s & np.isfinite(vgap); bb, tt, r2 = hac(raw[m] * 1e4, np.column_stack([np.abs(vgap[m]) * 10, np.abs(gapq[m]) * 100]))
    print(f"|VIX 跳空| 与 |QQQ 跳空| 同时进入:b(|VIX 跳空|,每 10%)= {bb[1]:+.2f}bp(t {tt[1]:.2f});b(|QQQ 跳空|,每 1%)= {bb[2]:+.2f}bp(t {tt[2]:.2f})")
    X["vix_gap"] = pd.Series(vgap, index=days).fillna(0.0); X["c_tsm"] = cr["TSM"].C; X["c_tsm_vg"] = X.c_tsm * X.vix_gap
    assert X.loc[sel, ["vix_gap", "c_tsm"]].notna().all().all()
    R = {"v4": v4}
    for nm, ft in (("Q1 v4 + vix_gap", F4 + ["vix_gap"]), ("Q2 v4 + vix_gap + c_tsm + c_tsm×vix_gap", F4 + ["vix_gap", "c_tsm", "c_tsm_vg"])):
        R[nm] = R3.assemble(R3.sizing(X, ft, Lv4, I, sel, days, "mv", "lin"), Lv4, I, om)
    oi = np.where(om.to_numpy())[0]; dev = yrs[oi] <= 2022; BI = M.block_idx(len(oi), 10, 2000, 0); srv = lambda v: v.mean() / v.std() * np.sqrt(252)
    ref = R["v4"]["port"].to_numpy()[oi]; refb = np.array([srv(ref[q_]) for q_ in BI]); rows = []
    for nm, r_ in R.items():
        x_ = r_["port"].to_numpy()[oi]; db = np.array([srv(x_[q_]) for q_ in BI]) - refb
        rows.append(dict(方案=nm, 组合SR_OOS=srv(x_), ΔSR=srv(x_) - srv(ref), lo=np.percentile(db, 2.5), hi=np.percentile(db, 97.5), dev=srv(x_[dev]) - srv(ref[dev]), test=srv(x_[~dev]) - srv(ref[~dev]),
                         **{str(y): srv(x_[yrs[oi] == y]) - srv(ref[yrs[oi] == y]) for y in range(2021, 2027)}))
    print("\n=== 4b. 开盘跳空进入 v4 仓位模型(组合 OOS 2021–26,相对 v4)==="); print(pd.DataFrame(rows).to_string(index=False))
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
