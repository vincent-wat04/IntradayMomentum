"""收益隐含拥挤度(Feng 2027,按微信解读还原)—— docs/PROPOSAL_crowding_vol_timing.md §2。
C_t = (z(run-up 12m) + z(AR(1) 60m) − z(β_ΔVIX 60m)) / 3;z 用截至上月的 120 个月均值/标准差;状态 = 自身此前 60 个月三分位。
三个"因子":MOM(French 截面动量)、TSM(NDX 时序动量)、MKT(NDX 超额收益)。
python3 crowding_index.py   → 自检 + outputs/figs/crowding_index.png + data/_crowding.pkl
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as _P

RAW = _P.RAW; DATA = _P.DATA
W_RUN, W_REG, W_STD, W_STATE = 12, 60, 120, 60


def _french(fname, cols):
    rows = []
    for ln in open(RAW / "french" / fname).read().splitlines():
        p = [x.strip() for x in ln.split(",")]
        if len(p) >= len(cols) + 1 and p[0].isdigit():
            rows.append(p[: len(cols) + 1])
        elif rows:
            break
    d = pd.DataFrame(rows, columns=["d"] + cols)
    fmt = "%Y%m" if len(d.d.iloc[0]) == 6 else "%Y%m%d"
    d.index = pd.to_datetime(d.d, format=fmt); d = d.drop(columns="d").astype(float) / 100.0
    if fmt == "%Y%m": d.index = d.index + pd.offsets.MonthEnd(0)
    return d


def load_monthly():
    mom = _french("F-F_Momentum_Factor.csv", ["MOM"]).MOM
    rf = _french("F-F_Research_Data_Factors.csv", ["MKT_RF", "SMB", "HML", "RF"]).RF
    ndx = pd.read_parquet(RAW / "yahoo" / "IDX_NDX.parquet")["close"]; ndx.index = pd.to_datetime(ndx.index).tz_localize(None)
    ndx_m = ndx.resample("ME").last(); r = ndx_m.pct_change()
    vix = pd.read_parquet(RAW / "cboe" / "VIX_long.parquet")["close"]; vix.index = pd.to_datetime(vix.index).tz_localize(None)
    dvix = vix.resample("ME").last().diff()
    past12 = r.rolling(12).sum().shift(1)                                       # 月 m−12..m−1
    tsm = np.sign(past12) * r
    mkt = r - rf.reindex(r.index)
    return pd.DataFrame({"MOM": mom, "TSM": tsm, "MKT": mkt, "dVIX": dvix})


def _ols_slope(y, x):
    xm, ym = x.mean(), y.mean(); vx = ((x - xm) ** 2).sum()
    return float(((x - xm) * (y - ym)).sum() / vx) if vx > 0 else np.nan


def crowding(f: pd.Series, dvix: pd.Series, W_RUN=W_RUN, W_REG=W_REG, W_STD=W_STD, W_STATE=W_STATE):
    """逐期独立计算(确定性循环);第 t 行只用 ≤ t 的数据。默认窗口为月频(文章);日频改写时传入日数。"""
    idx = f.index; fv = f.to_numpy(float); dv = dvix.reindex(idx).to_numpy(float); n = len(fv)
    run = np.full(n, np.nan); ar = np.full(n, np.nan); beta = np.full(n, np.nan)
    for t in range(n):
        if t >= W_RUN - 1 and np.isfinite(fv[t - W_RUN + 1:t + 1]).all(): run[t] = fv[t - W_RUN + 1:t + 1].sum()
        if t >= W_REG:
            y, xl = fv[t - W_REG + 1:t + 1], fv[t - W_REG:t]
            if np.isfinite(y).all() and np.isfinite(xl).all(): ar[t] = _ols_slope(y, xl)
            xv = dv[t - W_REG + 1:t + 1]
            if np.isfinite(y).all() and np.isfinite(xv).all(): beta[t] = _ols_slope(y, xv)
    def z(x):
        out = np.full(n, np.nan)
        for t in range(W_STD, n):
            h = x[t - W_STD:t]
            if np.isfinite(h).all() and np.isfinite(x[t]) and h.std(ddof=1) > 0: out[t] = (x[t] - h.mean()) / h.std(ddof=1)
        return out
        
    zr, za, zb = z(run), z(ar), z(beta); C = (zr + za - zb) / 3.0
    st = np.full(n, np.nan)
    for t in range(W_STATE, n):
        h = C[t - W_STATE:t]
        if np.isfinite(h).all() and np.isfinite(C[t]):
            q1, q2 = np.quantile(h, [1 / 3, 2 / 3]); st[t] = 0 if C[t] <= q1 else (2 if C[t] >= q2 else 1)
    return pd.DataFrame({"run": run, "ar": ar, "beta": beta, "z_run": zr, "z_ar": za, "z_beta": zb, "C": C, "state": st}, index=idx)


def to_daily(Cm: pd.DataFrame, days, extra_lag=0):
    """月末 t 的值作用于 t+1+extra_lag 月的全部交易日。"""
    s = Cm.copy(); s.index = s.index + pd.offsets.MonthEnd(1 + extra_lag)
    key = pd.DatetimeIndex(days) + pd.offsets.MonthEnd(0)
    return s.reindex(key).set_axis(pd.DatetimeIndex(days))


def build(days=None):
    Mo = load_monthly(); out = {}
    for nm in ("MOM", "TSM", "MKT"):
        out[nm] = crowding(Mo[nm], Mo.dVIX)
    return Mo, out


def main():
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from cjkfont import use_cjk
    pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:8.3f}")
    Mo, CI = build()
    print("月度样本:", {k: (v.dropna().index[0].date(), v.dropna().index[-1].date()) for k, v in Mo.items()})
    # 自检 1:投毒(某月之后的因子收益与 VIX ×(−3),该月及之前的 C/状态逐位不变)
    worst = 0.0
    for cut in ("2012-06-30", "2018-03-31", "2022-09-30"):
        Mp = Mo.copy(); m = Mp.index > cut; Mp.loc[m, ["MOM", "TSM", "MKT", "dVIX"]] *= -3.0
        for nm in ("MOM", "TSM", "MKT"):
            a, b = CI[nm].loc[:cut], crowding(Mp[nm], Mp.dVIX).loc[:cut]
            worst = max(worst, float(np.nanmax(np.abs(a.to_numpy() - b.to_numpy()))), float((a.isna() != b.isna()).to_numpy().sum()))
    print(f"[投毒自检] 3 个切点 × 3 个指标:截至切点的所有列最大差 {worst:.2e}(须为 0)"); assert worst == 0.0
    # 自检 2:PIT 映射
    d0 = pd.DatetimeIndex(["2021-03-01", "2021-03-31"]); dm = to_daily(CI["MOM"], d0, extra_lag=1)
    assert np.allclose(dm.C.to_numpy(), CI["MOM"].loc["2021-01-31", "C"]), "PIT 映射错误"
    print("[PIT] 2021-03 的 C_MOM = 2021-01 月末值(多滞后一个月)✓")
    for nm, d in CI.items():
        v = d.dropna(subset=["C"]); print(f"{nm}: C 首个可用 {v.index[0].date()};状态首个可用 {d.dropna(subset=['state']).index[0].date()};"
                                          f" 2019+ 高拥挤月比例 {float((d.loc['2019':].state == 2).mean()):.1%};分项相关 {v[['z_run', 'z_ar', 'z_beta']].corr().round(2).values[np.triu_indices(3, 1)]}")
    print("\n2019 年以来的状态(0 低 / 1 中 / 2 高):")
    S = pd.DataFrame({k: v.state for k, v in CI.items()}).loc["2018-12":]; print(S.T.astype("Int64").to_string())
    print("\n拥挤度之间的相关(2005+):"); print(pd.DataFrame({k: v.C for k, v in CI.items()}).loc["2005":].corr().round(2).to_string())
    pd.to_pickle(dict(M=Mo, CI=CI), DATA / "_crowding.pkl")
    use_cjk(plt); fig, ax = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    vz = (Mo.dVIX.cumsum()); vlev = pd.read_parquet(RAW / "cboe" / "VIX_long.parquet")["close"].resample("ME").last()
    for a_, nm in zip(ax, ("MOM", "TSM", "MKT")):
        d = CI[nm].loc["2005":]; a_.plot(d.C, color="k", lw=1.2, label=f"C_{nm}")
        for s_, c_ in ((2, "tab:red"), (0, "tab:green")):
            a_.fill_between(d.index, -3, 3, where=(d.state == s_).to_numpy(), color=c_, alpha=.12, step="mid", label="高拥挤" if s_ == 2 else "低拥挤")
        b_ = a_.twinx(); b_.plot(vlev.loc["2005":], color="tab:blue", lw=.8, alpha=.6); b_.set_ylabel("VIX", color="tab:blue")
        a_.axvline(pd.Timestamp("2019-01-01"), color="gray", ls="--"); a_.set_ylim(-2.5, 2.5); a_.set_title(f"C_{nm}(收益隐含拥挤度)与状态;蓝线 VIX", loc="left"); a_.legend(loc="upper left", fontsize=8); a_.grid(alpha=.3)
    out = Path("../outputs/figs/crowding_index.png"); fig.savefig(out, dpi=110, bbox_inches="tight"); print(f"图:{out}")


if __name__ == "__main__":
    main()
