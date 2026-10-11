"""研报因子的真正横截面复现：用 moomoo 5 分钟 bar，不再用日频代理。

数据：77 只美股的 5 分钟前复权 bar（2019-01 → 2026-10，每只约 151k 根）。
moomoo 历史 K 线有 300 标的/7 天的额度上限，已用满，所以是 77 只而不是 98 只。

因子：`pdf_volume_factors.py` 里按研报公式实现的 21 个（13 个 level + 8 个 stability）。
协议：`report_protocol.py` 的研报口径——去极值 → 标准化 → 行业+市值中性化 → τ=20 平滑 →
五分组 → 周频调仓 → 乘方向（先验 / dev 两种口径）。

和日频代理的对比是本脚本的重点：之前 PART6 的结论"IC 水平比研报低一个量级"是在日频代理上得到的，
现在可以判断那个差距里有多少是代理造成的。
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pdf_volume_factors as PV
import report_protocol as RP

SRC = Path("data/min5_us")
OUT = Path("outputs/full/protocol"); OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path("data/xs/pdf5m_factors.pkl")
SPLIT = "2023-01-03"
P = 5


def one_ticker(path: Path):
    """Daily level+stability factors for one name, from its 5-minute bars."""
    try:
        raw = pd.read_csv(path, usecols=["time_key", "open", "high", "low", "close", "volume"])
    except Exception:
        return path.stem, None
    raw["dt"] = pd.to_datetime(raw.time_key)
    raw["day"] = raw.dt.dt.strftime("%Y-%m-%d")
    raw["clock"] = raw.dt.dt.strftime("%H:%M")
    raw = raw[(raw.clock >= "09:30") & (raw.clock < "16:00")]
    n_per_day = raw.groupby("day").size()
    full = n_per_day[n_per_day >= 70].index          # drop half days / broken sessions
    raw = raw[raw.day.isin(full)]
    if raw.day.nunique() < 500:
        return path.stem, None
    dayclose = raw.groupby("day").close.last()
    prev = dayclose.shift(1)
    segs = dict(open30=("09:30", "10:00"), pm_open30=("13:00", "13:30"),
                mid_close30=("11:30", "12:00"), close30=("15:30", "16:00"))
    rows = {}
    for day, g in raw.groupby("day", sort=True):
        g = g.sort_values("dt")
        clock = g.clock.to_numpy()
        seg = {k: (clock >= a) & (clock < b) for k, (a, b) in segs.items()}
        seg["hi"] = g.high.to_numpy(float); seg["lo"] = g.low.to_numpy(float)
        rows[day] = PV.daily_raw(g.open.to_numpy(float), g.close.to_numpy(float),
                                 g.volume.to_numpy(float), clock, float(dayclose[day]), seg)
    D = pd.DataFrame(rows).T.replace([np.inf, -np.inf], np.nan)
    out = {}
    for n in PV.LEVEL_NAMES:
        if n in D:
            out[n] = PV.smooth(D[n], PV.TAU)
    for n in PV.STAB_OF:
        if n in D:
            out[f"{n}_STAB"] = PV.stability(D[n], PV.TAU)
    return path.stem, pd.DataFrame(out)


def build_panel():
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    files = sorted(SRC.glob("*.csv"))
    print(f"building 5-minute factors for {len(files)} names...", flush=True)
    res = {}
    with Pool(8) as pool:
        for i, (tk, df) in enumerate(pool.imap_unordered(one_ticker, files)):
            if df is not None:
                res[tk] = df
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(files)}", flush=True)
    names = sorted({c for d in res.values() for c in d.columns})
    dates = sorted({d for df in res.values() for d in df.index})
    F = {n: pd.DataFrame({tk: res[tk][n] for tk in res if n in res[tk]}, index=dates) for n in names}
    pd.to_pickle(F, CACHE)
    print(f"built {len(F)} factors x {len(next(iter(F.values())).columns)} names x {len(dates)} days")
    return F


def main():
    F5 = build_panel()
    tickers = list(next(iter(F5.values())).columns)
    dates = next(iter(F5.values())).index

    # price panel + sector for the same names, from the daily store
    px_all = pd.read_parquet("data/xs_big/close.parquet")
    vo_all = pd.read_parquet("data/xs_big/volume.parquet")
    U = pd.read_csv("data/xs_big/universe.csv")
    cols = [t for t in tickers if t in px_all.columns]
    px = px_all[cols].reindex(dates).astype(float)
    vo = vo_all[cols].reindex(dates).astype(float)
    sec = pd.Series(U.set_index("ticker").sector.reindex(cols).fillna("Other").to_dict())
    r = px.pct_change()
    size = np.log1p(px * vo.rolling(252, min_periods=60).mean())
    split = int(np.searchsorted(px.index, SPLIT))
    print(f"\nreplication universe: {len(cols)} names, {len(dates)} days, {sec.nunique()} sectors")

    # 研报预处理（因子里已含 tau=20 平滑）
    Z = {}
    for k, v in F5.items():
        v = v[cols].reindex(dates)
        Z[k] = RP.neutral_ind_size(RP.winsor_std(v), sec, size)

    fwd = RP.fwd_ret(px, P)
    rows = []
    for k, v in Z.items():
        icf = RP.weekly_ic(v.iloc[:split], fwd.iloc[:split], P)
        dsign = float(np.sign(icf.mean()) or 1.0)
        for mode, sg in (("report(-1)", -1.0), ("dev", dsign)):
            sc = v * sg
            ic = RP.weekly_ic(sc.iloc[split:], fwd.iloc[split:], P)
            rets, ls, turns = RP.group_books(sc, r, P)
            lng, lsx = rets[RP.N_GROUP].iloc[split:], ls.iloc[split:]
            bh = RP.ann(r.mean(axis=1).iloc[split:])
            rows.append(dict(factor=k, sign_mode=mode, sign=sg, is_stab=k.endswith("_STAB"),
                             rank_ic=float(ic.mean()),
                             icir=float(ic.mean() / ic.std()) if ic.std() else np.nan,
                             ic_std=float(ic.std()),
                             long_ann=RP.ann(lng), long_excess=RP.ann(lng) - bh,
                             long_turn=turns[RP.N_GROUP], ls_ann=RP.ann(lsx),
                             ls_vol=float(lsx.std() * np.sqrt(252)),
                             ls_sharpe=float(lsx.mean() / lsx.std() * np.sqrt(252)) if lsx.std() else np.nan,
                             ls_mdd=RP.mdd(lsx)))
    R = pd.DataFrame(rows); R.to_csv(OUT / "pdf_5min_replication.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_rows", 200)

    REPORT_IC = {"LOGV_STD": 1.55, "LOGV_SKEW": 5.94, "VOL_RATIO_1": 4.61, "VOL_INTRA_STD_1": 4.33,
                 "VOL_RATIO_1_5": 5.56, "VOL_RATIO_8_4": 2.58, "SIG_UP_PV_RATIO": 6.30,
                 "SIG_UP_PV_STD": 5.93, "V_P_SKEWNESS": 2.93, "V_P_REVERSAL": 5.50}
    REPORT_SR = {"LOGV_STD": 1.68, "LOGV_SKEW": 3.16, "VOL_RATIO_1": 1.50, "VOL_INTRA_STD_1": 3.24,
                 "VOL_RATIO_1_5": 4.09, "VOL_RATIO_8_4": 3.90, "SIG_UP_PV_RATIO": 2.95,
                 "SIG_UP_PV_STD": 3.34, "V_P_SKEWNESS": 1.30, "V_P_REVERSAL": 3.29}
    d = R[R.sign_mode == "dev"].set_index("factor")
    rp = R[R.sign_mode == "report(-1)"].set_index("factor")
    cmp = pd.DataFrame({
        "研报_RankIC%": pd.Series(REPORT_IC), "研报_多空SR": pd.Series(REPORT_SR),
        "美股_RankIC%(dev向)": (d.rank_ic * 100).round(3), "美股_ICIR": d.icir.round(3),
        "美股_多空SR(dev向)": d.ls_sharpe.round(3), "美股_多空SR(先验向)": rp.ls_sharpe.round(3),
        "美股_多空年化": d.ls_ann.round(3), "美股_多头超额": d.long_excess.round(3),
    }).dropna(subset=["研报_RankIC%"])
    print("\n=== 研报 10 个因子 vs 美股 5 分钟真实复现（77 只，周频五分组，行业市值中性）===")
    print(cmp.to_string())
    print(f"\n  IC 比率（美股/研报）中位: {(cmp['美股_RankIC%'.replace('%','%')+'(dev向)'].abs()/cmp['研报_RankIC%']).median():.3f}"
          if False else "")
    ratio = (cmp["美股_RankIC%(dev向)"].abs() / cmp["研报_RankIC%"])
    print(f"  IC 比率（美股/研报）：中位 {ratio.median():.2f}，范围 {ratio.min():.2f} – {ratio.max():.2f}")
    print(f"  先验方向与 dev 方向一致的比例: {(d.loc[cmp.index, 'sign'] < 0).mean():.1%}")

    print("\n=== 全部 21 个因子，按 dev 向多空 Sharpe 排序 ===")
    print(R[R.sign_mode == "dev"].sort_values("ls_sharpe", ascending=False)[
        ["factor", "is_stab", "rank_ic", "icir", "long_ann", "long_excess", "ls_ann", "ls_vol",
         "ls_sharpe", "ls_mdd", "long_turn"]].round(4).to_string(index=False))

    print("\n=== level vs stability（研报的核心结构性想法）===")
    print(R[R.sign_mode == "dev"].groupby("is_stab").agg(
        n=("factor", "size"), mean_ic=("rank_ic", lambda s: s.abs().mean()),
        mean_icir=("icir", lambda s: s.abs().mean()), mean_sr=("ls_sharpe", "mean"),
        best_sr=("ls_sharpe", "max")).round(4).to_string())

    # 和日频代理比
    try:
        prox = pd.read_csv(OUT / "universe_compare.csv")
        p5 = prox[(prox.universe == "sp500") & (prox.sign_mode == "dev")]
        print("\n=== 5 分钟真因子 vs 日频代理（同为 dev 向）===")
        print(f"  5 分钟真因子 (N=77):   |IC| 中位 {d.rank_ic.abs().median()*100:.3f}%  "
              f"多空SR 中位 {d.ls_sharpe.median():.3f}  最好 {d.ls_sharpe.max():.3f}")
        print(f"  日频代理   (N=479):   |IC| 中位 {p5.rank_ic.abs().median()*100:.3f}%  "
              f"多空SR 中位 {p5.ls_sharpe.median():.3f}  最好 {p5.ls_sharpe.max():.3f}")
    except Exception as e:
        print("proxy comparison unavailable:", e)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    ax = axes[0]
    xx = np.arange(len(cmp))
    ax.barh(xx - 0.2, cmp["研报_多空SR"], 0.4, label="研报 (A股全市场 ~5000只)")
    ax.barh(xx + 0.2, cmp["美股_多空SR(dev向)"], 0.4, label="美股 5分钟真因子 (77只)")
    ax.set_yticks(xx); ax.set_yticklabels(cmp.index, fontsize=9)
    ax.axvline(0, color="k", lw=.6); ax.set_xlabel("多空 Sharpe")
    ax.set_title("研报因子：A股 vs 美股（同一协议）"); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="x")
    ax = axes[1]
    q = R[R.sign_mode == "dev"]
    for flag, lab, c in [(False, "level", "tab:blue"), (True, "stability", "tab:orange")]:
        s = q[q.is_stab == flag]
        ax.scatter(s.rank_ic.abs() * 100, s.ls_sharpe, s=70, c=c, label=lab, alpha=.85)
        for _, rr in s.iterrows():
            ax.annotate(rr.factor.replace("_STAB", ""), (abs(rr.rank_ic) * 100, rr.ls_sharpe),
                        fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.axhline(0, color="k", lw=.6); ax.set_xlabel("|周度 Rank IC| (%)"); ax.set_ylabel("多空 Sharpe")
    ax.set_title("5 分钟真因子：level vs stability"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "fig_pdf_5min_replication.png", dpi=110); plt.close()
    print("\nfigure written")
    return R, cmp


if __name__ == "__main__":
    main()
