"""Large vs mid vs small cap, under the exact 研报 protocol, plus factor correlation and style exposure.

Answers:
  Q1  can we reproduce the report's Sharpe > 3?   -> breadth is NOT the main gap; see the IC comparison
  Q3  are the factors better in small caps?       -> run the identical protocol on S&P 500 / 400 / 600
      and on all 1350 names, and report the factor correlation and Barra-style exposure the report
      mentions ("整体上因子间相关性不高，各因子在 Barra 风险因子中略偏向流动性与特异性波动率")

Only the report's own horizon is run (weekly signal, weekly rebalance, 5 groups, tau=20 smoothing,
winsorise -> standardise -> industry+size neutral), so the four universes fit in one pass.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pdf_volume_factors as PV
import report_protocol as RP
import xs_factor_library as XL

OUT = RP.OUT
UNIVERSES = ("sp500", "sp400", "sp600", "ALL")
P = RP.P_REPORT


def prep(which):
    (px, op, hi, lo, vo), sec = RP.load_universe(which)
    F0, r = XL.build(px, op, hi, lo, vo)
    size = np.log1p(px * vo.rolling(252, min_periods=60).mean())
    rng_ = (hi - lo).replace(0, np.nan)
    locf = ((px - lo) / rng_).clip(0, 1)
    dv = px * vo
    F0["OFI_D"] = ((2 * locf - 1) * dv).rolling(RP.TAU, min_periods=10).sum() / dv.rolling(RP.TAU, min_periods=10).sum()
    F = dict(F0)
    F.update({f"{k}_STAB": PV.stability(v, RP.TAU) for k, v in F0.items()})
    F = {k: v.replace([np.inf, -np.inf], np.nan) for k, v in F.items()}
    Z = {}
    for k, v in F.items():
        Z[k] = RP.neutral_ind_size(RP.winsor_std(v), sec, size).rolling(RP.TAU, min_periods=RP.TAU // 2).mean()
    return px, r, sec, size, F0, Z


def main():
    allrows, corrs, expos = [], {}, {}
    for which in UNIVERSES:
        px, r, sec, size, F0, Z = prep(which)
        split = int(np.searchsorted(px.index, RP.SPLIT))
        fwd = RP.fwd_ret(px, P)
        print(f"{which}: {px.shape[1]} names", flush=True)
        for k, v in Z.items():
            ic_fit = RP.weekly_ic(v.iloc[:split], fwd.iloc[:split], P)
            dsign = float(np.sign(ic_fit.mean()) or 1.0)
            for mode, sg in (("report(-1)", -1.0), ("dev", dsign)):
                sc = v * sg
                ic = RP.weekly_ic(sc.iloc[split:], fwd.iloc[split:], P)
                rets, ls, turns = RP.group_books(sc, r, P)
                lng, lsx = rets[RP.N_GROUP].iloc[split:], ls.iloc[split:]
                bh = RP.ann(r.mean(axis=1).iloc[split:])
                allrows.append(dict(universe=which, n_names=px.shape[1], factor=k, sign_mode=mode,
                                    rank_ic=float(ic.mean()),
                                    icir=float(ic.mean() / ic.std()) if ic.std() else np.nan,
                                    ic_std=float(ic.std()),
                                    long_ann=RP.ann(lng), long_excess=RP.ann(lng) - bh,
                                    long_turn=turns[RP.N_GROUP], ls_ann=RP.ann(lsx),
                                    ls_vol=float(lsx.std() * np.sqrt(252)),
                                    ls_sharpe=float(lsx.mean() / lsx.std() * np.sqrt(252)) if lsx.std() else np.nan,
                                    ls_mdd=RP.mdd(lsx)))
        # correlation and style exposure on the last 400 days
        last = px.index[-400:]
        M = pd.DataFrame({k: Z[k].loc[last].stack(future_stack=True) for k in Z})
        C = M.corr().to_numpy(copy=True)
        np.fill_diagonal(C, np.nan)
        C = pd.DataFrame(C, index=list(Z), columns=list(Z))
        corrs[which] = C
        styles = {"SIZE": size, "BM_PROXY": -np.log(px)}
        for s in ("BETA", "VOL_21", "MOM_252", "AMIHUD", "IVOL", "TURN"):
            if s in F0:
                styles[s] = F0[s]
        SZ = {k: RP.winsor_std(v) for k, v in styles.items()}
        e = {}
        for k in Z:
            a = Z[k].loc[last].stack(future_stack=True)
            row = {}
            for s, sv in SZ.items():
                j = pd.concat([a, sv.loc[last].stack(future_stack=True)], axis=1).dropna()
                row[s] = float(j.iloc[:, 0].corr(j.iloc[:, 1])) if len(j) > 2000 else np.nan
            e[k] = row
        expos[which] = pd.DataFrame(e).T
        C.to_csv(OUT / f"corr_{which}.csv"); expos[which].to_csv(OUT / f"style_{which}.csv")

    R = pd.DataFrame(allrows); R.to_csv(OUT / "universe_compare.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_rows", 300)

    print("\n=== Q3  does the signal get stronger in smaller caps?  (研报口径, 先验方向) ===")
    q = R[R.sign_mode == "report(-1)"]
    print(q.groupby(["universe", "n_names"]).agg(
        n_fac=("factor", "size"), mean_abs_ic=("rank_ic", lambda s: s.abs().mean()),
        mean_icir=("icir", lambda s: s.abs().mean()), ic_std=("ic_std", "mean"),
        mean_ls_sharpe=("ls_sharpe", "mean"), best_ls_sharpe=("ls_sharpe", "max"),
        share_sr_gt1=("ls_sharpe", lambda s: (s > 1).mean()),
        mean_turn=("long_turn", "mean")).round(4).to_string())
    print("\n  同上，但用 dev 符号（样本内定向）：")
    d = R[R.sign_mode == "dev"]
    print(d.groupby("universe").agg(mean_abs_ic=("rank_ic", lambda s: s.abs().mean()),
                                    mean_ls_sharpe=("ls_sharpe", "mean"),
                                    best=("ls_sharpe", "max"),
                                    share_sr_gt1=("ls_sharpe", lambda s: (s > 1).mean())).round(4).to_string())

    print("\n=== Q1  IC 对比：这才是和研报差距的真正来源 ===")
    print(f"  研报 Rank IC（周度）范围      1.55% – 9.12%，ICIR 0.22 – 1.34")
    for u in UNIVERSES:
        s = d[d.universe == u]
        print(f"  {u:6s} N={int(s.n_names.iloc[0]):5d}  |IC| 中位 {s.rank_ic.abs().median()*100:.3f}%  "
              f"p90 {s.rank_ic.abs().quantile(.9)*100:.3f}%  |ICIR| 中位 {s.icir.abs().median():.3f}  "
              f"IC 噪声 std {s.ic_std.median():.4f}")
    print("  => 广度(N)只影响 IC 的噪声；研报与我们的差距主要在 IC 的**水平**，相差约一个量级。")

    print("\n=== 各股票池最好的 10 个因子（dev 符号，多空 Sharpe）===")
    for u in UNIVERSES:
        s = d[d.universe == u].nlargest(10, "ls_sharpe")
        print(f"\n  --- {u} (N={int(s.n_names.iloc[0])})")
        print(s[["factor", "rank_ic", "icir", "long_ann", "long_excess", "ls_ann", "ls_vol",
                 "ls_sharpe", "ls_mdd", "long_turn"]].round(4).to_string(index=False))

    print("\n=== 因子相关性（研报：整体上因子间相关性不高）===")
    for u in UNIVERSES:
        C = corrs[u]
        ev = np.linalg.eigvalsh(np.nan_to_num(C.fillna(0).to_numpy()) + np.eye(len(C)))
        ev = ev[ev > 1e-8]
        print(f"  {u:6s} 平均|corr| {C.abs().mean().mean():.3f} | "
              f"中位 {C.abs().stack().median():.3f} | "
              f">0.7 的对数占比 {(C.abs() > 0.7).sum().sum() / (len(C) * (len(C) - 1)):.1%} | "
              f"有效独立因子数 {(ev.sum() ** 2 / (ev ** 2).sum()):.1f}/{len(C)}")

    print("\n=== 风格暴露（|相关| 均值，研报：略偏向流动性与特异性波动率）===")
    E = pd.DataFrame({u: expos[u].abs().mean() for u in UNIVERSES})
    print(E.round(3).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    ax = axes[0]
    g = q.groupby("universe").ls_sharpe.mean().reindex(UNIVERSES)
    g2 = d.groupby("universe").ls_sharpe.mean().reindex(UNIVERSES)
    xx = np.arange(len(UNIVERSES))
    ax.bar(xx - 0.2, g.values, 0.4, label="报告先验方向")
    ax.bar(xx + 0.2, g2.values, 0.4, label="dev 符号（样本内定向）")
    ax.set_xticks(xx); ax.set_xticklabels([f"{u}\nN={int(R[R.universe==u].n_names.iloc[0])}" for u in UNIVERSES])
    ax.axhline(0, color="k", lw=.6); ax.set_ylabel("平均多空 Sharpe")
    ax.set_title("universe vs signal strength"); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    ax = axes[1]
    for u in UNIVERSES:
        s = d[d.universe == u]
        ax.scatter(s.ic_std, s.rank_ic.abs(), s=14, alpha=.6, label=f"{u} N={int(s.n_names.iloc[0])}")
    ax.set_xlabel("IC 的时间序列标准差"); ax.set_ylabel("|平均 Rank IC|")
    ax.axhline(0.0155, color="tab:red", ls="--", lw=1, label="研报最低 IC 1.55%")
    ax.axhline(0.0912, color="tab:purple", ls="--", lw=1, label="研报最高 IC 9.12%")
    ax.set_title("IC 水平 vs IC 噪声"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax = axes[2]
    E.plot(kind="barh", ax=ax, width=.8)
    ax.set_xlabel("|相关| 均值"); ax.set_title("风格暴露"); ax.grid(alpha=.3, axis="x")
    plt.tight_layout(); plt.savefig(OUT / "fig_universe_compare.png", dpi=110); plt.close()
    print("\nfigure written")
    return R, corrs, expos


if __name__ == "__main__":
    main()
