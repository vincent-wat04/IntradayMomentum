"""A 股中小盘横截面（中证500 + 中证1000 成分，剔除 ST），研报协议。

数据限制必须先说清楚：moomoo 历史 K 线有 **300 标的 / 7 天**的额度上限。本轮额度被
98 只美股 5 分钟 + A 股下载共同用满，所以 1499 只成分股只拿到 **126 只**日线。
126 只的横截面对五分组来说每组只有 25 只，统计功效很弱，**这是一次初步测试而不是结论**。
额度 7 天后释放，届时可以补全。

行业分类：moomoo 的 plate 接口没返回行业，这里用"中证500 / 中证1000"的指数归属 +
交易所（SH/SZ）作为粗分组做中性化，并额外做市值中性化（研报口径）。
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pdf_volume_factors as PV
import report_protocol as RP
import xs_factor_library as XL

SRC = Path("../cn_daily")
OUT = Path("outputs/full/cn_xs"); OUT.mkdir(parents=True, exist_ok=True)
SPLIT = "2023-01-03"
P = 5


def load():
    mem = json.loads((SRC / "members.json").read_text())
    grp = {}
    for idx, recs in mem.items():
        for m in recs:
            grp.setdefault(m["code"], idx)
    frames = {}
    for f in sorted(SRC.glob("*.csv")):
        try:
            d = pd.read_csv(f, usecols=["time_key", "open", "high", "low", "close", "volume", "turnover"])
        except Exception:
            continue
        if len(d) < 800:
            continue
        d["day"] = pd.to_datetime(d.time_key).dt.strftime("%Y-%m-%d")
        d = d.drop_duplicates("day").set_index("day")
        frames[f.stem] = d
    if not frames:
        raise SystemExit("no A-share data")
    dates = sorted(set().union(*[set(d.index) for d in frames.values()]))
    out = {}
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.DataFrame({k: v[col] for k, v in frames.items()}, index=dates).astype(float)
    cov = out["close"].notna().mean()
    keep = cov[cov > 0.9].index
    out = {k: v[keep] for k, v in out.items()}
    # 粗分组：指数归属 + 交易所
    sec = pd.Series({c: f"{grp.get(c,'other')}_{c.split('.')[0]}" for c in keep})
    return out, sec


def main():
    D, sec = load()
    px, op, hi, lo, vo = D["close"], D["open"], D["high"], D["low"], D["volume"]
    print(f"A 股中小盘：{px.shape[1]} 只（额度上限所致），{px.shape[0]} 天，{sec.nunique()} 个粗分组")
    print(f"  指数归属分布: {sec.str.split('_').str[0].value_counts().to_dict()}")
    F0, r = XL.build(px, op, hi, lo, vo)
    size = np.log1p(px * vo.rolling(252, min_periods=60).mean())
    rng_ = (hi - lo).replace(0, np.nan)
    locf = ((px - lo) / rng_).clip(0, 1)
    dv = px * vo
    F0["OFI_D"] = ((2 * locf - 1) * dv).rolling(20, min_periods=10).sum() / dv.rolling(20, min_periods=10).sum()
    F = dict(F0); F.update({f"{k}_STAB": PV.stability(v, 20) for k, v in F0.items()})
    F = {k: v.replace([np.inf, -np.inf], np.nan) for k, v in F.items()}
    split = int(np.searchsorted(px.index, SPLIT))
    print(f"  dev {px.index[0]}..{px.index[split-1]} | test {px.index[split]}..{px.index[-1]}")

    Z = {}
    for k, v in F.items():
        Z[k] = RP.neutral_ind_size(RP.winsor_std(v), sec, size).rolling(20, min_periods=10).mean()
    fwd = RP.fwd_ret(px, P)
    rows = []
    for k, v in Z.items():
        icf = RP.weekly_ic(v.iloc[:split], fwd.iloc[:split], P)
        dsign = float(np.sign(icf.mean()) or 1.0)
        sc = v * dsign
        ic = RP.weekly_ic(sc.iloc[split:], fwd.iloc[split:], P)
        rets, ls, turns = RP.group_books(sc, r, P, fee_bp=12.0)   # A股：佣金+印花税+滑点，单边约12bp
        lng, lsx = rets[RP.N_GROUP].iloc[split:], ls.iloc[split:]
        bh = RP.ann(r.mean(axis=1).iloc[split:])
        rows.append(dict(factor=k, is_stab=k.endswith("_STAB"), sign=dsign,
                         rank_ic=float(ic.mean()), icir=float(ic.mean() / ic.std()) if ic.std() else np.nan,
                         long_ann=RP.ann(lng), long_excess=RP.ann(lng) - bh, long_turn=turns[RP.N_GROUP],
                         ls_ann=RP.ann(lsx), ls_vol=float(lsx.std() * np.sqrt(252)),
                         ls_sharpe=float(lsx.mean() / lsx.std() * np.sqrt(252)) if lsx.std() else np.nan,
                         ls_mdd=RP.mdd(lsx)))
    R = pd.DataFrame(rows); R.to_csv(OUT / "cn_xs_factors.csv", index=False)
    pd.set_option("display.width", 290); pd.set_option("display.max_rows", 300)
    print(f"\n  等权池子年化（test）: {RP.ann(r.mean(axis=1).iloc[split:]):.1%}")
    print("\n=== A 股中小盘，研报协议（周频五分组、行业市值中性、12bp/腿）TOP 20 ===")
    print(R.nlargest(20, "ls_sharpe")[["factor", "is_stab", "rank_ic", "icir", "long_ann",
                                       "long_excess", "ls_ann", "ls_vol", "ls_sharpe", "ls_mdd",
                                       "long_turn"]].round(4).to_string(index=False))
    print("\n=== level vs stability ===")
    print(R.groupby("is_stab").agg(n=("factor", "size"), mean_ic=("rank_ic", lambda s: s.abs().mean()),
                                   mean_sr=("ls_sharpe", "mean"), best=("ls_sharpe", "max"),
                                   share_gt1=("ls_sharpe", lambda s: (s > 1).mean())).round(4).to_string())
    print("\n=== 和美股同协议对比 ===")
    print(f"  A 股中小盘 (N={px.shape[1]:3d}): |IC| 中位 {R.rank_ic.abs().median()*100:.3f}%  "
          f"多空SR 中位 {R.ls_sharpe.median():.3f}  最好 {R.ls_sharpe.max():.3f}  >1 的占比 {(R.ls_sharpe>1).mean():.1%}")
    for u, n in [("sp500", 479), ("sp400", 353), ("sp600", 518)]:
        try:
            p = pd.read_csv(RP.OUT / "universe_compare.csv")
            s = p[(p.universe == u) & (p.sign_mode == "dev")]
            print(f"  美股 {u} (N={n:3d}): |IC| 中位 {s.rank_ic.abs().median()*100:.3f}%  "
                  f"多空SR 中位 {s.ls_sharpe.median():.3f}  最好 {s.ls_sharpe.max():.3f}  "
                  f">1 的占比 {(s.ls_sharpe>1).mean():.1%}")
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(9, 7))
    t = R.nlargest(22, "ls_sharpe").sort_values("ls_sharpe")
    ax.barh(np.arange(len(t)), t.ls_sharpe, color=["tab:orange" if s else "tab:blue" for s in t.is_stab])
    ax.set_yticks(np.arange(len(t))); ax.set_yticklabels(t.factor, fontsize=8)
    ax.axvline(0, color="k", lw=.6); ax.set_xlabel("多空 Sharpe (12bp/腿)")
    ax.set_title(f"A 股中小盘横截面 TOP 22（N={px.shape[1]}，橙=stability）")
    ax.grid(alpha=.3, axis="x")
    plt.tight_layout(); plt.savefig(OUT / "fig_cn_xs.png", dpi=110); plt.close()
    print("\nfigure written")
    return R


if __name__ == "__main__":
    main()
