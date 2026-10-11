"""论文 SOTA 配置（vol targeting 4x + volume 调整 + GEX 调整）在 TQQQ 上的正确测试。

这个脚本修正 PART8 第 1 节的**三个**错误：

错误 1（最严重）：成本的名义金额算错了，正好多收 3 倍。
    要拿到 3 倍 QQQ 敞口，走 TQQQ 只需要买 **1/3** 的名义金额（因为 TQQQ 自己就动 3 倍）。
    PART8 里我按 3 倍名义金额收了 TQQQ 的价差和佣金。
    资金口径：QQQ 保证金 3×(0.176+0.324)=1.50bp；TQQQ 1×(2.42+0.32)=2.75bp -> 只差 1.8 倍。

错误 2：用了后复权价算每股成本，而每股成本取决于**真实成交价**。
    TQQQ 在 2025-11-20 做了 1:2 拆股，所以 2025-11 之前的后复权价是真实价的一半。
    PART8 说"2023 年初 TQQQ 价格 $8.1"——真实成交价是 **$17.3**。

错误 3：测的不是论文最好的那一版。
    PART8 测的是固定 1x 仓位；论文 SOTA 是 `sizing="voltarget"`（2% 日波动目标、4x 上限），
    再叠加我们自己在 E24b 做的成交量调整噪声区域、以及 E25 的 GEX 状态缩放。

关键结论的方向因此反过来：论文 SOTA 真正需要的是 **4 倍敞口**，而 TQQQ 恰好是小资金量
拿到 4 倍敞口**唯一不需要 4:1 日内保证金**的途径（4 倍敞口只占 1.33 倍名义金额）。

方法：
  仓位路径与成本无关（已验证引擎里 cost 只累加、不反馈进决策），所以
      net_t = gross_t − turnover_t × rate_t
  是**精确**的。用 cost_bp=1.0 跑一次就能取出 gross 和 turnover，然后套任意逐日成本模型。
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

from etft.intraday.bars import load_bars
from etft.intraday.strategy import NoiseCfg, run, buy_hold

from paths import ETFT
OUT = Path("outputs/full/intraday"); OUT.mkdir(parents=True, exist_ok=True)
HALF_SPREAD, COMMISSION = 0.005, 0.0045     # $/股，与 E24b/E25 的 per_share 口径一致
DEV, TEST = ("2019", "2022"), ("2023", "2026")


def actual_price(sym: str) -> pd.Series:
    """真实成交价（把未来的拆股还原回去）。per-share 成本必须用真实价，不能用后复权价。"""
    import yfinance as yf
    t = yf.Ticker(sym)
    h = yf.download(sym, start="2018-10-01", end="2026-10-05", auto_adjust=False,
                    progress=False, threads=False)
    px = h["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    sp = t.splits
    if len(sp):
        sp.index = pd.to_datetime(sp.index).tz_localize(None)
        # Close 已按拆股调整过 -> 乘回 t 之后发生的所有拆股倍数，得到当时的挂牌价
        fwd = pd.Series(1.0, index=px.index)
        for d, r in sp.items():
            fwd[px.index < d] *= r
        px = px * fwd
    return px.rename(sym)


def rf_daily() -> pd.Series:
    d = pd.read_parquet(ETFT / "data/raw/fred/DTB3.parquet")
    s = d.iloc[:, 0].astype(float) / 100.0
    return (s / 252.0).reindex(pd.date_range(s.index[0], "2026-12-31")).ffill()


def gex_scale(days, dev_ret) -> pd.Series:
    """E25 口径的 GEX 状态缩放：t−1 的 GEX 取 252 日滚动分位，再滞后一天保险。
    规则只在 dev 期上定（高分位组 Sharpe 是否 > 低分位组），原封不动用到 test。"""
    g = pd.read_parquet(ETFT / "data/raw/squeeze_dix_gex.parquet")["gex"]
    g.index = pd.to_datetime(g.index)
    p = g.rolling(252, min_periods=120).apply(lambda x: (x[-1] > x).mean(), raw=True).shift(2)
    p = p.reindex(days).ffill()
    terc = pd.cut(p, [-0.01, 1 / 3, 2 / 3, 1.01], labels=[1, 2, 3])
    d = dev_ret.loc[DEV[0]:DEV[1]]
    tt = terc.reindex(d.index)
    sr = {k: (d[tt == k].mean() / d[tt == k].std() if (tt == k).sum() > 40 else np.nan)
          for k in (1, 2, 3)}
    hi_better = (sr[3] > sr[1])
    mp = {1: 0.5, 2: 1.0, 3: 1.5} if hi_better else {1: 1.5, 2: 1.0, 3: 0.5}
    print(f"    GEX dev 分位组 Sharpe: T1 {sr[1]:.3f} T2 {sr[2]:.3f} T3 {sr[3]:.3f} "
          f"-> {'高GEX加仓' if hi_better else '低GEX加仓'}")
    return terc.map(mp).astype(float).reindex(days).fillna(1.0), hi_better


def metrics(x, bench=None):
    x = x.dropna()
    if len(x) < 60 or x.std() == 0:
        return dict(sharpe=np.nan)
    eq = (1 + x).cumprod(); dd = eq / eq.cummax() - 1
    ann_ = float(eq.iloc[-1] ** (252 / len(x)) - 1)
    o = dict(sharpe=float(x.mean() / x.std() * np.sqrt(252)), ann=ann_,
             vol=float(x.std() * np.sqrt(252)), mdd=float(dd.min()),
             calmar=ann_ / abs(dd.min()) if dd.min() < 0 else np.nan,
             t_stat=float(x.mean() / x.std() * np.sqrt(len(x))))
    if bench is not None:
        b = bench.reindex(x.index).fillna(0)
        if b.std() > 0:
            o["corr_bh"] = float(x.corr(b))
            o["beta"] = float(np.cov(x, b)[0, 1] / b.var())
            o["alpha"] = float((x - o["beta"] * b).mean() * 252)
    return o


def main():
    print("=== 真实成交价（还原拆股）===")
    pq, pt = actual_price("QQQ"), actual_price("TQQQ")
    adj = pd.read_csv("data/lev/levetf_close.csv", index_col=0)
    adj.index = pd.to_datetime(adj.index)
    for d in ("2023-01-03", "2024-01-03", "2025-01-03", "2026-01-02"):
        print(f"  {d}: TQQQ 真实挂牌价 ${pt.asof(pd.Timestamp(d)):7.2f} | "
              f"后复权价 ${adj['TQQQ'].asof(pd.Timestamp(d)):6.2f} | "
              f"QQQ ${pq.asof(pd.Timestamp(d)):7.2f}")

    b = load_bars("QQQ")
    days = b.days
    rf = rf_daily().reindex(days).fillna(0.0)
    bh_ex = buy_hold(b) - rf                      # 买入持有的超额收益
    PQ = pq.reindex(days).ffill()
    PT = pt.reindex(days).ffill()

    # 策略资金闲置时赚无风险利率（日内策略隔夜为 0），所以策略的原始收益**本身就是超额收益**
    print("\n=== 跑各档配置（cost_bp=1.0 取出 gross 与 turnover，仓位路径与成本无关）===")
    base = dict(stop="vwap", cost_bp=1.0)
    cfgs = {
        "A. PART8 测的版本（固定 1x）": NoiseCfg(**base),
        "B. 论文 SOTA（voltarget 4x）": NoiseCfg(sizing="voltarget", max_lev=4.0, **base),
        "C. SOTA + 成交量调整 rvol^0.5": NoiseCfg(sizing="voltarget", max_lev=4.0,
                                                band_volume="power", band_gamma=0.5, **base),
        "D. SOTA + 成交量时钟 volclock": NoiseCfg(sizing="voltarget", max_lev=4.0,
                                                band_volume="volclock", **base),
    }
    raw = {}
    for nm, cfg in cfgs.items():
        r = run(b, cfg)
        raw[nm] = r
        print(f"  {nm}: 日均敞口 {r.exposure_cap.mean():.2f}x (p95 {r.exposure_cap.quantile(.95):.2f}x), "
              f"日均换手 {(r['cost']*1e4).mean():.2f}x名义")

    # D + GEX 状态缩放（规则只在 dev 上定）
    print("  E. SOTA + volclock + GEX 状态缩放:")
    dev_proxy = raw["D. SOTA + 成交量时钟 volclock"]["overlay_gross"]
    gs, hi_better = gex_scale(days, dev_proxy)
    r_e = run(b, NoiseCfg(sizing="voltarget", max_lev=4.0, band_volume="volclock",
                          day_scale=gs, **base))
    raw["E. SOTA + volclock + GEX"] = r_e
    print(f"     日均敞口 {r_e.exposure_cap.mean():.2f}x, 日均换手 {(r_e['cost']*1e4).mean():.2f}x名义")

    # ---------- 三套成本模型
    per_share = HALF_SPREAD + COMMISSION
    cost_rates = {
        "QQQ 日内保证金": per_share / PQ,                 # 每单位 QQQ 敞口的名义金额 = 1
        "TQQQ（正确：名义金额 1/3）": (per_share / PT) / 3.0,
        "TQQQ（PART8 的错法：按 3 倍名义收）": per_share / PT,
        "TQQQ（错法 × 后复权价，PART8 原样）": per_share / adj["TQQQ"].reindex(days).ffill(),
    }
    print("\n=== 每单位 QQQ 等价敞口的单边成本（bp，test 期中位）===")
    for k, v in cost_rates.items():
        vv = v.loc[TEST[0]:TEST[1]] * 1e4
        print(f"  {k}: 中位 {vv.median():.3f}bp  (区间 {vv.min():.3f} ~ {vv.max():.3f})")

    rows, curves = [], {}
    for nm, r in raw.items():
        gross, turn = r["overlay_gross"], r["cost"] * 1e4
        for ck, rate in cost_rates.items():
            net = gross - turn * rate.reindex(days).ffill()
            for per, plab in ((TEST, "test23-26"), (("2019", "2026"), "all19-26")):
                m = metrics(net.loc[per[0]:per[1]], bh_ex.loc[per[0]:per[1]])
                rows.append(dict(variant=nm, cost_model=ck, period=plab,
                                 expo=r.exposure_cap.loc[per[0]:per[1]].mean(), **m))
            curves[(nm, ck)] = net
        curves[(nm, "gross")] = gross
    for per, plab in ((TEST, "test23-26"), (("2019", "2026"), "all19-26")):
        rows.append(dict(variant="QQQ 买入持有（超额）", cost_model="-", period=plab, expo=1.0,
                         **metrics(bh_ex.loc[per[0]:per[1]], bh_ex.loc[per[0]:per[1]])))

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "sota_tqqq.csv", index=False)
    pd.set_option("display.width", 320); pd.set_option("display.max_rows", 300)
    c = ["variant", "cost_model", "expo", "sharpe", "ann", "vol", "mdd", "calmar", "t_stat",
         "corr_bh", "beta", "alpha"]
    for plab in ("test23-26", "all19-26"):
        print(f"\n=== {plab} ===")
        print(R[R.period == plab][c].round(3).to_string(index=False))

    # ---------- 资金占用：这才是 TQQQ 的真正理由
    print("\n=== 资金占用（这是 TQQQ 真正的理由，不是成本）===")
    e = raw["D. SOTA + 成交量时钟 volclock"].exposure_cap.loc[TEST[0]:TEST[1]].dropna()
    print(f"  SOTA 的目标敞口：均值 {e.mean():.2f}x，中位 {e.median():.2f}x，"
          f"p95 {e.quantile(.95):.2f}x，打到 4x 上限的天数占比 {(e >= 3.99).mean():.1%}")
    print(f"  走 QQQ：需要 {e.quantile(.95):.2f}:1 的**日内**买力 -> 必须有 PDT 资格（账户≥$25,000）")
    print(f"  走 TQQQ：名义金额只要 {e.quantile(.95)/3:.2f}x -> "
          f"Reg-T 常规 2:1 保证金就够，甚至 {(e/3 <= 1).mean():.0%} 的交易日现金账户就能做")

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.8))
    ax = axes[0]
    for nm in raw:
        s = curves[(nm, "QQQ 日内保证金")].loc[TEST[0]:TEST[1]]
        ax.plot((1 + s).cumprod(), lw=1.2, label=f"{nm[:2]} SR {metrics(s)['sharpe']:.2f}")
    s = bh_ex.loc[TEST[0]:TEST[1]]
    ax.plot((1 + s).cumprod(), "k--", lw=1.8, label=f"QQQ B&H SR {metrics(s)['sharpe']:.2f}")
    ax.set_title("配置阶梯（QQQ 日内保证金成本，test）"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = axes[1]
    nm = "D. SOTA + 成交量时钟 volclock"
    for ck in cost_rates:
        s = curves[(nm, ck)].loc[TEST[0]:TEST[1]]
        ax.plot((1 + s).cumprod(), lw=1.3, label=f"{ck[:22]} SR {metrics(s)['sharpe']:.2f}")
    ax.set_title("同一策略，四种成本口径（SOTA+volclock）"); ax.legend(fontsize=7.5); ax.grid(alpha=.3)
    ax = axes[2]
    ax.plot(PT.loc[TEST[0]:TEST[1]], lw=1.3, label="TQQQ 真实挂牌价")
    ax.plot(adj["TQQQ"].reindex(days).loc[TEST[0]:TEST[1]], lw=1.3, ls="--", label="TQQQ 后复权价")
    ax.set_title("2025-11-20 的 1:2 拆股：后复权价不能用来算每股成本")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "fig_sota_tqqq.png", dpi=110); plt.close()
    pd.to_pickle(curves, OUT / "sota_tqqq_curves.pkl")
    print("\nfigure written")
    return R, curves, raw


if __name__ == "__main__":
    main()
