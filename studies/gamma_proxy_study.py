"""做市商 gamma：论文用 RSI(5) 做代理，我们手上有真实 GEX —— 谁在带信息？

论文 §4.5 的做法和主张（Zarattini, Aziz & Barbon 2024）：
  他们**没有**用 GEX 数据，而是用 **SPY 的 5 日 RSI（前一日收盘）** 当做市商 gamma 的代理。
  逻辑：市场近期上涨 -> 价格靠近大量被买入的 call 行权价 -> 做市商**正** gamma
        -> 他们逆势对冲（涨了卖、跌了买）-> 日内趋势短命 -> 策略收益低。
  回归  Ret_Mom,t = α + β × RSI_{SPY,t−1} + ε   他们得到 **β = −3.25, p = 0.001**
  （Ret 以 bp 计）

本脚本做四件事：
  1. 在 QQQ 上复现这个回归（dev / test 分开）
  2. 用**真实 GEX** 跑同一个回归，对比谁更强
  3. **伪证检验**：如果 RSI(5) 真的是 dealer gamma 的代理，
     它和真实 GEX 应该**正相关**。如果不相关，那"gamma 解释"就是错的，
     RSI 起作用（若有）只是普通的短期反转，和期权做市商无关。
  4. 两者同时入回归，看谁在控制对方后还活着
全部严格 PIT（RSI 用 t−1，GEX 用 t−2 的 E25 口径）。
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etft.intraday.bars import load_bars

from paths import ETFT
OUT = Path("outputs/full/intraday")
DEV, TEST = ("2019", "2022"), ("2023", "2026")


def rsi(close, n=5):
    """Wilder RSI，严格只用过去（输出在 t 日，使用时还要再 shift(1)）。"""
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = au / ad.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def ols(y, X, names):
    """返回 (系数, t 值, R²)。X 不含常数项，内部加。"""
    m = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y2, X2 = y[m], X[m]
    A = np.column_stack([np.ones(len(X2)), X2])
    beta, *_ = np.linalg.lstsq(A, y2, rcond=None)
    res = y2 - A @ beta
    s2 = res @ res / (len(y2) - A.shape[1])
    cov = s2 * np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    r2 = 1 - res.var() / y2.var()
    return dict(zip(["const"] + names, beta)), dict(zip(["const"] + names, beta / se)), r2, int(m.sum())


def main():
    b = load_bars("QQQ"); days = b.days
    cur = pd.read_pickle(OUT / "stop_hard_tail_curves.pkl")
    ret = cur["vwap（当前基线）"]            # 30 相位均值 + TQQQ 成本
    ret_bp = ret * 1e4

    px = b.day_close
    R5 = rsi(px, 5).shift(1)                 # 前一日收盘的 RSI(5)，论文口径
    R2 = rsi(px, 2).shift(1)
    R14 = rsi(px, 14).shift(1)

    g = pd.read_parquet(ETFT / "data/raw/squeeze_dix_gex.parquet")
    g.index = pd.to_datetime(g.index)
    GEX = g["gex"].reindex(days).ffill().shift(2)
    GEXp = GEX.rolling(252, min_periods=120).apply(lambda x: (x[-1] > x[:-1]).mean(), raw=True)

    print("=" * 104)
    print("1. 复现论文 §4.5 的回归：Ret(bp) = α + β × RSI5(t−1)")
    print("=" * 104)
    print(f"   论文在 SPY 上报告 β = −3.25, p = 0.001（样本 2007-2024）\n")
    for nm, x in (("RSI5", R5), ("RSI2", R2), ("RSI14", R14)):
        for per, pl in ((DEV, "dev19-22"), (TEST, "test23-26"), (("2019", "2026"), "全样本")):
            y = ret_bp.loc[per[0]:per[1]].to_numpy()
            X = x.loc[per[0]:per[1]].to_numpy()[:, None]
            bta, tv, r2, n = ols(y, X, [nm])
            print(f"   {nm:6s} {pl:10s} β={bta[nm]:+7.3f}  t={tv[nm]:+6.2f}  "
                  f"R²={r2:7.4f}  n={n}")
        print()

    print("=" * 104)
    print("2. 用真实 GEX 跑同一个回归")
    print("=" * 104)
    for nm, x in (("GEX", GEX), ("GEX分位", GEXp)):
        for per, pl in ((DEV, "dev19-22"), (TEST, "test23-26"), (("2019", "2026"), "全样本")):
            y = ret_bp.loc[per[0]:per[1]].to_numpy()
            X = x.loc[per[0]:per[1]].to_numpy()[:, None]
            bta, tv, r2, n = ols(y, X, [nm])
            print(f"   {nm:8s} {pl:10s} β={bta[nm]:+10.4f}  t={tv[nm]:+6.2f}  "
                  f"R²={r2:7.4f}  n={n}")
        print()

    print("=" * 104)
    print("3. 伪证检验：RSI(5) 如果真是 dealer gamma 的代理，应该和真实 GEX 正相关")
    print("=" * 104)
    for per, pl in ((DEV, "dev19-22"), (TEST, "test23-26"), (("2019", "2026"), "全样本")):
        a = R5.loc[per[0]:per[1]]; c = GEX.loc[per[0]:per[1]]; cp = GEXp.loc[per[0]:per[1]]
        m = a.notna() & c.notna()
        print(f"   {pl:10s} corr(RSI5, GEX) = {a[m].corr(c[m]):+.4f}   "
              f"corr(RSI5, GEX分位) = {a[m].corr(cp[m]):+.4f}   n={int(m.sum())}")
    print("""
   论文的逻辑链是「市场涨 -> 靠近 call 行权价 -> 做市商正 gamma」。
   若成立，RSI5 与真实 GEX 必须显著正相关。""")

    print("=" * 104)
    print("4. 两者同时入回归：控制对方后谁还活着")
    print("=" * 104)
    for per, pl in ((DEV, "dev19-22"), (TEST, "test23-26"), (("2019", "2026"), "全样本")):
        y = ret_bp.loc[per[0]:per[1]].to_numpy()
        X = np.column_stack([R5.loc[per[0]:per[1]].to_numpy(),
                             GEXp.loc[per[0]:per[1]].to_numpy()])
        bta, tv, r2, n = ols(y, X, ["RSI5", "GEX分位"])
        print(f"   {pl:10s} β_RSI5={bta['RSI5']:+7.3f}(t={tv['RSI5']:+5.2f})  "
              f"β_GEX={bta['GEX分位']:+8.3f}(t={tv['GEX分位']:+5.2f})  "
              f"R²={r2:7.4f}  n={n}")

    print("\n" + "=" * 104)
    print("5. 当成过滤器：低 RSI5 的日子才交易（dev 定阈值，test 套用，置换检验）")
    print("=" * 104)
    rg = np.random.default_rng(5)

    def perm_p(s, mask, nperm=800):
        s = s.dropna(); mk = mask.reindex(s.index).fillna(False)
        k = int(mk.sum())
        if k < 60:
            return np.nan, np.nan, 0
        real = s[mk]
        srr = real.mean() / real.std() * np.sqrt(252)
        nulls = []
        for _ in range(nperm):
            idx = np.sort(rg.choice(len(s), k, replace=False))
            x = s.iloc[idx]
            nulls.append(x.mean() / x.std() * np.sqrt(252))
        nulls = np.array([v for v in nulls if np.isfinite(v)])
        return srr, float((nulls >= srr).mean()), k

    for thr in (30, 40, 50, 60, 70):
        mk = (R5 < thr)
        line = f"   RSI5 < {thr:<3}"
        for per, pl in ((DEV, "dev"), (TEST, "test")):
            srr, p, k = perm_p(ret.loc[per[0]:per[1]], mk)
            line += f" | {pl} {k:4d}天 SR {srr:+.3f} (p={p:.3f})"
        print(line, flush=True)
    print("\n   对照：GEX 分位过滤")
    for thr in (0.33, 0.5, 0.67):
        mk = (GEXp <= thr)
        line = f"   GEX分位 <= {thr:<5}"
        for per, pl in ((DEV, "dev"), (TEST, "test")):
            srr, p, k = perm_p(ret.loc[per[0]:per[1]], mk)
            line += f" | {pl} {k:4d}天 SR {srr:+.3f} (p={p:.3f})"
        print(line, flush=True)

    pd.DataFrame(dict(ret_bp=ret_bp, rsi5=R5, rsi2=R2, gex=GEX, gex_pct=GEXp)).to_csv(
        OUT / "gamma_proxy.csv")
    print(f"\n已写入 {OUT}/gamma_proxy.csv")


if __name__ == "__main__":
    main()
