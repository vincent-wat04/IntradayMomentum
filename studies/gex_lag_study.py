"""GEX 的滞后代价：t−2（我用的）vs t−1（合法最小）vs t−0（前视，只做上界）。

为什么要做这个：
  SqueezeMetrics GEX 是**收盘后发布**的日频序列，GEX(t) 用 t 日的未平仓量和 t 日收盘价，
  所以 t 日开盘时不可知 -> 合法的最小滞后是 t−1。
  我之前一律用 t−2（沿用 E25 的"多滞后一天保险"），**这一天是没检验过的保守**。

t−0 是**前视**，不能用于任何结论，但它给出一个关键的上界：
  **如果连前视版本都很弱，那么追求"实时 gamma"就不值得做。**
  反之如果前视版本显著强于 t−1，就说明 gamma 信息衰减很快，
  实时计算（从期权链现算）才有价值。
这是决定要不要投入做实时 gamma 记录器的最便宜的证据。
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
from etft.intraday.noise import daily_vol

from paths import ETFT
OUT = Path("outputs/full/intraday")
DEV, TEST = ("2019", "2022"), ("2023", "2026")


def sr(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.mean() / x.std() * np.sqrt(252) if len(x) > 30 and x.std() > 0 else np.nan


def boot_pair(a, b, nb=4000, L=10, seed=0):
    idx = a.index.intersection(b.index)
    A, B = a.reindex(idx).fillna(0).to_numpy(), b.reindex(idx).fillna(0).to_numpy()
    n = len(A); nblk = int(np.ceil(n / L)); rg = np.random.default_rng(seed)
    out = np.empty(nb)
    for i in range(nb):
        st = rg.integers(0, max(n - L, 1), size=nblk)
        sel = np.concatenate([np.arange(s, min(s + L, n)) for s in st])[:n]
        out[i] = sr(A[sel]) - sr(B[sel])
    return sr(A) - sr(B), out


def ols_t(y, x):
    m = np.isfinite(y) & np.isfinite(x)
    y2, x2 = y[m], x[m]
    A = np.column_stack([np.ones(len(x2)), x2])
    beta, *_ = np.linalg.lstsq(A, y2, rcond=None)
    res = y2 - A @ beta
    s2 = res @ res / (len(y2) - 2)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(A.T @ A)))
    return beta[1], beta[1] / se[1], 1 - res.var() / y2.var(), int(m.sum())


def main():
    b = load_bars("QQQ"); days = b.days
    cur = pd.read_pickle(OUT / "stop_hard_tail_curves.pkl")
    base = cur["vwap（当前基线）"]
    ret_bp = base * 1e4

    g = pd.read_parquet(ETFT / "data/raw/squeeze_dix_gex.parquet")
    g.index = pd.to_datetime(g.index)
    raw = g["gex"].reindex(days).ffill()

    print("=" * 108)
    print("1. 回归 t 值：Ret(bp) = α + β × GEX(t−k)。k=0 是前视，仅作上界参考")
    print("=" * 108)
    print(f"   {'滞后':>6} {'合法性':>8} | {'dev β(t)':>18} | {'test β(t)':>18} | {'全样本 β(t)':>18}")
    for k in (0, 1, 2, 3, 5):
        x = raw.shift(k)
        legal = "**前视**" if k == 0 else ("合法" if k >= 1 else "")
        line = f"   t−{k:<4} {legal:>8} |"
        for per in (DEV, TEST, ("2019", "2026")):
            bt, tv, r2, n = ols_t(ret_bp.loc[per[0]:per[1]].to_numpy(),
                                  x.loc[per[0]:per[1]].to_numpy())
            line += f" {bt:+.2e}({tv:+5.2f}) |"
        print(line)

    print("\n" + "=" * 108)
    print("2. 自相关：GEX 自己衰减得多快（决定实时计算有多大价值）")
    print("=" * 108)
    for k in (1, 2, 3, 5, 10, 22):
        print(f"   corr(GEX_t, GEX_t−{k:<2}) = {raw.corr(raw.shift(k)):+.4f}")
    dg = raw.diff()
    print(f"\n   GEX 日变化的标准差 / GEX 水平的标准差 = {dg.std()/raw.std():.3f}")
    print("   -> 这个比值越大，说明 GEX 每天变动越剧烈、滞后代价越高")

    # 当日价格变动对 GEX 的影响：gamma 是价格相对行权价的函数
    r_day = (b.day_close / b.prev_close_adj - 1)
    m = dg.notna() & r_day.notna()
    bt, tv, r2, n = ols_t(dg[m].to_numpy(), r_day[m].to_numpy())
    print(f"\n   ΔGEX_t 对当日收益回归：β={bt:+.3e}  t={tv:+.2f}  R²={r2:.4f}")
    print("   -> GEX 的日内变化**确实**被当日价格驱动（gamma 是价格相对行权价的函数），")
    print("      这正是'实时 gamma 比昨日收盘 GEX 更有信息'的理论依据")

    print("\n" + "=" * 108)
    print("3. 仓位缩放：不同滞后下的表现 + 配对自助（λ=0.3，cap 0.5，PIT z-score）")
    print("=" * 108)
    rv = daily_vol(b, 14).reindex(days)
    lrv = np.log(rv.where(rv > 0))
    rows = []
    print(f"   {'滞后':>6} | {'dev SR':>8} {'dev ΔSR(p)':>16} | {'test SR':>8} "
          f"{'test ΔSR(p)':>16}")
    for k in (0, 1, 2, 3):
        x = raw.shift(k)
        mu = x.rolling(252, min_periods=120).mean()
        sd = x.rolling(252, min_periods=120).std()
        z = (x - mu) / sd
        s_ = base * (1.0 - 0.3 * z).clip(0.5, 1.5).fillna(1.0)
        line = f"   t−{k:<4} |"
        rec = dict(lag=k)
        for per, pl in ((DEV, "dev"), (TEST, "test")):
            a = s_.loc[per[0]:per[1]].dropna()
            bb = base.loc[per[0]:per[1]].dropna()
            real, dist = boot_pair(a, bb, seed=100 + k)
            p = float((dist <= 0).mean())
            line += f" {sr(a):+8.3f} {real:+8.3f}({p:.3f}) |"
            rec[f"{pl}_sr"] = sr(a); rec[f"{pl}_d"] = real; rec[f"{pl}_p"] = p
        print(line + ("   <- **前视，上界**" if k == 0 else ""), flush=True)
        rows.append(rec)

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "gex_lag_study.csv", index=False)

    print("\n" + "=" * 108)
    print("4. 关键问题：前视版（t−0）比合法版（t−1）好多少")
    print("=" * 108)
    r0, r1, r2_ = R[R.lag == 0].iloc[0], R[R.lag == 1].iloc[0], R[R.lag == 2].iloc[0]
    print(f"   dev :  t−0 ΔSR {r0.dev_d:+.3f}(p={r0.dev_p:.3f})   "
          f"t−1 {r1.dev_d:+.3f}(p={r1.dev_p:.3f})   t−2 {r2_.dev_d:+.3f}(p={r2_.dev_p:.3f})")
    print(f"   test:  t−0 ΔSR {r0.test_d:+.3f}(p={r0.test_p:.3f})   "
          f"t−1 {r1.test_d:+.3f}(p={r1.test_p:.3f})   t−2 {r2_.test_d:+.3f}(p={r2_.test_p:.3f})")
    gain_dev = r0.dev_d - r1.dev_d
    gain_test = r0.test_d - r1.test_d
    print(f"\n   前视相对合法最小滞后的额外收益：dev {gain_dev:+.3f}  test {gain_test:+.3f}")
    print(f"   我多滞后一天的代价：dev {r1.dev_d - r2_.dev_d:+.3f}  "
          f"test {r1.test_d - r2_.test_d:+.3f}")
    verdict = ("**值得**做实时 gamma" if max(gain_dev, gain_test) > 0.15
               else "**不值得**做实时 gamma（连前视上界都不够大）")
    print(f"\n   结论：{verdict}")
    print(f"   （判据：审计 5.1 的噪声护栏是 ΔSharpe 0.15）")


if __name__ == "__main__":
    main()
