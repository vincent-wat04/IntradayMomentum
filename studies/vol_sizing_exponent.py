"""仓位 ∝ σ^β 的指数扫描：从 vol targeting（β=−1）一直扫到反向（β=+1）。

动机：PART8 §3 发现策略收益单调依赖**当日实现波动**（平静日 −10bp、大波动日 +46bp），
而 vol targeting 的做法是 e ∝ 1/σ，也就是**波动越高仓位越小** —— 方向似乎相反。
gex_vol_forecast.py 进一步发现：把波动预测做得更准（GEX 把样本外 R² 从 0.009 提到 0.035），
收益反而在 test 上变差（ΔSR −0.10）。所以自然的问题是：**反着做会不会更好？**

必须先区分两个"波动"：
  当日实现波动   凸性针对的是这个，但它几乎不可预测（R²=0.035）
  趋势波动 σ_14  vol targeting 用的是这个，可预测，但策略 Sharpe 对它的依赖
                在 dev/test 上矛盾（PART9 §2：高波动区间 dev +1.33/test +0.60，
                低波动区间 dev +0.28/test +1.20）
所以 β 的最优值本质上是在押"哪种 regime 的 Sharpe 更高"，而这一点样本内外不一致。

做法：
  e(β) = clip(c · σ^β, 0, 4)，c 在**训练窗口**上标定使平均敞口与基线一致（敞口中性）
  ret(β) = base_ret × e(β)/e_base   —— 这是精确的（仓位路径不依赖规模、成本成正比）
  每个 β 报 Sharpe / 年化 / 波动 / **最大回撤 / 最差单日 / 偏度**，并做配对区块自助。

**反向 vol targeting 的风险必须量化**：它在波动最大的时候加杠杆，
所以最大回撤和最差单日是比 Sharpe 更重要的输出。
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
from etft.intraday.strategy import buy_hold

from paths import ETFT
OUT = Path("outputs/full/intraday")
DEV, TEST = ("2019", "2022"), ("2023", "2026")
TARGET_VOL, MAX_LEV = 0.02, 4.0


def sr(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std() * np.sqrt(252)) if len(x) > 30 and x.std() > 0 else np.nan


def risk(x):
    x = pd.Series(x).dropna()
    if len(x) < 40 or x.std() == 0:
        return dict(sharpe=np.nan)
    eq = (1 + x).cumprod(); dd = eq / eq.cummax() - 1
    q5 = x.quantile(0.05)
    return dict(sharpe=sr(x), ann=float(x.mean() * 252),
                vol=float(x.std() * np.sqrt(252)), mdd=float(dd.min()),
                worst=float(x.min()), CVaR5=float(x[x <= q5].mean()),
                skew=float(((x - x.mean()) ** 3).mean() / x.std() ** 3))


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


def main():
    b = load_bars("QQQ"); days = b.days; bh = buy_hold(b)
    cur = pd.read_pickle(OUT / "stop_hard_tail_curves.pkl")
    base_ret = cur["vwap（当前基线）"]

    sig = daily_vol(b, 14).reindex(days)                  # 现用的 σ（已 shift(1)）
    e_base = np.minimum(MAX_LEV, TARGET_VOL / sig)        # β=−1 的敞口
    tr = slice(DEV[0], f"{DEV[1]}-12-31")

    def expo(beta, s=sig):
        raw = s.where(s > 0) ** beta
        k = 1.0
        e = raw * k
        # 标定：训练窗口上平均敞口对齐基线（敞口中性）
        k = e_base.loc[tr].mean() / e.loc[tr].mean()
        return (raw * k).clip(upper=MAX_LEV)

    betas = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    print("=" * 120)
    print("1. 仓位 ∝ σ^β（β=−1 是现用的 vol targeting，β=0 是固定仓位，β>0 是反向）")
    print("   全部敞口中性：c 在 dev 上标定使平均敞口与基线一致")
    print("=" * 120)
    print(f"   {'β':>5} {'平均敞口':>8} | {'dev SR':>7} {'dev年化':>8} {'dev回撤':>8} "
          f"{'dev最差日':>9} | {'test SR':>8} {'test年化':>9} {'test回撤':>9} "
          f"{'test最差日':>10} {'偏度':>6}")
    curves = {}
    for bt in betas:
        e = expo(bt)
        r = base_ret * (e / e_base)
        curves[bt] = r
        rd, rt = risk(r.loc[tr]), risk(r.loc[TEST[0]:f"{TEST[1]}-12-31"])
        mark = "  <-现用" if bt == -1.0 else ("  <-固定仓位" if bt == 0.0 else "")
        print(f"   {bt:+5.1f} {e.loc[TEST[0]:].mean():8.3f} | {rd['sharpe']:+7.3f} "
              f"{rd['ann']:+8.2%} {rd['mdd']:+8.1%} {rd['worst']:+9.2%} | "
              f"{rt['sharpe']:+8.3f} {rt['ann']:+9.2%} {rt['mdd']:+9.1%} "
              f"{rt['worst']:+10.2%} {rt['skew']:+6.2f}{mark}", flush=True)

    print("\n" + "=" * 120)
    print("2. 配对区块自助（vs β=−1 现用）")
    print("=" * 120)
    base = curves[-1.0]
    print(f"   {'β':>5} | {'dev ΔSR':>9} {'95%区间':>20} {'p':>7} | "
          f"{'test ΔSR':>9} {'95%区间':>20} {'p':>7}")
    rows = []
    for bt in betas:
        if bt == -1.0:
            continue
        line = f"   {bt:+5.1f} |"
        rec = dict(beta=bt)
        for per, pl in ((DEV, "dev"), (TEST, "test")):
            s = slice(per[0], f"{per[1]}-12-31")
            a = curves[bt].loc[s].dropna(); bb = base.loc[s].dropna()
            real, dist = boot_pair(a, bb, seed=int((bt + 2) * 1000))
            lo, hi = np.percentile(dist, [2.5, 97.5])
            p = float((dist <= 0).mean())
            line += f" {real:+9.3f} [{lo:+7.3f},{hi:+7.3f}] {p:7.3f} |"
            rec[f"{pl}_d"] = real; rec[f"{pl}_p"] = p
        print(line, flush=True)
        rows.append(rec)

    print("\n" + "=" * 120)
    print("3. 逐年 Sharpe：反向仓位在哪些年份活、哪些年份死")
    print("=" * 120)
    Y = pd.DataFrame({f"β={bt:+.1f}": {y: sr(curves[bt].loc[f"{y}-01-01":f"{y}-12-31"])
                                       for y in range(2019, 2027)} for bt in betas}).T
    print(Y.round(3).to_string())

    print("\n" + "=" * 120)
    print("4. **风险代价**：反向仓位在危机年的表现（这是它的真实风险）")
    print("=" * 120)
    crises = {"2020 新冠崩盘": ("2020-02-15", "2020-04-15"),
              "2022 熊市全年": ("2022-01-01", "2022-12-31"),
              "2025-04 关税冲击": ("2025-03-25", "2025-04-25")}
    print(f"   {'区间':20s} " + " ".join(f"{f'β={bt:+.1f}':>10}" for bt in betas))
    for nm, (a, c) in crises.items():
        vals = []
        for bt in betas:
            x = curves[bt].loc[a:c].dropna()
            vals.append(f"{(1+x).prod()-1:+10.2%}" if len(x) > 5 else f"{'n/a':>10}")
        print(f"   {nm:20s} " + " ".join(vals))
    print("\n   （累计收益；反向仓位在危机期**加杠杆**，这里能看到代价或红利）")

    print("\n" + "=" * 120)
    print("5. 直接检验推理前提：策略 Sharpe 到底在高趋势波动还是低趋势波动的区间更高")
    print("=" * 120)
    unit = base_ret / e_base                     # 单位敞口的收益（剥离仓位规模）
    for per, pl in ((DEV, "dev"), (TEST, "test")):
        s = slice(per[0], f"{per[1]}-12-31")
        x = unit.loc[s].dropna()
        q = pd.qcut(sig.reindex(x.index).dropna(), 4, labels=["Q1低σ", "Q2", "Q3", "Q4高σ"])
        t = pd.DataFrame({"r": x.reindex(q.index)}).groupby(q, observed=True).agg(
            天数=("r", "size"), 日均bp=("r", lambda v: v.mean() * 1e4), Sharpe=("r", sr))
        print(f"\n   {pl}（单位敞口收益，已剥离仓位规模）：")
        print("     " + t.round(3).to_string().replace("\n", "\n     "))
    print("\n   如果高 σ 区间的 **Sharpe** 更高 -> 该 β>0（反向）；更低 -> 该 β<0（现用）")

    pd.DataFrame(rows).to_csv(OUT / "vol_sizing_exponent.csv", index=False)
    Y.to_csv(OUT / "vol_sizing_by_year.csv")
    print(f"\n已写入 {OUT}/vol_sizing_exponent.csv")


if __name__ == "__main__":
    main()
