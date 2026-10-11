"""用**原始 GEX**（而非分位）做仓位缩放 + 配对区块自助检验。

为什么重做：PART9 §2.4 测的是 **GEX 分位**缩放，置换 p=0.092。
而 gamma_proxy_study.py 发现**原始 GEX 的回归 t 值远强于分位**：
    原始 GEX   dev t=−1.06   test t=−4.07   全样本 t=−2.56
    GEX 分位   dev t=−0.75   test t=−1.03   全样本 t=−1.19
分位化把**幅度**信息丢掉了，而 gamma 的机制本来就是幅度效应
（gamma 越负，做市商顺势对冲的**金额**越大）。所以应该用原始值。

一个关键的实现事实：仓位缩放可以**事后**作用在日收益上，精确无误。
因为 (a) 开平仓条件不依赖仓位大小，所以仓位路径不变；
    (b) 成本与仓位成正比。
所以 ret_scaled(t) = s(t) × ret(t) 是**精确**的，不需要重跑引擎。

缩放函数（事前定好，不在 test 上调）：
    z(t)  = (GEX(t) − 滚动均值) / 滚动标准差    252 日滚动，严格 PIT，t−2 口径
    s(t)  = clip(1 − λ·z(t), 1−λ_cap, 1+λ_cap)
λ 在 dev 上选，原封不动套到 test。另外测正交化版本
（把 GEX 对 log σ_trailing 回归后取残差），因为 PART9 §2.2 的教训是
与趋势波动率相关的缩放会抵消 vol targeting（corr(neg_gex, log rv14)=+0.41）。
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


def risk(x):
    x = pd.Series(x).dropna()
    if len(x) < 40 or x.std() == 0:
        return dict(sharpe=np.nan)
    eq = (1 + x).cumprod(); dd = eq / eq.cummax() - 1
    return dict(sharpe=sr(x), ann=float(eq.iloc[-1] ** (252 / len(x)) - 1),
                vol=float(x.std() * np.sqrt(252)), mdd=float(dd.min()),
                worst=float(x.min()),
                skew=float(((x - x.mean()) ** 3).mean() / x.std() ** 3))


def main():
    b = load_bars("QQQ"); days = b.days; bh = buy_hold(b)
    cur = pd.read_pickle(OUT / "stop_hard_tail_curves.pkl")
    base = cur["vwap（当前基线）"]
    cand = cur["band + 硬尾0.5σ"]

    g = pd.read_parquet(ETFT / "data/raw/squeeze_dix_gex.parquet")
    g.index = pd.to_datetime(g.index)
    GEX = g["gex"].reindex(days).ffill().shift(2)          # t−2，E25 口径

    # PIT z-score（252 日滚动，只用过去）
    mu = GEX.rolling(252, min_periods=120).mean()
    sd = GEX.rolling(252, min_periods=120).std()
    Z = (GEX - mu) / sd

    # 正交化：GEX 对 log σ_trailing 做扩展窗口回归后取残差
    rv = daily_vol(b, 14).reindex(days)
    lrv = np.log(rv.where(rv > 0))
    Zo = pd.Series(np.nan, index=days)
    gv, zv = GEX.to_numpy(), lrv.to_numpy()
    for i in range(260, len(days)):
        ok = np.isfinite(gv[:i]) & np.isfinite(zv[:i])
        if ok.sum() < 200 or not (np.isfinite(gv[i]) and np.isfinite(zv[i])):
            continue
        A = np.column_stack([np.ones(ok.sum()), zv[:i][ok]])
        bb = np.linalg.lstsq(A, gv[:i][ok], rcond=None)[0]
        Zo.iloc[i] = gv[i] - (bb[0] + bb[1] * zv[i])
    muo = Zo.rolling(252, min_periods=120).mean()
    sdo = Zo.rolling(252, min_periods=120).std()
    Zorth = (Zo - muo) / sdo

    print("=" * 110)
    print("0. 先确认：原始 GEX 与趋势波动率的相关（PART9 §2.2 的教训）")
    print("=" * 110)
    m = GEX.notna() & lrv.notna()
    print(f"   corr(GEX, log σ_trailing) = {GEX[m].corr(lrv[m]):+.4f}")
    m2 = Zorth.notna() & lrv.notna()
    print(f"   corr(正交化GEX, log σ_trailing) = {Zorth[m2].corr(lrv[m2]):+.4f}"
          f"  <- 正交化后应接近 0")

    # ---------- 缩放
    def scaled(ret, z, lam, cap):
        s = (1.0 - lam * z).clip(1 - cap, 1 + cap).fillna(1.0)
        return ret * s, s

    print("\n" + "=" * 110)
    print("1. λ 在 dev 上选（cap 固定 0.5，即仓位在 0.5x~1.5x 之间）")
    print("=" * 110)
    for nm, z in (("原始GEX z", Z), ("正交化GEX z", Zorth)):
        print(f"\n   --- {nm}")
        print(f"   {'λ':>5} | {'dev SR':>8} {'dev年化':>8} | {'test SR':>8} {'test年化':>9} "
              f"{'平均仓位':>8}")
        for lam in (0.0, 0.1, 0.2, 0.3, 0.5):
            s_, sc = scaled(base, z, lam, 0.5)
            rd, rt = risk(s_.loc[DEV[0]:DEV[1]]), risk(s_.loc[TEST[0]:TEST[1]])
            print(f"   {lam:5.2f} | {rd['sharpe']:+8.3f} {rd['ann']:+8.2%} | "
                  f"{rt['sharpe']:+8.3f} {rt['ann']:+9.2%} "
                  f"{sc.loc[TEST[0]:TEST[1]].mean():8.3f}")

    # ---------- 配对自助：对 dev 选出的 λ
    print("\n" + "=" * 110)
    print("2. 配对区块自助检验（vs 未缩放基线）")
    print("=" * 110)
    print(f"   {'配置':28s} | {'dev ΔSR':>9} {'95%区间':>20} {'p':>7} | "
          f"{'test ΔSR':>9} {'95%区间':>20} {'p':>7}")
    rows = []
    for nm, z in (("原始GEX", Z), ("正交化GEX", Zorth)):
        for lam in (0.2, 0.3, 0.5):
            s_, _ = scaled(base, z, lam, 0.5)
            line = f"   {nm+f' λ={lam}':28s} |"
            rec = dict(配置=f"{nm} λ={lam}")
            for per, pl in ((DEV, "dev"), (TEST, "test")):
                a = s_.loc[per[0]:per[1]].dropna()
                bb2 = base.loc[per[0]:per[1]].dropna()
                real, dist = boot_pair(a, bb2, seed=abs(hash(nm + str(lam))) % 9999)
                lo, hi = np.percentile(dist, [2.5, 97.5])
                p = float((dist <= 0).mean())
                line += f" {real:+9.3f} [{lo:+7.3f},{hi:+7.3f}] {p:7.3f} |"
                rec[f"{pl}_d"] = real; rec[f"{pl}_p"] = p
            print(line, flush=True)
            rows.append(rec)

    # ---------- 用在 band+硬尾0.5σ 上（我们打算切换过去的配置）
    print("\n" + "=" * 110)
    print("3. 同样的缩放用在 band+硬尾0.5σ 上（准备切换的配置）")
    print("=" * 110)
    for lam in (0.0, 0.2, 0.3):
        s_, sc = scaled(cand, Z, lam, 0.5)
        rd, rt = risk(s_.loc[DEV[0]:DEV[1]]), risk(s_.loc[TEST[0]:TEST[1]])
        ov = risk((bh + s_.reindex(bh.index).fillna(0)).loc[TEST[0]:TEST[1]])
        print(f"   λ={lam:4.2f}  dev {rd['sharpe']:+.3f}  test {rt['sharpe']:+.3f} "
              f"年化 {rt['ann']:+.2%} 最差日 {rt['worst']:+.2%}  |  "
              f"叠加 test {ov['sharpe']:.3f}")
        if lam > 0:
            for per, pl in ((DEV, "dev"), (TEST, "test")):
                a = s_.loc[per[0]:per[1]].dropna()
                bb2 = cand.loc[per[0]:per[1]].dropna()
                real, dist = boot_pair(a, bb2, seed=777 + int(lam * 100))
                print(f"          {pl} ΔSR {real:+.3f}  p={float((dist<=0).mean()):.3f}")

    # ---------- 逐年
    print("\n" + "=" * 110)
    print("4. 逐年（原始GEX λ=0.3 作用在 vwap 基线上）")
    print("=" * 110)
    s_, _ = scaled(base, Z, 0.3, 0.5)
    Y = pd.DataFrame({
        "未缩放": {y: sr(base.loc[f"{y}-01-01":f"{y}-12-31"]) for y in range(2019, 2027)},
        "GEX缩放": {y: sr(s_.loc[f"{y}-01-01":f"{y}-12-31"]) for y in range(2019, 2027)},
    }).T
    print(Y.round(3).to_string())

    pd.DataFrame(rows).to_csv(OUT / "gex_scaling_raw.csv", index=False)
    print(f"\n已写入 {OUT}/gex_scaling_raw.csv")


if __name__ == "__main__":
    main()
