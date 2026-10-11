"""止损改动到底显著吗：配对区块自助法（paired block bootstrap）。

为什么必须做这一步：dev 期约 1008 个交易日、Sharpe 0.665，
Sharpe 的标准误约 sqrt(252/N)·sqrt(1+SR²/2) ≈ 0.55。
也就是说 dev 上 0.665 和 0.960 的差距（+0.295）**远在一个标准误之内**。
直接比 Sharpe 大小会把噪声当发现 —— 这正是本项目反复踩的坑。

正确的做法是**配对**检验：两个配置在同一批交易日上运行，
对**每日收益之差**做区块自助法（区块长度 10 天，保留自相关），
得到 ΔSharpe 的分布和 p 值。配对能消掉共同的市场波动，功效远高于独立比较。
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

OUT = Path("outputs/full/intraday")
DEV, TEST = ("2019", "2022"), ("2023", "2026")


def sr(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.mean() / x.std() * np.sqrt(252) if len(x) > 30 and x.std() > 0 else np.nan


def block_boot_pair(a, b, nb=4000, L=10, seed=0):
    """配对区块自助：对 (a,b) 同步重采样区块，返回 ΔSharpe 的分布。"""
    idx = a.index.intersection(b.index)
    A, B = a.reindex(idx).fillna(0).to_numpy(), b.reindex(idx).fillna(0).to_numpy()
    n = len(A); nblk = int(np.ceil(n / L))
    rg = np.random.default_rng(seed)
    out = np.empty(nb)
    for i in range(nb):
        starts = rg.integers(0, max(n - L, 1), size=nblk)
        sel = np.concatenate([np.arange(s, min(s + L, n)) for s in starts])[:n]
        out[i] = sr(A[sel]) - sr(B[sel])
    return sr(A) - sr(B), out


def main():
    cur = pd.read_pickle(OUT / "stop_design_curves.pkl")
    base_key = "vwap（当前）"
    base = cur[base_key]

    cands = [
        ("tail k=0.25", "对称：只切尾部 k=0.25（单笔最多亏 0.5% 资金）"),
        ("tail k=0.5", "对称：只切尾部 k=0.5（单笔最多亏 1% 资金）"),
        ("tail k=1.0", "对称：只切尾部 k=1.0"),
        ("band", "对称：穿带止损"),
        ("none", "对称：不止损"),
        ("giveback g=0.6", "对称：浮盈回吐 60% 止盈"),
        ("多tail/空vwap", "不对称：多头只切尾部、空头保持 vwap（dev 最优）"),
        ("多band/空vwap", "不对称：多头穿带、空头保持 vwap"),
        ("多vwap/空band", "不对称：多头 vwap、空头穿带"),
        ("多band/空band", "不对称：两侧穿带"),
    ]

    print("=" * 116)
    print(f"配对区块自助法（区块 10 天，4000 次）：各配置 vs 基线「{base_key}」")
    print("=" * 116)
    print(f"{'配置':34s} | {'dev ΔSR':>9} {'95%区间':>20} {'p':>7} | "
          f"{'test ΔSR':>9} {'95%区间':>20} {'p':>7}")
    rows = []
    for key, desc in cands:
        if key not in cur:
            print(f"  {key} 不在曲线文件里，跳过"); continue
        s = cur[key]
        line = f"{key:34s} |"
        rec = dict(配置=key, 说明=desc)
        for per, pl in ((DEV, "dev"), (TEST, "test")):
            a = s.loc[per[0]:per[1]].dropna()
            bb = base.loc[per[0]:per[1]].dropna()
            real, dist = block_boot_pair(a, bb, seed=hash(key) % 10000)
            lo, hi = np.percentile(dist, [2.5, 97.5])
            p = float((dist <= 0).mean())           # 单边：改进 <= 0 的概率
            line += f" {real:+9.3f} [{lo:+7.3f},{hi:+7.3f}] {p:7.3f} |"
            rec[f"{pl}_dSR"] = real; rec[f"{pl}_lo"] = lo
            rec[f"{pl}_hi"] = hi; rec[f"{pl}_p"] = p
        print(line, flush=True)
        rows.append(rec)

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "stop_significance.csv", index=False)

    print("\n" + "=" * 116)
    print("解读：p 是「改进 ≤ 0」的单边概率。95% 区间跨 0 就说明分不出好坏。")
    print("=" * 116)
    sig_dev = R[(R.dev_p < 0.05)]
    sig_both = R[(R.dev_p < 0.10) & (R.test_p < 0.10)]
    print(f"  dev 上单边 p<0.05 的配置：{list(sig_dev.配置) or '（无）'}")
    print(f"  dev 和 test 都 p<0.10 的配置：{list(sig_both.配置) or '（无）'}")

    # Sharpe 本身的标准误（说明为什么必须配对检验）
    for per, pl in ((DEV, "dev"), (TEST, "test")):
        x = base.loc[per[0]:per[1]].dropna()
        n = len(x); s0 = sr(x)
        se = np.sqrt(252 / n) * np.sqrt(1 + s0 ** 2 / 2)
        print(f"  参考：{pl} 期 {n} 天，基线 Sharpe {s0:.3f}，"
              f"**非配对**标准误约 {se:.3f} —— 所以直接比大小毫无意义")

    # 逐笔均值的检验（观测数更多，功效更高）
    print("\n" + "=" * 116)
    print("补充：逐笔收益均值的差异检验（观测数 1100~1500，比日频 Sharpe 功效高）")
    print("=" * 116)
    import importlib
    RI = importlib.import_module("audit.reimpl_noise")
    from etft.intraday.bars import load_bars
    b = load_bars("QQQ")

    def trades(**kw):
        _, F = RI.simulate(b, sizing="voltarget", max_lev=4.0, reentry="hold",
                           half_spread=0.005, commission=0.0045, **kw)
        F = F.copy(); F["day"] = pd.to_datetime(F.day)
        F["cmin"] = F.clock.map(lambda s: int(str(s)[:2]) * 60 + int(str(s)[3:5]))
        tr = []
        for d, g in F.groupby("day", sort=True):
            g = g.sort_values("cmin").reset_index(drop=True); op = None
            for _, row in g.iterrows():
                if row.kind.startswith("ENTRY"):
                    op = row
                elif op is not None:
                    sd_ = 1 if op.shares > 0 else -1
                    tr.append(sd_ * (row.price / op.price - 1)); op = None
        return np.array(tr)

    t0 = trades(stop_long="vwap", stop_short="vwap")
    print(f"  基线 vwap：{len(t0)} 笔，均值 {t0.mean()*1e4:+.2f}bp，"
          f"标准误 {t0.std()/np.sqrt(len(t0))*1e4:.2f}bp")
    for kw, nm in ((dict(stop_long="tail", stop_short="tail", tail_k=0.25), "tail k=0.25"),
                   (dict(stop_long="tail", stop_short="tail", tail_k=0.5), "tail k=0.5"),
                   (dict(stop_long="band", stop_short="band"), "band")):
        t1 = trades(**kw)
        d = t1.mean() - t0.mean()
        se = np.sqrt(t1.var() / len(t1) + t0.var() / len(t0))
        print(f"  {nm:12s}：{len(t1)} 笔，均值 {t1.mean()*1e4:+.2f}bp，"
              f"Δ {d*1e4:+.2f}bp，标准误 {se*1e4:.2f}bp，**t={d/se:+.2f}**")
    print("\n  注：两组交易不是同一批（止损变了，交易就变了），所以这是非配对检验，")
    print("      只能说明'均值差异是否超出抽样噪声'，不能当成配对证据。")


if __name__ == "__main__":
    main()
